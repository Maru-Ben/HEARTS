#!/usr/bin/env python
"""
run_finetune.py

A fine-tuning script for HyTREL. This script loads a pre-trained checkpoint (from a 
directory structure similar to run_pretrain.py), sets up the data folder for fine-tuning,
and then trains the model on a (typically smaller) dataset.
"""

import os
import sys
import time
import json
import psutil
import shutil
from datetime import datetime
from collections import OrderedDict

import torch
torch.set_float32_matmul_precision('high')

import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

import transformers
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from dataclasses import dataclass, field, fields
from typing import Optional

# Import model and data modules
from model import Encoder, ContrastiveLoss
from data import TableDataModule

try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

################################################################################
#                             ARGUMENT CLASSES                                 #
################################################################################

@dataclass
class DataArguments:
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={"help": "Tokenizer configuration (e.g., bert-base-uncased)"}
    )
    data_path: str = field(
        default='./data/santos/',
        metadata={"help": "Path to the fine-tuning dataset"}
    )
    max_token_length: int = field(
        default=128,
        metadata={"help": "Maximum token length after tokenization"}
    )
    max_row_length: int = field(
        default=30,
        metadata={"help": "Maximum number of rows per table"}
    )
    max_column_length: int = field(
        default=20,
        metadata={"help": "Maximum number of columns per table"}
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"}
    )
    valid_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_epoch: int = field(
        default=10,
        metadata={"help": "Maximum number of fine-tuning epochs"}
    )
    electra: bool = field(
        default=False,
        metadata={"help": "Whether to use the ELECTRA objective"}
    )
    mask_ratio: float = field(
        default=0.15,
        metadata={"help": "Masking ratio for training"}
    )
    contrast_bipartite_edge: bool = field(
        default=True,
        metadata={"help": "Whether to use contrastive bipartite edge objective"}
    )
    bipartite_edge_corrupt_ratio: float = field(
        default=0.3,
        metadata={"help": "Corruption ratio for bipartite edges"}
    )
    checkpoint_dir: str = field(
        default='checkpoints_finetune',
        metadata={"help": "Directory to save fine-tuning checkpoints"}
    )

@dataclass
class OptimizerConfig:
    batch_size: int = field(
        default=16,
        metadata={"help": "Training batch size"}
    )
    base_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Base learning rate for fine-tuning"}
    )
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: transformers.SchedulerType = "linear"
    warmup_step_ratio: float = 0.05
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int = field(
        default=1,
        metadata={"help": "Save checkpoint every N epochs"}
    )
    save_top_k: int = field(
        default=3,
        metadata={"help": "Number of best checkpoints to keep"}
    )
    # Use the same argument name as in run_pretrain.py for consistency.
    checkpoint_path: str = field(
        default="checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt",
        metadata={"help": "Path to pre-trained HyTREL checkpoint directory (e.g., ...epoch=4-step=32690.ckpt)"}
    )

    @classmethod
    def dict(cls):
        return {field.name: getattr(cls, field.name) for field in fields(cls)}

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {"adam": AdamW if self.adam_w_mode else Adam}[optimizer]
        kwargs = {"lr": learning_rate, "eps": self.adam_epsilon, "betas": (self.adam_beta1, self.adam_beta2)}
        return optim_cls(optim_groups, **kwargs)

################################################################################
#                           MODEL DEFINITION (PLModule)                          #
################################################################################

import os
import time
import json
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
from transformers.optimization import get_scheduler

import psutil

try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

from model import Encoder, ContrastiveLoss  # Make sure these are correctly imported

class PlModel(pl.LightningModule):
    def __init__(self, model_config, optimizer_cfg):
        super().__init__()
        self.model = Encoder(model_config)
        self.model_config = model_config
        self.optimizer_cfg = optimizer_cfg
        self.save_hyperparameters()

        if self.model_config.electra:
            self.dense = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
            self.act = nn.GELU()
            self.dense_prediction = nn.Linear(self.model_config.hidden_size, 1)
            self.criterion = nn.BCEWithLogitsLoss()
            self.pre = BinaryPrecision(threshold=0.5)
            self.rec = BinaryRecall(threshold=0.5)
            self.f1 = BinaryF1Score(threshold=0.5)
            self.acc = BinaryAccuracy(threshold=0.5)
        elif self.model_config.contrast_bipartite_edge:
            self.con_loss = ContrastiveLoss(temperature=0.07)

        self.epoch_start_time = None
        self.epoch_times = []
        self.peak_gpu_memory = 0

    def on_save_checkpoint(self, checkpoint):
        """
        This hook post-processes the checkpoint's state_dict just before saving.
        It removes the unwanted 'le.model.' prefix from the keys.
        """
        new_state_dict = OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            if key.startswith("le.model."):
                new_key = key[len("le.model."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        checkpoint['state_dict'] = new_state_dict
        return checkpoint

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu_memory)
            if NVIDIA_SMI_AVAILABLE:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                self.log_dict({
                    'gpu_memory_used_mb': info.used / (1024**2),
                    'gpu_utilization': gpu_util.gpu
                })
        self.log('epoch_time_seconds', epoch_time)

    def on_train_end(self):
        stats = {
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_epoch_time_seconds': sum(self.epoch_times) / len(self.epoch_times),
            'total_training_time_seconds': sum(self.epoch_times),
            'peak_gpu_memory_mb': float(self.peak_gpu_memory),
            'num_epochs': len(self.epoch_times),
            'epoch_times': self.epoch_times,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_ram_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
        }
        stats_path = os.path.join(self.trainer.logger.log_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def training_step(self, batch, batch_idx):
        if self.model_config.electra:
            outputs = self.model(batch)
            cell_embeds = outputs[0]
            hyperedge_outputs = outputs[1]
            col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
            all_embeds = torch.cat([cell_embeds, col_embeds], axis=0)
            hidden_states = self.dense(all_embeds)
            hidden_states = self.act(hidden_states)
            logits = self.dense_prediction(hidden_states).view(-1)
            c_lbls = batch.electra_c
            h_lbls = batch.electra_h
            lbls = torch.cat([c_lbls, h_lbls])
            loss_pos = self.criterion(logits[lbls == 1.], lbls[lbls == 1.])
            loss_neg = self.criterion(logits[lbls == 0.], lbls[lbls == 0.])
            loss = loss_pos + loss_neg
        elif self.model_config.contrast_bipartite_edge:
            self.model_config.update({'edge_neg_view': 1})
            outputs1 = self.model(batch)
            hyperedge_outputs1 = outputs1[1]
            hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
            self.model_config.update({'edge_neg_view': 2})
            outputs2 = self.model(batch)
            hyperedge_outputs2 = outputs2[1]
            hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
            loss = self.con_loss(hyper_embeds1, hyper_embeds2)

        self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_config.electra:
            outputs = self.model(batch)
            cell_embeds = outputs[0]
            hyperedge_outputs = outputs[1]
            col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
            all_embeds = torch.cat([cell_embeds, col_embeds], axis=0)
            hidden_states = self.dense(all_embeds)
            hidden_states = self.act(hidden_states)
            logits = self.dense_prediction(hidden_states).view(-1)
            c_lbls = batch.electra_c
            h_lbls = batch.electra_h
            lbls = torch.cat([c_lbls, h_lbls])
            loss_pos = self.criterion(logits[lbls == 1.], lbls[lbls == 1.])
            loss_neg = self.criterion(logits[lbls == 0.], lbls[lbls == 0.])
            loss = loss_pos + loss_neg
            self.log("validation_loss", loss, prog_bar=True)
            return {"logits": logits, "labels": lbls}
        elif self.model_config.contrast_bipartite_edge:
            self.model_config.update({'edge_neg_view': 1})
            outputs1 = self.model(batch)
            hyperedge_outputs1 = outputs1[1]
            hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
            self.model_config.update({'edge_neg_view': 2})
            outputs2 = self.model(batch)
            hyperedge_outputs2 = outputs2[1]
            hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
            loss = self.con_loss(hyper_embeds1, hyper_embeds2)
            self.log("validation_loss", loss, prog_bar=True)
            return loss

    def validation_epoch_end(self, outputs):
        if self.model_config.electra:
            logits = torch.cat([out["logits"] for out in outputs], dim=0)
            labels = torch.cat([out["labels"] for out in outputs], dim=0).long()
            probs = torch.sigmoid(logits)
            precision = self.pre(probs, labels)
            recall = self.rec(probs, labels)
            f1_score = self.f1(probs, labels)
            acc = self.acc(probs, labels)
            self.log_dict({
                'val_f1': f1_score,
                'acc': acc,
                'val_precision': precision,
                'val_recall': recall
            }, prog_bar=True)

    def configure_optimizers(self):
        from dataclasses import asdict
        self.logger.log_hyperparams(asdict(self.optimizer_cfg))
        learning_rate = self.optimizer_cfg.base_learning_rate
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.optimizer_cfg.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = self.optimizer_cfg.get_optimizer(optim_groups, learning_rate)
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        num_warmup_steps = int(self.optimizer_cfg.warmup_step_ratio * num_training_steps)
        scheduler = get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": "validation_loss",
        }]


################################################################################
#                         CHECKPOINT FLATTENING UTILITY                          #
################################################################################

def flatten_deepspeed_checkpoint(root_dir):
    """
    Moves DeepSpeed checkpoint files out of nested directories.
    For example, moves the files from ...epoch=4-step=32690.ckpt/checkpoint/
    to ...epoch=4-step=32690.ckpt/best/ or similar.
    """
    import shutil
    # Handle best checkpoint
    best_ckpt_dir = os.path.join(root_dir, 'best.ckpt')
    if os.path.exists(best_ckpt_dir):
        checkpoint_subdir = os.path.join(best_ckpt_dir, 'checkpoint')
        if os.path.exists(checkpoint_subdir):
            best_dir = os.path.join(root_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)
            for f in os.listdir(checkpoint_subdir):
                shutil.move(os.path.join(checkpoint_subdir, f), best_dir)
            shutil.rmtree(best_ckpt_dir, ignore_errors=True)

    # Handle last checkpoint
    last_ckpt_dir = os.path.join(root_dir, 'last.ckpt')
    if os.path.exists(last_ckpt_dir):
        checkpoint_subdir = os.path.join(last_ckpt_dir, 'checkpoint')
        if os.path.exists(checkpoint_subdir):
            last_dir = os.path.join(root_dir, 'last')
            os.makedirs(last_dir, exist_ok=True)
            for f in os.listdir(checkpoint_subdir):
                shutil.move(os.path.join(checkpoint_subdir, f), last_dir)
            shutil.rmtree(last_ckpt_dir, ignore_errors=True)


def freeze_early_layers(model, freeze_embedding=True, freeze_encoder_layers=True, num_encoder_layers_to_freeze=None):
    """
    Freezes parts of the encoder.
    
    Parameters:
      model: the encoder model (instance of Encoder)
      freeze_embedding (bool): if True, freeze the embedding layer.
      freeze_encoder_layers (bool): if True, freeze encoder layers.
      num_encoder_layers_to_freeze (int or None): how many encoder layers to freeze.
          If None, freeze all but the last encoder layer.
    """
    if freeze_embedding:
        for param in model.embed_layer.parameters():
            param.requires_grad = False
        print("Embedding layer frozen.")
        
    if freeze_encoder_layers:
        total_layers = len(model.layer)
        if num_encoder_layers_to_freeze is None:
            # Freeze all but the last layer by default.
            num_encoder_layers_to_freeze = total_layers - 1
        for i, encoder_layer in enumerate(model.layer):
            if i < num_encoder_layers_to_freeze:
                for param in encoder_layer.parameters():
                    param.requires_grad = False
                print(f"Encoder layer {i} frozen.")
            else:
                print(f"Encoder layer {i} left unfrozen.")

################################################################################
#                                   MAIN                                       #
################################################################################

def main():
    # Set up logging.
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Parse command-line arguments.
    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)
    data_args, optimizer_cfg, trainer_args = parser.parse_args_into_dataclasses()

    # Delete the checkpoint directory if it exists.
    if os.path.exists(data_args.checkpoint_dir):
        shutil.rmtree(data_args.checkpoint_dir)

    # Create checkpoint directory if it doesn't exist.
    os.makedirs(data_args.checkpoint_dir, exist_ok=True)

    # Configure TensorBoard logger.
    tb_logger = TensorBoardLogger(
        save_dir=data_args.checkpoint_dir,
        name="finetune",
        default_hp_metric=True
    )

    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Optimizer config: {optimizer_cfg}")
    logger.info(f"Trainer arguments: {trainer_args}")

    if data_args.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    pl.utilities.seed.seed_everything(data_args.seed)

    # Set up the tokenizer and extend its vocabulary.
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    tokenizer.add_tokens(new_tokens)
    logger.info(f"Added new tokens: {new_tokens}")

    # Set up the model configuration.
    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({
        'vocab_size': len(tokenizer),
        "pre_norm": False,
        "activation_dropout": 0.1,
        "gated_proj": False,
        "electra": data_args.electra,
        "contrast_bipartite_edge": data_args.contrast_bipartite_edge
    })
    logger.info(f"Model config: {model_config}")

    # Set up the data module.
    data_module = TableDataModule(
        tokenizer=tokenizer,
        data_args=data_args,
        seed=data_args.seed,
        batch_size=optimizer_cfg.batch_size,
        py_logger=logger,
        objective='electra' if model_config.electra else 'contrast'
    )

    # Initialize the model module.
    if optimizer_cfg.checkpoint_path:
        logger.info(f"Loading checkpoint from {optimizer_cfg.checkpoint_path}")
        # If the checkpoint path is a directory, load the internal state file.
        if os.path.isdir(optimizer_cfg.checkpoint_path):
            ckpt_file = os.path.join(optimizer_cfg.checkpoint_path, "checkpoint", "mp_rank_00_model_states.pt")
        else:
            ckpt_file = optimizer_cfg.checkpoint_path

        state_dict = torch.load(ckpt_file,
                                map_location='cuda' if torch.cuda.is_available() else 'cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict['module'].items():
            if 'model' in k:
                name = k[13:]  # remove "module.model." prefix
                new_state_dict[name] = v

        model_module = PlModel(model_config, optimizer_cfg)
        model_module.model.load_state_dict(new_state_dict, strict=True)
        logger.info("Loaded pre-trained weights into the encoder.")
    else:
        model_module = PlModel(model_config, optimizer_cfg)

    freeze_early_layers(model_module.model, freeze_embedding=False, freeze_encoder_layers=True)

    # ================================================================
    # Run baseline validation before training starts to establish a baseline.
    # ================================================================
    logger.info("Running baseline validation to establish a baseline before training starts...")
    baseline_trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        strategy="deepspeed_stage_1",
        logger=tb_logger,
        precision='bf16',
        enable_checkpointing=False,  # Disable checkpointing during baseline validation
        max_epochs=0,                # No training; only run validation
    )
    baseline_results = baseline_trainer.validate(model_module, datamodule=data_module)
    logger.info(f"Baseline validation results: {baseline_results}")

    # Configure callbacks for checkpointing.
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=data_args.checkpoint_dir,
        filename='best',
        save_top_k=1,
        every_n_epochs=1,
        monitor="validation_loss" if data_args.contrast_bipartite_edge else "val_f1",
        mode="min" if data_args.contrast_bipartite_edge else "max",
        save_last=False,
    )

    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=data_args.checkpoint_dir,
        filename='last',
        save_top_k=0,
        every_n_epochs=1,
        save_last=True
    )

    callbacks = [
        best_checkpoint_callback,
        last_checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
    ]

    # Memory tracking callback.
    class MemoryTracker(pl.callbacks.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    callbacks.append(MemoryTracker())

    if trainer_args.gpus == -1:
        trainer_args.gpus = torch.cuda.device_count()

    assert trainer_args.replace_sampler_ddp == False, "replace_sampler_ddp must be False for correct data sampling"

    # Initialize the trainer.
    trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        strategy="deepspeed_stage_1",
        callbacks=callbacks,
        logger=tb_logger,
        max_epochs=data_args.max_epoch,
        precision='bf16',
        accumulate_grad_batches=getattr(trainer_args, 'accumulate_grad_batches', 1),
        gradient_clip_val=getattr(trainer_args, 'gradient_clip_val', None),
    )

    # Start fine-tuning.
    trainer.fit(model_module, data_module)

    # After training, flatten the DeepSpeed checkpoint structure if needed.
    flatten_deepspeed_checkpoint(data_args.checkpoint_dir)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
