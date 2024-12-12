import os
import sys
import logging
import time
from datetime import datetime
import json
import psutil
try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

import torch
import torch.nn as nn
from torch.optim import Adam
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

import transformers
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from dataclasses import dataclass, field, fields
from typing import Optional

from model import Encoder, ContrastiveLoss
from data import TableDataModule

@dataclass
class DataArguments:
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "bert-base-cased, bert-base-uncased etc"
        },
    )
    data_path: str = field(default='./data/santos/', metadata={"help": "data path"})
    max_token_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input token length for cell/caption/header after tokenization."
        },
    )
    max_row_length: int = field(
        default=100,
        metadata={
            "help": "The maximum total input rows for a table"
        },
    )
    max_column_length: int = field(
        default=100,
        metadata={
            "help": "The maximum total input columns for a table"
        },
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"},
    )
    valid_ratio: float = field(
        default=0.01,
        metadata={"help": "Validation split ratio"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_epoch: int = field(
        default=5,
        metadata={"help": "Maximum number of training epochs"}
    )
    electra: bool = field(
        default=False,
        metadata={"help": "Whether to use ELECTRA objective"}
    )
    mask_ratio: float = field(
        default=0.15,
        metadata={"help": "Masking ratio for training"}
    )
    contrast_bipartite_edge: bool = field(
        default=False,
        metadata={"help": "Whether to use contrastive bipartite edge objective"}
    )
    bipartite_edge_corrupt_ratio: float = field(
        default=0.3,
        metadata={"help": "Corruption ratio for bipartite edges"}
    )
    checkpoint_dir: str = field(
        default='checkpoints',
        metadata={"help": "Directory to save checkpoints"}
    )

@dataclass
class OptimizerConfig:
    batch_size: int = field(
        default=128,
        metadata={"help": "Training batch size"}
    )
    base_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Base learning rate"}
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
    checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to pretrained checkpoint for finetuning"}
    )

    @classmethod
    def dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {
            "adam": AdamW if self.adam_w_mode else Adam,
        }[optimizer]

        kwargs = {
            "lr": learning_rate,
            "eps": self.adam_epsilon,
            "betas": (self.adam_beta1, self.adam_beta2),
        }
        optimizer = optim_cls(optim_groups, **kwargs)
        return optimizer


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

        # Add training time tracking
        self.epoch_start_time = None
        self.epoch_times = []
        self.peak_gpu_memory = 0
        
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
            self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu_memory)
            
            # Additional detailed GPU info if nvidia-smi is available
            if NVIDIA_SMI_AVAILABLE:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU 0
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                
                self.log_dict({
                    'gpu_memory_used_mb': info.used / (1024**2),
                    'gpu_utilization': gpu_util.gpu
                })
        
        self.log('epoch_time_seconds', epoch_time)

    def on_train_end(self):
        # Save training statistics
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
        
        # Save to checkpoint directory
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
            loss_pos = self.criterion(logits[lbls==1.], lbls[lbls==1.])
            loss_neg = self.criterion(logits[lbls==0.], lbls[lbls==0.])
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
            loss_pos = self.criterion(logits[lbls==1.], lbls[lbls==1.])
            loss_neg = self.criterion(logits[lbls==0.], lbls[lbls==0.])
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
        params_decay = [
            p for n, p in self.named_parameters() 
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters()
            if any(nd in n for nd in no_decay)
        ]
        
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

def flatten_deepspeed_checkpoint(root_dir):
    """
    This function moves DeepSpeed checkpoint files out of the nested .ckpt directories.
    It looks for 'best.ckpt' and 'last.ckpt', moves their contents directly into 'best/'
    and 'last/' directories, and removes the unnecessary nesting.
    """
    import shutil

    # Handle best checkpoint
    best_ckpt_dir = os.path.join(root_dir, 'best.ckpt')
    if os.path.exists(best_ckpt_dir):
        checkpoint_subdir = os.path.join(best_ckpt_dir, 'checkpoint')
        if os.path.exists(checkpoint_subdir):
            # Move all files from the checkpoint_subdir to root_dir/best
            best_dir = os.path.join(root_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)
            for f in os.listdir(checkpoint_subdir):
                shutil.move(os.path.join(checkpoint_subdir, f), best_dir)
            # Remove the entire best.ckpt directory
            shutil.rmtree(best_ckpt_dir, ignore_errors=True)

    # Handle last checkpoint
    last_ckpt_dir = os.path.join(root_dir, 'last.ckpt')
    if os.path.exists(last_ckpt_dir):
        checkpoint_subdir = os.path.join(last_ckpt_dir, 'checkpoint')
        if os.path.exists(checkpoint_subdir):
            # Move all files to root_dir/last
            last_dir = os.path.join(root_dir, 'last')
            os.makedirs(last_dir, exist_ok=True)
            for f in os.listdir(checkpoint_subdir):
                shutil.move(os.path.join(checkpoint_subdir, f), last_dir)
            # Remove the entire last.ckpt directory
            shutil.rmtree(last_ckpt_dir, ignore_errors=True)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)
    
    data_args, optimizer_cfg, trainer_args = parser.parse_args_into_dataclasses()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(data_args.checkpoint_dir, exist_ok=True)
    
    # Configure tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=data_args.checkpoint_dir,
        name="pretrain",
        default_hp_metric=True
    )

    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Optimizer config: {optimizer_cfg}")
    logger.info(f"Trainer arguments: {trainer_args}")

    if data_args.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    pl.utilities.seed.seed_everything(data_args.seed)

    # Set up tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    tokenizer.add_tokens(new_tokens)
    logger.info(f"Added new tokens: {new_tokens}")

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

    # Set up data module
    data_module = TableDataModule(
        tokenizer=tokenizer,
        data_args=data_args,
        seed=data_args.seed,
        batch_size=optimizer_cfg.batch_size,
        py_logger=logger,
        objective='electra' if model_config.electra else 'contrast'
    )

    if optimizer_cfg.checkpoint_path:
        logger.info(f"Loading checkpoint from {optimizer_cfg.checkpoint_path}")
        state_dict = torch.load(optimizer_cfg.checkpoint_path, 
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict['module'].items():
            if 'model' in k:
                name = k[13:]  # remove `module.model.`
                new_state_dict[name] = v
        
        model_module = PlModel(model_config, optimizer_cfg)
        model_module.model.load_state_dict(new_state_dict, strict=True)
        
        # Run initial validation
        logger.info("Running initial validation with loaded checkpoint...")
        trainer = pl.Trainer.from_argparse_args(
            trainer_args,
            strategy="deepspeed_stage_1",
            logger=tb_logger,
            precision='bf16',
            enable_checkpointing=False,  # Disable checkpointing for validation only
            max_epochs=0,  # No training, just validation
        )
        trainer.validate(model_module, datamodule=data_module)
    else:
        model_module = PlModel(model_config, optimizer_cfg)

    # Configure callbacks for best and last checkpoints
    # This will save best checkpoints under `best.ckpt` directory and last under `last.ckpt`
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

    class MemoryTracker(pl.callbacks.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
    
    callbacks.append(MemoryTracker())

    if trainer_args.gpus == -1:
        trainer_args.gpus = torch.cuda.device_count()
    
    assert trainer_args.replace_sampler_ddp == False, "replace_sampler_ddp must be False for correct data sampling"

    # Initialize trainer
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

    # Start training
    trainer.fit(model_module, data_module)

    # After training, flatten the DeepSpeed checkpoint structure
    flatten_deepspeed_checkpoint(data_args.checkpoint_dir)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
