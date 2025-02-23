import os
import torch
import pickle
import argparse
import pandas as pd
import json
import time
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
import re
from typing import List, Tuple

# Ignore warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoConfig
from model import Encoder
from data import BipartiteData, CAP_TAG, HEADER_TAG, ROW_TAG, MISSING_CAP_TAG, MISSING_CELL_TAG, MISSING_HEADER_TAG
from torch_geometric.data import Batch
from run_pretrain import OptimizerConfig, DataArguments


def extract_vectors(model, input_data):
    """Extract embeddings from the model"""
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        embeddings = model(input_data[0])
    duration = time.time() - start_time
    return embeddings, duration


def remove_special_characters(text):
    """Remove special characters from text"""
    return ''.join(char for char in text if ord(char) != 0x7f)


class EmbeddingGenerator:
    def __init__(self, checkpoint_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
        self.tokenizer.add_tokens(new_tokens)
        
        # Initialize model config
        model_config = AutoConfig.from_pretrained('bert-base-uncased')
        model_config.update({
            'vocab_size': len(self.tokenizer),
            "pre_norm": False,
            "activation_dropout": 0.1,
            "gated_proj": False,
            "contrast_bipartite_edge": True
        })
        
        # Initialize model
        self.model = Encoder(model_config)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_dir):
        """Load standard or DeepSpeed checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        try:
            state_dict = torch.load(checkpoint_dir)
            new_state_dict = OrderedDict()
            
            # Try DeepSpeed format first
            try:
                for k, v in state_dict['module'].items():
                    if k.startswith('_forward_module.model.'):
                        name = k[22:]
                    elif k.startswith('module.model.'):
                        name = k[13:]
                    elif k.startswith('model.'):
                        name = k[6:]
                    else:
                        name = k
                    new_state_dict[name] = v
                logger.info("Loaded DeepSpeed checkpoint")
            except KeyError:
                # Regular format
                for k, v in state_dict.items():
                    if k.startswith('_forward_module.model.'):
                        name = k[22:]
                    elif k.startswith('module.model.'):
                        name = k[13:]
                    elif k.startswith('model.'):
                        name = k[6:]
                    else:
                        name = k
                    new_state_dict[name] = v
                logger.info("Loaded regular checkpoint")
                        
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.success("Successfully loaded checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _tokenize_word(self, word: str) -> Tuple[List[str], List[int]]:
        number_pattern = re.compile(r"(\d+)\.?(\d*)")

        def number_repl(matchobj):
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                exponent = len(pre) - 1
            else:
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return f"{pre}{post.rstrip('0')} scinotexp {exponent}"

        def apply_scientific_notation(line):
            return re.sub(number_pattern, number_repl, line)

        word = apply_scientific_notation(word)
        wordpieces = self.tokenizer.tokenize(word)[:128]
        mask = [1] * len(wordpieces) + [0] * (128 - len(wordpieces))
        wordpieces += ['[PAD]'] * (128 - len(wordpieces))
        return wordpieces, mask

    def _table2graph_columns_only(self, df: pd.DataFrame, max_rows=50, max_cols=20) -> BipartiteData:
        max_token_length = 128
        pad_sequence = ['[PAD]'] * max_token_length
        default_mask = [0] * max_token_length
        
        default_cell_pieces = [MISSING_CELL_TAG] + pad_sequence[1:]
        default_cell_mask = [1] + default_mask[1:]
        default_row_pieces = ['[ROW]'] + pad_sequence[1:]
        default_row_mask = [1] + default_mask[1:]
        default_tab_pieces = [MISSING_CAP_TAG] + pad_sequence[1:]
        default_tab_mask = [1] + default_mask[1:]
        default_head_pieces = [MISSING_HEADER_TAG] + pad_sequence[1:]
        default_head_mask = [1] + default_mask[1:]

        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        if df.empty or len(df.columns) == 0:
            logger.warning("Skipping empty table")
            return None
        
        header = [str(col).strip() for col in df.columns]
        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []
        
        # Table token
        wordpieces_xt_all.append(default_tab_pieces)
        mask_xt_all.append(default_tab_mask)
        
        # Headers
        for head in header:
            if not head or pd.isna(head):
                wordpieces_xt_all.append(default_head_pieces)
                mask_xt_all.append(default_head_mask)
            else:
                w, m = self._tokenize_word(head)
                wordpieces_xt_all.append(w)
                mask_xt_all.append(m)
        
        # Rows
        for _ in range(min(len(df), max_rows)):
            wordpieces_xt_all.append(default_row_pieces)
            mask_xt_all.append(default_row_mask)
        
        for row_i, row in enumerate(df.itertuples()):
            if row_i >= max_rows:
                break
            for col_i, cell in enumerate(row[1:]):
                if col_i >= max_cols:
                    break
                
                if pd.isna(cell):
                    wordpieces_xs_all.append(default_cell_pieces)
                    mask_xs_all.append(default_cell_mask)
                else:
                    word = remove_special_characters(' '.join(str(cell).split()[:max_token_length]))
                    w, m = self._tokenize_word(word)
                    wordpieces_xs_all.append(w)
                    mask_xs_all.append(m)
                
                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.extend([
                    [node_id, 0],
                    [node_id, col_i + 1],
                    [node_id, row_i + len(header) + 1]
                ])
        
        xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
        xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        # Column mask
        col_mask = torch.zeros(len(wordpieces_xt_all), dtype=torch.long)
        col_mask[1:len(header)+1] = 1

        return BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, col_mask=col_mask)

    def extract_columns(self, outputs: torch.Tensor, num_cols: int) -> torch.Tensor:
        return outputs[1][1:num_cols+1]

    def process_directory(self, csv_dir, output_vectors_path, batch_size=4, max_rows=50, max_cols=20):
        logger.info(f"Processing directory: {csv_dir}")
        logger.info(f"Using max_rows={max_rows}, max_cols={max_cols}")
        os.makedirs(os.path.dirname(output_vectors_path), exist_ok=True)
        
        data_embeds = []
        inference_time = 0
        total_time = 0
        skipped_files = 0
        
        total_start = time.time()
        csv_files = list(Path(csv_dir).glob('*.csv'))
        
        for csv_file in tqdm(csv_files, desc=f"Processing {Path(csv_dir).name}"):
            try:
                df = pd.read_csv(csv_file)
                if len(df) == 0:
                    skipped_files += 1
                    continue
                
                graph = self._table2graph_columns_only(df, max_rows=max_rows, max_cols=max_cols)
                if graph is None:
                    skipped_files += 1
                    continue
                
                graph = graph.to(self.device)
                
                inference_start = time.time()
                with torch.no_grad():
                    outputs = self.model(graph)
                inference_time += time.time() - inference_start
                
                num_cols = len(df.columns) if len(df.columns) <= max_cols else max_cols
                col_embeddings = self.extract_columns(outputs, num_cols)
                
                data_embeds.append((csv_file.stem, col_embeddings.cpu().numpy()))
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                skipped_files += 1
                continue
        
        total_time = time.time() - total_start
        
        logger.info(f"Saving embeddings to {output_vectors_path}")
        with open(output_vectors_path, 'wb') as f:
            pickle.dump(data_embeds, f)
        
        stats = {
            "total_time": total_time,
            "inference_time": inference_time,
            "processed_tables": len(data_embeds),
            "skipped_files": skipped_files
        }
        
        return stats


def process_benchmark(benchmark, generator, args):
    logger.info(f"Processing benchmark: {benchmark}")
    
    # We'll store embeddings in: vectors/hytrel/pretrained/{benchmark}/
    base_vectors_path = os.path.join('vectors', 'hytrel', 'pretrained', benchmark)
    os.makedirs(base_vectors_path, exist_ok=True)

    # Handle both original and p-col variants if p-col exists
    variants = ['original']
    dataset_path = os.path.join('data', benchmark)
    datalake_path = os.path.join(dataset_path, 'datalake_hytrel')
    p_col_path = os.path.join(dataset_path, 'datalake_hytrel_p-col')
    query_path = os.path.join(dataset_path, 'query')

    if os.path.isdir(p_col_path):
        variants.append('p-col')

    all_stats = {}
    query_run_done = False

    for variant in variants:
        variant_suffix = f"_{variant}" if variant != 'original' else ''
        datalake_vectors_path = os.path.join(base_vectors_path, f'datalake_vectors{variant_suffix}.pkl')
        timing_path = os.path.join(base_vectors_path, f'timing_stats{variant_suffix}.json')

        timing_stats = {
            "benchmark": benchmark,
            "variant": variant,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": args.checkpoint_dir,
            "model_type": "pretrained"
        }

        if variant == 'original':
            current_datalake_path = datalake_path
        else:
            current_datalake_path = p_col_path

        logger.info(f"Processing datalake directory for {benchmark}/{variant}...")
        datalake_stats = generator.process_directory(current_datalake_path, datalake_vectors_path, 
                                                     max_rows=args.max_rows, max_cols=args.max_cols)
        
        timing_stats["datalake"] = datalake_stats

        # Only process queries once (for the original variant)
        if variant == 'original' and not query_run_done:
            query_vectors_path = os.path.join(base_vectors_path, f'query_vectors.pkl')
            logger.info(f"Processing query directory for {benchmark} (no variant, just once)...")
            query_stats = generator.process_directory(query_path, query_vectors_path, 
                                                      max_rows=args.max_rows, max_cols=args.max_cols)
            timing_stats["query"] = query_stats
            query_run_done = True
        else:
            logger.info(f"Skipping queries for {benchmark}/{variant} because queries are the same.")
        
        logger.info(f"Saving timing statistics to {timing_path}")
        with open(timing_path, 'w') as f:
            json.dump(timing_stats, f, indent=4)
        
        all_stats[variant] = timing_stats

    return all_stats

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for table columns using HyTrel')
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--max_rows', type=int, default=30)
    parser.add_argument('--max_cols', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    logger.info("Model type: pretrained")
    generator = EmbeddingGenerator(args.checkpoint_dir)
    all_stats = process_benchmark(args.benchmark, generator, args)

    # Save combined stats
    base_vectors_path = os.path.join('vectors', 'hytrel', 'pretrained', args.benchmark)
    combined_stats_path = os.path.join(base_vectors_path, 'combined_stats.json')
    with open(combined_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)

    logger.success("All benchmark processing completed successfully!")

if __name__ == '__main__':
    main()
