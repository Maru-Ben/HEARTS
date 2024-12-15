from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import time
import sys
import argparse
from tqdm import tqdm
import os
import json
import transformers

transformers.logging.set_verbosity_error()

def extractVectors(dfs, model, trainset, batch_size=64, return_serialized=False):
    """Extract vectors from tables with optional serialization return.
    
    Args:
        dfs: List of (filename, DataFrame) tuples
        model: The Starmie model
        trainset: Training dataset
        batch_size: Batch size for processing
        return_serialized: Whether to return serialized strings
        
    Returns:
        If return_serialized=False:
            - dataEmbeds: List of (filename, embedding) tuples
            - timing_stats: Dictionary of timing statistics
        If return_serialized=True:
            - dataEmbeds: List of (filename, embedding) tuples
            - serialized_tables: Dictionary mapping filenames to serialized strings
            - timing_stats: Dictionary of timing statistics
    """
    num_tables = len(dfs)
    dataEmbeds = []
    serialized_tables = {}  # New dictionary for serialized strings
    
    # Initialize timing stats
    timing_stats = {
        "total_time": 0,
        "inference_time": 0,
        "preprocessing_time": 0,
        "postprocessing_time": 0,
        "num_tables": num_tables,
        "num_batches": (num_tables + batch_size - 1) // batch_size,
        "batch_times": []
    }
    
    start_time = time.time()
    for i in tqdm(range(0, num_tables, batch_size), desc="Processing tables"):
        batch_start_time = time.time()
        batch_dfs = dfs[i:i + batch_size]
        
        try:
            # Extract just the DataFrames for inference
            preprocess_start = time.time()
            inference_dfs = [df for _, df in batch_dfs]
            timing_stats["preprocessing_time"] += time.time() - preprocess_start
            
            # Extract model vectors and optionally get serialized strings
            inference_start = time.time()
            if return_serialized:
                cl_features, batch_serialized = inference_on_tables(
                    inference_dfs, model, trainset, 
                    batch_size=batch_size, 
                    return_serialized=True
                )
            else:
                cl_features = inference_on_tables(
                    inference_dfs, model, trainset, 
                    batch_size=batch_size
                )
            
            inference_time = time.time() - inference_start
            timing_stats["inference_time"] += inference_time

            # Store results with filenames
            postprocess_start = time.time()
            for j, (filename, _) in enumerate(batch_dfs):
                dataEmbeds.append((filename, np.array(cl_features[j])))
                if return_serialized:
                    serialized_tables[filename] = batch_serialized[j]
            timing_stats["postprocessing_time"] += time.time() - postprocess_start

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            for filename, df in batch_dfs:
                EMBEDDING_DIM = 768  # Adjust this to match your model's dimension
                zero_embeddings = np.zeros((len(df.columns), EMBEDDING_DIM))
                dataEmbeds.append((filename, zero_embeddings))
                if return_serialized:
                    serialized_tables[filename] = ""  # Empty string for failed serialization
            continue
        
        # Record batch timing
        batch_time = time.time() - batch_start_time
        timing_stats["batch_times"].append(batch_time)

    # Compute final timing stats
    timing_stats["total_time"] = time.time() - start_time
    timing_stats["average_batch_time"] = np.mean(timing_stats["batch_times"])
    timing_stats["average_time_per_table"] = timing_stats["total_time"] / num_tables
    timing_stats["throughput_tables_per_second"] = num_tables / timing_stats["total_time"]

    if return_serialized:
        return dataEmbeds, serialized_tables, timing_stats
    return dataEmbeds, timing_stats

def get_df(dataFolder):
    ''' Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''
    dataFiles = glob.glob(dataFolder + "/*.csv")
    dataDFs = {}
    for file in dataFiles:
        try:
            df = pd.read_csv(file, lineterminator='\n')
            if len(df) > 1000:
                df = df.head(1000)
            filename = file.split("/")[-1]
            dataDFs[filename] = (filename, df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Warning: Skipping problematic file {file}: {str(e)}")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error reading file {file}: {str(e)}")
            continue
    
    if not dataDFs:
        print(f"Warning: No valid CSV files were found in {dataFolder}")
    
    return dataDFs

def get_base_and_variant(benchmark):
    """Extract base benchmark name and variant from benchmark string."""
    parts = benchmark.split('-')
    base = parts[0]
    variant = '-'.join(parts[1:]) if len(parts) > 1 else None
    return base, variant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="santos")
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--return_serialized", dest="return_serialized", action="store_true", 
                       help="Whether to return and save serialized table strings")

    hp = parser.parse_args()
    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    base_benchmark, variant = get_base_and_variant(hp.benchmark)

    # Define augmentation options to process
    augmentation_options = ['shuffle_col']  # Start with shuffle_col
    
    # Add benchmark-specific ao based on the base benchmark
    if base_benchmark in ['santos', 'wdc', 'pylon', 'ugen_v1', 'ugen_v2']:
        benchmark_ao = 'drop_col' if not isSingleCol else 'drop_cell'
    elif base_benchmark in ['tus', 'tusLarge']:
        benchmark_ao = 'drop_cell'
    
    augmentation_options.append(benchmark_ao)

    # Process for each augmentation option
    for ao in augmentation_options:
        print(f"\n=== Processing with augmentation option: {ao} ===")
        
        # Set similarity model based on base benchmark
        if base_benchmark in ['santos', 'wdc', 'pylon', 'ugen_v1', 'ugen_v2']:
            sm = 'tfidf_entity'
        elif base_benchmark in ['tus', 'tusLarge']:
            sm = 'tfidf_entity' if base_benchmark == 'tusLarge' else 'alphaHead'

        run_id = hp.run_id
        table_order = hp.table_order

        # Modify model path to include augmentation option
        model_path = f"checkpoints/starmie/{base_benchmark}/model_{ao}_{sm}_{table_order}_{run_id}.pt"
        ckpt = torch.load(model_path, map_location=torch.device('cuda'))
        model, trainset = load_checkpoint(ckpt)

        # Setup data paths
        DATAPATH = f"data/{base_benchmark}/"
        
        # Process query data
        query_output_path = f"vectors/starmie/{base_benchmark}/query_vectors.pkl"
        if variant and os.path.exists(query_output_path):
            print("Query vectors already exist, skipping query processing")
            process_query = False
        else:
            process_query = True

        # Define data directories to process
        if process_query:
            dirs_to_process = ['query', 'datalake']
        else:
            dirs_to_process = ['datalake']

        # Adjust datalake path for variants
        if variant:
            DATAPATH = {
                'query': f"data/{base_benchmark}/query",
                'datalake': f"data/{base_benchmark}/datalake-{variant}"
            }
        
        total_inference_time = 0
        total_tables = 0
        overall_timing_stats = {}

        for dir in dirs_to_process:
            print(f"//==== Processing {dir}")
            
            # Get correct data folder
            DATAFOLDER = DATAPATH[dir] if isinstance(DATAPATH, dict) else os.path.join(DATAPATH, dir)
            
            dfs = get_df(DATAFOLDER)
            num_tables = len(dfs)
            total_tables += num_tables
            print("num dfs:", num_tables)

            if hp.save_model:
                if hp.return_serialized:
                    dataEmbeds, serialized_tables, timing_stats = extractVectors(
                        list(dfs.values()), model, trainset, 
                        batch_size=64, 
                        return_serialized=True
                    )
                else:
                    dataEmbeds, timing_stats = extractVectors(
                        list(dfs.values()), model, trainset, 
                        batch_size=64
                    )
                
                total_inference_time += timing_stats["inference_time"]

                # Define output paths
                output_dir = f"vectors/starmie/{base_benchmark}/{ao}/"  # Add ao to path
                os.makedirs(output_dir, exist_ok=True)

                if dir == 'query':
                    output_path = os.path.join(output_dir, "query_vectors.pkl")
                else:  # datalake
                    suffix = f"_{variant}" if variant else ""
                    output_path = os.path.join(output_dir, f"datalake_vectors{suffix}.pkl")

                pickle.dump(dataEmbeds, open(output_path, "wb"))
                
                if hp.return_serialized:
                    serialized_suffix = "_serialized"
                    serialized_path = output_path.replace(".pkl", f"{serialized_suffix}.pkl")
                    pickle.dump(serialized_tables, open(serialized_path, "wb"))

                overall_timing_stats[dir] = timing_stats

        overall_timing_stats["overall"] = {
            "total_tables": total_tables,
            "total_inference_time": total_inference_time,
            "avg_time_per_table": total_inference_time / total_tables if total_tables > 0 else 0
        }

        # Modify timing stats file path
        suffix = f"_{variant}" if variant else "_original"
        timing_file = f"vectors/starmie/{base_benchmark}/{ao}/timing_stats{suffix}.json"
        os.makedirs(os.path.dirname(timing_file), exist_ok=True)
        with open(timing_file, 'w') as f:
            json.dump(overall_timing_stats, f, indent=4)

        print(f"=== Completed processing with {ao} ===")
        print("Benchmark: ", dataFolder)
        print("--- Total Inference Time: %s seconds ---" % (total_inference_time))
        print(f"--- Average Time per Table: {total_inference_time / total_tables:.2f} seconds ---")

