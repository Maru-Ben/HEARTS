import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Add tqdm for progress bars
from tqdm import tqdm

# Assuming sdd is in PYTHONPATH or adjust imports accordingly
from sdd.pretrain import load_checkpoint, inference_on_tables
from sdd.dataset import PretrainTableDataset # Used by load_checkpoint

def get_all_tables(data_dir_path):
    all_tables = {}
    for filename in os.listdir(data_dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir_path, filename)
            try:
                df = pd.read_csv(file_path, lineterminator='\n')
                all_tables[filename] = df
            except Exception as e:
                print(f"Warning: Skipping problematic file {filename}: {str(e)}")
    return all_tables

def extract_all_column_embeddings(model, unlabeled_dataset, tables_dict, batch_size=32):
    all_column_data = []
    # Ensure model is on the correct device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    table_names = list(tables_dict.keys())
    dataframes = [tables_dict[name] for name in table_names]

    # Process tables in batches for inference_on_tables
    # inference_on_tables expects a list of DataFrames
    # Note: The original inference_on_tables might need adjustments or careful batching
    # For simplicity, let's process one by one if batching is complex to adapt here
    # Or, if inference_on_tables handles list of tables efficiently:
    
    # This part needs to align with how `inference_on_tables` expects input and gives output.
    # The `inference_on_tables` from sdd.pretrain.py processes a list of tables.
    # Its output `results` is a list, where each element corresponds to a table,
    # and contains a list of column embeddings for that table.

    # We'll iterate and call inference_on_tables for each table to simplify mapping
    # This might be less efficient than batching tables if inference_on_tables supports it well for this purpose
    print("Extracting column embeddings...")
    for table_name, df in tqdm(tables_dict.items(), desc="Processing tables for embeddings"):
        if df.empty or len(df.columns) == 0:
            print(f"Skipping empty or column-less table: {table_name}")
            continue
        
        # `inference_on_tables` returns a list of results, one per table in the input list.
        # Since we pass one table, we expect a list containing one list of column embeddings.
        # Example: [[col1_emb, col2_emb, ...]] for the input table df
        column_embeddings_list_of_lists = inference_on_tables(
            tables=[df], 
            model=model, 
            unlabeled=unlabeled_dataset, 
            batch_size=1 # Batch size for columns within the table processing by model
        )

        if not column_embeddings_list_of_lists or not column_embeddings_list_of_lists[0]:
            print(f"Warning: No embeddings returned for table {table_name}")
            continue
            
        column_embeddings_for_table = column_embeddings_list_of_lists[0]

        if len(df.columns) != len(column_embeddings_for_table):
            print(f"Warning: Mismatch columns ({len(df.columns)}) vs embeddings ({len(column_embeddings_for_table)}) for {table_name}. Skipping.")
            # This can happen if hp.table_order is not 'column' or tokenization issues.
            continue

        for i, col_name in enumerate(df.columns):
            all_column_data.append({
                "table_name": table_name,
                "column_name": col_name,
                "embedding": np.array(column_embeddings_for_table[i])
            })
    return all_column_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find semantically similar columns using Starmie.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Starmie model (.pt file).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV tables.")
    parser.add_argument("--target_column_name", type=str, required=True, help="Name of the column to find similarities for.")
    parser.add_argument("--target_table_name", type=str, default=None, help="(Optional) Name of the table containing the target column.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of similar columns to return.")
    
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    # Ensure CUDA is available if model was trained on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model, unlabeled_dataset = load_checkpoint(ckpt) # unlabeled_dataset is the PretrainTableDataset instance
    
    # Critical: Check table_order. This logic assumes 'column' order for direct mapping.
    if hasattr(model, 'hp') and model.hp.table_order != 'column':
        print(f"Warning: Model's table_order is '{model.hp.table_order}'. Embedding-to-column name mapping might be incorrect.")
        print("This script assumes 'column' order for direct mapping of embeddings to df.columns.")

    print(f"Loading tables from {args.data_dir}...")
    all_tables_dict = get_all_tables(args.data_dir)
    if not all_tables_dict:
        print("No tables found or loaded. Exiting.")
        exit()

    all_column_data = extract_all_column_embeddings(model, unlabeled_dataset, all_tables_dict)
    
    if not all_column_data:
        print("No column embeddings extracted. Exiting.")
        exit()

    target_embeddings = []
    for col_data in all_column_data:
        if col_data["column_name"] == args.target_column_name:
            if args.target_table_name is None or col_data["table_name"] == args.target_table_name:
                target_embeddings.append(col_data["embedding"])
    
    if not target_embeddings:
        print(f"Target column '{args.target_column_name}' (table: {args.target_table_name or 'any'}) not found or no embedding. Exiting.")
        exit()
    
    # For simplicity, using the first found target embedding.
    # Could average if multiple are found and no specific table is given.
    target_embedding = target_embeddings[0].reshape(1, -1) 

    print(f"\nFinding columns similar to '{args.target_column_name}':")
    
    similarities = []
    for col_data in all_column_data:
        # Avoid comparing the column with itself if it's the exact same instance
        is_self = col_data["column_name"] == args.target_column_name and \
                    (args.target_table_name is None or col_data["table_name"] == args.target_table_name) and \
                    np.array_equal(col_data["embedding"], target_embedding.flatten())
        if is_self and len(target_embeddings) == 1: # Only skip if it's the unique target
            continue

        sim = cosine_similarity(target_embedding, col_data["embedding"].reshape(1, -1))[0][0]
        similarities.append({
            "table_name": col_data["table_name"],
            "column_name": col_data["column_name"],
            "similarity": sim
        })
        
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    for i, item in enumerate(similarities[:args.top_n]):
        print(f"{i+1}. Table: {item['table_name']}, Column: {item['column_name']}, Similarity: {item['similarity']:.4f}")

    # Add tqdm for progress bars
    from tqdm import tqdm