import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
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

def extract_all_column_embeddings(model, unlabeled_dataset, tables_dict, batch_size_for_inference=32):
    all_column_data = []
    # Ensure model is on the correct device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()

    print("Extracting column embeddings...")
    for table_name, df in tqdm(tables_dict.items(), desc="Processing tables for embeddings"):
        if df.empty or len(df.columns) == 0:
            print(f"Skipping empty or column-less table: {table_name}")
            continue
        
        column_embeddings_list_of_lists = inference_on_tables(
            tables=[df], 
            model=model, 
            unlabeled=unlabeled_dataset, 
            batch_size=batch_size_for_inference # Use the passed batch size for inference_on_tables
        )

        if not column_embeddings_list_of_lists or not column_embeddings_list_of_lists[0]:
            print(f"Warning: No embeddings returned for table {table_name}")
            continue
            
        column_embeddings_for_table = column_embeddings_list_of_lists[0]

        if len(df.columns) != len(column_embeddings_for_table):
            print(f"Warning: Mismatch columns ({len(df.columns)}) vs embeddings ({len(column_embeddings_for_table)}) for {table_name}. Skipping.")
            continue

        for i, col_name in enumerate(df.columns):
            all_column_data.append({
                "table_name": table_name,
                "column_name": col_name,
                "embedding": np.array(column_embeddings_for_table[i])
            })
    return all_column_data

def build_faiss_index(all_column_data_list):
    if not all_column_data_list:
        return None, None
    
    embeddings = np.array([col_data["embedding"] for col_data in all_column_data_list]).astype('float32')
    
    if embeddings.ndim == 1: # Should ideally not happen if there's more than one column globally
        if embeddings.shape[0] == 0: # No embeddings at all
             return None, None
        embeddings = embeddings.reshape(1, -1) # Reshape if it's a single embedding vector
    elif embeddings.shape[0] == 0: # No rows
        return None, None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # For cosine similarity (Inner Product on normalized vectors)
    
    # Normalize embeddings before adding to IndexFlatIP for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Create a mapping from FAISS index to original column info
    idx_to_info = {i: {"table_name": col_data["table_name"], "column_name": col_data["column_name"]} 
                   for i, col_data in enumerate(all_column_data_list)}
    return index, idx_to_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find semantically similar columns using Starmie and FAISS.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Starmie model (.pt file).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV tables.")
    parser.add_argument("--target_column_name", type=str, required=True, help="Name of the column to find similarities for.")
    parser.add_argument("--target_table_name", type=str, default=None, help="(Optional) Name of the table containing the target column.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of similar columns to return.")
    parser.add_argument("--embedding_batch_size", type=int, default=32, help="Batch size for inference_on_tables when extracting embeddings.")
    
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

    all_column_data = extract_all_column_embeddings(model, unlabeled_dataset, all_tables_dict, batch_size_for_inference=args.embedding_batch_size)
    
    if not all_column_data:
        print("No column embeddings extracted. Exiting.")
        exit()

    print("Building FAISS index...")
    faiss_index, idx_to_info_map = build_faiss_index(all_column_data)
    if faiss_index is None:
        print("Failed to build FAISS index (perhaps no embeddings found). Exiting.")
        exit()
        
    # Find the target column's embedding to use as a query
    query_embedding_vector = None
    query_column_info = None # To store info of the column used for query

    for col_data in all_column_data:
        if col_data["column_name"] == args.target_column_name:
            if args.target_table_name is None or col_data["table_name"] == args.target_table_name:
                # Use the first match as the query embedding
                query_embedding_vector = col_data["embedding"].astype('float32').reshape(1, -1)
                query_column_info = col_data
                break # Found our query vector source
    
    if query_embedding_vector is None:
        print(f"Target column '{args.target_column_name}' (table: {args.target_table_name or 'any'}) not found in the dataset. Exiting.")
        exit()
    
    # Normalize the query vector (same as done for index)
    faiss.normalize_L2(query_embedding_vector)

    print(f"\nFinding columns similar to '{query_column_info['table_name']}.{query_column_info['column_name']}':")
    
    # Search using FAISS
    # Retrieve top_n + 1 to allow filtering the query item itself if it appears
    num_to_retrieve = args.top_n + 1 
    
    distances, indices = faiss_index.search(query_embedding_vector, num_to_retrieve)
    
    results = []
    for i in range(indices.shape[1]): # Iterate through retrieved items
        faiss_idx = indices[0][i]
        if faiss_idx == -1: # Should not happen with IndexFlatIP unless k > N or index is empty
            continue
            
        retrieved_col_info = idx_to_info_map[faiss_idx]
        similarity_score = distances[0][i]

        # Check if the retrieved item is the exact same column instance used for the query
        is_query_self = (retrieved_col_info["table_name"] == query_column_info["table_name"] and
                         retrieved_col_info["column_name"] == query_column_info["column_name"])
        
        if is_query_self:
            continue # Skip the query column itself

        results.append({
            "table_name": retrieved_col_info["table_name"],
            "column_name": retrieved_col_info["column_name"],
            "similarity": similarity_score
        })

        if len(results) >= args.top_n:
            break # Collected enough results
            
    for i, item in enumerate(results):
        print(f"{i+1}. Table: {item['table_name']}, Column: {item['column_name']}, Similarity: {item['similarity']:.4f}")
    # Add tqdm for progress bars
    from tqdm import tqdm