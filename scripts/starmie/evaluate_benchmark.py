import os
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
import time
import re

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import only the allowed searchers
from naive_search import NaiveSearcher
from lsh_search import LSHSearcher
from hnsw_search import HNSWSearcher
from faiss_search import FaissSearcher
from cluster_search import ClusterSearcher
from checkPrecisionRecall import calcMetrics, loadDictionaryFromPickleFile

def normalize_table_name(table_name):
    # Split the string on both forward and back slashes and return the last element.
    return re.split(r'[\\/]', table_name)[-1]

def setup_directories(benchmark): 
    """Create output directory structure for the benchmark"""
    base_dir = Path(f"output/{benchmark}")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def load_embeddings(benchmark, ao=None):
    """Load query and datalake embeddings for a benchmark from the starmie directory"""
    if ao:
        base_path = Path(f"vectors/starmie/{benchmark}/{ao}")
    else:
        base_path = Path(f"vectors/starmie/{benchmark}")

    # Query embeddings (only one set, always from original)
    query_vectors_path = base_path / "query_vectors.pkl"
    queries = None
    if query_vectors_path.exists():
        queries = loadDictionaryFromPickleFile(query_vectors_path)
        queries.sort(key=lambda x: x[0])

    # Check for variants
    datalake_embeddings = {}
    original_path = base_path / "datalake_vectors.pkl"
    if original_path.exists():
        datalake_embeddings["original"] = loadDictionaryFromPickleFile(original_path)

    pcol_path = base_path / "datalake_vectors_p-col.pkl"
    if pcol_path.exists():
        datalake_embeddings["p-col"] = loadDictionaryFromPickleFile(pcol_path)

    return queries, datalake_embeddings

def load_table_structure(table_path):
    """Load CSV table to get column names and order"""
    try:
        df = pd.read_csv(table_path)
        return list(df.columns)
    except Exception:
        return None

def load_serializations(benchmark, ao=None):
    """Load serialized strings for original and p-col variants"""
    if ao:
        base_path = Path(f"vectors/starmie/{benchmark}/{ao}")
    else:
        base_path = Path(f"vectors/starmie/{benchmark}")
    
    serializations = {}
    orig_path = base_path / "datalake_vectors_serialized.pkl"
    if orig_path.exists():
        with open(orig_path, 'rb') as f:
            serializations["original"] = pickle.load(f)
            
    pcol_path = base_path / "datalake_vectors_p-col_serialized.pkl"
    if pcol_path.exists():
        with open(pcol_path, 'rb') as f:
            serializations["p-col"] = pickle.load(f)
            
    return serializations

def calculate_detailed_similarity_metrics(original_embeddings, variant_embeddings, data_dir, variant, benchmark, ao=None):
    """Calculate column-level similarity metrics between original and variant embeddings"""
    detailed_metrics = {"tables": []}
    serializations = load_serializations(benchmark, ao)

    orig_indices = {normalize_table_name(x[0]): i for i, x in enumerate(original_embeddings)}
    var_indices  = {normalize_table_name(x[0]): i for i, x in enumerate(variant_embeddings)}
    
    for table_id in orig_indices:
        if table_id not in var_indices:
            continue
            
        table_base = table_id  # table_id is assumed to be like "311_calls_historic_data_0.csv"
        orig_path = Path("data") / benchmark / "datalake" / table_base
        var_path = None
        if variant == "original":
            var_path = orig_path
        elif variant == "p-col":
            var_path = Path("data") / benchmark / "datalake-p-col" / table_base

        if var_path is None or not var_path.exists():
            continue

        orig_columns = load_table_structure(orig_path)
        var_columns = load_table_structure(var_path)
        if not orig_columns or not var_columns:
            continue

        orig_embeddings_table = original_embeddings[orig_indices[table_id]][1]
        var_embeddings_table = variant_embeddings[var_indices[table_id]][1]
        if len(orig_embeddings_table) != len(var_embeddings_table):
            continue

        cos_sim = cosine_similarity(orig_embeddings_table, var_embeddings_table)
        column_similarities = []
        euclidean_distances = []
        cosine_similarities = []
        
        for col_name in orig_columns:
            if col_name in var_columns:
                orig_idx = orig_columns.index(col_name)
                var_idx = var_columns.index(col_name)
                orig_emb = orig_embeddings_table[orig_idx]
                var_emb = var_embeddings_table[var_idx]
                cos_sim_val = float(cos_sim[orig_idx][var_idx])
                euc_dist = float(np.linalg.norm(orig_emb - var_emb))
                column_similarities.append({
                    "column_name": col_name,
                    "original_position": orig_idx,
                    "permuted_position": var_idx,
                    "euclidean_distance": euc_dist,
                    "cosine_similarity": cos_sim_val
                })
                euclidean_distances.append(euc_dist)
                cosine_similarities.append(cos_sim_val)
        
        if column_similarities:
            table_metrics = {
                "table_name": normalize_table_name(table_id),
                "num_columns": len(orig_columns),
                "column_similarities": column_similarities,
                "aggregate_metrics": {
                    "mean_euclidean": float(np.mean(euclidean_distances)),
                    "mean_cosine": float(np.mean(cosine_similarities))
                },
                "serialization": {
                    "original_serialization": serializations.get("original", {}).get(table_id, ""),
                    "permuted_serialization": serializations.get("p-col", {}).get(table_id, "")
                }
            }
            detailed_metrics["tables"].append(table_metrics)
    
    return detailed_metrics

def instantiate_searcher(searcher_type, table_path, scale=1.0, pooling='mean'):
    """Create appropriate searcher instance with allowed types:
       naive, bounds, hnsw, lsh, faiss, cluster.
       If searcher type is 'faiss', the pooling parameter is passed.
    """
    if searcher_type in ['naive', 'bounds']:
        return NaiveSearcher(table_path, scale)
    elif searcher_type == 'lsh':
        hash_func_num = 16
        hash_table_num = 100
        return LSHSearcher(table_path, hash_func_num, hash_table_num, scale)
    elif searcher_type == 'hnsw':
        index_path = table_path + "_hnsw.index"
        return HNSWSearcher(table_path, index_path, scale)
    elif searcher_type == 'faiss':
        return FaissSearcher(table_path, scale=scale, pooling=pooling)
    elif searcher_type == 'cluster':
        return ClusterSearcher(table_path, scale=scale)
    else:
        raise ValueError("Unknown searcher type: " + searcher_type)

def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark with specified searcher, and compute similarity metrics")
    parser.add_argument("benchmark", 
                        choices=['santos', 'tus', 'tusLarge', 'pylon', 'wiki_union'],
                        help="Benchmark to evaluate")
    parser.add_argument("--searcher_type",
                        choices=['naive', 'bounds', 'hnsw', 'lsh', 'faiss', 'cluster'],
                        default='bounds',
                        help="Type of searcher to use")
    parser.add_argument("--distances_only",
                        action="store_true",
                        help="Only recalculate raw distances without redoing evaluation")
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        help="Optional: Override default data directory path")
    parser.add_argument("--ao", type=str, choices=['default', 'shuffle_col'], default='default',
                        help="Augmentation option to evaluate")
    parser.add_argument("--pooling",
                        choices=['mean', 'max', 'None'],
                        default='None',
                        help="Pooling method for FAISS searcher. Use 'None' for no aggregation (column-level).")
    args = parser.parse_args()

    pooling_value = None if args.pooling == "None" else args.pooling

    if args.data_dir:
        data_path = Path(args.data_dir)
    else:
        data_path = Path(f"data/{args.benchmark}")

    params = {
        'santos': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'pylon': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tus': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 150, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tusLarge': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 100, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'wiki_union': {
            'max_k': 10,
            'k_range': 1,
            'sample_size': 100,
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        }
    }[args.benchmark]

    if args.ao == 'default':
        if args.benchmark in ['santos', 'pylon', 'wiki_union']:
            ao = 'drop_col'
        elif args.benchmark in ['tus', 'tusLarge']:
            ao = 'drop_cell'
    else:
        ao = args.ao

    output_dir = setup_directories(args.benchmark) / ao
    output_dir.mkdir(parents=True, exist_ok=True)
    base_vectors_path = Path(f"vectors/starmie/{args.benchmark}/{ao}")
    gt_path = data_path / "benchmark.pkl"

    # Load embeddings with augmentation option
    queries, datalake_embeddings = load_embeddings(args.benchmark, ao)
    variants = list(datalake_embeddings.keys())
    
    if "original" in variants and "p-col" in variants:
        original_datalake = datalake_embeddings["original"]
        variant_datalake = datalake_embeddings["p-col"]
        detailed_metrics = calculate_detailed_similarity_metrics(
            original_datalake, 
            variant_datalake,
            data_path,
            "p-col",
            args.benchmark,
            ao
        )
        distances_file = f'starmie_distances_{ao}.json'
        with open(output_dir / distances_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"Distances computed and saved to {distances_file}")

    if args.distances_only:
        return

    if queries is None:
        print("No query embeddings found. Cannot proceed with evaluation.")
        return

    if params['sample_size'] is not None and params['sample_size'] < len(queries):
        np.random.seed(42)
        indices = np.random.choice(len(queries), size=params['sample_size'], replace=False)
        queries = [queries[i] for i in indices]

    # Evaluate each variant and save metrics
    for variant in variants:
        temp_path = base_vectors_path / ("datalake_vectors.pkl" if variant == "original" else f"datalake_vectors_{variant}.pkl")
        if not temp_path.exists():
            print(f"Warning: {temp_path} does not exist, but embeddings were loaded. Skipping {variant}.")
            continue

        # Start overall timing (index build + query search)
        variant_start_time = time.time()

        if args.searcher_type == 'faiss':
            searcher = instantiate_searcher(args.searcher_type, str(temp_path), scale=params['scale'], pooling=pooling_value)
        else:
            searcher = instantiate_searcher(args.searcher_type, str(temp_path), scale=params['scale'])

        returnedResults = {}
        total_query_time = 0.0
        num_queries = len(queries)
        for query in tqdm(queries, desc=f"Processing queries ({variant})", unit="query"):
            query_start = time.time()
            if args.searcher_type == 'bounds':
                search_results = searcher.topk_bounds(
                    enc=params['encoder'],
                    query=query,
                    K=params['max_k'], 
                    threshold=params['threshold']
                )
            else:
                search_results, _ = searcher.topk(
                    enc=params['encoder'],
                    query=query,
                    K=params['max_k'],
                    # N=50,
                    threshold=params['threshold']
                )
            query_end = time.time()
            total_query_time += (query_end - query_start)
            returnedResults[normalize_table_name(query[0])] = [normalize_table_name(r[1]) for r in search_results]

        variant_end_time = time.time()
        overall_variant_time = variant_end_time - variant_start_time
        avg_query_time = total_query_time / num_queries if num_queries > 0 else 0.0
        print(f"Variant '{variant}' using '{args.searcher_type}': Avg query time: {avg_query_time:.8f} sec; Overall time: {overall_variant_time:.8f} sec")

        metrics = calcMetrics(
            max_k=params['max_k'],
            k_range=params['k_range'],
            resultFile=returnedResults,
            gtPath=gt_path,
            record=False,
            verbose=False
        )

        # Add timing details into the metrics
        metrics["search_time"] = {
            "total_time": float("{:.8f}".format(overall_variant_time)),
            "avg_query_time": float("{:.8f}".format(avg_query_time))
        }

        if args.searcher_type == 'faiss':
            pooling_label = pooling_value if pooling_value is not None else "col"
            metrics_filename = f"starmie_metrics_{variant}_{args.searcher_type}_{ao}_{pooling_label}.json"
        else:
            metrics_filename = f"starmie_metrics_{variant}_{args.searcher_type}_{ao}.json"
        
        with open(output_dir / metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_filename}")

    print(f"Evaluation and distance calculations completed successfully for {ao}")

if __name__ == "__main__":
    main()
