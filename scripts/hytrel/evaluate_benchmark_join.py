import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import logging
import tempfile  # For temporary file creation
import random    # For sampling queries
import pandas as pd
from faiss_search_join import FaissSearcher

# Minimal logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure parent directory is on sys.path.
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# --- Helper functions for CSV reading and delimiter detection ---
def determine_delimiter(filepath):
    with open(filepath, 'r', encoding='utf8', errors='ignore') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        elif '\t' in first_line:
            return '\t'
        else:
            return ','

import pandas as pd
from loguru import logger

def robust_get_df(file_path, expected_column=None, chunksize=100000, sample_nrows=100, size_threshold=100*1024*1024):
    """
    Read a CSV file using a detected delimiter. If expected_column is provided and not found,
    try alternative common delimiters. If no attempt yields the expected column, return None.
    
    For very large CSV files (>= size_threshold bytes), the file is first sampled to check for the 
    expected column, and then loaded in chunks to reduce memory usage.
    """
    file_size = os.path.getsize(file_path)
    use_chunks = file_size >= size_threshold
    detected = determine_delimiter(file_path)
    candidates = [detected] + [d for d in [',', ';', '\t'] if d != detected]
    used = set()
    last_df = None

    for delim in candidates:
        if delim in used:
            continue
        used.add(delim)
        try:
            if use_chunks:
                df_sample = pd.read_csv(file_path, delimiter=delim, nrows=sample_nrows,
                                        on_bad_lines='skip', engine='python')
                if expected_column is None or expected_column in df_sample.columns:
                    chunks = []
                    for chunk in pd.read_csv(file_path, delimiter=delim, chunksize=chunksize,
                                             on_bad_lines='skip', engine='python'):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = df_sample
            else:
                df = pd.read_csv(file_path, delimiter=delim, on_bad_lines='skip', engine='python')
        except Exception as e:
            logger.debug(f"Failed reading {file_path} with delimiter '{delim}': {e}")
            continue

        if expected_column is None or expected_column in df.columns:
            if expected_column is not None and delim != detected:
                logger.info(f"Recovered column '{expected_column}' using alternative delimiter '{delim}' in {file_path}")
            return df

        last_df = df

    if expected_column is not None:
        logger.error(f"Could not read expected column '{expected_column}' from {file_path}. Skipping table.")
        return None
    return last_df

# --- LSH Ensemble Helper Functions for Re-ranking ---
from datasketch import MinHashLSHEnsemble, MinHash

def find_table_file(benchmark, table_filename):
    """
    Look for table_filename under:
      data/{benchmark}/datalake/ and data/{benchmark}/query/
    Returns the first found full path or raises FileNotFoundError.
    """
    benchmark_dir = Path(f"data/{benchmark}")
    candidates = [
        benchmark_dir / "datalake" / table_filename,
        benchmark_dir / "query" / table_filename
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Could not find {table_filename} in {benchmark_dir / 'datalake'} or {benchmark_dir / 'query'}")

# --- Caching for Candidate MinHash Signatures ---
# This cache stores a mapping from candidate table identifier (tuple: (table_filename, column_name))
# to its precomputed (MinHash object, candidate_set).
candidate_minhash_cache = {}

def get_candidate_signature(benchmark, table, num_perm):
    """
    For the given candidate table (a tuple of (table_filename, column_name)), compute and return
    its MinHash signature and candidate set. Cache the result to avoid recomputation.
    """
    if table in candidate_minhash_cache:
        return candidate_minhash_cache[table]
    try:
        table_file = find_table_file(benchmark, table[0])
    except FileNotFoundError:
        return None
    c_df = robust_get_df(table_file, expected_column=table[1])
    if c_df is None:
        logger.warning(f"Skipping candidate table {table[0]} due to CSV reading issues.")
        return None
    try:
        candidate_set = set(c_df[table[1]].dropna().astype(str).tolist())
    except Exception as e:
        logger.debug(f"Error processing candidate column '{table[1]}' in {table[0]}: {e}")
        return None
    if len(candidate_set) == 0:
        logger.warning(f"Empty cell set for candidate column '{table[1]}' in {table[0]}")
        return None
    m2 = MinHash(num_perm=num_perm)
    for cell in candidate_set:
        m2.update(cell.encode('utf8'))
    candidate_minhash_cache[table] = (m2, candidate_set)
    return candidate_minhash_cache[table]

def lsh_ensemble_all_tables(benchmark, candidate_tables, query_table, query_column,
                            num_perm=128, threshold=0.8, num_part=32):
    try:
        query_file = find_table_file(benchmark, query_table)
    except FileNotFoundError as e:
        logger.warning(f"Query file not found for {query_table}: {e}")
        return [], {}
    q_df = robust_get_df(query_file, expected_column=query_column)
    if q_df is None:
        logger.warning(f"Skipping query table {query_table} due to CSV reading issues.")
        return [], {}
    query_set = set(q_df[query_column].dropna().astype(str).tolist())
    if len(query_set) == 0:
        logger.warning(f"Empty cell set for query column '{query_column}' in {query_table}")
        return [], {}
    m1 = MinHash(num_perm=num_perm)
    for cell in query_set:
        m1.update(cell.encode('utf8'))
    
    # Deduplicate candidate tables to avoid duplicate keys.
    unique_candidates = []
    seen = set()
    for table in candidate_tables:
        if table not in seen:
            unique_candidates.append(table)
            seen.add(table)
    
    lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
    list_indices = []
    containment_dict = {}
    for table in unique_candidates:
        sig = get_candidate_signature(benchmark, table, num_perm)
        if sig is None:
            continue
        m2, candidate_set = sig
        list_indices.append((table, m2, len(candidate_set)))
        containment = len(query_set.intersection(candidate_set)) / len(query_set)
        containment_dict[table] = containment
    
    if not list_indices:
        return [], containment_dict
    lshensemble.index(list_indices)
    refined = list(lshensemble.query(m1, len(query_set)))
    if refined is None:
        refined = []
    return refined, containment_dict

def rerank_candidates(query, candidates, benchmark, num_perm=128, threshold=0.8, num_part=32, final_K=10):
    query_table = query["table_name"] if query["table_name"].endswith(".csv") else query["table_name"] + ".csv"
    query_column = query["column_name"]
    refined, containment_dict = lsh_ensemble_all_tables(benchmark, candidates, query_table, query_column,
                                                          num_perm=num_perm,
                                                          threshold=threshold,
                                                          num_part=num_part)
    if not refined:
        return candidates
    refined_sorted = sorted(refined, key=lambda x: containment_dict.get(x, 0), reverse=True)
    return refined_sorted[:final_K]

# --- End LSH Ensemble Helpers ---

def setup_directories(benchmark, model_type):
    output_dir = Path(f"output/{benchmark}/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_embeddings(benchmark, model_type, groundtruth):
    base_path = Path(f"vectors/hytrel/{model_type}/{benchmark}")
    query_file = base_path / "query_vectors.pkl"
    datalake_file = base_path / "datalake_vectors.pkl"
    if not datalake_file.exists():
        logger.error("Datalake embedding file not found.")
        return None, None
    if query_file.exists():
        with open(query_file, "rb") as f:
            queries = pickle.load(f)
        logger.info(f"Loaded {len(queries)} query columns from query file.")
    else:
        with open(datalake_file, "rb") as f:
            all_datalake_embeddings = pickle.load(f)
        queries = []
        for entry in all_datalake_embeddings:
            normalized_table = entry["table_name"] if entry["table_name"].endswith(".csv") else entry["table_name"] + ".csv"
            query_key = (normalized_table, entry["column_name"])
            if query_key in groundtruth:
                queries.append(entry)
        logger.warning("Query embedding file not found. Falling back to datalake embeddings for ground truth queries.")
        logger.info(f"Filtered {len(queries)} query columns from datalake file.")
    return queries, None

def load_benchmark(benchmark, benchmark_file=None):
    if benchmark_file is None:
        gt_path = Path(f"data/{benchmark}/benchmark.pkl")
    else:
        gt_path = Path(benchmark_file)
    if not gt_path.exists():
        logger.error(f"Benchmark file not found at {gt_path}.")
        return None
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    logger.info(f"Loaded ground truth with {len(gt)} entries from {gt_path}.")
    return gt

def calc_metrics(max_k, resultFile, groundtruth):
    system_precision = np.zeros(max_k)
    system_recall    = np.zeros(max_k)
    system_ap        = np.zeros(max_k)
    f1_lists = [[] for _ in range(max_k)]
    per_query_metrics = {}
    
    for query_key, retrieved in resultFile.items():
        if query_key not in groundtruth:
            continue
        gt_set = set(groundtruth[query_key])
        total_relevant = len(gt_set)
        query_metrics = {
            'candidates': retrieved,
            'ground_truth': list(gt_set),
            'precision': [],
            'recall': [],
            'f1': [],
            'ap': []
        }
        for k in range(1, max_k + 1):
            retrieved_k = retrieved[:k]
            num_relevant = sum(1 for item in retrieved_k if item in gt_set)
            precision = num_relevant / k
            recall = num_relevant / total_relevant if total_relevant > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            ap = 0.0
            relevant_count = 0
            for i, item in enumerate(retrieved_k, start=1):
                if item in gt_set:
                    relevant_count += 1
                    ap += relevant_count / i
            norm = min(total_relevant, k)
            ap = ap / norm if norm > 0 else 0.0
            query_metrics['precision'].append(precision)
            query_metrics['recall'].append(recall)
            query_metrics['f1'].append(f1)
            query_metrics['ap'].append(ap)
            system_precision[k - 1] += precision
            system_recall[k - 1]    += recall
            system_ap[k - 1]        += ap
            f1_lists[k - 1].append(f1)
        per_query_metrics[query_key] = query_metrics

    num_queries = len(per_query_metrics)
    if num_queries > 0:
        system_precision /= num_queries
        system_recall    /= num_queries
        system_ap        /= num_queries
        system_f1 = np.array([np.mean(f1_list) if f1_list else 0.0 for f1_list in f1_lists])
    else:
        system_f1 = np.zeros(max_k)
    
    used_k = list(range(1, max_k + 1))
    metrics_at_k = {
        k: {
            'precision': float(system_precision[k - 1]),
            'recall':    float(system_recall[k - 1]),
            'f1':        float(system_f1[k - 1]),
            'map':       float(system_ap[k - 1])
        }
        for k in used_k
    }
    
    return {
        'system_metrics': {
            'precision': system_precision.tolist(),
            'recall':    system_recall.tolist(),
            'f1':        system_f1.tolist(),
            'map':       system_ap.tolist(),
            'used_k':    used_k,
            'metrics_at_k': metrics_at_k
        },
        'per_query_metrics': per_query_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate joinable column search with optional LSH re-ranking.")
    parser.add_argument("benchmark", help="Benchmark name (folder in data/)")
    parser.add_argument("--benchmark_file", type=str, default=None,
                        help="Path to benchmark file. Defaults to data/{benchmark}/benchmark.pkl if not specified.")
    # Removed --model_type argument. Model type is always "pretrained".
    parser.add_argument("--K", type=int, default=10, help="Final top K candidates to use per query")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for datalake columns (for FAISS). Adjusts the magnitude of embeddings.")
    parser.add_argument("--eval_fraction", type=float, default=1.0,
                        help="Fraction of evaluation entries to use (default=1.0, i.e. all).")
    parser.add_argument("--seed", type=int, default=94, help="Random seed for evaluation sampling")
    parser.add_argument("--rerank", action="store_true", help="Enable LSH Ensemble re-ranking")
    parser.add_argument("--rerank_factor", type=int, default=3,
                        help="Factor to multiply K for initial candidate retrieval when re-ranking")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations for MinHash")
    parser.add_argument("--threshold", type=float, default=0.95, help="LSH Ensemble threshold")
    parser.add_argument("--num_part", type=int, default=32, help="Number of partitions for LSH Ensemble")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="Optional output file name for metrics. If not specified, a default name is used.")
    args = parser.parse_args()

    random.seed(args.seed)

    # Model type is always "pretrained"
    model_type = "pretrained"
    output_dir = setup_directories(args.benchmark, model_type)
    groundtruth = load_benchmark(args.benchmark, benchmark_file=args.benchmark_file)
    if groundtruth is None:
        logger.error("Failed to load benchmark ground truth.")
        return

    queries, _ = load_embeddings(args.benchmark, model_type, groundtruth)
    if queries is None:
        logger.error("Failed to load query embeddings.")
        return

    filtered_queries = []
    for q in queries:
        normalized_table = q["table_name"] if q["table_name"].endswith(".csv") else q["table_name"] + ".csv"
        q_key = (normalized_table, q["column_name"])
        if q_key in groundtruth:
            filtered_queries.append(q)
    logger.info(f"Evaluating {len(filtered_queries)} queries (filtered from {len(queries)}).")

    if args.eval_fraction < 1.0:
        sample_size = int(len(filtered_queries) * args.eval_fraction)
        filtered_queries = random.sample(filtered_queries, sample_size)
        logger.info(f"Sampling {sample_size} queries for evaluation (fraction={args.eval_fraction}).")

    # --- Build FAISS Index from Precomputed Embeddings ---
    base_vectors_path = Path(f"vectors/hytrel/{model_type}/{args.benchmark}")
    datalake_file = base_vectors_path / "datalake_vectors.pkl"
    index_file = base_vectors_path / "faiss_index.bin"
    column_ids_file = base_vectors_path / "column_ids.pkl"

    with open(datalake_file, "rb") as f:
        all_datalake_embeddings = pickle.load(f)
    filtered_datalake_embeddings = []
    for entry in all_datalake_embeddings:
        normalized_table = entry["table_name"] if entry["table_name"].endswith(".csv") else entry["table_name"] + ".csv"
        candidate_key = (normalized_table, entry["column_name"])
        if candidate_key in groundtruth:
            filtered_datalake_embeddings.append(entry)
    logger.info(f"Filtered FAISS candidate set to {len(filtered_datalake_embeddings)} entries.")

    # Always rebuild the index, regardless of whether one exists.
    faiss_searcher = FaissSearcher(columns=filtered_datalake_embeddings, scale=args.scale)
    faiss_searcher.save_index(str(index_file), str(column_ids_file))

    # --- Retrieve candidates for each query ---
    resultFile = {}
    for q in tqdm(filtered_queries, desc="Retrieving candidates"):
        normalized_table = q["table_name"] if q["table_name"].endswith(".csv") else q["table_name"] + ".csv"
        q_key = (normalized_table, q["column_name"])

        if args.rerank:
            try:
                query_file = find_table_file(args.benchmark, normalized_table)
            except FileNotFoundError as e:
                logger.warning(f"Skipping query {normalized_table} {q['column_name']}: {e}")
                continue
            q_df = robust_get_df(query_file, expected_column=q["column_name"])
            if q_df is None:
                logger.warning(f"Skipping query {normalized_table} {q['column_name']} due to CSV issues.")
                continue

        if args.rerank:
            initial_k = args.K * args.rerank_factor
            retrieved, _ = faiss_searcher.topk(q, initial_k)
            candidates = [cand for score, cand in retrieved]
            refined = rerank_candidates(q, candidates, args.benchmark,
                                        num_perm=args.num_perm,
                                        threshold=args.threshold,
                                        num_part=args.num_part,
                                        final_K=args.K)
            final_candidates = refined
        else:
            retrieved, _ = faiss_searcher.topk(q, args.K)
            final_candidates = [cand for score, cand in retrieved]

        # Ensure self-match is included.
        self_candidate = (normalized_table, q["column_name"])
        if self_candidate not in final_candidates:
            final_candidates.insert(0, self_candidate)
            final_candidates = final_candidates[:args.K]
        resultFile[q_key] = final_candidates

    metrics = calc_metrics(args.K, resultFile, groundtruth)
    converted = {f"{key[0]}||{key[1]}": value for key, value in metrics['per_query_metrics'].items()}
    metrics['per_query_metrics'] = converted

    if args.metrics_file:
        metrics_path = output_dir / args.metrics_file
    else:
        filename = f"join_metrics_faiss_K{args.K}_seed{args.seed}"
        if args.rerank:
            filename += f"_rerank_rf{args.rerank_factor}_thr{args.threshold}_np{args.num_part}"
        filename += ".json"
        metrics_path = output_dir / filename

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
