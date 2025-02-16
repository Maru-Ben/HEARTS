import pickle
import pandas as pd
from matplotlib import *
from matplotlib import pyplot as plt
import numpy as np
import mlflow

def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary.
    
    Args:
        dictionaryPath: path to the pickle file
    Return:
        dictionary from the pickle file
    '''
    with open(dictionaryPath, 'rb') as filePointer:
        dictionary = pickle.load(filePointer)
    return dictionary

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    ''' Save dictionary as a pickle file.
    
    Args:
        dictionary: dictionary to be saved
        dictionaryPath: filepath to which the dictionary will be saved
    '''
    with open(dictionaryPath, 'wb') as filePointer:
        pickle.dump(dictionary, filePointer, protocol=pickle.HIGHEST_PROTOCOL)

def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True, verbose=False):
    '''Calculate and log the performance metrics, both system-wide and per-query.
    
    Args:
        max_k: maximum K value (10 for SANTOS, 60 for TUS)
        k_range: step size for K values
        resultFile: dictionary containing search results
        gtPath: path to groundtruth pickle file
        resPath: (deprecated) path to results file
        record: whether to log to MLFlow
        verbose: whether to print intermediate results
        
    Returns:
        Dictionary containing both system-wide metrics and per-query metrics.
    '''
    groundtruth = loadDictionaryFromPickleFile(gtPath)
    
    # Initialize system-wide metrics arrays
    system_precision = np.zeros(max_k)
    system_recall = np.zeros(max_k)
    system_map = np.zeros(max_k)
    
    # For system F1, we now store the per-query F1 values to average later.
    system_f1_list = [[] for _ in range(max_k)]
    
    # Initialize per-query results
    per_query_metrics = {}
    
    # Process each query
    for query_id, results in resultFile.items():
        if query_id not in groundtruth:
            continue
            
        query_metrics = {
            'candidates': results,              # Retrieved results
            'ground_truth': groundtruth[query_id],  # Ground truth for the query
            'precision': [],
            'recall': [],
            'ap': [],  # Average precision at each k
            'f1': []   # Per-query F1 at each k
        }
        
        gt_set = set(groundtruth[query_id])
        total_relevant = len(gt_set)
        
        # Calculate metrics at each k for this query
        for k in range(1, max_k + 1):
            current_results = results[:k]
            # Compute precision and recall for current k
            intersect = [item for item in current_results if item in gt_set]
            precision = len(intersect) / k if k > 0 else 0
            recall = len(intersect) / total_relevant if total_relevant > 0 else 0
            
            query_metrics['precision'].append(precision)
            query_metrics['recall'].append(recall)
            
            # Add to system-wide sums for precision and recall
            system_precision[k-1] += precision
            system_recall[k-1] += recall
            
            # Correct AP@k calculation:
            ap_k = 0.0
            relevant_count = 0
            for i, candidate in enumerate(current_results, start=1):
                if candidate in gt_set:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    ap_k += precision_at_i
            # Normalize by min(k, total_relevant)
            norm = min(total_relevant, k) if total_relevant > 0 else 0
            ap_k = ap_k / norm if norm > 0 else 0.0
            query_metrics['ap'].append(ap_k)
            
            # Calculate F1@k for the query: use the current precision and recall
            if (precision + recall) > 0:
                f1_k = 2 * precision * recall / (precision + recall)
            else:
                f1_k = 0.0
            query_metrics['f1'].append(f1_k)
            
            # Store this query's F1@k for system-level averaging later
            system_f1_list[k-1].append(f1_k)
        
        per_query_metrics[query_id] = query_metrics
    
    # Compute system-wide averages
    num_queries = len(per_query_metrics)
    if num_queries > 0:
        system_precision /= num_queries
        system_recall /= num_queries
        system_map = np.mean([metrics['ap'] for metrics in per_query_metrics.values()], axis=0)
        
        # Compute system F1@k by averaging per-query F1 values for each k.
        system_f1 = np.array([np.mean(f1_values) if f1_values else 0.0 for f1_values in system_f1_list])
    else:
        system_f1 = np.zeros(max_k)
    
    # Determine k values used for evaluation based on k_range
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k+1, k_range):
            used_k.append(i)
    
    # Store system-wide metrics at the specified k points
    metrics_at_k = {}
    for k in used_k:
        metrics_at_k[k] = {
            'precision': float(system_precision[k-1]),
            'recall': float(system_recall[k-1]),
            'map': float(system_map[k-1]),
            'f1': float(system_f1[k-1])
        }
    
    if record:
        mlflow.log_metric("mean_avg_precision", system_map[-1])
        mlflow.log_metric("prec_k", system_precision[-1])
        mlflow.log_metric("recall_k", system_recall[-1])
        mlflow.log_metric("f1_k", system_f1[-1])
    
    return {
        'system_metrics': {
            'precision': system_precision.tolist(),
            'recall': system_recall.tolist(),
            'map': system_map.tolist(),
            'f1': system_f1.tolist(),
            'used_k': used_k,
            'metrics_at_k': metrics_at_k
        },
        'per_query_metrics': per_query_metrics
    }