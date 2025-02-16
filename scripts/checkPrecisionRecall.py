# import pickle
# import pandas as pd
# from matplotlib import *
# from matplotlib import pyplot as plt
# import numpy as np
# import mlflow

# def loadDictionaryFromPickleFile(dictionaryPath):
#     ''' Load the pickle file as a dictionary
#     Args:
#         dictionaryPath: path to the pickle file
#     Return: dictionary from the pickle file
#     '''
#     filePointer=open(dictionaryPath, 'rb')
#     dictionary = pickle.load(filePointer)
#     filePointer.close()
#     return dictionary

# def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
#     ''' Save dictionary as a pickle file
#     Args:
#         dictionary to be saved
#         dictionaryPath: filepath to which the dictionary will be saved
#     '''
#     filePointer=open(dictionaryPath, 'wb')
#     pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
#     filePointer.close()


# def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True, verbose=False):
#     '''Calculate and log the performance metrics, both system-wide and per-query
#     Args:
#         max_k: maximum K value (10 for SANTOS, 60 for TUS)
#         k_range: step size for K values
#         resultFile: dictionary containing search results
#         gtPath: path to groundtruth pickle file
#         resPath: (deprecated) path to results file
#         record: whether to log to MLFlow
#         verbose: whether to print intermediate results
#     Returns:
#         Dictionary containing both system-wide metrics and per-query metrics
#     '''
#     groundtruth = loadDictionaryFromPickleFile(gtPath)
    
#     # Initialize system-wide metrics
#     system_precision = np.zeros(max_k)
#     system_recall = np.zeros(max_k)
#     system_map = np.zeros(max_k)
#     system_f1 = np.zeros(max_k)
    
#     # Initialize per-query results
#     per_query_metrics = {}
    
#     # Process each query
#     for query_id, results in resultFile.items():
#         if query_id not in groundtruth:
#             continue
            
#         query_metrics = {
#             'candidates': results,  # Store retrieved results
#             'ground_truth': groundtruth[query_id],  # Store ground truth
#             'precision': [],
#             'recall': [],
#             'ap': []  # Average precision at each k
#         }
        
#         gt_set = set(groundtruth[query_id])
        
#         # Calculate metrics at each k for this query
#         for k in range(1, max_k + 1):
#             result_set = set(results[:k])
#             intersect = result_set.intersection(gt_set)
            
#             # Calculate precision and recall for this k
#             precision = len(intersect) / k if k > 0 else 0
#             recall = len(intersect) / len(gt_set) if len(gt_set) > 0 else 0
            
#             # Store metrics for this query
#             query_metrics['precision'].append(precision)
#             query_metrics['recall'].append(recall)
            
#             # Add to system-wide metrics
#             system_precision[k-1] += precision
#             system_recall[k-1] += recall
            
#             # Calculate AP up to this k
#             ap_k = sum(query_metrics['precision'][:k]) / k
#             query_metrics['ap'].append(ap_k)
        
#         per_query_metrics[query_id] = query_metrics
    
#     # Calculate system-wide averages
#     num_queries = len(per_query_metrics)
#     if num_queries > 0:
#         system_precision /= num_queries
#         system_recall /= num_queries
#         system_map = np.mean([metrics['ap'] for metrics in per_query_metrics.values()], axis=0)
        
#         # Calculate F1 scores
#         system_f1 = 2 * (system_precision * system_recall) / (system_precision + system_recall)
#         system_f1 = np.nan_to_num(system_f1)  # Replace NaN with 0
    
#     # Get k values used for evaluation
#     used_k = [k_range]
#     if max_k > k_range:
#         for i in range(k_range * 2, max_k+1, k_range):
#             used_k.append(i)
    
#     # Store system-wide metrics at specific k points
#     metrics_at_k = {}
#     for k in used_k:
#         metrics_at_k[k] = {
#             'precision': float(system_precision[k-1]),
#             'recall': float(system_recall[k-1]),
#             'map': float(system_map[k-1]),
#             'f1': float(system_f1[k-1])
#         }
    
#     if record:
#         mlflow.log_metric("mean_avg_precision", system_map[-1])
#         mlflow.log_metric("prec_k", system_precision[-1])
#         mlflow.log_metric("recall_k", system_recall[-1])
#         mlflow.log_metric("f1_k", system_f1[-1])
    
#     return {
#         'system_metrics': {
#             'precision': system_precision.tolist(),
#             'recall': system_recall.tolist(),
#             'map': system_map.tolist(),
#             'f1': system_f1.tolist(),
#             'used_k': used_k,
#             'metrics_at_k': metrics_at_k
#         },
#         'per_query_metrics': per_query_metrics
#     } 

import pickle
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

def loadDictionaryFromPickleFile(dictionaryPath):
    """Load the pickle file as a dictionary."""
    with open(dictionaryPath, 'rb') as fp:
        return pickle.load(fp)

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    """Save dictionary as a pickle file."""
    with open(dictionaryPath, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)

def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True, verbose=False):
    """
    Calculate and log the performance metrics (system-wide and per-query) 
    using scikit-learn metrics instead of manual averaging.
    
    Args:
        max_k: maximum cutoff (e.g. 10 for SANTOS, 60 for TUS)
        k_range: step size for reporting k values
        resultFile: dict mapping query_id -> list of retrieved candidates (in rank order)
        gtPath: path to ground truth pickle file (dictionary mapping query_id to ground truth set)
        resPath: (deprecated) path to results file
        record: whether to log metrics to MLflow
        verbose: whether to print intermediate results
    Returns:
        Dictionary with system-wide metrics and per-query metrics.
    """
    # Load ground truth dictionary
    groundtruth = loadDictionaryFromPickleFile(gtPath)
    
    # Initialize arrays for system-wide metrics for each k from 1 to max_k.
    system_precision = np.zeros(max_k)
    system_recall    = np.zeros(max_k)
    system_map       = np.zeros(max_k)
    system_f1      = np.zeros(max_k)
    
    per_query_metrics = {}
    
    # Process each query in the results
    for query_id, results in resultFile.items():
        if query_id not in groundtruth:
            continue
        
        # Convert ground truth list to a set for faster lookup.
        gt_set = set(groundtruth[query_id])
        
        # Initialize per-query metric storage.
        query_metrics = {
            'candidates': results,      # Retrieved list for this query
            'ground_truth': list(gt_set),
            'precision': [],
            'recall': [],
            'f1': [],
            'ap': []   # Average precision computed at each k
        }
        
        # For each cutoff k from 1 to max_k:
        for k in range(1, max_k + 1):
            # Build a binary vector indicating whether each of the top k candidates is relevant.
            y_true = np.array([1 if candidate in gt_set else 0 for candidate in results[:k]])
            # Since by design we treat the entire top-k as “predicted relevant,” the predicted labels are all ones.
            y_pred = np.ones_like(y_true)
            
            # Compute precision, recall, and f1 using scikit-learn. (Note that with y_pred all ones,
            # precision becomes (number of relevant items in top k)/k and recall becomes (number of relevant items in top k)/(total relevant).)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            
            # To compute average precision (AP) at k using scikit-learn, we need a score for each candidate.
            # Here we simply assign a descending score (highest for rank 1).
            scores = np.arange(k, 0, -1)
            # average_precision_score computes AP over the provided ordering.
            ap = average_precision_score(y_true, scores) if np.sum(y_true) > 0 else 0.0
            
            # Store per-query metrics.
            query_metrics['precision'].append(prec)
            query_metrics['recall'].append(rec)
            query_metrics['f1'].append(f1)
            query_metrics['ap'].append(ap)
            
            # Accumulate for system-wide metrics.
            system_precision[k-1] += prec
            system_recall[k-1]    += rec
            system_f1[k-1]      += f1
            system_map[k-1]       += ap
        
        per_query_metrics[query_id] = query_metrics
    
    # Average system-wide metrics over all queries.
    num_queries = len(per_query_metrics)
    if num_queries > 0:
        system_precision /= num_queries
        system_recall    /= num_queries
        system_f1      /= num_queries
        system_map       /= num_queries
    
    # Determine the k cutoffs at which we want to report metrics.
    used_k = list(range(k_range, max_k+1, k_range))
    metrics_at_k = {}
    for k in used_k:
        metrics_at_k[k] = {
            'precision': float(system_precision[k-1]),
            'recall':    float(system_recall[k-1]),
            'map':       float(system_map[k-1]),
            'f1':        float(system_f1[k-1])
        }
    
    # Optionally log final k-th metrics to MLflow.
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