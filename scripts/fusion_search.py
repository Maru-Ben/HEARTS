# fusion_search.py

import numpy as np
import pickle
import time
import random
import faiss
import hdbscan

class FusionSearcher(object):
    """
    FusionSearcher combines two complementary signals:
      1) Global similarity from aggregated (mean-pooled) column embeddings (indexed by FAISS),
      2) Local similarity from cluster memberships (obtained via HDBSCAN).
    
    For each table in the data lake, it computes:
      - An aggregated embedding (mean pooling over column embeddings, normalized),
      - A set of cluster labels derived from clustering all individual column embeddings.
    
    At query time, given a query table:
      - Its aggregated embedding is computed and used to retrieve candidate tables from the FAISS index.
      - Its individual columns are assigned cluster labels via HDBSCAN’s approximate_predict to form a query cluster set.
      - For each candidate table, a cluster similarity score is computed as:
            score_cluster = |(query clusters ∩ candidate clusters)| / |(query clusters ∪ candidate clusters)|
      - The final score is computed as a weighted sum:
            fused_score = weight_faiss * (FAISS score) + weight_cluster * (cluster score)
    The method returns the top-K candidate tables sorted by the fused score.
    """
    def __init__(self, table_path, scale=1.0, weight_faiss=0.5, weight_cluster=0.5, min_cluster_size=2):
        start_time = time.time()
        with open(table_path, "rb") as f:
            tables = pickle.load(f)
        if scale < 1.0:
            tables = random.sample(tables, int(scale * len(tables)))
        self.tables = tables
        
        # Build aggregated embeddings for each table (max pooling over columns)
        self.agg_embeddings = []
        self.table_ids = []
        # Also, collect all column embeddings for clustering.
        all_columns = []
        self.col_table_ids = []  # maps each column to its table index
        for idx, table in enumerate(self.tables):
            table_id, columns = table
            cols_arr = np.array(columns)
            agg_emb = np.max(cols_arr, axis=0)
            norm = np.linalg.norm(agg_emb)
            if norm > 0:
                agg_emb = agg_emb / norm
            self.agg_embeddings.append(agg_emb)
            self.table_ids.append(table_id)
            for col in columns:
                all_columns.append(col)
                self.col_table_ids.append(idx)
        self.agg_embeddings = np.vstack(self.agg_embeddings).astype('float32')
        d = self.agg_embeddings.shape[1]
        # Build FAISS index (IndexFlatIP for cosine similarity on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.agg_embeddings)
        
        # Cluster all column embeddings using HDBSCAN with prediction data enabled.
        all_columns = np.vstack(all_columns).astype('float32')
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        self.cluster_labels = self.clusterer.fit_predict(all_columns)
        
        # For each table, compute the set of cluster labels (ignore noise, label -1)
        self.table_clusters = {idx: set() for idx in range(len(self.tables))}
        for i, label in enumerate(self.cluster_labels):
            if label == -1:
                continue
            table_idx = self.col_table_ids[i]
            self.table_clusters[table_idx].add(label)
        
        self.weight_faiss = weight_faiss
        self.weight_cluster = weight_cluster
        
        print("FusionSearcher built in {:.4f} seconds.".format(time.time() - start_time))
        
    def topk(self, enc, query, K, **kwargs):
        """
        Retrieve top-K tables for a given query table.
        Args:
          query: tuple (query_table_id, query_column_embeddings)
          K: number of top candidates to return.
        Returns:
          - A list of tuples (fused_score, table_id) sorted by descending score.
          - Total number of candidates evaluated.
        """
        import hdbscan
        query_id, query_columns = query
        # Compute aggregated query embedding (mean pooling)
        query_arr = np.array(query_columns)
        agg_query = np.max(query_arr, axis=0)
        norm = np.linalg.norm(agg_query)
        if norm > 0:
            agg_query = agg_query / norm
        agg_query = np.array([agg_query]).astype('float32')
        # Retrieve candidates from FAISS (retrieve more than K to allow fusion)
        faiss_scores, faiss_indices = self.faiss_index.search(agg_query, K * 5)
        faiss_scores = faiss_scores.flatten()
        faiss_indices = faiss_indices.flatten()
        candidate_indices = set(faiss_indices.tolist())
        
        # Compute query cluster set using approximate_predict
        query_columns_arr = np.vstack(query_columns).astype('float32')
        query_labels, _ = hdbscan.approximate_predict(self.clusterer, query_columns_arr)
        query_cluster_set = set(label for label in query_labels if label != -1)
        
        fused_scores = {}
        for idx in candidate_indices:
            # Retrieve FAISS score for candidate table idx from aggregated index.
            faiss_score = 0.0
            for s, cand in zip(faiss_scores, faiss_indices):
                if cand == idx:
                    faiss_score = s
                    break
            # Compute cluster similarity: intersection over union of candidate and query clusters.
            candidate_cluster_set = self.table_clusters.get(idx, set())
            union = query_cluster_set.union(candidate_cluster_set)
            if len(union) == 0:
                cluster_score = 0.0
            else:
                intersection = query_cluster_set.intersection(candidate_cluster_set)
                cluster_score = len(intersection) / len(union)
            fused_score = self.weight_faiss * faiss_score + self.weight_cluster * cluster_score
            fused_scores[idx] = fused_score
        
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_candidates[:K]:
            table_id = self.tables[idx][0]
            results.append((score, table_id))
        return results, len(sorted_candidates)
        
if __name__ == "__main__":
    test_table_path = "datalake_vectors.pkl"
    try:
        searcher = FusionSearcher(test_table_path, scale=1.0)
        test_query = searcher.tables[0]  # use first table as query
        top_results, total = searcher.topk(enc=None, query=test_query, K=5)
        print("Top results:", top_results)
    except Exception as e:
        print("Error during FusionSearcher test:", e)
