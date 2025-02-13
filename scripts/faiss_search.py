# faiss_search.py

import numpy as np
import pickle
import time
import random
import faiss

class FaissSearcher(object):
    """
    FAISS-based searcher with two modes:
      1. Aggregated table embeddings mode: if pooling is 'mean' or 'max',
         each table’s column embeddings are aggregated into a single table embedding,
         normalized, and indexed.
      2. Column-level mode: if pooling is None, each individual column embedding is normalized,
         indexed separately, and a query table’s candidate tables are determined by searching
         on each query column and summing the similarity scores.
    """
    
    def __init__(self, table_path, scale=1.0, pooling='mean'):
        """
        Initialize the FAISS searcher.
        
        Args:
            table_path (str): Path to a pickle file containing a list of tables.
                              Each table is expected to be a tuple: (table_id, column_embeddings),
                              where column_embeddings is a list (or array) of vectors.
            scale (float): A scaling factor (0 < scale <= 1.0) to use a subset of the tables.
            pooling (str or None): Determines how to aggregate embeddings.
                                   - 'mean' or 'max': aggregate each table’s column embeddings
                                     using the specified pooling method.
                                   - None: no aggregation; index individual column embeddings.
        """
        start_time = time.time()
        
        # Load tables from pickle
        with open(table_path, "rb") as f:
            tables = pickle.load(f)
        if scale < 1.0:
            tables = random.sample(tables, int(scale * len(tables)))
        self.tables = tables
        
        # Mode 1: Aggregated (pooling provided as 'mean' or 'max')
        if pooling is not None:
            self.pooling = pooling
            self.agg_embeddings = []
            self.table_ids = []  # mapping: index in self.agg_embeddings -> table id
            for table in self.tables:
                table_id, columns = table
                cols_arr = np.array(columns)
                if pooling == 'mean':
                    agg_emb = np.mean(cols_arr, axis=0)
                elif pooling == 'max':
                    agg_emb = np.max(cols_arr, axis=0)
                else:
                    raise ValueError("Unsupported pooling method: {}".format(pooling))
                # Normalize aggregated embedding to unit length
                norm = np.linalg.norm(agg_emb)
                if norm > 0:
                    agg_emb = agg_emb / norm
                self.agg_embeddings.append(agg_emb)
                self.table_ids.append(table_id)
            # Stack embeddings into a matrix (num_tables x vector_dim)
            self.agg_embeddings = np.vstack(self.agg_embeddings).astype('float32')
            d = self.agg_embeddings.shape[1]
            # Build FAISS index using inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.agg_embeddings)
        else:
            # Mode 2: Column-level indexing (no aggregation)
            self.pooling = None
            self.col_embeddings = []
            self.col_table_ids = []  # mapping: each column embedding -> its table id
            for table in self.tables:
                table_id, columns = table
                for col in columns:
                    col = np.array(col)
                    # Normalize column embedding
                    norm = np.linalg.norm(col)
                    if norm > 0:
                        col = col / norm
                    self.col_embeddings.append(col)
                    self.col_table_ids.append(table_id)
            self.col_embeddings = np.vstack(self.col_embeddings).astype('float32')
            d = self.col_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.col_embeddings)
        
        print("FAISS index built with {} mode in {:.4f} seconds.".format(
            "aggregated" if pooling is not None else "column-level",
            time.time() - start_time))

    def topk(self, enc, query, K, N=5, threshold=None):
        """
        Retrieve the top-K most similar tables for the given query table.
        
        Args:
            enc: (Ignored) Encoder type (for interface compatibility).
            query (tuple): A tuple (query_table_id, query_column_embeddings).
            K (int): Number of top tables to return.
            N (int): For column-level mode (pooling is None), number of similar columns
                     to retrieve per query column.
            threshold: (Ignored) Similarity threshold (not used in FAISS searcher).
        
        Returns:
            A tuple (results, total) where:
              - results: list of tuples (score, table_id) sorted by descending score.
              - total: total number of candidate tables in the index.
        """
        if self.pooling is not None:
            # Aggregated mode
            _, query_columns = query
            query_arr = np.array(query_columns)
            if self.pooling == 'mean':
                agg_query = np.mean(query_arr, axis=0)
            elif self.pooling == 'max':
                agg_query = np.max(query_arr, axis=0)
            else:
                raise ValueError("Unsupported pooling method: {}".format(self.pooling))
            norm = np.linalg.norm(agg_query)
            if norm > 0:
                agg_query = agg_query / norm
            agg_query = np.array([agg_query]).astype('float32')
            scores, indices = self.index.search(agg_query, K)
            scores = scores.flatten()
            indices = indices.flatten()
            results = []
            for score, idx in zip(scores, indices):
                table_id = self.table_ids[idx]
                results.append((float(score), table_id))
            results.sort(key=lambda x: x[0], reverse=True)
            return results, len(self.table_ids)
        else:
            # Column-level mode: for each query column, search for top N similar columns,
            # then aggregate candidate scores per table.
            _, query_columns = query
            candidate_scores = {}  # table_id -> cumulative similarity score
            query_arr = np.array(query_columns)
            for col in query_arr:
                col = np.array(col)
                norm = np.linalg.norm(col)
                if norm > 0:
                    col = col / norm
                col = np.array([col]).astype('float32')
                scores, indices = self.index.search(col, N)
                scores = scores.flatten()
                indices = indices.flatten()
                for score, idx in zip(scores, indices):
                    table_id = self.col_table_ids[idx]
                    candidate_scores[table_id] = candidate_scores.get(table_id, 0.0) + float(score)
            # Sort candidate tables by the cumulative score (descending)
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            results = [(score, table_id) for table_id, score in sorted_candidates[:K]]
            return results, len(set(self.col_table_ids))

