import numpy as np
import pickle
import time
import random
import hdbscan
from umap import UMAP

class ClusterSearcher(object):
    """
    ClusterSearcher uses UMAP to reduce dimensions of all column embeddings,
    then applies HDBSCAN to cluster them. For each table in the data lake, it computes a set of
    cluster labels assigned to its columns. Given a query table, it reduces its column embeddings
    via UMAP, assigns cluster labels (using approximate prediction), and then computes a unionability score:
    
         score = |S_query ∩ S_candidate| / |S_query ∪ S_candidate|
    
    where S_query and S_candidate are the sets of cluster labels (ignoring noise) for the query and candidate table.
    """
    def __init__(self, table_path, min_cluster_size=2, scale=1.0):
        start_time = time.time()
        with open(table_path, "rb") as f:
            tables = pickle.load(f)
        if scale < 1.0:
            tables = random.sample(tables, int(scale * len(tables)))
        self.tables = tables

        # Gather all column embeddings from all tables and record the mapping.
        self.all_columns = []
        self.col_table_ids = []  # maps each column to its table (index in self.tables)
        for idx, table in enumerate(self.tables):
            table_id, columns = table
            for col in columns:
                self.all_columns.append(col)
                self.col_table_ids.append(idx)
        self.all_columns = np.vstack(self.all_columns).astype('float32')

        # First, reduce dimensions using UMAP with the default configuration.
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        self.reduced_columns = self.umap_model.fit_transform(self.all_columns)

        # Cluster the UMAP-reduced column embeddings using HDBSCAN.
        # IMPORTANT: Enable prediction_data so that approximate_predict can work.
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        self.cluster_labels = self.clusterer.fit_predict(self.reduced_columns)
        
        # For each table, build a set of cluster labels from its columns (ignoring noise, label -1)
        self.table_clusters = {idx: set() for idx in range(len(self.tables))}
        for i, label in enumerate(self.cluster_labels):
            if label == -1:
                continue
            table_idx = self.col_table_ids[i]
            self.table_clusters[table_idx].add(label)
            
        print("ClusterSearcher built in {:.4f} seconds.".format(time.time() - start_time))
        
    def topk(self, enc, query, K, **kwargs):
        """
        For a given query table (tuple: (table_id, query_column_embeddings)),
        reduce its embeddings via UMAP, assign cluster labels (using approximate prediction),
        then compute the unionability score against each table in the data lake.
        
        Returns:
          - A list of tuples (score, table_id) sorted in descending order (top K first)
          - The total number of candidate tables evaluated.
        """
        import hdbscan
        query_id, query_columns = query
        query_columns = np.vstack(query_columns).astype('float32')
        # Reduce query embeddings with the fitted UMAP model.
        query_reduced = self.umap_model.transform(query_columns)
        # Use approximate_predict from hdbscan on the reduced query embeddings.
        query_labels, _ = hdbscan.approximate_predict(self.clusterer, query_reduced)
        query_set = set(label for label in query_labels if label != -1)
        
        scores = []
        for idx, clusters in self.table_clusters.items():
            union = query_set.union(clusters)
            if len(union) == 0:
                score = 0.0
            else:
                intersection = query_set.intersection(clusters)
                score = len(intersection) / len(union)
            table_id = self.tables[idx][0]
            scores.append((score, table_id))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:K], len(scores)