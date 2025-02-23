import numpy as np
import pickle
import time
import faiss
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FaissSearcher(object):
    """
    FAISS-based searcher for joinable column search.
    It can either build an index from a list of column records (each with an embedding)
    or load a prebuilt index from disk.
    Each column record is expected to be a dict with keys: 'table_name', 'column_name', 'embedding'.
    """
    def __init__(self, table_path=None, columns=None, scale=1.0, index_path=None, column_ids_path=None):
        # If prebuilt index files are provided, load them.
        if index_path is not None and column_ids_path is not None:
            self.index = faiss.read_index(index_path)
            with open(column_ids_path, "rb") as f:
                self.column_ids = pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_path} with {len(self.column_ids)} columns.")
        # Otherwise, build the index from a provided list of columns
        elif columns is not None:
            start_time = time.time()
            self.columns = columns
            # Optionally, subsample columns if scale < 1.0
            if scale < 1.0:
                self.columns = random.sample(self.columns, int(scale * len(self.columns)))
            self.embeddings = []
            self.column_ids = []  # Each entry: (table_name_with_.csv, column_name)
            for record in self.columns:
                try:
                    emb = np.array(record["embedding"])
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    self.embeddings.append(emb)
                    tname = record["table_name"]
                    normalized_tname = tname if tname.endswith(".csv") else tname + ".csv"
                    self.column_ids.append((normalized_tname, record["column_name"]))
                except Exception:
                    continue
            if len(self.embeddings) == 0:
                raise ValueError("No valid embeddings found.")
            self.embeddings = np.vstack(self.embeddings).astype("float32")
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)
            elapsed = time.time() - start_time
            logger.info(f"FAISS index built with {len(self.column_ids)} columns in {elapsed:.2f} seconds.")
        # Alternatively, load from a file containing all columns
        elif table_path:
            start_time = time.time()
            with open(table_path, "rb") as f:
                columns = pickle.load(f)
            if scale < 1.0:
                columns = random.sample(columns, int(scale * len(columns)))
            self.columns = columns
            self.embeddings = []
            self.column_ids = []
            for record in self.columns:
                try:
                    emb = np.array(record["embedding"])
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    self.embeddings.append(emb)
                    tname = record["table_name"]
                    normalized_tname = tname if tname.endswith(".csv") else tname + ".csv"
                    self.column_ids.append((normalized_tname, record["column_name"]))
                except Exception:
                    continue
            if len(self.embeddings) == 0:
                raise ValueError("No valid embeddings found.")
            self.embeddings = np.vstack(self.embeddings).astype("float32")
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)
            elapsed = time.time() - start_time
            logger.info(f"FAISS index built with {len(self.column_ids)} columns in {elapsed:.2f} seconds.")
        else:
            raise ValueError("Either index_path+column_ids_path, columns, or table_path must be provided.")

    def save_index(self, index_path, column_ids_path):
        """Save the FAISS index and the column IDs to disk."""
        faiss.write_index(self.index, index_path)
        with open(column_ids_path, "wb") as f:
            pickle.dump(self.column_ids, f)
        logger.info(f"Saved FAISS index to {index_path} and column IDs to {column_ids_path}.")

    def topk(self, query, K):
        query_emb = np.array(query["embedding"])
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        query_emb = np.array([query_emb]).astype("float32")
        scores, indices = self.index.search(query_emb, K)
        scores = scores.flatten()
        indices = indices.flatten()
        results = [(float(score), self.column_ids[idx]) for score, idx in zip(scores, indices)]
        results.sort(key=lambda x: x[0], reverse=True)
        return results, len(self.column_ids)
