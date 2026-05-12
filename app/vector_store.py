import faiss
import numpy as np
from typing import List, Tuple


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_embeddings(self, embeddings: np.ndarray, docs: List[str]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "document": self.documents[idx]
            })

        return results