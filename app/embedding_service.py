from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    """
    Simulates Vertex AI TextEmbeddingModel
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        return self.model.encode(documents)