from typing import List
import numpy as np

from app.mock_vertex_ai import TextEmbeddingModel


class EmbeddingService:
    """
    Wraps the (mocked) Vertex AI TextEmbeddingModel and exposes a simple
    embed_text / embed_documents interface to the rest of the pipeline.

    In production this class is unchanged -- only the import at the top
    swaps from `app.mock_vertex_ai` to `vertexai.language_models`.
    """

    def __init__(self, model_name: str = "textembedding-gecko@003"):
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        embeddings = self.model.get_embeddings([text])

        return np.array(embeddings[0].values, dtype="float32")

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        embeddings = self.model.get_embeddings(documents)

        return np.array(
            [embedding.values for embedding in embeddings],
            dtype="float32"
        )