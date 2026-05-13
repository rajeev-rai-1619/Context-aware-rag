"""
Mock implementations of the Vertex AI SDK classes used in this project.

These mocks intentionally mirror the *real* Google Cloud Vertex AI SDK
class names and method signatures so that swapping them for the real SDK
in production is a single-line change.

Real SDK reference:
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel

    embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = embed_model.get_embeddings(["some text"])
    vector = embeddings[0].values

    gen_model = GenerativeModel("gemini-1.5-pro")
    response = gen_model.generate_content("rewrite this query ...")
    text = response.text
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Mock of vertexai.language_models.TextEmbeddingModel
# ---------------------------------------------------------------------------

class TextEmbedding:
    """Mirror of vertexai.language_models.TextEmbedding result object."""

    def __init__(self, values: List[float]):
        self.values = values


class TextEmbeddingModel:
    """
    Mock of vertexai.language_models.TextEmbeddingModel.

    Backed locally by a sentence-transformers model so we can produce
    deterministic embeddings without any network / GCP credentials.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str = "textembedding-gecko@003"):
        """
        Matches the real SDK factory signature:
            TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

        The Vertex AI model name is accepted for API compatibility but the
        local sentence-transformers model is used to actually compute
        embeddings.
        """

        instance = cls(model_name="all-MiniLM-L6-v2")
        instance._model_name = model_name

        return instance

    def get_embeddings(self, texts: List[str]) -> List[TextEmbedding]:
        """Mirrors TextEmbeddingModel.get_embeddings()."""

        vectors = self._model.encode(texts)

        return [TextEmbedding(values=vector.tolist()) for vector in vectors]


# ---------------------------------------------------------------------------
# Mock of vertexai.generative_models.GenerativeModel
# ---------------------------------------------------------------------------

class GenerationResponse:
    """Mirror of vertexai.generative_models.GenerationResponse."""

    def __init__(self, text: str):
        self.text = text


class GenerativeModel:
    """
    Mock of vertexai.generative_models.GenerativeModel used for the
    query-expansion step (Strategy B).

    The mock returns hand-crafted expansions for the benchmark queries
    so that the retrieval comparison is deterministic and reproducible.
    """

    _EXPANSIONS = {
        "How does the system handle peak load?": (
            "How does the system manage high traffic, autoscaling, "
            "load balancing, caching, and traffic spikes?"
        ),
        "How is database performance improved?": (
            "How does the architecture optimize database throughput, "
            "replication, caching, read scalability, and query performance?"
        ),
        "How are failures monitored?": (
            "How does the platform monitor system health, latency, "
            "resource utilization, alerts, and operational failures?"
        ),
    }

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self._model_name = model_name

    def generate_content(self, prompt: str) -> GenerationResponse:
        """
        Mirrors GenerativeModel.generate_content().

        Returns a GenerationResponse whose .text attribute is the expanded
        query. Falls back to the original prompt for unknown queries so the
        pipeline still works on arbitrary input.
        """

        expanded = self._EXPANSIONS.get(prompt, prompt)

        return GenerationResponse(text=expanded)
