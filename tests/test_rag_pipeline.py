"""
End-to-end tests for the RAGPipeline.

Uses unittest.mock to *fully* replace the Vertex AI SDK mocks with
lightweight fakes so these tests run fast and don't require downloading
the sentence-transformers model.

This pattern is exactly how the production Vertex AI SDK would be
mocked in a CI pipeline (substitute real `vertexai.language_models` /
`vertexai.generative_models` symbols).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.rag_pipeline import RAGPipeline


# A tiny corpus to embed -- three orthogonal "topics" so we can reason
# about retrieval deterministically.
DOCS = ["load balancing topic", "database topic", "monitoring topic"]

# Pre-computed orthogonal embeddings (4-dim) for each document.
DOC_VECTORS = {
    "load balancing topic":  [1.0, 0.0, 0.0, 0.0],
    "database topic":        [0.0, 1.0, 0.0, 0.0],
    "monitoring topic":      [0.0, 0.0, 1.0, 0.0],
}

# Query embeddings: the "load" query points at doc 0, the expanded
# version is sharper / cleaner along that axis.
QUERY_VECTORS = {
    "peak load":                              [0.7, 0.1, 0.1, 0.0],
    "high traffic autoscaling load balancing": [0.99, 0.0, 0.0, 0.0],
}


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeTextEmbeddingModel:
    """Stand-in for the real Vertex AI TextEmbeddingModel."""

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def get_embeddings(self, texts):
        out = []
        for text in texts:
            if text in DOC_VECTORS:
                out.append(_FakeEmbedding(DOC_VECTORS[text]))
            elif text in QUERY_VECTORS:
                out.append(_FakeEmbedding(QUERY_VECTORS[text]))
            else:
                # Default to a small random-ish but deterministic vector.
                out.append(_FakeEmbedding([0.0, 0.0, 0.0, 1.0]))
        return out


class _FakeGenerationResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenerativeModel:
    """Stand-in for the real Vertex AI GenerativeModel."""

    def __init__(self, *_args, **_kwargs):
        pass

    def generate_content(self, prompt):
        if prompt == "peak load":
            return _FakeGenerationResponse(
                "high traffic autoscaling load balancing"
            )
        return _FakeGenerationResponse(prompt)


@pytest.fixture
def pipeline(tmp_path):
    """A RAGPipeline wired up with fully-mocked Vertex AI SDK classes."""

    with patch(
        "app.embedding_service.TextEmbeddingModel",
        _FakeTextEmbeddingModel,
    ), patch(
        "app.rag_pipeline.GenerativeModel",
        _FakeGenerativeModel,
    ):
        pipe = RAGPipeline()

        corpus = tmp_path / "corpus.txt"
        corpus.write_text("\n\n".join(DOCS), encoding="utf-8")
        pipe.ingest_documents(str(corpus))

        yield pipe


def test_pipeline_ingests_all_documents(pipeline):
    assert len(pipeline.vector_store.documents) == 3


def test_strategy_a_returns_top_k_ranked_results(pipeline):
    results = pipeline.search_raw("peak load", top_k=3)

    assert len(results) == 3
    assert [r["rank"] for r in results] == [1, 2, 3]
    # The "load balancing" doc must be ranked first.
    assert results[0]["document"] == "load balancing topic"


def test_strategy_b_uses_expanded_query(pipeline):
    output = pipeline.search_expanded("peak load", top_k=3)

    assert output["original_query"] == "peak load"
    assert output["expanded_query"] == (
        "high traffic autoscaling load balancing"
    )
    assert output["results"][0]["document"] == "load balancing topic"


def test_strategy_b_score_is_higher_than_strategy_a(pipeline):
    """
    Query expansion should produce an embedding more closely aligned
    with the target document, yielding a higher cosine score.
    """

    a_top = pipeline.search_raw("peak load", top_k=1)[0]
    b_top = pipeline.search_expanded("peak load", top_k=1)["results"][0]

    assert b_top["score"] > a_top["score"]
