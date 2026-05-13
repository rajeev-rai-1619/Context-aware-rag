"""
Tests for the Vertex AI mock classes.

These tests assert that our mocks faithfully mirror the public surface
of the real Vertex AI SDK so the production swap is a no-op.
"""

import pytest

from app.mock_vertex_ai import (
    GenerationResponse,
    GenerativeModel,
    TextEmbedding,
    TextEmbeddingModel,
)


@pytest.fixture(scope="module")
def embedding_model():
    """
    Loads the sentence-transformers model once for the module.

    Skips all tests in this fixture's scope if the model cannot be
    fetched (offline / SSL / proxy environment) -- the network
    download itself is not what we are testing.
    """
    try:
        return TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    except Exception as exc:
        pytest.skip(f"Embedding model unavailable: {exc}")


# ---------------------------------------------------------------------------
# TextEmbeddingModel
# ---------------------------------------------------------------------------

def test_text_embedding_model_from_pretrained_returns_instance(embedding_model):
    assert isinstance(embedding_model, TextEmbeddingModel)


def test_get_embeddings_returns_text_embedding_objects(embedding_model):
    results = embedding_model.get_embeddings(["hello world", "second sentence"])

    assert len(results) == 2
    assert all(isinstance(item, TextEmbedding) for item in results)
    assert all(isinstance(item.values, list) for item in results)
    assert len(results[0].values) > 0


def test_get_embeddings_is_deterministic(embedding_model):
    first = embedding_model.get_embeddings(["repeatable text"])[0].values
    second = embedding_model.get_embeddings(["repeatable text"])[0].values

    assert first == second


# ---------------------------------------------------------------------------
# GenerativeModel
# ---------------------------------------------------------------------------

def test_generative_model_returns_generation_response():
    model = GenerativeModel("gemini-1.5-pro")

    response = model.generate_content("How does the system handle peak load?")

    assert isinstance(response, GenerationResponse)
    assert isinstance(response.text, str)
    assert response.text != ""


def test_generative_model_expands_known_query():
    model = GenerativeModel()

    response = model.generate_content("How does the system handle peak load?")

    # The expansion must enrich the query with related vocabulary.
    assert "autoscaling" in response.text
    assert "load balancing" in response.text


def test_generative_model_falls_back_for_unknown_query():
    model = GenerativeModel()

    response = model.generate_content("totally novel question")

    assert response.text == "totally novel question"
