"""
Tests for the FAISS-backed VectorStore.

These tests use deterministic hand-crafted embeddings (no model required)
so they run fast and don't depend on sentence-transformers.
"""

import numpy as np
import pytest

from app.vector_store import VectorStore


def _make_store():
    store = VectorStore(dimension=3)

    # Three documents on three orthogonal axes -> easy to reason about.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],   # doc 0
            [0.0, 1.0, 0.0],   # doc 1
            [0.0, 0.0, 1.0],   # doc 2
        ],
        dtype="float32",
    )

    docs = ["doc-x-axis", "doc-y-axis", "doc-z-axis"]

    store.add_embeddings(embeddings, docs)
    return store


def test_search_returns_top_k_results():
    store = _make_store()

    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=2)

    assert len(results) == 2


def test_search_ranks_most_similar_first():
    store = _make_store()

    query = np.array([0.9, 0.1, 0.0], dtype="float32")
    results = store.search(query, top_k=3)

    assert results[0]["document"] == "doc-x-axis"
    assert results[1]["document"] == "doc-y-axis"
    assert results[2]["document"] == "doc-z-axis"


def test_search_scores_are_cosine_similarity():
    """
    L2-normalised inputs + inner-product index == cosine similarity.
    An exact match must score ~1.0; an orthogonal vector must score ~0.0.
    """
    store = _make_store()

    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=3)

    top = results[0]
    assert top["document"] == "doc-x-axis"
    assert top["score"] == pytest.approx(1.0, abs=1e-5)

    # The other two are orthogonal -> cosine similarity 0.
    assert results[1]["score"] == pytest.approx(0.0, abs=1e-5)
    assert results[2]["score"] == pytest.approx(0.0, abs=1e-5)


def pytest_approx(value, tol=1e-5):
    import pytest
    return pytest.approx(value, abs=tol)
