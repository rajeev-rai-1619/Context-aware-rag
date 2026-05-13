"""
Tests for the Retriever class -- verifies rank assignment and
delegation to VectorStore.
"""

import numpy as np
from unittest.mock import MagicMock

from app.retriever import Retriever


def test_retriever_assigns_sequential_ranks():
    embedding_service = MagicMock()
    embedding_service.embed_text.return_value = np.array(
        [1.0, 0.0], dtype="float32"
    )

    vector_store = MagicMock()
    vector_store.search.return_value = [
        {"score": 0.9, "document": "a"},
        {"score": 0.5, "document": "b"},
        {"score": 0.1, "document": "c"},
    ]

    retriever = Retriever(embedding_service, vector_store)
    results = retriever.retrieve("any query", top_k=3)

    assert [r["rank"] for r in results] == [1, 2, 3]
    embedding_service.embed_text.assert_called_once_with("any query")
    vector_store.search.assert_called_once()
