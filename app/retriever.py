from typing import List, Dict

from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:

        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        return results