from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.dataset import DatasetLoader


class RAGPipeline:

    def __init__(self):
        self.embedding_service = EmbeddingService()

        self.vector_store = None
        self.retriever = None

    def ingest_documents(self, file_path: str):

        documents = DatasetLoader.load_text_file(file_path)

        embeddings = self.embedding_service.embed_documents(documents)

        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_embeddings(
            embeddings=embeddings,
            docs=documents
        )

        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )

    def search(self, query: str, top_k: int = 3):

        return self.retriever.retrieve(
            query=query,
            top_k=top_k
        )