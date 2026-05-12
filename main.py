# from app.embedding_service import EmbeddingService
# from app.vector_store import VectorStore

# docs = [
#     "Kubernetes autoscaling handles peak traffic loads.",
#     "Redis caching reduces database pressure.",
#     "Load balancers distribute traffic evenly."
# ]

# embedding_service = EmbeddingService()

# doc_embeddings = embedding_service.embed_documents(docs)

# dimension = doc_embeddings.shape[1]

# vector_store = VectorStore(dimension)

# vector_store.add_embeddings(doc_embeddings, docs)

# query = "How does the system manage heavy traffic?"

# query_embedding = embedding_service.embed_text(query)

# results = vector_store.search(query_embedding)

# for result in results:
#     print(result)


from app.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

pipeline.ingest_documents("data/sample_docs.txt")

query = "How does the system handle peak load?"

results = pipeline.search(query)

print("\n=== Retrieval Results ===\n")

for idx, result in enumerate(results, start=1):

    print(f"Rank {idx}")
    print(f"Score: {result['score']:.4f}")
    print(result["document"])
    print("-" * 60)