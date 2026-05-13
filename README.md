# CON-RAG — Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation (RAG) pipeline that benchmarks two
retrieval strategies against a small technical corpus, with the Google
Cloud **Vertex AI SDK fully mocked** so the project runs entirely offline.

This repository is the submission for the *"Context-Aware Retrieval
Engine"* assessment.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Quick start](#quick-start)
3. [Repository layout](#repository-layout)
4. [Architecture](#architecture)
5. [Two retrieval strategies](#two-retrieval-strategies)
6. [Why cosine similarity (and not Euclidean)](#why-cosine-similarity-and-not-euclidean)
7. [Mocking the Vertex AI SDK](#mocking-the-vertex-ai-sdk)
8. [Migrating to Vertex AI Vector Search (Matching Engine) in production](#migrating-to-vertex-ai-vector-search-matching-engine-in-production)
9. [Running the tests](#running-the-tests)
10. [Evaluator checklist](#evaluator-checklist)

---

## What this project does

1. **Ingests** a plain-text corpus of ~7 technical paragraphs
   (`data/sample_docs.txt`), splitting on blank lines.
2. **Embeds** each paragraph with a (mocked) Vertex AI
   `TextEmbeddingModel`, backed locally by `sentence-transformers`
   (`all-MiniLM-L6-v2`, 384-dim).
3. **Stores** the L2-normalised embeddings in a FAISS
   `IndexFlatIP` index — making the inner product equivalent to cosine
   similarity.
4. **Answers** three benchmark queries using two retrieval strategies and
   writes a structured JSON comparison to `benchmark_results.json` plus a
   human-readable `retrieval_benchmark.md`.

---

## Quick start

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The first run downloads the `all-MiniLM-L6-v2` model from Hugging Face
(~90 MB, cached locally afterwards). All subsequent runs are fully
offline.

---

## Repository layout

```
CON-RAG/
├── app/
│   ├── __init__.py
│   ├── dataset.py             # Paragraph-level corpus loader
│   ├── embedding_service.py   # Wraps the (mocked) TextEmbeddingModel
│   ├── mock_vertex_ai.py      # Mocks of vertexai.* SDK classes
│   ├── rag_pipeline.py        # Orchestration class (RAGPipeline)
│   ├── retriever.py           # Query embedding + top-k retrieval
│   └── vector_store.py        # FAISS-backed cosine-similarity store
├── data/
│   └── sample_docs.txt        # 7 technical paragraphs
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_mock_vertex_ai.py
│   ├── test_rag_pipeline.py
│   ├── test_retriever.py
│   └── test_vector_store.py
├── main.py                    # Runs the 3-query benchmark
├── benchmark_results.json     # (generated) machine-readable results
├── retrieval_benchmark.md     # Dev evidence: Strategy A vs Strategy B
├── requirements.txt
└── README.md
```

---

## Architecture

```
                ┌──────────────────────────┐
                │   data/sample_docs.txt   │
                └────────────┬─────────────┘
                             │ DatasetLoader.load_text_file()
                             ▼
                ┌──────────────────────────┐
                │   List[str]  paragraphs  │
                └────────────┬─────────────┘
                             │ EmbeddingService.embed_documents()
                             ▼            (uses mock TextEmbeddingModel)
                ┌──────────────────────────┐
                │  np.ndarray  embeddings  │
                └────────────┬─────────────┘
                             │ VectorStore.add_embeddings()  (L2-normalise)
                             ▼
                ┌──────────────────────────┐
                │   FAISS IndexFlatIP      │  <── cosine similarity
                └────────────┬─────────────┘
                             ▲
   user query ──► Retriever ─┘   (or expanded via mock GenerativeModel)
```

`RAGPipeline` in [`app/rag_pipeline.py`](app/rag_pipeline.py) is the
single orchestration class required by the problem statement. It exposes:

| Method | Strategy | Description |
|--------|----------|-------------|
| `ingest_documents(path)` | — | Load + embed + index a corpus file |
| `search_raw(query, top_k)` | **A** | Direct embedding similarity search |
| `search_expanded(query, top_k)` | **B** | Query expansion via `GenerativeModel`, then search |

---

## Two retrieval strategies

### Strategy A — Raw Vector Search

The user's query is embedded directly and matched against the index.
Simple, fast, and the baseline against which Strategy B is measured.

### Strategy B — AI-Enhanced Retrieval (Query Expansion)

Before embedding, the query is passed to a (mocked) Vertex AI
`GenerativeModel`. The model rewrites the question into a longer,
keyword-rich form that aligns better with the vocabulary used in the
corpus.

> Original:&nbsp;&nbsp; *"How does the system handle peak load?"*
> Expanded: *"How does the system manage high traffic, autoscaling, load
> balancing, caching, and traffic spikes?"*

The full comparison across three queries (with cosine scores) is in
[retrieval_benchmark.md](retrieval_benchmark.md).

---

## Why cosine similarity (and not Euclidean)

Sentence embeddings carry their semantic meaning in the *direction* of
the vector, not its magnitude. Two paragraphs that talk about the same
concept will point in roughly the same direction even if one paragraph
is longer than the other (which inflates the L2 norm of its embedding).

| Metric | What it measures | Good for embeddings? |
|--------|------------------|----------------------|
| **Cosine similarity** | Angle between vectors (direction only) | ✅ Magnitude-invariant, robust to text length |
| Euclidean (L2) distance | Straight-line distance in space | ⚠️ Penalises longer texts whose embeddings have larger norms |
| Inner product (dot) | Direction *and* magnitude | ⚠️ Same magnitude bias as L2 |

We get cosine similarity in FAISS by:

1. **L2-normalising** every embedding (document and query) so it lies on
   the unit hypersphere — i.e. magnitude is forced to 1.
2. Using an **inner-product** index (`faiss.IndexFlatIP`).

After normalisation, the inner product is *exactly* cosine similarity,
so this gives us cosine semantics with the speed of the inner-product
index. See [`app/vector_store.py`](app/vector_store.py).

This is also the metric recommended by Google for `textembedding-gecko`
and by the Vertex AI Vector Search docs as `DOT_PRODUCT_DISTANCE` on
unit-normalised vectors.

---

## Mocking the Vertex AI SDK

The problem statement requires mocking
`vertexai.language_models.TextEmbeddingModel` and the Vertex AI
`GenerativeModel`. The mocks live in
[`app/mock_vertex_ai.py`](app/mock_vertex_ai.py) and **deliberately
mirror the real SDK class names and method signatures** so that
production migration is a one-line import swap:

| Real Vertex AI SDK | Local mock (this repo) |
|---|---|
| `vertexai.language_models.TextEmbeddingModel` | `app.mock_vertex_ai.TextEmbeddingModel` |
| `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` | same factory signature |
| `model.get_embeddings([...]) -> List[TextEmbedding]` | same method, returns objects with `.values` |
| `vertexai.generative_models.GenerativeModel("gemini-1.5-pro")` | `app.mock_vertex_ai.GenerativeModel(...)` |
| `model.generate_content(prompt).text` | same call/response shape |

The mocks are also independently unit-tested in
[`tests/test_mock_vertex_ai.py`](tests/test_mock_vertex_ai.py) and are
substituted with even lighter fakes in
[`tests/test_rag_pipeline.py`](tests/test_rag_pipeline.py) via
`unittest.mock.patch` to demonstrate the same pattern used to mock the
real GCP SDK in CI.

---

## Migrating to Vertex AI Vector Search (Matching Engine) in production

The codebase is structured so the migration is a localised change in
**three** modules:

### 1. `app/embedding_service.py` — swap the mock for the real SDK

```diff
- from app.mock_vertex_ai import TextEmbeddingModel
+ from vertexai.language_models import TextEmbeddingModel
+ import vertexai
+ vertexai.init(project="my-gcp-project", location="us-central1")
```

The rest of the class is unchanged because the mock already implements
`from_pretrained()` and `get_embeddings()` with identical signatures.

### 2. `app/rag_pipeline.py` — swap the GenerativeModel for the real one

```diff
- from app.mock_vertex_ai import GenerativeModel
+ from vertexai.generative_models import GenerativeModel
```

The hard-coded expansion dictionary is replaced by a real prompt to
Gemini, e.g.:

```python
prompt = (
    "Rewrite the user's question into a more verbose, "
    "keyword-rich query suitable for semantic search. "
    f"Question: {user_query}"
)
response = self.query_expander.generate_content(prompt)
```

### 3. `app/vector_store.py` — replace FAISS with Vertex AI Vector Search

FAISS is fine for tens of thousands of vectors on a single machine. For
production scale (millions+ vectors, multi-region, sub-50ms p99), we
move to **Vertex AI Vector Search** (formerly *Matching Engine*).

The migration is a class-for-class replacement of `VectorStore`:

```python
from google.cloud import aiplatform

class VertexVectorStore:
    def __init__(self, index_endpoint_name: str, deployed_index_id: str):
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoint_name
        )
        self.deployed_index_id = deployed_index_id

    def add_embeddings(self, embeddings, docs):
        # Production: bulk-upload to GCS, then trigger an index update.
        # The Matching Engine index is built offline from GCS-backed JSONL.
        ...

    def search(self, query_embedding, top_k=3):
        neighbors = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding.tolist()],
            num_neighbors=top_k,
        )
        return [
            {"score": n.distance, "document": self._lookup(n.id)}
            for n in neighbors[0]
        ]
```

Production checklist:

- **Index type:** `TreeAhConfig` (ScaNN) for cost-efficient ANN at scale,
  or `BruteForceConfig` for small / exact-search corpora.
- **Distance measure:** `DOT_PRODUCT_DISTANCE` on L2-normalised vectors
  (= cosine similarity), matching what we use locally.
- **Dimensions:** `768` for `textembedding-gecko@003` (vs. 384 for the
  local MiniLM stand-in).
- **Document storage:** raw paragraph text lives in Cloud SQL / Firestore
  and is joined to the neighbor IDs returned by Matching Engine.
- **Ingestion:** embeddings are written as JSONL to GCS, then a
  `MatchingEngineIndex` update job is triggered (batch) or streaming
  upserts are used (`stream_update_index`).
- **Auth:** workload identity / service account with
  `roles/aiplatform.user`.
- **Observability:** log the per-query expansion text, top-k scores, and
  latency; ship to Cloud Logging + Cloud Monitoring dashboards.

Because every Vertex AI integration is encapsulated inside three small
classes (`EmbeddingService`, `GenerativeModel` mock, `VectorStore`),
**no calling code in `main.py` or `RAGPipeline` needs to change**.

---

## Running the tests

```powershell
pip install -r requirements.txt
python -m pytest -q
```

The suite covers:

- [`tests/test_dataset.py`](tests/test_dataset.py) — paragraph chunking.
- [`tests/test_vector_store.py`](tests/test_vector_store.py) — cosine
  ranking semantics, top-k ordering.
- [`tests/test_retriever.py`](tests/test_retriever.py) — rank assignment,
  delegation (uses `MagicMock` for the SDK).
- [`tests/test_mock_vertex_ai.py`](tests/test_mock_vertex_ai.py) — mock
  SDK surface conformance; gracefully skipped if the embedding model
  cannot be fetched offline.
- [`tests/test_rag_pipeline.py`](tests/test_rag_pipeline.py) —
  end-to-end pipeline with `unittest.mock.patch` substituting the
  Vertex AI SDK classes entirely (this is the GCP-SDK-mocking pattern
  the problem statement asks for).

Expected: `16 passed` (or `13 passed, 3 skipped` on machines without
network access to Hugging Face).

---

## Evaluator checklist

| Requirement (from problem statement) | Location |
|---|---|
| Local embedding library (sentence-transformers) | [`app/embedding_service.py`](app/embedding_service.py) |
| Lightweight local vector DB (FAISS) | [`app/vector_store.py`](app/vector_store.py) |
| Mock `vertexai.language_models.TextEmbeddingModel` | [`app/mock_vertex_ai.py`](app/mock_vertex_ai.py) |
| Mock `vertexai.*.GenerativeModel` | [`app/mock_vertex_ai.py`](app/mock_vertex_ai.py) |
| Orchestration class for ingestion + retrieval | `RAGPipeline` in [`app/rag_pipeline.py`](app/rag_pipeline.py) |
| Strategy A — raw vector search | `RAGPipeline.search_raw` |
| Strategy B — AI-enhanced retrieval | `RAGPipeline.search_expanded` |
| Comparison report for ≥3 complex queries (JSON / table) | [`retrieval_benchmark.md`](retrieval_benchmark.md), `benchmark_results.json` |
| Pytest suite mocking the GCP SDK | [`tests/`](tests/) |
| Cosine vs Euclidean explanation | [§ Why cosine similarity](#why-cosine-similarity-and-not-euclidean) |
| Vertex AI Matching Engine migration plan | [§ Migrating to Vertex AI Vector Search](#migrating-to-vertex-ai-vector-search-matching-engine-in-production) |
