# Retrieval Benchmark — Strategy A vs Strategy B

This document is the **dev evidence** for the comparison described in the
problem statement. It contains the output of `python main.py` for three
complex queries.

- **Strategy A — Raw Vector Search:** the user query is embedded directly
  with the (mocked) Vertex AI `TextEmbeddingModel` and matched against the
  document index.
- **Strategy B — AI-Enhanced Retrieval:** the (mocked) Vertex AI
  `GenerativeModel` first rewrites/expands the user query, then the
  expanded text is embedded and matched.

Embedding model: `all-MiniLM-L6-v2` (local stand-in for
`textembedding-gecko@003`)
Similarity metric: **cosine similarity** (L2-normalised vectors +
FAISS `IndexFlatIP`)
Corpus: `data/sample_docs.txt` (7 paragraphs)

> The full machine-readable output is also written to
> [benchmark_results.json](benchmark_results.json) by `main.py`.

---

## Query 1 — "How does the system handle peak load?"

**Expanded query (Strategy B):**
> *"How does the system manage high traffic, autoscaling, load balancing,
> caching, and traffic spikes?"*

### Strategy A — Raw Vector Search

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.5921 | The system uses horizontal scaling to handle peak traffic loads. Additional application instances are automatically provisioned using Kubernetes autoscaling policies. |
| 2 | 0.3274 | Asynchronous message queues are used to decouple services and smooth traffic spikes. Tasks are processed in the background to avoid blocking client requests. |
| 3 | 0.2988 | The architecture includes a load balancer that distributes incoming traffic evenly across backend services. This prevents any single node from becoming overloaded. |

### Strategy B — AI-Enhanced Retrieval

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.6843 | The system uses horizontal scaling to handle peak traffic loads. Additional application instances are automatically provisioned using Kubernetes autoscaling policies. |
| 2 | 0.5612 | The architecture includes a load balancer that distributes incoming traffic evenly across backend services. This prevents any single node from becoming overloaded. |
| 3 | 0.4870 | Caching is implemented using Redis to reduce database pressure during periods of high request volume. Frequently accessed records are served directly from memory. |

**Observation:** Strategy B pulls in the *caching* paragraph that Strategy A
missed entirely, and it boosts the top-1 cosine score by ~0.09. The
expansion vocabulary ("autoscaling", "load balancing", "caching") aligns
the query vector with multiple architectural facets of "handling load".

---

## Query 2 — "How is database performance improved?"

**Expanded query (Strategy B):**
> *"How does the architecture optimize database throughput, replication,
> caching, read scalability, and query performance?"*

### Strategy A — Raw Vector Search

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.5407 | Database replication is configured to improve read throughput and provide high availability. Read replicas serve analytical queries separately from transactional workloads. |
| 2 | 0.4128 | Caching is implemented using Redis to reduce database pressure during periods of high request volume. Frequently accessed records are served directly from memory. |
| 3 | 0.1875 | The system uses horizontal scaling to handle peak traffic loads. Additional application instances are automatically provisioned using Kubernetes autoscaling policies. |

### Strategy B — AI-Enhanced Retrieval

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.6491 | Database replication is configured to improve read throughput and provide high availability. Read replicas serve analytical queries separately from transactional workloads. |
| 2 | 0.5733 | Caching is implemented using Redis to reduce database pressure during periods of high request volume. Frequently accessed records are served directly from memory. |
| 3 | 0.2912 | Asynchronous message queues are used to decouple services and smooth traffic spikes. Tasks are processed in the background to avoid blocking client requests. |

**Observation:** Both strategies surface the two truly relevant chunks
(replication, Redis caching), but Strategy B lifts both scores
substantially (+0.10 and +0.16). At rank 3 Strategy B replaces the
mostly-irrelevant "horizontal scaling" paragraph with the more
database-adjacent message-queue paragraph.

---

## Query 3 — "How are failures monitored?"

**Expanded query (Strategy B):**
> *"How does the platform monitor system health, latency, resource
> utilization, alerts, and operational failures?"*

### Strategy A — Raw Vector Search

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.5012 | Monitoring tools continuously track CPU, memory usage, and request latency. Alerts are triggered when predefined thresholds are exceeded. |
| 2 | 0.2643 | Rate limiting protects the API from abuse and ensures fair resource allocation among users during periods of elevated demand. |
| 3 | 0.1981 | Asynchronous message queues are used to decouple services and smooth traffic spikes. Tasks are processed in the background to avoid blocking client requests. |

### Strategy B — AI-Enhanced Retrieval

| Rank | Score | Document |
|------|-------|----------|
| 1 | 0.6720 | Monitoring tools continuously track CPU, memory usage, and request latency. Alerts are triggered when predefined thresholds are exceeded. |
| 2 | 0.3608 | Rate limiting protects the API from abuse and ensures fair resource allocation among users during periods of elevated demand. |
| 3 | 0.2854 | The system uses horizontal scaling to handle peak traffic loads. Additional application instances are automatically provisioned using Kubernetes autoscaling policies. |

**Observation:** The monitoring paragraph is clearly identified by both
strategies, but Strategy B's expanded query ("system health", "latency",
"resource utilization") drives the top-1 cosine score from 0.50 -> 0.67 —
a much sharper match.

---

## Summary

| Query | Top-1 score (A) | Top-1 score (B) | Δ |
|-------|-----------------|-----------------|---|
| Peak load | 0.5921 | 0.6843 | **+0.0922** |
| Database performance | 0.5407 | 0.6491 | **+0.1084** |
| Failures monitored | 0.5012 | 0.6720 | **+0.1708** |

Across all three queries, **Strategy B (AI-enhanced query expansion)
consistently improves the cosine similarity of the top-ranked result**
and frequently surfaces additional relevant chunks at lower ranks. The
gain is largest on terse, abstract queries (Q3) where the original
embedding has the least lexical overlap with the corpus.

---

### Reproducing

```powershell
pip install -r requirements.txt
python main.py
```

`main.py` writes both the human-readable console output above and the
machine-readable `benchmark_results.json`.
