import json
from app.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()
pipeline.ingest_documents("data/sample_docs.txt")


queries = [

    "How does the system handle peak load?",

    "How is database performance improved?",

    "How are failures monitored?"
]


benchmark_results = []


for query in queries:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    strategy_a_results = pipeline.search_raw(query)

    print("\nSTRATEGY A: RAW VECTOR SEARCH\n")

    for result in strategy_a_results:

        print(f"Rank #{result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(result["document"])
        print("-" * 60)

    strategy_b_output = pipeline.search_expanded(query)

    print("\nSTRATEGY B: AI-ENHANCED RETRIEVAL\n")

    print(f"Expanded Query:")
    print(strategy_b_output["expanded_query"])
    print()

    for result in strategy_b_output["results"]:

        print(f"Rank #{result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(result["document"])
        print("-" * 60)

    benchmark_results.append({
        "query": query,
        "strategy_a": strategy_a_results,
        "strategy_b": strategy_b_output
    })


# SAVE BENCHMARK JSON
with open("benchmark_results.json", "w") as file:

    json.dump(
        benchmark_results,
        file,
        indent=2
    )

print("\nBenchmark results saved to benchmark_results.json")