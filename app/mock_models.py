class MockGenerativeModel:
    """
    Mock version of Vertex AI GenerativeModel
    used for query expansion.
    """

    def generate_content(self, query: str) -> str:

        expansions = {

            "How does the system handle peak load?":
                (
                    "How does the system manage high traffic, "
                    "autoscaling, load balancing, caching, "
                    "and traffic spikes?"
                ),

            "How is database performance improved?":
                (
                    "How does the architecture optimize database throughput, "
                    "replication, caching, read scalability, "
                    "and query performance?"
                ),

            "How are failures monitored?":
                (
                    "How does the platform monitor system health, "
                    "latency, resource utilization, alerts, "
                    "and operational failures?"
                )
        }

        return expansions.get(query, query)