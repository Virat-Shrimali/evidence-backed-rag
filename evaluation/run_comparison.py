"""Runs multi-strategy evaluation comparisons across retrieval configurations."""



def generate_comparison_markdown_table(results: dict[str, dict[str, float]]) -> str:
    """Format benchmark results dict into markdown table."""
    headers = [
        "Retriever Strategy",
        "Recall@5",
        "Precision@5",
        "MRR",
        "Refusal Accuracy",
        "p95 Latency (ms)",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for strategy, metrics in results.items():
        row = [
            f"**{strategy}**",
            f"{metrics.get('recall@5', 0.0):.2%}",
            f"{metrics.get('precision@5', 0.0):.2%}",
            f"{metrics.get('mrr', 0.0):.3f}",
            f"{metrics.get('refusal_accuracy', 0.0):.2%}",
            f"{metrics.get('p95_latency_ms', 0.0):.1f}",
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows) + "\n"
