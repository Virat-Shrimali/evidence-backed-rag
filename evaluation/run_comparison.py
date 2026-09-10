"""Runs data-driven multi-strategy evaluation comparisons across retrieval configurations.

Compares:
1. BM25-only (sparse keyword search)
2. Dense-only (semantic vector embeddings)
3. Hybrid / RRF (Reciprocal Rank Fusion)
4. Hybrid + Cross-Encoder Reranker (two-stage neural reranking)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evaluation.build_golden_set import GoldenQAPair, load_golden_qa_set
from evaluation.retrieval_metrics import (
    RetrievalEvaluationReport,
    evaluate_retriever,
)
from src.config import settings
from src.index.bm25_index import BM25Index
from src.index.embed import DenseIndex
from src.ingest.chunk import Chunk, chunk_document
from src.ingest.parse import parse_directory, parse_document
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
)

logger = logging.getLogger(__name__)


def generate_comparison_markdown_table(
    results: dict[str, RetrievalEvaluationReport | dict[str, float]],
    top_k: int = 5,
) -> str:
    """Format benchmark evaluation results into a clean markdown table."""
    headers = [
        "Retriever Strategy",
        f"Recall@{top_k}",
        f"Precision@{top_k}",
        "MRR",
        "Refusal Accuracy",
        "p50 Latency (ms)",
        "p95 Latency (ms)",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for strategy, metrics in results.items():
        if isinstance(metrics, RetrievalEvaluationReport):
            rec = metrics.recall_at_k
            prec = metrics.precision_at_k
            mrr = metrics.mrr
            ref_acc = metrics.refusal_accuracy
            p50 = metrics.latency_p50_ms
            p95 = metrics.latency_p95_ms
        else:
            rec = metrics.get(f"recall@{top_k}", metrics.get("recall", 0.0))
            prec = metrics.get(f"precision@{top_k}", metrics.get("precision", 0.0))
            mrr = metrics.get("mrr", 0.0)
            ref_acc = metrics.get("refusal_accuracy", 0.0)
            p50 = metrics.get("p50_latency_ms", 0.0)
            p95 = metrics.get("p95_latency_ms", 0.0)

        row = [
            f"**{strategy}**",
            f"{rec:.2%}",
            f"{prec:.2%}",
            f"{mrr:.3f}",
            f"{ref_acc:.2%}",
            f"{p50:.1f}",
            f"{p95:.1f}",
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows) + "\n"


class RetrievalBenchmarkHarness:
    """Configurable, data-driven harness evaluating retrieval configurations over any corpus and golden set."""

    def __init__(
        self,
        chunks: list[Chunk],
        golden_dataset: list[GoldenQAPair],
        top_k: int = 5,
        dense_index: DenseIndex | None = None,
        bm25_index: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
        collection_name: str = "eval_bench_collection",
        evidence_confidence_threshold: float = 0.65,
    ):
        self.chunks = chunks
        self.golden_dataset = golden_dataset
        self.top_k = top_k
        self.collection_name = collection_name
        self.evidence_confidence_threshold = evidence_confidence_threshold

        # Initialize and populate dense index
        self.dense_index = dense_index or DenseIndex(
            persist_dir=settings.chroma_persist_dir,
            model_name=settings.embedding_model_name,
        )
        if self.chunks:
            self.dense_index.delete_collection(self.collection_name)
            self.dense_index.index_chunks(self.chunks, collection_name=self.collection_name)

        # Initialize and populate BM25 index
        self.bm25_index = bm25_index or BM25Index(self.chunks)

        # Initialize neural reranker
        self.reranker = reranker or CrossEncoderReranker(
            model_name=settings.reranker_model_name
        )

    def run_benchmark(self) -> dict[str, RetrievalEvaluationReport]:
        """Execute all 4 retrieval strategies across the golden evaluation dataset."""
        dense_retriever = DenseRetriever(
            dense_index=self.dense_index,
            top_k=settings.dense_top_k,
            collection_name=self.collection_name,
        )
        bm25_retriever = BM25Retriever(
            bm25_index=self.bm25_index,
            top_k=settings.sparse_top_k,
        )
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=settings.rrf_k,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            final_top_k=self.top_k,
        )
        hybrid_for_rerank = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=settings.rrf_k,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            final_top_k=settings.candidate_top_k,
        )
        reranked_retriever = RerankedRetriever(
            base_retriever=hybrid_for_rerank,
            reranker=self.reranker,
            candidate_top_k=settings.candidate_top_k,
            final_top_k=self.top_k,
        )

        strategies = [
            ("BM25-only", bm25_retriever, None),
            ("Dense-only", dense_retriever, self.evidence_confidence_threshold),
            ("Hybrid (RRF)", hybrid_retriever, None),
            ("Hybrid + Cross-Encoder", reranked_retriever, 0.0),
        ]

        results: dict[str, RetrievalEvaluationReport] = {}
        for name, retriever, threshold in strategies:
            logger.info("Evaluating strategy: %s ...", name)
            report = evaluate_retriever(
                retriever=retriever,
                dataset=self.golden_dataset,
                top_k=self.top_k,
                score_threshold=threshold,
                strategy_name=name,
            )
            results[name] = report

        return results

    def cleanup(self) -> None:
        """Clean up test collection from vector store."""
        try:
            self.dense_index.delete_collection(self.collection_name)
        except Exception:
            pass


def load_corpus_and_chunk(
    corpus_path: Path | str,
    strategy: str = "strategy_a",
) -> list[Chunk]:
    """Ingest raw documents from a directory or single file and generate chunks."""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {path}")

    if path.is_file():
        docs = [parse_document(path)]
    else:
        docs = parse_directory(path)

    if not docs:
        raise ValueError(f"No parseable documents found in {path}")

    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc, strategy=strategy)
        all_chunks.extend(chunks)

    return all_chunks


def run_comparison_suite(
    corpus_path: Path | str = "data/raw",
    golden_qa_path: Path | str = "data/golden_qa/qa_pairs.jsonl",
    output_table_path: Path | str = "evaluation/results/comparison_table.md",
    output_json_path: Path | str | None = None,
    chunking_strategy: str = "strategy_a",
    top_k: int = 5,
    collection_name: str = "eval_bench_collection",
) -> tuple[dict[str, RetrievalEvaluationReport], str]:
    """Orchestrate end-to-end benchmark comparison and write markdown summary table."""
    golden_path = Path(golden_qa_path)
    golden_dataset = load_golden_qa_set(golden_path)
    if not golden_dataset:
        raise ValueError(f"No golden QA pairs found at: {golden_path}")

    chunks = load_corpus_and_chunk(corpus_path, strategy=chunking_strategy)
    harness = RetrievalBenchmarkHarness(
        chunks=chunks,
        golden_dataset=golden_dataset,
        top_k=top_k,
        collection_name=collection_name,
    )

    try:
        results = harness.run_benchmark()
        markdown_table = generate_comparison_markdown_table(results, top_k=top_k)

        out_table = Path(output_table_path)
        out_table.parent.mkdir(parents=True, exist_ok=True)
        out_table.write_text(markdown_table, encoding="utf-8")

        if output_json_path:
            out_json = Path(output_json_path)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            serialized = {k: v.model_dump() for k, v in results.items()}
            out_json.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

        return results, markdown_table
    finally:
        harness.cleanup()


def main() -> None:
    """CLI entrypoint for running multi-strategy retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Run multi-strategy retrieval evaluation benchmark."
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=str(settings.raw_data_dir),
        help="Path to raw document corpus directory or file.",
    )
    parser.add_argument(
        "--golden-qa",
        type=str,
        default=str(settings.golden_qa_path),
        help="Path to golden QA dataset JSONL.",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default=str(settings.eval_results_dir / "comparison_table.md"),
        help="Path to write the markdown comparison table.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write raw JSON results.",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="strategy_a",
        choices=["strategy_a", "strategy_b"],
        help="Chunking strategy to apply to corpus.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K chunks to retrieve.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Running retrieval comparison on corpus: {args.corpus_path}")
    print(f"Using golden QA dataset: {args.golden_qa}")

    results, table = run_comparison_suite(
        corpus_path=args.corpus_path,
        golden_qa_path=args.golden_qa,
        output_table_path=args.output_table,
        output_json_path=args.output_json,
        chunking_strategy=args.chunking_strategy,
        top_k=args.top_k,
    )

    print("\nBenchmark Results:")
    print(table)
    print(f"Summary table written to: {args.output_table}")


if __name__ == "__main__":
    main()

