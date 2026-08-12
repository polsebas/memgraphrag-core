"""Multihop retrieval benchmark: Precision@K, Recall@K, MRR, latency."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever, RetrievalResult

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "multihop_sample.json"


@dataclass
class QueryMetrics:
    query_id: str
    precision_at_k_facts: float
    recall_at_k_facts: float
    precision_at_k_passages: float
    recall_at_k_passages: float
    mrr_facts: float
    mrr_passages: float
    latency_ms: float


@dataclass
class BenchmarkReport:
    k: int
    n_queries: int
    mean_precision_facts: float
    mean_recall_facts: float
    mean_precision_passages: float
    mean_recall_passages: float
    mean_mrr_facts: float
    mean_mrr_passages: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_query: List[QueryMetrics]


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in relevant)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # Nearest-rank style
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


async def build_benchmark_index() -> tuple[
    GlobalMemoryState, NetworkXGraphStore, InMemoryVectorStore
]:
    """Synthetic hierarchical corpus used by the sample multihop dataset."""
    memory = GlobalMemoryState(tau=2)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    for _ in range(2):
        memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
        memory.register_schema_candidate("Winery", "PRODUCES", "Wine")

    passages = [
        Passage(
            id="pas_001",
            text="Catena Zapata winery is located in the Mendoza region, Argentina.",
        ),
        Passage(
            id="pas_002",
            text="Catena Zapata produces the Gran Enemigo wine in Gualtallary, Mendoza.",
        ),
        Passage(
            id="pas_003",
            text="Norton winery is located in Lujan de Cuyo and produces Malbec Reserva.",
        ),
    ]
    for p in passages:
        memory.add_passage(p)
        await vectors.add_texts(texts=[p.text], ids=[p.id], namespace="M_pas")
        await graph.add_node(GraphNode(id=f"pas_{p.id}", label="Passage"))

    facts = [
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_001"],
            confidence=0.95,
        ),
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="Gran Enemigo",
            tail_type="Wine",
            passage_ids=["pas_002"],
            confidence=0.93,
        ),
        Fact(
            head_entity="Norton",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Lujan de Cuyo",
            tail_type="Region",
            passage_ids=["pas_003"],
            confidence=0.9,
        ),
        Fact(
            head_entity="Norton",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="Malbec Reserva",
            tail_type="Wine",
            passage_ids=["pas_003"],
            confidence=0.88,
        ),
    ]
    for fact in facts:
        accepted = memory.add_fact(fact)
        assert accepted is not None
        await vectors.add_texts(
            texts=[f"{fact.head_entity} {fact.relation} {fact.tail_entity}"],
            ids=[fact.fact_key],
            namespace="M_fac",
        )
        fid = f"fact_{fact.fact_key}"
        await graph.add_node(GraphNode(id=fid, label="Fact"))
        await graph.add_edge(
            GraphEdge(
                source_id=f"ont_{fact.schema_key}",
                target_id=fid,
                relation="INSTANTIATES",
            )
        )
        for pid in fact.passage_ids:
            await graph.add_edge(
                GraphEdge(source_id=fid, target_id=f"pas_{pid}", relation="EVIDENCE_IN")
            )

    for schema in memory.ontologies.values():
        await vectors.add_texts(
            texts=[f"{schema.head_type} {schema.relation} {schema.tail_type}"],
            ids=[schema.schema_key],
            namespace="M_ont",
        )
        await graph.add_node(
            GraphNode(
                id=f"ont_{schema.schema_key}",
                label="Schema",
                properties={"frequency": schema.frequency},
            )
        )

    return memory, graph, vectors


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_result(
    query_id: str,
    result: RetrievalResult,
    ground_truth_fact_ids: Iterable[str],
    ground_truth_passage_ids: Iterable[str],
    *,
    k: int,
    latency_ms: float,
) -> QueryMetrics:
    fact_ids = [f.fact_key for f in result.facts]
    passage_ids = [p.id for p in result.passages]
    gt_facts = set(ground_truth_fact_ids)
    gt_passages = set(ground_truth_passage_ids)
    return QueryMetrics(
        query_id=query_id,
        precision_at_k_facts=precision_at_k(fact_ids, gt_facts, k),
        recall_at_k_facts=recall_at_k(fact_ids, gt_facts, k),
        precision_at_k_passages=precision_at_k(passage_ids, gt_passages, k),
        recall_at_k_passages=recall_at_k(passage_ids, gt_passages, k),
        mrr_facts=mrr(fact_ids, gt_facts),
        mrr_passages=mrr(passage_ids, gt_passages),
        latency_ms=latency_ms,
    )


async def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    k: int = 5,
    alpha_hybrid: float = 0.55,
) -> BenchmarkReport:
    dataset = load_dataset(dataset_path)
    memory, graph, vectors = await build_benchmark_index()
    retriever = MemoryGuidedRetriever(
        memory_state=memory,
        graph_store=graph,
        vector_store=vectors,
        layer_weights={"M_ont": 0.25, "M_fac": 0.5, "M_pas": 0.25},
        alpha_hybrid=alpha_hybrid,
        top_k_facts=k,
        top_k_passages=k,
    )

    per_query: List[QueryMetrics] = []
    for row in dataset:
        t0 = time.perf_counter()
        result = await retriever.retrieve(row["query"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        per_query.append(
            evaluate_result(
                row["id"],
                result,
                row.get("ground_truth_fact_ids", []),
                row.get("ground_truth_passage_ids", []),
                k=k,
                latency_ms=latency_ms,
            )
        )

    latencies = [q.latency_ms for q in per_query]
    return BenchmarkReport(
        k=k,
        n_queries=len(per_query),
        mean_precision_facts=statistics.fmean(q.precision_at_k_facts for q in per_query),
        mean_recall_facts=statistics.fmean(q.recall_at_k_facts for q in per_query),
        mean_precision_passages=statistics.fmean(
            q.precision_at_k_passages for q in per_query
        ),
        mean_recall_passages=statistics.fmean(q.recall_at_k_passages for q in per_query),
        mean_mrr_facts=statistics.fmean(q.mrr_facts for q in per_query),
        mean_mrr_passages=statistics.fmean(q.mrr_passages for q in per_query),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        per_query=per_query,
    )


def format_report(report: BenchmarkReport) -> str:
    lines = [
        "=" * 64,
        "MemGraphRAG Multihop Benchmark",
        "=" * 64,
        f"Queries: {report.n_queries} | K={report.k}",
        f"Facts     P@{report.k}={report.mean_precision_facts:.3f}  "
        f"R@{report.k}={report.mean_recall_facts:.3f}  "
        f"MRR={report.mean_mrr_facts:.3f}",
        f"Passages  P@{report.k}={report.mean_precision_passages:.3f}  "
        f"R@{report.k}={report.mean_recall_passages:.3f}  "
        f"MRR={report.mean_mrr_passages:.3f}",
        f"Latency   P50={report.latency_p50_ms:.2f} ms  "
        f"P95={report.latency_p95_ms:.2f} ms",
        "-" * 64,
    ]
    for q in report.per_query:
        lines.append(
            f"{q.query_id}: fact P/R/MRR="
            f"{q.precision_at_k_facts:.2f}/{q.recall_at_k_facts:.2f}/{q.mrr_facts:.2f} "
            f"| pas P/R/MRR="
            f"{q.precision_at_k_passages:.2f}/{q.recall_at_k_passages:.2f}/{q.mrr_passages:.2f} "
            f"| {q.latency_ms:.2f} ms"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    payload = asdict(report)
    return payload


async def _amain(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MemGraphRAG multihop benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to multihop JSON dataset",
    )
    parser.add_argument("--k", type=int, default=5, help="Cutoff K for Precision/Recall")
    parser.add_argument("--alpha-hybrid", type=float, default=0.55)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )
    args = parser.parse_args(argv)

    report = await run_benchmark(
        args.dataset, k=args.k, alpha_hybrid=args.alpha_hybrid
    )
    print(format_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report_to_dict(report), indent=2), encoding="utf-8"
        )
        print(f"Wrote JSON report to {args.output}")
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    raise SystemExit(asyncio.run(_amain(argv)))


if __name__ == "__main__":
    main()
