"""Reproducible MemoryTax / MemDev-Bench comparison on a travel-agent session.

Run::

    python -m benchmarks.verify_memory_tax --mode deterministic
    python -m benchmarks.verify_memory_tax --mode llm

``deterministic`` replays the annotated graph in the scenario file and does not
call a model. ``llm`` extracts facts with an OpenAI-compatible endpoint
(Ollama by default) and then scores the same queries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from memgraphrag_core.domain.models import (
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    NODE_FACT,
    NODE_PASSAGE,
    NODE_SCHEMA,
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    Fact,
    FactCategory,
    GlobalMemoryState,
    Passage,
)
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.ingestion.extractor import LLMExtractor
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.llm_provider import BaseLLMProvider
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever

SCENARIO_PATH = Path(__file__).resolve().parent / "scenarios" / "travel_agent_session.json"
_SAFE_TOPICS = ("clima", "equipaje", "seguro", "auto")


@dataclass(frozen=True)
class MetricRow:
    engine: str
    process_recall: float
    constraint_violation: float
    tokens: int


def load_scenario(path: Path = SCENARIO_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def _contains_all(haystack: str, needles: Sequence[str]) -> bool:
    folded = haystack.casefold()
    return all(needle.casefold() in folded for needle in needles)


def _contains_any(haystack: str, needles: Sequence[str]) -> bool:
    folded = haystack.casefold()
    return any(needle.casefold() in folded for needle in needles)


def lossy_window_summary(turns: Sequence[Mapping[str, Any]]) -> str:
    """Compact a window the way a summarizer does: keep the trip sketch, drop process.

    Rejection motives and tool error codes are not copied into the paragraph.
    """
    topics: List[str] = []
    for turn in turns:
        folded = str(turn.get("text", "")).casefold()
        for topic in _SAFE_TOPICS:
            if topic in folded and topic not in topics:
                topics.append(topic)
    topic_clause = f" y consultó {', '.join(topics)}" if topics else ""
    return (
        "El cliente viaja a Madrid con 4 personas, presupuesto USD 4200"
        f"{topic_clause}."
    )


def baseline_memory_text(turns: Sequence[Mapping[str, Any]]) -> str:
    """Rewrite history every 4 turns into lossy paragraphs."""
    paragraphs: List[str] = []
    window: List[Mapping[str, Any]] = []
    for turn in turns:
        window.append(turn)
        if len(window) == 4:
            paragraphs.append(lossy_window_summary(window))
            window = []
    if window:
        paragraphs.append(lossy_window_summary(window))
    return " ".join(paragraphs)


def full_history_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(turn["text"]) for turn in turns)


def _passage_id(turn_number: int) -> str:
    return f"turn_{turn_number:02d}"


async def _index_fact(
    memory: GlobalMemoryState,
    vectors: InMemoryVectorStore,
    graph: NetworkXGraphStore,
    fact: Fact,
) -> None:
    memory.register_schema_candidate(fact.head_type, fact.relation, fact.tail_type)
    accepted = memory.add_fact(fact)
    if accepted is None:
        return
    await vectors.add_texts(
        texts=[f"{accepted.head_type} {accepted.relation} {accepted.tail_type}"],
        ids=[accepted.schema_key],
        namespace=LAYER_M_ONT,
    )
    await vectors.add_texts(
        texts=[f"{accepted.head_entity} {accepted.relation} {accepted.tail_entity}"],
        ids=[accepted.fact_key],
        namespace=LAYER_M_FAC,
    )
    await graph.add_node(GraphNode(id=f"ont_{accepted.schema_key}", label=NODE_SCHEMA))
    await graph.add_node(
        GraphNode(
            id=f"fact_{accepted.fact_key}",
            label=NODE_FACT,
            properties=accepted.graph_properties(),
        )
    )
    await graph.add_edge(
        GraphEdge(
            source_id=f"ont_{accepted.schema_key}",
            target_id=f"fact_{accepted.fact_key}",
            relation=EDGE_INSTANTIATES,
        )
    )
    for pid in accepted.passage_ids:
        await graph.add_edge(
            GraphEdge(
                source_id=f"fact_{accepted.fact_key}",
                target_id=f"pas_{pid}",
                relation=EDGE_EVIDENCE_IN,
            )
        )


async def build_memgraph_retriever(
    turns: Sequence[Mapping[str, Any]],
    *,
    use_annotations: bool,
    extractor: Optional[LLMExtractor] = None,
) -> MemoryGuidedRetriever:
    memory = GlobalMemoryState(tau=1)
    vectors = InMemoryVectorStore()
    graph = NetworkXGraphStore()
    for turn in turns:
        turn_number = int(turn["turn"])
        text = str(turn["text"])
        pid = _passage_id(turn_number)
        passage = Passage(
            id=pid,
            text=text,
            metadata={"turn": turn_number, "role": turn.get("role", "user")},
        )
        memory.add_passage(passage)
        await vectors.add_texts(texts=[text], ids=[pid], namespace=LAYER_M_PAS)
        await graph.add_node(GraphNode(id=f"pas_{pid}", label=NODE_PASSAGE))

        if use_annotations:
            specs = list(turn.get("facts") or [])
        else:
            if extractor is None:
                raise RuntimeError("llm mode requires an extractor")
            extracted = await extractor.extract_from_text(text)
            specs = [item.model_dump() for item in extracted.facts]

        for spec in specs:
            fact = Fact(
                head_entity=str(spec["head_entity"]),
                head_type=str(spec["head_type"]),
                relation=str(spec["relation"]),
                tail_entity=str(spec["tail_entity"]),
                tail_type=str(spec["tail_type"]),
                passage_ids=[pid],
                source_passage_id=pid,
                category=spec.get("category", FactCategory.DECLARATIVE),
            )
            await _index_fact(memory, vectors, graph, fact)
    return MemoryGuidedRetriever(
        memory,
        graph,
        vectors,
        top_k_passages=2,
        top_k_facts=4,
        alpha_hybrid=0.45,
    )


def _render_memgraph_context(result: Any) -> str:
    chunks: List[str] = []
    seen: set[str] = set()
    for fact in result.facts:
        line = (
            f"[{fact.category.value}] {fact.head_entity} {fact.relation} {fact.tail_entity}"
        )
        if line not in seen:
            chunks.append(line)
            seen.add(line)
    for passage in result.passages:
        if passage.text not in seen:
            chunks.append(passage.text)
            seen.add(passage.text)
    return "\n".join(chunks)


async def _contexts_for_queries(
    retriever: MemoryGuidedRetriever, queries: Sequence[Mapping[str, Any]]
) -> List[str]:
    contexts: List[str] = []
    for query in queries:
        result = await retriever.retrieve(str(query["text"]))
        contexts.append(_render_memgraph_context(result))
    return contexts


def score_engine(
    engine: str,
    queries: Sequence[Mapping[str, Any]],
    contexts: Sequence[str],
    *,
    footprint_text: str,
) -> MetricRow:
    process_hits = 0
    process_total = 0
    violations = 0
    constraint_total = 0
    for query, context in zip(queries, contexts):
        kind = query["kind"]
        if kind == "process":
            process_total += 1
            if _contains_all(context, query["needles"]):
                process_hits += 1
        elif kind == "constraint":
            constraint_total += 1
            if not _contains_any(context, query["block_if_any"]):
                violations += 1
    process_recall = 100.0 * process_hits / process_total if process_total else 0.0
    violation_rate = 100.0 * violations / constraint_total if constraint_total else 0.0
    return MetricRow(
        engine=engine,
        process_recall=process_recall,
        constraint_violation=violation_rate,
        tokens=whitespace_tokens(footprint_text),
    )


async def evaluate(scenario: Mapping[str, Any], *, mode: str = "deterministic") -> List[MetricRow]:
    turns = list(scenario["turns"])
    queries = list(scenario["queries"])
    if len(turns) != 12:
        raise ValueError(f"expected 12 turns, found {len(turns)}")
    if len(queries) != 5:
        raise ValueError(f"expected 5 control queries, found {len(queries)}")

    summary = baseline_memory_text(turns)
    history = full_history_text(turns)
    baseline_contexts = [summary for _ in queries]
    history_contexts = [history for _ in queries]

    if mode == "deterministic":
        retriever = await build_memgraph_retriever(turns, use_annotations=True)
    elif mode == "llm":
        retriever = await build_memgraph_retriever(
            turns,
            use_annotations=False,
            extractor=LLMExtractor(_provider_from_env()),
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    graph_contexts = await _contexts_for_queries(retriever, queries)
    constraint_index = next(i for i, query in enumerate(queries) if query["kind"] == "constraint")
    return [
        score_engine(
            "BaselineSummarizerMemory",
            queries,
            baseline_contexts,
            footprint_text=baseline_contexts[constraint_index],
        ),
        score_engine(
            "MemGraphRAGCore",
            queries,
            graph_contexts,
            footprint_text=graph_contexts[constraint_index],
        ),
        score_engine(
            "FullRawHistory",
            queries,
            history_contexts,
            footprint_text=history,
        ),
    ]


def render_markdown(rows: Iterable[MetricRow]) -> str:
    lines = [
        "| Engine | Process Recall (%) | Constraint Violation Rate (%) | Context Token Footprint |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.engine} | {row.process_recall:.1f} | {row.constraint_violation:.1f} | {row.tokens} |"
        )
    return "\n".join(lines)


class _OpenAICompatibleProvider(BaseLLMProvider):
    """Minimal chat-completions client for ``--mode llm`` (Ollama or OpenAI)."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        return await asyncio.to_thread(
            self._complete_sync, prompt, system_prompt, temperature
        )

    async def complete_structured(
        self,
        prompt: str,
        response_model: type,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        schema = json.dumps(response_model.model_json_schema())
        raw = await self.complete(
            prompt + "\n\nReturn only a JSON object matching this schema:\n" + schema,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = match.group(0) if match else raw
        return response_model.model_validate_json(payload)

    def _complete_sync(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {"model": self.model, "temperature": temperature, "messages": messages}
        ).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(
                "LLM endpoint unreachable. Set MEMGRAPHRAG_LLM_BASE_URL / "
                "MEMGRAPHRAG_LLM_MODEL, or rerun with --mode deterministic. "
                f"Detail: {exc}"
            ) from exc
        return str(payload["choices"][0]["message"]["content"])


def _provider_from_env() -> BaseLLMProvider:
    base_url = os.environ.get("MEMGRAPHRAG_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    model = os.environ.get("MEMGRAPHRAG_LLM_MODEL", "llama3.2")
    api_key = os.environ.get("MEMGRAPHRAG_LLM_API_KEY") or os.environ.get(
        "OPENAI_API_KEY", "ollama"
    )
    return _OpenAICompatibleProvider(base_url, model, api_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "llm"),
        default="deterministic",
        help="deterministic uses the annotated scenario; llm calls the configured provider",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    scenario = load_scenario()
    rows = asyncio.run(evaluate(scenario, mode=args.mode))
    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
