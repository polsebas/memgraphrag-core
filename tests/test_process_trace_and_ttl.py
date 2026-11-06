"""Process categories, M_pas drill-down, and TTL pruning inside PPR."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pytest

from memgraphrag_core.domain.models import (
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    Fact,
    FactCategory,
    GlobalMemoryState,
    Passage,
)
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import BaseVectorStore, VectorSearchResult
from memgraphrag_core.retrieval.ppr_engine import ALPHA_CONSTRAINT, W_EXPIRED
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever

T0 = datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc)


class ScriptedVectorStore(BaseVectorStore):
    """Returns a fixed hit list per namespace. Ignores the query text."""

    def __init__(self, hits: Dict[str, List[VectorSearchResult]]) -> None:
        self.hits = hits

    async def add_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[dict]] = None,
        namespace: str = "default",
    ) -> List[str]:
        return list(ids)

    async def search_similarity(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter_metadata: Optional[dict] = None,
    ) -> List[VectorSearchResult]:
        return list(self.hits.get(namespace, []))[:k]

    async def delete(self, ids: List[str], namespace: str = "default") -> None:
        return None


async def _link(graph: NetworkXGraphStore, fact: Fact) -> None:
    passage_id = fact.source_passage_id
    await graph.add_node(GraphNode(id=f"ont_{fact.schema_key}", label="Schema"))
    await graph.add_node(
        GraphNode(id=f"fact_{fact.fact_key}", label="Fact", properties=fact.graph_properties())
    )
    await graph.add_node(GraphNode(id=f"pas_{passage_id}", label="Passage"))
    await graph.add_edge(
        GraphEdge(
            source_id=f"ont_{fact.schema_key}",
            target_id=f"fact_{fact.fact_key}",
            relation="INSTANTIATES",
        )
    )
    await graph.add_edge(
        GraphEdge(
            source_id=f"fact_{fact.fact_key}",
            target_id=f"pas_{passage_id}",
            relation="EVIDENCE_IN",
        )
    )


def _memory_with(*facts: Fact, passages: List[Passage]) -> GlobalMemoryState:
    memory = GlobalMemoryState(tau=1)
    for passage in passages:
        memory.add_passage(passage)
    seen = set()
    for fact in facts:
        if fact.schema_key not in seen:
            memory.register_schema_candidate(fact.head_type, fact.relation, fact.tail_type)
            seen.add(fact.schema_key)
        assert memory.add_fact(fact) is not None
    return memory


@pytest.mark.asyncio
async def test_rejected_option_outranks_irrelevant_declarative_fact() -> None:
    assert ALPHA_CONSTRAINT == 1.35
    assert W_EXPIRED == 0.0
    miami = Passage(
        id="pas_miami",
        text="El cliente rechazó la escala en Miami por falta de visa de tránsito.",
    )
    weather = Passage(
        id="pas_weather",
        text="El clima en Madrid en noviembre es templado.",
    )
    rejected = Fact(
        head_entity="Cliente",
        head_type="Traveler",
        relation="REJECTED_ROUTE",
        tail_entity="Miami",
        tail_type="City",
        passage_ids=["pas_miami"],
        source_passage_id="pas_miami",
        category=FactCategory.REJECTED_OPTION,
        created_at=T0,
    )
    irrelevant = Fact(
        head_entity="Madrid",
        head_type="City",
        relation="HAS_WEATHER",
        tail_entity="Templado",
        tail_type="Weather",
        passage_ids=["pas_weather"],
        source_passage_id="pas_weather",
        category=FactCategory.DECLARATIVE,
        created_at=T0,
    )
    memory = _memory_with(rejected, irrelevant, passages=[miami, weather])
    graph = NetworkXGraphStore()
    await _link(graph, rejected)
    await _link(graph, irrelevant)
    vectors = ScriptedVectorStore(
        {
            LAYER_M_ONT: [
                VectorSearchResult(id=rejected.schema_key, text="route", score=1.0),
                VectorSearchResult(id=irrelevant.schema_key, text="weather", score=1.0),
            ],
            LAYER_M_FAC: [
                VectorSearchResult(id=rejected.fact_key, text="miami", score=0.5),
                VectorSearchResult(id=irrelevant.fact_key, text="weather", score=0.5),
            ],
            LAYER_M_PAS: [
                VectorSearchResult(id="pas_miami", text=miami.text, score=0.4),
                VectorSearchResult(id="pas_weather", text=weather.text, score=0.4),
            ],
        }
    )
    retriever = MemoryGuidedRetriever(
        memory,
        graph,
        vectors,
        top_k_facts=2,
        top_k_passages=2,
        alpha_hybrid=0.6,
    )
    result = await retriever.retrieve(
        "El cliente pregunta si la escala en Miami sigue disponible",
        as_of=T0,
    )
    order = [fact.fact_key for fact in result.facts]
    assert order[0] == rejected.fact_key
    assert irrelevant.fact_key in order
    assert result.facts[0].category is FactCategory.REJECTED_OPTION


@pytest.mark.asyncio
async def test_expired_ttl_fact_is_pruned_from_returned_context() -> None:
    live_passage = Passage(id="pas_live", text="Viaje familiar a Madrid para 4 personas.")
    stale_passage = Passage(id="pas_stale", text="Cotización vencida del hotel norte.")
    live = Fact(
        head_entity="Cliente",
        head_type="Traveler",
        relation="TRAVELS_TO",
        tail_entity="Madrid",
        tail_type="City",
        passage_ids=["pas_live"],
        source_passage_id="pas_live",
        category=FactCategory.DECLARATIVE,
        created_at=T0,
    )
    expired = Fact(
        head_entity="Hotel Norte",
        head_type="Hotel",
        relation="QUOTED_PRICE",
        tail_entity="900 USD",
        tail_type="Price",
        passage_ids=["pas_stale"],
        source_passage_id="pas_stale",
        category=FactCategory.DECLARATIVE,
        created_at=T0,
        ttl_seconds=3600,
    )
    memory = _memory_with(live, expired, passages=[live_passage, stale_passage])
    graph = NetworkXGraphStore()
    await _link(graph, live)
    await _link(graph, expired)
    vectors = ScriptedVectorStore(
        {
            LAYER_M_ONT: [
                VectorSearchResult(id=live.schema_key, text="trip", score=1.0),
                VectorSearchResult(id=expired.schema_key, text="quote", score=1.0),
            ],
            LAYER_M_FAC: [
                VectorSearchResult(id=live.fact_key, text="madrid", score=0.8),
                VectorSearchResult(id=expired.fact_key, text="quote", score=0.8),
            ],
            LAYER_M_PAS: [
                VectorSearchResult(id="pas_live", text=live_passage.text, score=0.7),
                VectorSearchResult(id="pas_stale", text=stale_passage.text, score=0.7),
            ],
        }
    )
    retriever = MemoryGuidedRetriever(memory, graph, vectors, top_k_facts=5, top_k_passages=5)
    result = await retriever.retrieve(
        "viaje a Madrid",
        as_of=T0 + timedelta(seconds=3600),
    )
    returned_ids = {fact.fact_key for fact in result.facts}
    returned_passages = {passage.id for passage in result.passages}
    assert live.fact_key in returned_ids
    assert expired.fact_key not in returned_ids
    assert "pas_stale" not in returned_passages
    assert all(not node["id"].endswith(expired.fact_key) for node in result.subgraph["nodes"])
    assert expired.fact_key in memory.facts


@pytest.mark.asyncio
async def test_expand_fact_evidence_returns_exact_raw_passage() -> None:
    raw = "Rechazó escala en Miami porque dos pasajeros no tienen visa de tránsito."
    passage = Passage(id="pas_visa", text=raw)
    fact = Fact(
        head_entity="Cliente",
        head_type="Traveler",
        relation="REJECTED_ROUTE",
        tail_entity="Miami",
        tail_type="City",
        passage_ids=["pas_visa"],
        source_passage_id="pas_visa",
        category=FactCategory.REJECTED_OPTION,
    )
    memory = _memory_with(fact, passages=[passage])
    retriever = MemoryGuidedRetriever(memory, NetworkXGraphStore(), ScriptedVectorStore({}))
    assert retriever.get_raw_passage("pas_visa").text == raw
    assert retriever.expand_fact_evidence(fact.fact_id or fact.fact_key) == raw
    assert fact.category is FactCategory.REJECTED_OPTION
    assert fact.source_passage_id == "pas_visa"
