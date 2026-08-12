"""Integration tests for MemoryGuidedRetriever (layered PPR + hybrid rerank)."""

from __future__ import annotations

import pytest

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


async def _build_wine_memory() -> tuple[GlobalMemoryState, NetworkXGraphStore, InMemoryVectorStore]:
    memory = GlobalMemoryState(tau=2)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")

    p1 = Passage(
        id="pas_001",
        text="Catena Zapata winery is located in the Mendoza region, Argentina.",
    )
    p2 = Passage(
        id="pas_002",
        text="Catena Zapata produces the Gran Enemigo wine in Gualtallary, Mendoza.",
    )
    p_noise = Passage(
        id="pas_noise",
        text="Generic tourism brochure about beaches hotels and nightlife.",
    )
    for p in (p1, p2, p_noise):
        memory.add_passage(p)

    f_loc = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_001"],
        )
    )
    f_prod = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="Gran Enemigo",
            tail_type="Wine",
            passage_ids=["pas_002"],
        )
    )
    assert f_loc and f_prod

    await vectors.add_texts(
        texts=[p1.text, p2.text, p_noise.text],
        ids=[p1.id, p2.id, p_noise.id],
        namespace="M_pas",
    )
    for s in memory.ontologies.values():
        await vectors.add_texts(
            texts=[f"{s.head_type} {s.relation} {s.tail_type}"],
            ids=[s.schema_key],
            namespace="M_ont",
        )
        await graph.add_node(GraphNode(id=f"ont_{s.schema_key}", label="Schema"))

    for fact in memory.facts.values():
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
            await graph.add_node(GraphNode(id=f"pas_{pid}", label="Passage"))
            await graph.add_edge(
                GraphEdge(source_id=fid, target_id=f"pas_{pid}", relation="EVIDENCE_IN")
            )

    await graph.add_node(GraphNode(id="pas_pas_noise", label="Passage"))
    return memory, graph, vectors


@pytest.mark.asyncio
async def test_retrieval_flow_hybrid_mode() -> None:
    memory, graph, vectors = await _build_wine_memory()
    retriever = MemoryGuidedRetriever(
        memory_state=memory,
        graph_store=graph,
        vector_store=vectors,
        layer_weights={"M_ont": 0.2, "M_fac": 0.5, "M_pas": 0.3},
        alpha_hybrid=0.55,
    )
    result = await retriever.retrieve(
        "Where is Catena Zapata winery located and what wine does it produce?"
    )
    assert result.retrieval_mode == "hybrid_rerank"
    assert len(result.facts) >= 1
    assert len(result.passages) >= 1
    assert result.hybrid_scores
    assert result.subgraph.get("nodes")


@pytest.mark.asyncio
async def test_memory_guided_beats_vector_only_on_noisy_corpus() -> None:
    memory, graph, vectors = await _build_wine_memory()
    # Make noise lexically attractive for a wine/location query.
    await vectors.add_texts(
        texts=["Catena Zapata Mendoza winery wine tourism brochure beaches hotels"],
        ids=["pas_noise"],
        namespace="M_pas",
    )

    query = "What wine does the winery located in Mendoza produce?"
    vec_hits = await vectors.search_similarity(query, k=3, namespace="M_pas")
    vec_ids = [h.id for h in vec_hits]

    retriever = MemoryGuidedRetriever(
        memory_state=memory,
        graph_store=graph,
        vector_store=vectors,
        layer_weights={"M_ont": 0.25, "M_fac": 0.55, "M_pas": 0.20},
        alpha_hybrid=0.35,
        top_k_passages=2,
        top_k_facts=5,
    )
    result = await retriever.retrieve(query)
    hybrid_ids = [p.id for p in result.passages]

    assert result.retrieval_mode == "hybrid_rerank"
    assert any(pid.startswith("pas_00") for pid in hybrid_ids)
    # Grounded evidence should appear; ordering should differ from pure vector when noise ranks high.
    if vec_ids and vec_ids[0] == "pas_noise":
        assert hybrid_ids[0] != "pas_noise"


@pytest.mark.asyncio
async def test_vector_fallback_when_graph_empty() -> None:
    memory = GlobalMemoryState(tau=2)
    vectors = InMemoryVectorStore()
    graph = NetworkXGraphStore()
    p = Passage(id="pas_only", text="Catena Zapata produces Gran Enemigo in Mendoza.")
    memory.add_passage(p)
    await vectors.add_texts(texts=[p.text], ids=[p.id], namespace="M_pas")

    retriever = MemoryGuidedRetriever(memory, graph, vectors)
    result = await retriever.retrieve("Gran Enemigo wine Mendoza")
    assert result.retrieval_mode == "vector_fallback"
    assert result.passages and result.passages[0].id == "pas_only"
    assert result.facts == []
