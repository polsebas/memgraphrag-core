"""MemoryGuidedRetriever tests with in-memory adapters."""

import pytest

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


@pytest.mark.asyncio
async def test_retrieve_memory_guided_ppr() -> None:
    memory = GlobalMemoryState(tau=2)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    p1 = Passage(
        id="pas_001",
        text="Catena Zapata winery is located in the Mendoza region, Argentina.",
    )
    p2 = Passage(
        id="pas_002",
        text="Catena Zapata produces the Gran Enemigo wine in Gualtallary, Mendoza.",
    )
    memory.add_passage(p1)
    memory.add_passage(p2)
    await vectors.add_texts(
        texts=[p1.text, p2.text], ids=[p1.id, p2.id], namespace="M_pas"
    )

    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")

    f1 = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_001"],
        )
    )
    f2 = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="Gran Enemigo",
            tail_type="Wine",
            passage_ids=["pas_002"],
        )
    )
    assert f1 and f2

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

    retriever = MemoryGuidedRetriever(memory, graph, vectors)
    result = await retriever.retrieve(
        "Where is Catena Zapata winery located and what wine does it produce?"
    )

    assert result.retrieval_mode in {"hybrid_rerank", "memory_guided_ppr"}
    assert len(result.facts) >= 1
    assert len(result.passages) >= 1
    assert result.hybrid_scores
