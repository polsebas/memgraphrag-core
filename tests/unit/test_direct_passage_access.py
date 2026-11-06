"""MemoryTax: raw M_pas stays readable when the writer never stored the span."""

from __future__ import annotations

import pytest

from memgraphrag_core.domain.models import LAYER_M_PAS, Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import VectorSearchResult
from memgraphrag_core.retrieval.direct_passage_access import (
    cited_passage_ids,
    merge_writer_omitted,
    select_direct_passages,
)
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


def test_select_direct_passages_ignores_writer_layers() -> None:
    passages = {
        "pas_low": Passage(id="pas_low", text="low"),
        "pas_high": Passage(id="pas_high", text="high"),
    }
    hits = [
        VectorSearchResult(id="pas_low", text="low", score=0.2),
        VectorSearchResult(id="pas_high", text="high", score=0.9),
        VectorSearchResult(id="missing", text="gone", score=1.0),
        VectorSearchResult(id="pas_zero", text="zero", score=0.0),
    ]
    chosen = select_direct_passages(hits, passages, k=1)
    assert [p.id for p in chosen] == ["pas_high"]
    assert select_direct_passages(hits, passages, k=0) == []


def test_merge_appends_uncited_passages_without_reordering() -> None:
    ranked = [
        Passage(id="pas_fact", text="cited first"),
        Passage(id="pas_other", text="cited second"),
    ]
    direct = [
        Passage(id="pas_secret", text="The cellar access code is 7341."),
        ranked[0],
    ]
    merged, omitted = merge_writer_omitted(ranked, direct, cited_ids={"pas_fact", "pas_other"})
    assert [p.id for p in merged] == ["pas_fact", "pas_other", "pas_secret"]
    assert omitted == ["pas_secret"]
    assert cited_passage_ids(
        {
            "k": Fact(
                head_entity="Catena",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
                passage_ids=["pas_fact"],
            )
        }
    ) == {"pas_fact"}


@pytest.mark.asyncio
async def test_reader_recovers_span_the_writer_never_stored() -> None:
    """A fact-only reader cannot see 7341; the direct M_pas channel can."""
    memory = GlobalMemoryState(tau=1)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    pas_loc = Passage(
        id="pas_loc",
        text="Catena Zapata winery is located in the Mendoza region.",
    )
    pas_secret = Passage(
        id="pas_secret",
        text="The cellar access code is 7341. Keep the cellar code private.",
    )
    memory.add_passage(pas_loc)
    memory.add_passage(pas_secret)
    fact = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_loc"],
        )
    )
    assert fact is not None

    await vectors.add_texts(
        texts=[pas_loc.text, pas_secret.text],
        ids=[pas_loc.id, pas_secret.id],
        namespace=LAYER_M_PAS,
    )
    await vectors.add_texts(
        texts=["Catena Zapata LOCATED_IN Mendoza"],
        ids=[fact.fact_key],
        namespace="M_fac",
    )
    await graph.add_node(GraphNode(id=f"ont_{fact.schema_key}", label="Schema"))
    await graph.add_node(GraphNode(id=f"fact_{fact.fact_key}", label="Fact"))
    await graph.add_node(GraphNode(id="pas_pas_loc", label="Passage"))
    await graph.add_node(GraphNode(id="pas_pas_secret", label="Passage"))
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
            target_id="pas_pas_loc",
            relation="EVIDENCE_IN",
        )
    )

    retriever = MemoryGuidedRetriever(
        memory,
        graph,
        vectors,
        top_k_passages=1,
        top_k_facts=5,
        direct_pas_k=2,
    )
    result = await retriever.retrieve("What is the cellar access code?")

    fact_blob = " ".join(
        f"{item.head_entity} {item.relation} {item.tail_entity}" for item in result.facts
    )
    assert "7341" not in fact_blob
    assert any("7341" in passage.text for passage in result.direct_passages)
    assert "pas_secret" in result.writer_omitted_passage_ids
    assert any(passage.id == "pas_secret" for passage in result.passages)
    assert result.retrieval_mode == "hybrid_rerank"
