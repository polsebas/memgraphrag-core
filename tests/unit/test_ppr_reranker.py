"""Advanced retrieval tests: layered PPR seeds and hybrid reranking."""

from __future__ import annotations

import pytest

from memgraphrag_core.domain.models import Fact, GlobalMemoryState, Passage
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import VectorSearchResult
from memgraphrag_core.retrieval.ppr_engine import LayeredPPRCalculator
from memgraphrag_core.retrieval.reranker import HybridReranker
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever


def test_layered_ppr_seed_weights_apply_layer_bias() -> None:
    calc = LayeredPPRCalculator(
        layer_weights={"M_ont": 0.2, "M_fac": 0.5, "M_pas": 0.3}
    )
    seeds = calc.build_seed_weights(
        ont_hits=[VectorSearchResult(id="s1", text="schema", score=1.0)],
        fac_hits=[VectorSearchResult(id="f1", text="fact", score=1.0)],
        pas_hits=[VectorSearchResult(id="p1", text="passage", score=1.0)],
    )
    assert seeds["ont_s1"] == pytest.approx(0.2)
    assert seeds["fact_f1"] == pytest.approx(0.5)
    assert seeds["pas_p1"] == pytest.approx(0.3)


def test_hybrid_reranker_changes_order_vs_vector_only() -> None:
    reranker = HybridReranker(alpha_hybrid=0.4, normalization="minmax")
    vec = {"a": 0.95, "b": 0.90, "c": 0.10}
    ppr = {"a": 0.05, "b": 0.80, "c": 0.90}

    vector_only_order = sorted(vec, key=vec.get, reverse=True)
    hybrid_order = reranker.rerank_ids(vec, ppr)

    assert vector_only_order[0] == "a"
    # Structural centrality should promote b/c over pure vector leader when alpha is low.
    assert hybrid_order[0] in {"b", "c"}
    assert hybrid_order != vector_only_order


def test_hybrid_reranker_minmax_bounds() -> None:
    reranker = HybridReranker(alpha_hybrid=0.5)
    ranked = reranker.fuse({"x": 1.0, "y": 3.0}, {"x": 10.0, "y": 20.0})
    assert ranked[0].item_id == "y"
    assert 0.0 <= ranked[0].vec_norm <= 1.0
    assert 0.0 <= ranked[0].ppr_norm <= 1.0


@pytest.mark.asyncio
async def test_layered_ppr_propagates_across_hierarchy() -> None:
    """Ontology seed should reach fact/passage nodes via bidirectional INSTANTIATES/EVIDENCE_IN."""
    graph = NetworkXGraphStore()
    await graph.add_node(GraphNode(id="ont_schema", label="Schema"))
    await graph.add_node(GraphNode(id="fact_f1", label="Fact"))
    await graph.add_node(GraphNode(id="pas_p1", label="Passage"))
    await graph.add_edge(
        GraphEdge(source_id="ont_schema", target_id="fact_f1", relation="INSTANTIATES")
    )
    await graph.add_edge(
        GraphEdge(source_id="fact_f1", target_id="pas_p1", relation="EVIDENCE_IN")
    )

    calc = LayeredPPRCalculator(
        layer_weights={"M_ont": 1.0, "M_fac": 0.0, "M_pas": 0.0},
        relation_weights={"INSTANTIATES": 1.0, "EVIDENCE_IN": 1.5},
        bidirectional=True,
    )
    scores = await calc.run(
        graph,
        ont_hits=[VectorSearchResult(id="schema", text="ont", score=1.0)],
        fac_hits=[],
        pas_hits=[],
    )
    assert scores
    assert scores.get("fact_f1", 0.0) > 0.0
    assert scores.get("pas_p1", 0.0) > 0.0


@pytest.mark.asyncio
async def test_retriever_hybrid_improves_over_vector_only_ranking() -> None:
    """A weakly similar but structurally central passage should surface via hybrid fusion."""
    memory = GlobalMemoryState(tau=1)
    graph = NetworkXGraphStore()
    vectors = InMemoryVectorStore()

    # Seed memory with a schema/fact linked to a passage that is only weakly lexical.
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    p_strong = Passage(
        id="pas_strong",
        text="random tourism brochure about beaches and hotels unrelated",
    )
    p_struct = Passage(
        id="pas_struct",
        text="The estate sits in the Andean foothills near Lujan.",
    )
    memory.add_passage(p_strong)
    memory.add_passage(p_struct)

    fact = memory.add_fact(
        Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="LOCATED_IN",
            tail_entity="Mendoza",
            tail_type="Region",
            passage_ids=["pas_struct"],
            confidence=0.95,
        )
    )
    assert fact is not None

    # Index texts so vector search prefers the noisy brochure for a wine query
    # (shared tokens like none) — craft scores via explicit documents.
    await vectors.add_texts(
        texts=["Winery LOCATED_IN Region"],
        ids=[fact.schema_key],
        namespace="M_ont",
    )
    await vectors.add_texts(
        texts=["Catena Zapata LOCATED_IN Mendoza winery region"],
        ids=[fact.fact_key],
        namespace="M_fac",
    )
    await vectors.add_texts(
        texts=[
            "Catena Zapata winery Mendoza wine location brochure tourism",
            p_struct.text,
        ],
        ids=["pas_strong", "pas_struct"],
        namespace="M_pas",
    )

    # Hierarchical graph: fact → ont, fact → evidence passage
    await graph.add_node(GraphNode(id=f"ont_{fact.schema_key}", label="Schema"))
    await graph.add_node(GraphNode(id=f"fact_{fact.fact_key}", label="Fact"))
    await graph.add_node(GraphNode(id="pas_pas_strong", label="Passage"))
    await graph.add_node(GraphNode(id="pas_pas_struct", label="Passage"))
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
            target_id="pas_pas_struct",
            relation="EVIDENCE_IN",
        )
    )

    query = "Where is Catena Zapata winery located in Mendoza?"

    # Vector-only baseline on M_pas
    vec_hits = await vectors.search_similarity(query, k=2, namespace="M_pas")
    vec_order = [h.id for h in vec_hits]
    assert vec_order[0] == "pas_strong"

    retriever = MemoryGuidedRetriever(
        memory_state=memory,
        graph_store=graph,
        vector_store=vectors,
        layer_weights={"M_ont": 0.25, "M_fac": 0.55, "M_pas": 0.20},
        alpha_hybrid=0.35,  # emphasize structural PPR
        top_k_passages=2,
        top_k_facts=3,
    )
    result = await retriever.retrieve(query)

    assert result.retrieval_mode == "hybrid_rerank"
    assert result.facts
    hybrid_passage_ids = [p.id for p in result.passages]
    # Structural evidence passage should outrank the lexically noisy brochure.
    assert "pas_struct" in hybrid_passage_ids
    assert hybrid_passage_ids[0] == "pas_struct"
    assert hybrid_passage_ids != vec_order[: len(hybrid_passage_ids)]
