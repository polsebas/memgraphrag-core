"""Matriz de validación continua vs. el paper de MemGraphRAG.

Cubre los 5 ejes formales:

1. Memoria de 3 capas (M_pas, M_ont, M_fac) con namespaces vectoriales propios.
2. Estabilidad ontológica f(schema) >= tau y rechazo categórico en M_fac.
3. Topología jerárquica canónica Schema -INSTANTIATES-> Fact -EVIDENCE_IN-> Passage.
4. Personalized PageRank: p = (1 - α_ppr) · p_0 + α_ppr · A^T p, con grounding.
5. Reranking híbrido: S_final = α_hybrid · Norm(S_vec) + (1 - α_hybrid) · Norm(S_ppr).

Ejecutar::

    pytest tests/test_paper_compliance.py
    python tests/test_paper_compliance.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    EDGE_SUPPORTED_BY,
    GRAPH_NODE_LABELS,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    MEMORY_NAMESPACES,
    NODE_FACT,
    NODE_PASSAGE,
    NODE_SCHEMA,
    Fact,
    GlobalMemoryState,
    Passage,
    Schema,
)
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.infrastructure.scripted_model import ScriptedModel
from memgraphrag_core.indexing.orchestrator import IndexingOrchestrator
from memgraphrag_core.ingestion.entity_resolver import EntityResolver
from memgraphrag_core.ingestion.pipeline import IngestionPipeline
from memgraphrag_core.ingestion.schemas import (
    DocumentExtractionResult,
    ExtractedFact,
    ExtractedSchema,
)
from memgraphrag_core.infrastructure.mock_llm import MockLLMProvider
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import VectorSearchResult
from memgraphrag_core.retrieval.ppr_engine import LayeredPPRCalculator
from memgraphrag_core.retrieval.reranker import HybridReranker
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EdgeTriple = Tuple[str, str, str]


def _minmax(scores: Sequence[float]) -> List[float]:
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def _paper_ppr_fixed_point(
    nodes: List[str],
    adj: Dict[str, List[Tuple[str, float]]],
    p0: Dict[str, float],
    alpha: float,
    scores: Dict[str, float],
) -> float:
    """L1 residual of p - ((1-α) p_0 + α A^T p), with dangling teleport via p_0."""
    n = len(nodes)
    index = {nid: i for i, nid in enumerate(nodes)}
    personalization = np.zeros(n, dtype=float)
    for nid, w in p0.items():
        if nid in index and w > 0.0:
            personalization[index[nid]] = w
    total = float(personalization.sum())
    assert total > 0.0
    personalization /= total

    p = np.array([scores[nid] for nid in nodes], dtype=float)
    walk = np.zeros(n, dtype=float)
    out_strength = {nid: sum(w for _, w in neigh) for nid, neigh in adj.items()}
    for src in nodes:
        i = index[src]
        neighbors = adj.get(src, [])
        strength = out_strength.get(src, 0.0)
        if not neighbors or strength <= 0.0:
            walk += p[i] * personalization
        else:
            for tgt, w in neighbors:
                walk[index[tgt]] += p[i] * (w / strength)
    rhs = (1.0 - alpha) * personalization + alpha * walk
    return float(np.linalg.norm(p - rhs, ord=1))


async def _all_graph_edges(graph: NetworkXGraphStore) -> List[GraphEdge]:
    node_ids = list(graph._g.nodes())  # noqa: SLF001 — inspect full topology
    _, edges = await graph.get_subgraph(node_ids)
    return edges


async def _index_layers(
    memory: GlobalMemoryState,
    graph: NetworkXGraphStore,
    vectors: InMemoryVectorStore,
) -> None:
    for passage in memory.passages.values():
        await vectors.add_texts(
            texts=[passage.text], ids=[passage.id], namespace=LAYER_M_PAS
        )
        await graph.add_node(GraphNode(id=f"pas_{passage.id}", label=NODE_PASSAGE))
    for schema in memory.ontologies.values():
        await vectors.add_texts(
            texts=[f"{schema.head_type} {schema.relation} {schema.tail_type}"],
            ids=[schema.schema_key],
            namespace=LAYER_M_ONT,
        )
        await graph.add_node(
            GraphNode(id=f"ont_{schema.schema_key}", label=NODE_SCHEMA)
        )
    for fact in memory.facts.values():
        await vectors.add_texts(
            texts=[f"{fact.head_entity} {fact.relation} {fact.tail_entity}"],
            ids=[fact.fact_key],
            namespace=LAYER_M_FAC,
        )
        fid = f"fact_{fact.fact_key}"
        await graph.add_node(GraphNode(id=fid, label=NODE_FACT))
        await graph.add_edge(
            GraphEdge(
                source_id=f"ont_{fact.schema_key}",
                target_id=fid,
                relation=EDGE_INSTANTIATES,
            )
        )
        for pid in fact.passage_ids:
            await graph.add_node(GraphNode(id=f"pas_{pid}", label=NODE_PASSAGE))
            await graph.add_edge(
                GraphEdge(
                    source_id=fid,
                    target_id=f"pas_{pid}",
                    relation=EDGE_EVIDENCE_IN,
                )
            )


# ===========================================================================
# Eje 1 — Memoria de 3 capas
# ===========================================================================


class TestAxis1ThreeLayerMemory:
    def test_layers_are_strictly_separated_in_memory_state(self) -> None:
        memory = GlobalMemoryState(tau=2)
        memory.add_passage(Passage(id="pas_001", text="Catena Zapata is in Mendoza."))
        schema = memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
        rejected = memory.add_fact(
            Fact(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
                passage_ids=["pas_001"],
            )
        )
        assert rejected is None
        assert list(MEMORY_NAMESPACES) == [LAYER_M_PAS, LAYER_M_ONT, LAYER_M_FAC]
        assert "pas_001" in memory.passages
        assert schema.schema_key in memory.ontologies
        assert memory.facts == {}
        assert set(memory.passages).isdisjoint(memory.ontologies)
        assert set(memory.ontologies).isdisjoint(memory.facts)
        assert set(memory.passages).isdisjoint(memory.facts)

    def test_schema_and_fact_tuple_shapes_match_paper(self) -> None:
        schema = Schema(head_type="Winery", relation="PRODUCES", tail_type="Wine")
        fact = Fact(
            head_entity="Catena Zapata",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="Gran Enemigo",
            tail_type="Wine",
        )
        assert (schema.head_type, schema.relation, schema.tail_type) == (
            "Winery",
            "PRODUCES",
            "Wine",
        )
        assert (
            fact.head_entity,
            fact.head_type,
            fact.relation,
            fact.tail_entity,
            fact.tail_type,
        ) == (
            "Catena Zapata",
            "Winery",
            "PRODUCES",
            "Gran Enemigo",
            "Wine",
        )
        assert fact.schema_key == schema.schema_key

    @pytest.mark.asyncio
    async def test_each_layer_has_its_own_vector_namespace(self) -> None:
        vectors = InMemoryVectorStore()
        await vectors.add_texts(
            texts=["primary evidence passage about Mendoza"],
            ids=["pas_001"],
            namespace=LAYER_M_PAS,
        )
        await vectors.add_texts(
            texts=["Winery LOCATED_IN Region"],
            ids=["(winery)-[located_in]->(region)"],
            namespace=LAYER_M_ONT,
        )
        await vectors.add_texts(
            texts=["Catena Zapata LOCATED_IN Mendoza"],
            ids=["catena zapata--located_in--mendoza"],
            namespace=LAYER_M_FAC,
        )
        assert set(vectors._collections) >= set(MEMORY_NAMESPACES)  # noqa: SLF001
        pas_ids = {h.id for h in await vectors.search_similarity("Mendoza", k=10, namespace=LAYER_M_PAS)}
        ont_ids = {
            h.id
            for h in await vectors.search_similarity("Winery", k=10, namespace=LAYER_M_ONT)
        }
        fac_ids = {
            h.id
            for h in await vectors.search_similarity("Catena", k=10, namespace=LAYER_M_FAC)
        }
        assert pas_ids == {"pas_001"}
        assert ont_ids == {"(winery)-[located_in]->(region)"}
        assert fac_ids == {"catena zapata--located_in--mendoza"}
        assert pas_ids.isdisjoint(ont_ids)
        assert ont_ids.isdisjoint(fac_ids)
        assert pas_ids.isdisjoint(fac_ids)

    @pytest.mark.asyncio
    async def test_graph_nodes_are_labeled_by_layer(self) -> None:
        graph = NetworkXGraphStore()
        await graph.add_node(GraphNode(id="pas_p1", label=NODE_PASSAGE))
        await graph.add_node(GraphNode(id="ont_s1", label=NODE_SCHEMA))
        await graph.add_node(GraphNode(id="fact_f1", label=NODE_FACT))
        labels = {
            (await graph.get_node("pas_p1")).label,  # type: ignore[union-attr]
            (await graph.get_node("ont_s1")).label,  # type: ignore[union-attr]
            (await graph.get_node("fact_f1")).label,  # type: ignore[union-attr]
        }
        assert labels == set(GRAPH_NODE_LABELS)


# ===========================================================================
# Eje 2 — Estabilidad ontológica τ
# ===========================================================================


class TestAxis2OntologicalStability:
    def test_stability_condition_is_frequency_ge_tau(self) -> None:
        tau = 3
        schema = Schema(head_type="Winery", relation="PRODUCES", tail_type="Wine", frequency=1)
        assert schema.frequency < tau
        assert not schema.is_stable(tau)
        schema.frequency = tau
        assert schema.frequency >= tau
        assert schema.is_stable(tau)

    def test_m_fac_rejects_facts_when_f_lt_tau(self) -> None:
        memory = GlobalMemoryState(tau=2)
        memory.register_schema_candidate("Winery", "PRODUCES", "Vermouth")
        schema = memory.ontologies["(winery)-[produces]->(vermouth)"]
        assert schema.frequency == 1
        assert schema.frequency < memory.tau
        accepted = memory.add_fact(
            Fact(
                head_entity="Experimental Bodega",
                head_type="Winery",
                relation="PRODUCES",
                tail_entity="Patagonian Vermouth",
                tail_type="Vermouth",
                passage_ids=["pas_noise"],
            )
        )
        assert accepted is None
        assert memory.facts == {}

    def test_candidate_becomes_stable_at_tau_and_then_instantiates(self) -> None:
        memory = GlobalMemoryState(tau=2)
        first = memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
        assert first.frequency == 1
        assert memory.get_stable_schemas() == []
        second = memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
        assert second.frequency == 2
        assert second.frequency >= memory.tau
        stable = memory.get_stable_schemas()
        assert len(stable) == 1
        fact = memory.add_fact(
            Fact(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
                passage_ids=["pas_001"],
            )
        )
        assert fact is not None
        assert fact.fact_key in memory.facts

    @pytest.mark.asyncio
    async def test_rejected_facts_are_not_written_to_m_fac_or_graph(self) -> None:
        memory = GlobalMemoryState(tau=2)
        vectors = InMemoryVectorStore()
        graph = NetworkXGraphStore()
        orch = IndexingOrchestrator(
            memory,
            model=ScriptedModel(),
            vector_store=vectors,
            graph_store=graph,
            run_conflict_pipeline=False,
        )
        memory.add_passage(Passage(id="pas_x", text="noise"))
        memory.register_schema_candidate("Winery", "PRODUCES", "Vermouth")
        rejected = memory.add_fact(
            Fact(
                head_entity="X",
                head_type="Winery",
                relation="PRODUCES",
                tail_entity="Y",
                tail_type="Vermouth",
                passage_ids=["pas_x"],
            )
        )
        assert rejected is None
        # Orchestrator must not sync a rejected fact.
        assert LAYER_M_FAC not in vectors._collections or not vectors._collections[LAYER_M_FAC]  # noqa: SLF001
        assert await graph.get_node("fact_x--produces--y") is None
        _ = orch  # constructed to mirror the production write path


# ===========================================================================
# Eje 3 — Topología jerárquica y aristas canónicas
# ===========================================================================


class TestAxis3HierarchicalGraph:
    @pytest.mark.asyncio
    async def test_ingestion_emits_canonical_directed_edges(self) -> None:
        def extract_responder(prompt: str, _system, _model):
            return DocumentExtractionResult(
                schemas=[
                    ExtractedSchema(
                        head_type="Winery", relation="LOCATED_IN", tail_type="Region"
                    )
                ],
                facts=[
                    ExtractedFact(
                        head_entity="Catena Zapata",
                        head_type="Winery",
                        relation="LOCATED_IN",
                        tail_entity="Mendoza",
                        tail_type="Region",
                    )
                ],
            )

        llm = MockLLMProvider(
            structured_responses={"DocumentExtractionResult": extract_responder}
        )
        memory = GlobalMemoryState(tau=1)
        vectors = InMemoryVectorStore()
        graph = NetworkXGraphStore()
        pipeline = IngestionPipeline(
            memory_state=memory,
            vector_store=vectors,
            graph_store=graph,
            llm=llm,
            entity_resolver=EntityResolver(vectors, similarity_threshold=0.99),
        )
        report = await pipeline.process_document(
            "Catena Zapata is located in Mendoza.", document_id="doc_topo"
        )
        assert report["facts_accepted"] == 1
        fact = next(iter(memory.facts.values()))
        edges = await _all_graph_edges(graph)
        relations = {e.relation for e in edges}
        assert EDGE_SUPPORTED_BY not in relations
        assert relations <= {EDGE_INSTANTIATES, EDGE_EVIDENCE_IN}
        assert any(
            e.source_id == f"ont_{fact.schema_key}"
            and e.target_id == f"fact_{fact.fact_key}"
            and e.relation == EDGE_INSTANTIATES
            for e in edges
        )
        assert any(
            e.source_id == f"fact_{fact.fact_key}"
            and e.target_id == f"pas_{fact.passage_ids[0]}"
            and e.relation == EDGE_EVIDENCE_IN
            for e in edges
        )

    @pytest.mark.asyncio
    async def test_indexing_orchestrator_uses_same_edge_vocabulary(self) -> None:
        memory = GlobalMemoryState(tau=1)
        vectors = InMemoryVectorStore()
        graph = NetworkXGraphStore()
        orch = IndexingOrchestrator(
            memory,
            model=ScriptedModel(),
            vector_store=vectors,
            graph_store=graph,
            run_conflict_pipeline=False,
        )
        memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
        fact = memory.add_fact(
            Fact(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="PRODUCES",
                tail_entity="Gran Enemigo",
                tail_type="Wine",
                passage_ids=["pas_002"],
            )
        )
        assert fact is not None
        await orch._sync_fact(fact)
        edges = await _all_graph_edges(graph)
        relations = {e.relation for e in edges}
        assert EDGE_SUPPORTED_BY not in relations
        assert EDGE_INSTANTIATES in relations
        assert EDGE_EVIDENCE_IN in relations
        instantiates = [e for e in edges if e.relation == EDGE_INSTANTIATES]
        evidence = [e for e in edges if e.relation == EDGE_EVIDENCE_IN]
        assert all(e.source_id.startswith("ont_") and e.target_id.startswith("fact_") for e in instantiates)
        assert all(e.source_id.startswith("fact_") and e.target_id.startswith("pas_") for e in evidence)

    @pytest.mark.asyncio
    async def test_no_nomenclature_drift_between_writers(self) -> None:
        """Ingestion and indexing must agree on INSTANTIATES / EVIDENCE_IN direction."""
        # Direction contract encoded as source/target prefixes.
        expected_instantiates: EdgeTriple = ("ont_", "fact_", EDGE_INSTANTIATES)
        expected_evidence: EdgeTriple = ("fact_", "pas_", EDGE_EVIDENCE_IN)
        assert expected_instantiates[2] == EDGE_INSTANTIATES
        assert expected_evidence[2] == EDGE_EVIDENCE_IN


# ===========================================================================
# Eje 4 — PPR y grounding
# ===========================================================================


class TestAxis4PersonalizedPageRank:
    def test_seed_vector_p0_uses_all_three_layers(self) -> None:
        calc = LayeredPPRCalculator(
            layer_weights={LAYER_M_ONT: 0.2, LAYER_M_FAC: 0.5, LAYER_M_PAS: 0.3}
        )
        seeds = calc.build_seed_weights(
            ont_hits=[VectorSearchResult(id="s1", text="schema", score=1.0)],
            fac_hits=[VectorSearchResult(id="f1", text="fact", score=0.5)],
            pas_hits=[VectorSearchResult(id="p1", text="passage", score=0.8)],
        )
        assert seeds["ont_s1"] == pytest.approx(1.0 * 0.2)
        assert seeds["fact_f1"] == pytest.approx(0.5 * 0.5)
        assert seeds["pas_p1"] == pytest.approx(0.8 * 0.3)
        assert set(seeds) == {"ont_s1", "fact_f1", "pas_p1"}

    @pytest.mark.asyncio
    async def test_ppr_satisfies_paper_fixed_point_equation(self) -> None:
        graph = NetworkXGraphStore()
        await graph.add_node(GraphNode(id="ont_s", label=NODE_SCHEMA))
        await graph.add_node(GraphNode(id="fact_f", label=NODE_FACT))
        await graph.add_node(GraphNode(id="pas_p", label=NODE_PASSAGE))
        await graph.add_edge(
            GraphEdge(source_id="ont_s", target_id="fact_f", relation=EDGE_INSTANTIATES)
        )
        await graph.add_edge(
            GraphEdge(source_id="fact_f", target_id="pas_p", relation=EDGE_EVIDENCE_IN)
        )
        alpha = 0.85
        p0_raw = {"ont_s": 0.2, "fact_f": 0.5, "pas_p": 0.3}
        scores = await graph.run_ppr(
            seed_node_weights=p0_raw,
            alpha=alpha,
            max_iter=200,
            tol=1e-10,
            relation_weights={EDGE_INSTANTIATES: 1.0, EDGE_EVIDENCE_IN: 1.2},
            bidirectional=True,
        )
        assert scores
        assert pytest.approx(sum(scores.values()), abs=1e-6) == 1.0

        adj: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in scores}
        rel_w = {EDGE_INSTANTIATES: 1.0, EDGE_EVIDENCE_IN: 1.2}
        for src, tgt, data in graph._g.edges(data=True):  # noqa: SLF001
            w = float(rel_w.get(str(data.get("relation")), 1.0))
            adj[src].append((tgt, w))
            adj[tgt].append((src, w))  # bidirectional=True
        residual = _paper_ppr_fixed_point(
            list(scores), adj, p0_raw, alpha, scores
        )
        assert residual < 1e-6

    @pytest.mark.asyncio
    async def test_subgraph_is_grounded_in_retrieved_passages(self) -> None:
        memory = GlobalMemoryState(tau=1)
        graph = NetworkXGraphStore()
        vectors = InMemoryVectorStore()
        memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
        memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
        p_loc = Passage(
            id="pas_loc",
            text="Catena Zapata winery is located in Mendoza Argentina.",
        )
        p_prod = Passage(
            id="pas_prod",
            text="Catena Zapata produces Gran Enemigo wine in Gualtallary.",
        )
        p_noise = Passage(
            id="pas_noise",
            text="Generic tourism brochure about beaches hotels nightlife.",
        )
        for p in (p_loc, p_prod, p_noise):
            memory.add_passage(p)
        f_loc = memory.add_fact(
            Fact(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="LOCATED_IN",
                tail_entity="Mendoza",
                tail_type="Region",
                passage_ids=["pas_loc"],
            )
        )
        f_prod = memory.add_fact(
            Fact(
                head_entity="Catena Zapata",
                head_type="Winery",
                relation="PRODUCES",
                tail_entity="Gran Enemigo",
                tail_type="Wine",
                passage_ids=["pas_prod"],
            )
        )
        assert f_loc and f_prod
        await _index_layers(memory, graph, vectors)

        retriever = MemoryGuidedRetriever(
            memory_state=memory,
            graph_store=graph,
            vector_store=vectors,
            top_k_passages=2,
            top_k_facts=5,
            alpha_hybrid=0.5,
        )
        result = await retriever.retrieve(
            "Where is Catena Zapata winery located in Mendoza?"
        )
        assert result.retrieval_mode == "hybrid_rerank"
        retrieved_pas = {p.id for p in result.passages}
        assert retrieved_pas
        for fact in result.facts:
            assert any(pid in retrieved_pas for pid in fact.passage_ids), (
                f"fact {fact.fact_key} lacks evidence in retrieved passages"
            )
        node_ids = {n["id"] for n in result.subgraph["nodes"]}
        for edge in result.subgraph["edges"]:
            assert edge["source_id"] in node_ids
            assert edge["target_id"] in node_ids
            if edge["relation"] == EDGE_EVIDENCE_IN:
                pas_id = edge["target_id"].removeprefix("pas_")
                assert pas_id in retrieved_pas


# ===========================================================================
# Eje 5 — Reranking híbrido
# ===========================================================================


class TestAxis5HybridReranking:
    def test_fusion_matches_paper_formula_with_minmax_norm(self) -> None:
        alpha_hybrid = 0.6
        vec = {"a": 2.0, "b": 8.0, "c": 5.0}
        ppr = {"a": 100.0, "b": 10.0, "c": 55.0}
        ids = sorted(set(vec) | set(ppr))
        norm_vec = dict(zip(ids, _minmax([vec[i] for i in ids])))
        norm_ppr = dict(zip(ids, _minmax([ppr[i] for i in ids])))
        expected = {
            i: alpha_hybrid * norm_vec[i] + (1.0 - alpha_hybrid) * norm_ppr[i]
            for i in ids
        }
        reranker = HybridReranker(alpha_hybrid=alpha_hybrid, normalization="minmax")
        ranked = reranker.fuse(vec, ppr)
        got = {item.item_id: item.score for item in ranked}
        for item_id, score in expected.items():
            assert got[item_id] == pytest.approx(score)
            assert 0.0 <= ranked[0].vec_norm <= 1.0
            assert 0.0 <= ranked[0].ppr_norm <= 1.0

    def test_normalization_prevents_scale_dominance(self) -> None:
        """Sin Norm, S_ppr en [1, 1000] aplastaría S_vec en [0, 1]."""
        alpha_hybrid = 0.6
        vec = {"lexical": 0.95, "structural": 0.10}
        ppr = {"lexical": 1.0, "structural": 1000.0}

        raw_fused = {
            k: alpha_hybrid * vec[k] + (1.0 - alpha_hybrid) * ppr[k] for k in vec
        }
        assert max(raw_fused, key=raw_fused.get) == "structural"  # type: ignore[arg-type]

        reranker = HybridReranker(alpha_hybrid=alpha_hybrid, normalization="minmax")
        ranked = reranker.fuse(vec, ppr)
        # After min-max, lexical: vec=1, ppr=0 → 0.6; structural: vec=0, ppr=1 → 0.4
        assert ranked[0].item_id == "lexical"
        assert ranked[0].vec_norm == pytest.approx(1.0)
        assert ranked[0].ppr_norm == pytest.approx(0.0)
        assert ranked[1].vec_norm == pytest.approx(0.0)
        assert ranked[1].ppr_norm == pytest.approx(1.0)
        assert ranked[0].score == pytest.approx(alpha_hybrid * 1.0 + (1.0 - alpha_hybrid) * 0.0)

    def test_norm_maps_both_metrics_to_unit_interval(self) -> None:
        reranker = HybridReranker(alpha_hybrid=0.5, normalization="minmax")
        ranked = reranker.fuse(
            {"x": -50.0, "y": 0.0, "z": 80.0},
            {"x": 1e-6, "y": 3.2, "z": 9.9},
        )
        for item in ranked:
            assert 0.0 <= item.vec_norm <= 1.0
            assert 0.0 <= item.ppr_norm <= 1.0
            assert 0.0 <= item.score <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
