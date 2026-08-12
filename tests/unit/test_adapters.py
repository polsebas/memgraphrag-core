"""Unit tests for vector/graph adapters (in-memory; Chroma optional)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from memgraphrag_core.adapters import VectorStoreFactory
from memgraphrag_core.adapters.chroma_vector_store import ChromaVectorStore
from memgraphrag_core.infrastructure.in_memory_graph_store import InMemoryGraphStore
from memgraphrag_core.infrastructure.in_memory_vector_store import InMemoryVectorStore
from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore
from memgraphrag_core.interfaces.graph_store import GraphEdge, GraphNode
from memgraphrag_core.interfaces.vector_store import BaseVectorStore, VectorSearchResult


class _DummyVectorStore(BaseVectorStore):
    def __init__(self, marker: str = "dummy") -> None:
        self.marker = marker
        self._docs: Dict[str, Dict[str, Any]] = {}

    async def add_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> List[str]:
        metas = metadatas or [{} for _ in texts]
        for text, doc_id, meta in zip(texts, ids, metas):
            self._docs[f"{namespace}:{doc_id}"] = {"text": text, "metadata": meta}
        return list(ids)

    async def search_similarity(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        hits: List[VectorSearchResult] = []
        q = query.lower()
        for key, doc in self._docs.items():
            ns, doc_id = key.split(":", 1)
            if ns != namespace:
                continue
            meta = doc["metadata"]
            if filter_metadata and any(meta.get(mk) != mv for mk, mv in filter_metadata.items()):
                continue
            if q in doc["text"].lower() or any(tok in doc["text"].lower() for tok in q.split()):
                hits.append(
                    VectorSearchResult(
                        id=doc_id, text=doc["text"], score=1.0, metadata=dict(meta)
                    )
                )
        return hits[:k]

    async def delete(self, ids: List[str], namespace: str = "default") -> None:
        for doc_id in ids:
            self._docs.pop(f"{namespace}:{doc_id}", None)


async def _exercise_vector_contract(store: BaseVectorStore) -> None:
    await store.add_texts(
        texts=[
            "Catena Zapata winery is located in Mendoza.",
            "Gran Enemigo is a Malbec wine.",
        ],
        ids=["pas_001", "pas_002"],
        metadatas=[{"layer": "M_pas"}, {"layer": "M_pas"}],
        namespace="M_pas",
    )
    await store.add_texts(
        texts=["Winery LOCATED_IN Region"],
        ids=["(winery)-[located_in]->(region)"],
        namespace="M_ont",
    )

    hits = await store.search_similarity("Catena Zapata Mendoza", k=2, namespace="M_pas")
    assert hits
    assert hits[0].id in {"pas_001", "pas_002"}
    assert hits[0].score > 0.0

    filtered = await store.search_similarity(
        "Catena", k=5, namespace="M_pas", filter_metadata={"layer": "M_pas"}
    )
    assert filtered

    ont_hits = await store.search_similarity("Winery Region", k=1, namespace="M_ont")
    assert ont_hits
    assert ont_hits[0].id == "(winery)-[located_in]->(region)"

    await store.delete(["pas_002"], namespace="M_pas")
    after_delete = await store.search_similarity("Malbec wine", k=5, namespace="M_pas")
    assert all(h.id != "pas_002" for h in after_delete)


def test_factory_creates_in_memory() -> None:
    store = VectorStoreFactory.create("in_memory")
    assert isinstance(store, InMemoryVectorStore)


def test_factory_creates_chroma() -> None:
    pytest.importorskip("chromadb")
    store = VectorStoreFactory.create("chroma", collection_prefix="unit_test_")
    assert isinstance(store, ChromaVectorStore)


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown vector store provider"):
        VectorStoreFactory.create("does_not_exist")


def test_register_custom_provider() -> None:
    VectorStoreFactory.register_provider("dummy_unit", _DummyVectorStore)
    store = VectorStoreFactory.create("dummy_unit", marker="ok")
    assert isinstance(store, _DummyVectorStore)
    assert store.marker == "ok"


@pytest.mark.asyncio
async def test_in_memory_vector_contract() -> None:
    await _exercise_vector_contract(VectorStoreFactory.create("in_memory"))


@pytest.mark.asyncio
async def test_chroma_vector_contract() -> None:
    pytest.importorskip("chromadb")
    await _exercise_vector_contract(
        VectorStoreFactory.create("chroma", collection_prefix="unit_chroma_")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_cls", [InMemoryGraphStore, NetworkXGraphStore])
async def test_graph_crud_and_neighbors(graph_cls: type) -> None:
    graph = graph_cls()
    await graph.add_node(GraphNode(id="a", label="Fact", properties={"k": 1}))
    await graph.add_node(GraphNode(id="b", label="Passage"))
    await graph.add_edge(GraphEdge(source_id="a", target_id="b", relation="EVIDENCE_IN"))

    node = await graph.get_node("a")
    assert node is not None
    assert node.label == "Fact"
    assert node.properties.get("k") == 1

    neighbors = await graph.get_neighbors("a", relation_type="EVIDENCE_IN")
    assert len(neighbors) == 1
    assert neighbors[0].id == "b"

    nodes, edges = await graph.get_subgraph(["a", "b"])
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].relation == "EVIDENCE_IN"


@pytest.mark.asyncio
async def test_networkx_ppr_with_relation_weights() -> None:
    graph = NetworkXGraphStore()
    await graph.add_node(GraphNode(id="ont_1", label="Schema"))
    await graph.add_node(GraphNode(id="fact_1", label="Fact"))
    await graph.add_node(GraphNode(id="pas_1", label="Passage"))
    await graph.add_edge(
        GraphEdge(source_id="ont_1", target_id="fact_1", relation="INSTANTIATES")
    )
    await graph.add_edge(
        GraphEdge(source_id="fact_1", target_id="pas_1", relation="EVIDENCE_IN")
    )

    scores = await graph.run_ppr(
        seed_node_weights={"ont_1": 1.0},
        relation_weights={"INSTANTIATES": 1.0, "EVIDENCE_IN": 2.0},
        bidirectional=True,
    )
    assert scores["fact_1"] > 0.0
    assert scores["pas_1"] > 0.0
