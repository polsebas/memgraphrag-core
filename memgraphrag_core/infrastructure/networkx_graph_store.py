"""NetworkX-backed graph store with Personalized PageRank."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from memgraphrag_core.interfaces.graph_store import BaseGraphStore, GraphEdge, GraphNode


class NetworkXGraphStore(BaseGraphStore):
    """In-memory directed graph; PPR via pure NumPy power iteration."""

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    async def add_node(self, node: GraphNode) -> None:
        self._g.add_node(node.id, label=node.label, **node.properties)

    async def add_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self._g:
            self._g.add_node(edge.source_id, label="unknown")
        if edge.target_id not in self._g:
            self._g.add_node(edge.target_id, label="unknown")
        self._g.add_edge(
            edge.source_id,
            edge.target_id,
            relation=edge.relation,
            **edge.properties,
        )

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        if node_id not in self._g:
            return None
        data = dict(self._g.nodes[node_id])
        label = data.pop("label", "unknown")
        return GraphNode(id=node_id, label=label, properties=data)

    async def get_neighbors(
        self, node_id: str, relation_type: Optional[str] = None
    ) -> List[GraphNode]:
        if node_id not in self._g:
            return []
        neighbors: List[GraphNode] = []
        for _, tgt, edata in self._g.out_edges(node_id, data=True):
            if relation_type is not None and edata.get("relation") != relation_type:
                continue
            node = await self.get_node(tgt)
            if node is not None:
                neighbors.append(node)
        return neighbors

    def _weighted_adjacency(
        self,
        relation_weights: Optional[Dict[str, float]],
        bidirectional: bool,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Build outgoing adjacency list with relation-aware edge weights."""
        adj: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in self._g.nodes()}
        rel_w = relation_weights or {}

        def edge_weight(edata: dict) -> float:
            relation = str(edata.get("relation", "RELATED"))
            base = float(edata.get("weight", 1.0))
            return base * float(rel_w.get(relation, 1.0))

        for src, tgt, edata in self._g.edges(data=True):
            w = edge_weight(edata)
            if w <= 0.0:
                continue
            adj[src].append((tgt, w))
            if bidirectional:
                adj[tgt].append((src, w))
        return adj

    def _run_ppr_sync(
        self,
        seed_node_weights: Dict[str, float],
        alpha: float,
        max_iter: int,
        tol: float,
        relation_weights: Optional[Dict[str, float]],
        bidirectional: bool,
    ) -> Dict[str, float]:
        """CPU-bound PPR implementation (run via ``asyncio.to_thread``)."""
        if self._g.number_of_nodes() == 0:
            return {}

        nodes = list(self._g.nodes())
        n = len(nodes)
        index = {nid: i for i, nid in enumerate(nodes)}

        personalization = np.zeros(n, dtype=float)
        for nid, w in seed_node_weights.items():
            if nid in index and w > 0.0:
                personalization[index[nid]] = w

        total = float(personalization.sum())
        if total <= 0.0:
            return {}
        personalization /= total

        adj = self._weighted_adjacency(relation_weights, bidirectional)
        out_strength = {
            nid: sum(w for _, w in neighbors) for nid, neighbors in adj.items()
        }

        scores = personalization.copy()
        # Paper: p = (1 - α_ppr) · p_0 + α_ppr · A^T p
        # A is the row-normalized weighted adjacency (dangling mass teleports via p_0).
        for _ in range(max_iter):
            new_scores = np.zeros(n, dtype=float)
            for src in nodes:
                i = index[src]
                neighbors = adj.get(src, [])
                strength = out_strength.get(src, 0.0)
                if not neighbors or strength <= 0.0:
                    new_scores += scores[i] * personalization
                else:
                    for tgt, w in neighbors:
                        new_scores[index[tgt]] += scores[i] * (w / strength)
            new_scores = (1.0 - alpha) * personalization + alpha * new_scores
            if np.linalg.norm(new_scores - scores, ord=1) < tol:
                scores = new_scores
                break
            scores = new_scores

        return {nid: float(scores[index[nid]]) for nid in nodes}

    async def run_ppr(
        self,
        seed_node_weights: Dict[str, float],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        relation_weights: Optional[Dict[str, float]] = None,
        bidirectional: bool = False,
    ) -> Dict[str, float]:
        """Personalized PageRank via power iteration (optional edge weights)."""
        return await asyncio.to_thread(
            self._run_ppr_sync,
            seed_node_weights,
            alpha,
            max_iter,
            tol,
            relation_weights,
            bidirectional,
        )

    async def get_subgraph(
        self, node_ids: List[str]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        existing = [nid for nid in node_ids if nid in self._g]
        if not existing:
            return [], []

        sub = self._g.subgraph(existing).copy()
        nodes: List[GraphNode] = []
        for nid, data in sub.nodes(data=True):
            props = dict(data)
            label = props.pop("label", "unknown")
            nodes.append(GraphNode(id=nid, label=label, properties=props))

        edges: List[GraphEdge] = []
        for src, tgt, edata in sub.edges(data=True):
            props = dict(edata)
            relation = props.pop("relation", "RELATED")
            edges.append(
                GraphEdge(
                    source_id=src,
                    target_id=tgt,
                    relation=relation,
                    properties=props,
                )
            )
        return nodes, edges
