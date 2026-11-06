"""Abstract port for graph stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class BaseGraphStore(ABC):
    @abstractmethod
    async def add_node(self, node: GraphNode) -> None:
        pass

    @abstractmethod
    async def add_edge(self, edge: GraphEdge) -> None:
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        pass

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, relation_type: Optional[str] = None
    ) -> List[GraphNode]:
        pass

    @abstractmethod
    async def run_ppr(
        self,
        seed_node_weights: Dict[str, float],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        relation_weights: Optional[Dict[str, float]] = None,
        bidirectional: bool = False,
        node_multipliers: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Personalized PageRank over the hierarchical graph.

        Args:
            relation_weights: Optional per-relation edge multipliers
                (e.g. ``{"INSTANTIATES": 1.0, "EVIDENCE_IN": 1.2}``).
            bidirectional: If True, treat hierarchical edges as traversable
                in both directions (useful for ont↔fac↔pas propagation).
            node_multipliers: Per-node personalization scales. ``0`` prunes the
                node and its incident edges. Values above 1 boost constraints.
        """
        pass

    @abstractmethod
    async def get_subgraph(
        self, node_ids: List[str]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        pass
