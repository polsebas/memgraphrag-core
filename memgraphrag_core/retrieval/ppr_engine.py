"""Layered / weighted Personalized PageRank for the three-tier memory graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    EDGE_SUPPORTED_BY,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
)
from memgraphrag_core.interfaces.graph_store import BaseGraphStore
from memgraphrag_core.interfaces.vector_store import VectorSearchResult

DEFAULT_LAYER_WEIGHTS: Dict[str, float] = {
    LAYER_M_ONT: 0.2,
    LAYER_M_FAC: 0.5,
    LAYER_M_PAS: 0.3,
}

DEFAULT_RELATION_WEIGHTS: Dict[str, float] = {
    EDGE_INSTANTIATES: 1.0,
    EDGE_EVIDENCE_IN: 1.2,
    # Deprecated alias kept for graphs written before edge unification.
    EDGE_SUPPORTED_BY: 1.2,
}


@dataclass
class LayeredPPRCalculator:
    """Build multilayer seed distributions and run hierarchical PPR.

    Seed weights combine vector similarity with configurable layer biases.
    Edge transfers can be reweighted by relation type (INSTANTIATES, EVIDENCE_IN).
    """

    layer_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_LAYER_WEIGHTS)
    )
    relation_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELATION_WEIGHTS)
    )
    alpha: float = 0.85
    bidirectional: bool = True

    def build_seed_weights(
        self,
        ont_hits: list[VectorSearchResult],
        fac_hits: list[VectorSearchResult],
        pas_hits: list[VectorSearchResult],
    ) -> Dict[str, float]:
        """Map vector hits to graph node ids with layer-biased seed mass."""
        w_ont = float(self.layer_weights.get(LAYER_M_ONT, 0.0))
        w_fac = float(self.layer_weights.get(LAYER_M_FAC, 0.0))
        w_pas = float(self.layer_weights.get(LAYER_M_PAS, 0.0))

        seeds: Dict[str, float] = {}
        for hit in ont_hits:
            if hit.score > 0.0 and w_ont > 0.0:
                nid = f"ont_{hit.id}"
                seeds[nid] = seeds.get(nid, 0.0) + hit.score * w_ont
        for hit in fac_hits:
            if hit.score > 0.0 and w_fac > 0.0:
                nid = f"fact_{hit.id}"
                seeds[nid] = seeds.get(nid, 0.0) + hit.score * w_fac
        for hit in pas_hits:
            if hit.score > 0.0 and w_pas > 0.0:
                nid = f"pas_{hit.id}"
                seeds[nid] = seeds.get(nid, 0.0) + hit.score * w_pas
        return seeds

    async def run(
        self,
        graph_store: BaseGraphStore,
        ont_hits: list[VectorSearchResult],
        fac_hits: list[VectorSearchResult],
        pas_hits: list[VectorSearchResult],
        *,
        alpha: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """Compute layered PPR scores keyed by graph node id."""
        seeds = self.build_seed_weights(ont_hits, fac_hits, pas_hits)
        if not seeds:
            return {}
        return await graph_store.run_ppr(
            seed_node_weights=seeds,
            alpha=self.alpha if alpha is None else alpha,
            max_iter=max_iter,
            tol=tol,
            relation_weights=self.relation_weights,
            bidirectional=self.bidirectional,
        )


def merge_layer_weights(
    base: Optional[Mapping[str, float]] = None,
    **overrides: float,
) -> Dict[str, float]:
    """Helper to build layer weight dicts with defaults."""
    out = dict(DEFAULT_LAYER_WEIGHTS)
    if base:
        out.update(base)
    out.update(overrides)
    return out
