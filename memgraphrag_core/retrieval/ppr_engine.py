"""Layered / weighted Personalized PageRank for the three-tier memory graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Set

from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    EDGE_SUPPORTED_BY,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    FactCategory,
    GlobalMemoryState,
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

# Active hard constraints and explicit rejections stay inside the attention budget.
ALPHA_CONSTRAINT = 1.35
# Invalidated or TTL-expired M_fac nodes and their incident edges drop out of PPR.
W_EXPIRED = 0.0
_PRIORITY_CATEGORIES = {FactCategory.HARD_CONSTRAINT, FactCategory.REJECTED_OPTION}


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
        memory: Optional[GlobalMemoryState] = None,
        query: str = "",
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Compute layered PPR scores keyed by graph node id."""
        seeds = self.build_seed_weights(ont_hits, fac_hits, pas_hits)
        multipliers: Dict[str, float] = {}
        if memory is not None:
            multipliers = applicability_multipliers(memory, ont_hits, query, as_of)
            floor = float(self.layer_weights.get(LAYER_M_FAC, 0.0))
            for nid, scale in multipliers.items():
                if scale <= 0.0:
                    seeds.pop(nid, None)
                    continue
                if nid not in seeds and floor > 0.0:
                    seeds[nid] = floor
        if not seeds:
            return {}
        return await graph_store.run_ppr(
            seed_node_weights=seeds,
            alpha=self.alpha if alpha is None else alpha,
            max_iter=max_iter,
            tol=tol,
            relation_weights=self.relation_weights,
            bidirectional=self.bidirectional,
            node_multipliers=multipliers or None,
        )


def linked_to_queried_ontology(
    schema_key: str,
    head_entity: str,
    tail_entity: str,
    ont_ids: Set[str],
    query: str,
) -> bool:
    """True when the fact instantiates a hit schema or names an entity in the query."""
    if schema_key in ont_ids:
        return True
    haystack = query.casefold()
    return any(token and token.casefold() in haystack for token in (head_entity, tail_entity))


def applicability_multipliers(
    memory: GlobalMemoryState,
    ont_hits: list[VectorSearchResult],
    query: str,
    as_of: Optional[datetime] = None,
) -> Dict[str, float]:
    """Personalization scales for M_fac nodes.

    Expired or invalidated facts get ``W_EXPIRED`` (0). Active hard constraints
    and rejected options that touch the queried ontology get ``ALPHA_CONSTRAINT``.
    """
    moment = as_of or datetime.now(timezone.utc)
    ont_ids = {hit.id for hit in ont_hits}
    scales: Dict[str, float] = {}
    for fact in memory.facts.values():
        nid = f"fact_{fact.fact_key}"
        if fact.is_suppressed(moment):
            scales[nid] = W_EXPIRED
            continue
        if fact.category in _PRIORITY_CATEGORIES and linked_to_queried_ontology(
            fact.schema_key, fact.head_entity, fact.tail_entity, ont_ids, query
        ):
            scales[nid] = ALPHA_CONSTRAINT
    return scales


def suppressed_fact_keys(
    memory: GlobalMemoryState, as_of: Optional[datetime] = None
) -> Set[str]:
    moment = as_of or datetime.now(timezone.utc)
    return {fact.fact_key for fact in memory.facts.values() if fact.is_suppressed(moment)}


def exclusive_suppressed_passages(
    memory: GlobalMemoryState, as_of: Optional[datetime] = None
) -> Set[str]:
    """Passage ids cited only by expired or invalidated facts."""
    moment = as_of or datetime.now(timezone.utc)
    suppressed: Set[str] = set()
    live: Set[str] = set()
    for fact in memory.facts.values():
        ids = set(fact.passage_ids)
        if fact.source_passage_id:
            ids.add(fact.source_passage_id)
        if fact.is_suppressed(moment):
            suppressed.update(ids)
        else:
            live.update(ids)
    return suppressed - live


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
