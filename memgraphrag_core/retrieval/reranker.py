"""Hybrid vector + PPR reranking utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence

Normalization = Literal["minmax", "zscore"]


@dataclass(frozen=True)
class RankedItem:
    """A candidate item with fused retrieval scores."""

    item_id: str
    score: float
    vec_score: float
    ppr_score: float
    vec_norm: float
    ppr_norm: float


@dataclass
class HybridReranker:
    """Fuse normalized vector similarity and PPR proximity scores.

    ``S_final = alpha_hybrid * S_vec_norm + (1 - alpha_hybrid) * S_ppr_norm``
    """

    alpha_hybrid: float = 0.6
    normalization: Normalization = "minmax"

    def normalize(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        if self.normalization == "zscore":
            return self._zscore(scores)
        return self._minmax(scores)

    @staticmethod
    def _minmax(scores: Sequence[float]) -> List[float]:
        lo = min(scores)
        hi = max(scores)
        if math.isclose(hi, lo):
            return [1.0 for _ in scores]
        span = hi - lo
        return [(s - lo) / span for s in scores]

    @staticmethod
    def _zscore(scores: Sequence[float]) -> List[float]:
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        if math.isclose(std, 0.0):
            return [0.0 for _ in scores]
        # Squash to ~[0, 1] via logistic for comparability with min-max.
        return [1.0 / (1.0 + math.exp(-(s - mean) / std)) for s in scores]

    def fuse(
        self,
        vec_scores: Mapping[str, float],
        ppr_scores: Mapping[str, float],
        *,
        alpha_hybrid: Optional[float] = None,
    ) -> List[RankedItem]:
        """Return items sorted by fused score (desc). Missing scores default to 0."""
        alpha = self.alpha_hybrid if alpha_hybrid is None else alpha_hybrid
        ids = sorted(set(vec_scores) | set(ppr_scores))
        if not ids:
            return []

        raw_vec = [float(vec_scores.get(i, 0.0)) for i in ids]
        raw_ppr = [float(ppr_scores.get(i, 0.0)) for i in ids]
        norm_vec = self.normalize(raw_vec)
        norm_ppr = self.normalize(raw_ppr)

        ranked: List[RankedItem] = []
        for i, item_id in enumerate(ids):
            final = alpha * norm_vec[i] + (1.0 - alpha) * norm_ppr[i]
            ranked.append(
                RankedItem(
                    item_id=item_id,
                    score=final,
                    vec_score=raw_vec[i],
                    ppr_score=raw_ppr[i],
                    vec_norm=norm_vec[i],
                    ppr_norm=norm_ppr[i],
                )
            )
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def rerank_ids(
        self,
        vec_scores: Mapping[str, float],
        ppr_scores: Mapping[str, float],
        *,
        top_k: Optional[int] = None,
        alpha_hybrid: Optional[float] = None,
    ) -> List[str]:
        ranked = self.fuse(vec_scores, ppr_scores, alpha_hybrid=alpha_hybrid)
        ids = [r.item_id for r in ranked]
        return ids if top_k is None else ids[:top_k]
