"""Direct reads of M_pas that bypass the writer's ceiling.

MemoryTax (arXiv:2610.29823) shows that a store written by a weaker model
caps every later reader: swapping in a stronger reader does not recover spans
the writer never stored. M_fac and M_ont are that store. This channel returns
raw passages from vector search on M_pas, including passages no accepted fact
cites, without reordering the hybrid ranking.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from memgraphrag_core.domain.models import Fact, Passage
from memgraphrag_core.interfaces.vector_store import VectorSearchResult


def select_direct_passages(
    hits: Sequence[VectorSearchResult],
    passages: Mapping[str, Passage],
    *,
    k: int,
) -> list[Passage]:
    """Take the top-k raw M_pas hits that still exist in memory.

    Score order is the vector rank. Non-positive scores and unknown ids are
    skipped. The result does not consult M_fac or M_ont.
    """
    if k <= 0:
        return []
    chosen: list[Passage] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        if hit.score <= 0.0 or hit.id in seen:
            continue
        passage = passages.get(hit.id)
        if passage is None:
            continue
        chosen.append(passage)
        seen.add(hit.id)
        if len(chosen) >= k:
            break
    return chosen


def cited_passage_ids(facts: Mapping[str, Fact]) -> set[str]:
    """Passage ids the writer attached to at least one accepted fact."""
    return {pid for fact in facts.values() for pid in fact.passage_ids}


def merge_writer_omitted(
    ranked: Sequence[Passage],
    direct: Sequence[Passage],
    cited_ids: set[str],
) -> tuple[list[Passage], list[str]]:
    """Append raw passages the writer never cited, preserving ``ranked`` order.

    Returns the merged passage list and the ids of direct hits absent from
    every accepted fact (the ceiling residue), whether or not hybrid ranking
    already surfaced them.
    """
    omitted_ids = [passage.id for passage in direct if passage.id not in cited_ids]
    present = {passage.id for passage in ranked}
    merged = list(ranked)
    for passage in direct:
        if passage.id in cited_ids or passage.id in present:
            continue
        merged.append(passage)
        present.add(passage.id)
    return merged, omitted_ids
