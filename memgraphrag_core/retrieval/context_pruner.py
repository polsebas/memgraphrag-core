"""Selective context pruning so remembered items do not degrade reasoning.

Remembering Is Not Enough (arXiv:2610.36652, DiscernMem) separates a bad rule
from a rule used in the wrong situation. Avoidance conditions outrank use
conditions, and a defective rule is not injected. Budget cuts drop the
lowest-ranked leftovers. They do not drop the evidence passage of a fact that
stays, nor a passage id listed in ``protected_passage_ids`` (the raw M_pas
hits the writer never stored).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from memgraphrag_core.domain.models import Fact, Passage
from memgraphrag_core.domain.process_memory import (
    ApplicabilityVerdict,
    ProcessRecord,
)


class PruneReason(str, Enum):
    """Why a candidate was kept in, or removed from, the reader context."""

    KEPT = "kept"
    AVOID_BOUNDARY = "avoid_boundary"
    OUT_OF_SCOPE = "out_of_scope"
    DEFECTIVE = "defective"
    EXPIRED = "expired"
    NOT_YET_APPLICABLE = "not_yet_applicable"
    REDUNDANT = "redundant"
    BUDGET = "budget"
    DEPENDENCY_KEPT = "dependency_kept"
    UNGROUNDED = "ungrounded"


class PruneDecision(BaseModel):
    """One keep/drop judgment."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: str
    reason: PruneReason

    @property
    def dropped(self) -> bool:
        return self.reason not in {PruneReason.KEPT, PruneReason.DEPENDENCY_KEPT}


class PrunedContext(BaseModel):
    """Context after selective pruning. Stores are not modified."""

    passages: List[Passage] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    process_records: List[ProcessRecord] = Field(default_factory=list)
    decisions: List[PruneDecision] = Field(default_factory=list)


def scope_reason(query: str, metadata: Mapping[str, Any]) -> Optional[PruneReason]:
    """Avoidance wins. Non-empty use conditions must match at least one phrase."""
    haystack = query.casefold()
    avoid = _phrases(metadata.get("avoid_conditions"))
    if any(phrase.casefold() in haystack for phrase in avoid):
        return PruneReason.AVOID_BOUNDARY
    use = _phrases(metadata.get("use_conditions"))
    if use and not any(phrase.casefold() in haystack for phrase in use):
        return PruneReason.OUT_OF_SCOPE
    return None


def _phrases(raw: Any) -> List[str]:
    if not raw:
        return []
    return [str(phrase).strip() for phrase in raw if str(phrase).strip()]


class SelectiveContextPruner:
    """Drop context that would be misapplied, leaving evidence of kept facts."""

    def __init__(
        self,
        max_passages: int = 32,
        max_facts: int = 32,
        max_process: int = 32,
        max_chars: int = 100_000,
    ) -> None:
        if min(max_passages, max_facts, max_process, max_chars) < 0:
            raise ValueError("prune budgets must be >= 0")
        self.max_passages = max_passages
        self.max_facts = max_facts
        self.max_process = max_process
        self.max_chars = max_chars

    def prune(
        self,
        *,
        query: str,
        passages: Sequence[Passage],
        facts: Sequence[Fact],
        process_records: Sequence[ProcessRecord] = (),
        as_of: Optional[datetime] = None,
        protected_passage_ids: Optional[Set[str]] = None,
    ) -> PrunedContext:
        decisions: List[PruneDecision] = []
        kept_facts, fact_decisions = self._select_facts(query, facts)
        decisions.extend(fact_decisions)

        eligible_passages, early_passage_decisions = self._eligible_passages(query, passages)
        decisions.extend(early_passage_decisions)
        evidence_ids = _evidence_ids(kept_facts, eligible_passages)
        protected = set(protected_passage_ids or ()) & {p.id for p in eligible_passages}
        pinned = evidence_ids | protected

        kept_passages, passage_reasons = self._select_passages(eligible_passages, pinned)
        decisions.extend(
            PruneDecision(item_id=pid, kind="passage", reason=reason)
            for pid, reason in passage_reasons.items()
        )

        support = {passage.id for passage in kept_passages}
        grounded: List[Fact] = []
        for fact in kept_facts:
            if fact.passage_ids and not any(pid in support for pid in fact.passage_ids):
                _replace_reason(
                    decisions, fact.fact_key, "fact", PruneReason.UNGROUNDED
                )
                continue
            grounded.append(fact)

        kept_records, record_decisions = self._select_process(
            query, process_records, as_of
        )
        decisions.extend(record_decisions)

        grounded, kept_passages, kept_records, decisions = self._trim_chars(
            grounded, kept_passages, kept_records, pinned, decisions
        )
        return PrunedContext(
            passages=kept_passages,
            facts=grounded,
            process_records=kept_records,
            decisions=decisions,
        )

    def _select_facts(
        self, query: str, facts: Sequence[Fact]
    ) -> tuple[List[Fact], List[PruneDecision]]:
        decisions: List[PruneDecision] = []
        survivors: List[Fact] = []
        seen: Set[str] = set()
        for fact in facts:
            if fact.fact_key in seen:
                decisions.append(
                    PruneDecision(
                        item_id=fact.fact_key,
                        kind="fact",
                        reason=PruneReason.REDUNDANT,
                    )
                )
                continue
            seen.add(fact.fact_key)
            if fact.metadata.get("defective") is True:
                decisions.append(
                    PruneDecision(
                        item_id=fact.fact_key,
                        kind="fact",
                        reason=PruneReason.DEFECTIVE,
                    )
                )
                continue
            scoped = scope_reason(query, fact.metadata)
            if scoped is not None:
                decisions.append(
                    PruneDecision(item_id=fact.fact_key, kind="fact", reason=scoped)
                )
                continue
            survivors.append(fact)

        if len(survivors) > self.max_facts:
            overflow = survivors[self.max_facts :]
            survivors = survivors[: self.max_facts]
            for fact in overflow:
                decisions.append(
                    PruneDecision(
                        item_id=fact.fact_key, kind="fact", reason=PruneReason.BUDGET
                    )
                )
        for fact in survivors:
            decisions.append(
                PruneDecision(
                    item_id=fact.fact_key, kind="fact", reason=PruneReason.KEPT
                )
            )
        return survivors, decisions

    def _eligible_passages(
        self, query: str, passages: Sequence[Passage]
    ) -> tuple[List[Passage], List[PruneDecision]]:
        eligible: List[Passage] = []
        early: List[PruneDecision] = []
        seen: Set[str] = set()
        for passage in passages:
            if passage.id in seen:
                early.append(
                    PruneDecision(
                        item_id=passage.id,
                        kind="passage",
                        reason=PruneReason.REDUNDANT,
                    )
                )
                continue
            seen.add(passage.id)
            if passage.metadata.get("defective") is True:
                early.append(
                    PruneDecision(
                        item_id=passage.id,
                        kind="passage",
                        reason=PruneReason.DEFECTIVE,
                    )
                )
                continue
            scoped = scope_reason(query, passage.metadata)
            if scoped is not None:
                early.append(
                    PruneDecision(item_id=passage.id, kind="passage", reason=scoped)
                )
                continue
            eligible.append(passage)
        return eligible, early

    def _select_passages(
        self,
        eligible: Sequence[Passage],
        pinned: Set[str],
    ) -> tuple[List[Passage], Dict[str, PruneReason]]:
        reasons: Dict[str, PruneReason] = {}
        pinned_here = {passage.id for passage in eligible if passage.id in pinned}
        room = max(0, self.max_passages - len(pinned_here))
        chosen: Set[str] = set(pinned_here)
        for passage in eligible:
            if passage.id in pinned_here:
                continue
            if room <= 0:
                reasons[passage.id] = PruneReason.BUDGET
                continue
            chosen.add(passage.id)
            room -= 1
        naive = {passage.id for passage in list(eligible)[: self.max_passages]}
        kept = [passage for passage in eligible if passage.id in chosen]
        for passage in kept:
            if passage.id in pinned_here and passage.id not in naive:
                reasons[passage.id] = PruneReason.DEPENDENCY_KEPT
            else:
                reasons[passage.id] = PruneReason.KEPT
        return kept, reasons

    def _select_process(
        self,
        query: str,
        records: Sequence[ProcessRecord],
        as_of: Optional[datetime],
    ) -> tuple[List[ProcessRecord], List[PruneDecision]]:
        decisions: List[PruneDecision] = []
        survivors: List[ProcessRecord] = []
        seen: Set[str] = set()
        for record in records:
            identity = f"{record.kind.value}:{record.summary.casefold()}"
            if identity in seen:
                decisions.append(
                    PruneDecision(
                        item_id=record.record_id,
                        kind="process",
                        reason=PruneReason.REDUNDANT,
                    )
                )
                continue
            seen.add(identity)
            if record.metadata.get("defective") is True:
                decisions.append(
                    PruneDecision(
                        item_id=record.record_id,
                        kind="process",
                        reason=PruneReason.DEFECTIVE,
                    )
                )
                continue
            timed = _time_reason(record, as_of)
            if timed is not None:
                decisions.append(
                    PruneDecision(
                        item_id=record.record_id, kind="process", reason=timed
                    )
                )
                continue
            scoped = scope_reason(query, record.metadata)
            if scoped is not None:
                decisions.append(
                    PruneDecision(
                        item_id=record.record_id, kind="process", reason=scoped
                    )
                )
                continue
            survivors.append(record)

        if len(survivors) > self.max_process:
            overflow = survivors[self.max_process :]
            survivors = survivors[: self.max_process]
            for record in overflow:
                decisions.append(
                    PruneDecision(
                        item_id=record.record_id,
                        kind="process",
                        reason=PruneReason.BUDGET,
                    )
                )
        for record in survivors:
            decisions.append(
                PruneDecision(
                    item_id=record.record_id,
                    kind="process",
                    reason=PruneReason.KEPT,
                )
            )
        return survivors, decisions

    def _trim_chars(
        self,
        facts: List[Fact],
        passages: List[Passage],
        records: List[ProcessRecord],
        pinned: Set[str],
        decisions: List[PruneDecision],
    ) -> tuple[List[Fact], List[Passage], List[ProcessRecord], List[PruneDecision]]:
        guard = 0
        while _char_size(facts, passages, records) > self.max_chars and guard < 10_000:
            guard += 1
            expendable = [passage for passage in passages if passage.id not in pinned]
            if expendable:
                victim = expendable[-1]
                passages = [passage for passage in passages if passage.id != victim.id]
                _replace_reason(decisions, victim.id, "passage", PruneReason.BUDGET)
                continue
            if records:
                victim_record = records[-1]
                records = records[:-1]
                _replace_reason(
                    decisions, victim_record.record_id, "process", PruneReason.BUDGET
                )
                continue
            if facts:
                victim_fact = facts[-1]
                facts = facts[:-1]
                _replace_reason(
                    decisions, victim_fact.fact_key, "fact", PruneReason.BUDGET
                )
                continue
            break

        support = {passage.id for passage in passages}
        grounded: List[Fact] = []
        for fact in facts:
            if fact.passage_ids and not any(pid in support for pid in fact.passage_ids):
                _replace_reason(decisions, fact.fact_key, "fact", PruneReason.UNGROUNDED)
                continue
            grounded.append(fact)
        return grounded, passages, records, decisions


def _evidence_ids(facts: Sequence[Fact], passages: Sequence[Passage]) -> Set[str]:
    available = {passage.id for passage in passages}
    pinned: Set[str] = set()
    for fact in facts:
        for pid in fact.passage_ids:
            if pid in available:
                pinned.add(pid)
                break
    return pinned


def _time_reason(
    record: ProcessRecord, as_of: Optional[datetime]
) -> Optional[PruneReason]:
    if as_of is None:
        return None
    verdict = record.applicability.judge(as_of)
    if verdict is ApplicabilityVerdict.EXPIRED:
        return PruneReason.EXPIRED
    if verdict is ApplicabilityVerdict.NOT_YET:
        return PruneReason.NOT_YET_APPLICABLE
    return None


def _char_size(
    facts: Sequence[Fact],
    passages: Sequence[Passage],
    records: Sequence[ProcessRecord],
) -> int:
    total = sum(len(passage.text) for passage in passages)
    total += sum(
        len(f"{fact.head_entity} {fact.relation} {fact.tail_entity}") for fact in facts
    )
    total += sum(len(record.summary) for record in records)
    return total


def _replace_reason(
    decisions: List[PruneDecision], item_id: str, kind: str, reason: PruneReason
) -> None:
    for index, decision in enumerate(decisions):
        if decision.item_id == item_id and decision.kind == kind:
            decisions[index] = PruneDecision(item_id=item_id, kind=kind, reason=reason)
            return
    decisions.append(PruneDecision(item_id=item_id, kind=kind, reason=reason))
