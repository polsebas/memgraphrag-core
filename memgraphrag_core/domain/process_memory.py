"""Process information and temporal applicability.

MemDev-Bench (arXiv:2610.27524) finds that compressed memories drop the solving
process — failed attempts, rejections, revisions, and negative constraints —
and that agents reuse stored experience without checking whether it still
applies. Records stay in ``GlobalMemoryState.process_log`` in execution order.
A read judges each window at ``as_of`` and injects only those that are applicable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from memgraphrag_core.domain.models import Fact, GlobalMemoryState


class ProcessKind(str, Enum):
    """Events compressed memories tend to discard."""

    FAILED_ATTEMPT = "failed_attempt"
    REJECTION = "rejection"
    NEGATIVE_CONSTRAINT = "negative_constraint"


class ApplicabilityVerdict(str, Enum):
    """Temporal judgment of a process record at a given instant."""

    NOT_YET = "not_yet"
    APPLICABLE = "applicable"
    EXPIRED = "expired"


def ensure_aware(moment: datetime) -> datetime:
    """Treat naive datetimes as UTC so window comparisons stay ordered."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


class ApplicabilityWindow(BaseModel):
    """Half-open interval ``[valid_from, valid_until)`` during which a record applies.

    ``valid_until is None`` means the record does not expire. A TTL of zero
    expires at ``valid_from`` (the instant is already outside the window).
    """

    model_config = ConfigDict(frozen=False)

    valid_from: datetime
    valid_until: Optional[datetime] = None

    @classmethod
    def from_ttl(
        cls,
        ttl_seconds: Optional[float],
        *,
        valid_from: Optional[datetime] = None,
    ) -> "ApplicabilityWindow":
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        start = ensure_aware(valid_from or datetime.now(timezone.utc))
        if ttl_seconds is None:
            return cls(valid_from=start, valid_until=None)
        return cls(
            valid_from=start,
            valid_until=start + timedelta(seconds=float(ttl_seconds)),
        )

    def judge(self, as_of: datetime) -> ApplicabilityVerdict:
        moment = ensure_aware(as_of)
        start = ensure_aware(self.valid_from)
        if moment < start:
            return ApplicabilityVerdict.NOT_YET
        if self.valid_until is not None and moment >= ensure_aware(self.valid_until):
            return ApplicabilityVerdict.EXPIRED
        return ApplicabilityVerdict.APPLICABLE


class ProcessRecord(BaseModel):
    """One preserved process event, with an explicit applicability window."""

    model_config = ConfigDict(frozen=False)

    record_id: str
    kind: ProcessKind
    summary: str
    passage_ids: List[str] = Field(default_factory=list)
    related_fact_key: Optional[str] = None
    constraint: Optional[str] = None
    applicability: ApplicabilityWindow
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ordinal: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)


def add_process_record(memory: GlobalMemoryState, record: ProcessRecord) -> ProcessRecord:
    """Append ``record``, keeping a unique id and a monotonic ordinal."""
    if record.record_id in memory.process_log:
        record = record.model_copy(
            update={"record_id": f"{record.record_id}#{len(memory.process_log) + 1}"}
        )
    if record.ordinal <= 0:
        record = record.model_copy(update={"ordinal": len(memory.process_log) + 1})
    memory.process_log[record.record_id] = record
    return record


def _window(
    ttl_seconds: Optional[float], valid_from: Optional[datetime]
) -> ApplicabilityWindow:
    return ApplicabilityWindow.from_ttl(ttl_seconds, valid_from=valid_from)


def record_failed_attempt(
    memory: GlobalMemoryState,
    summary: str,
    *,
    passage_id: Optional[str] = None,
    ttl_seconds: Optional[float] = None,
    valid_from: Optional[datetime] = None,
) -> ProcessRecord:
    return add_process_record(
        memory,
        ProcessRecord(
            record_id=f"fail:{len(memory.process_log) + 1:04d}",
            kind=ProcessKind.FAILED_ATTEMPT,
            summary=summary,
            passage_ids=[passage_id] if passage_id else [],
            applicability=_window(ttl_seconds, valid_from),
        ),
    )


def record_rejection(
    memory: GlobalMemoryState,
    fact: Fact,
    passage_id: str,
    *,
    reason: str,
    ttl_seconds: Optional[float] = None,
    valid_from: Optional[datetime] = None,
) -> ProcessRecord:
    """Preserve a refused triple outside M_fac."""
    passage_ids = [passage_id] if passage_id else list(fact.passage_ids)
    summary = (
        f"{reason}: {fact.head_entity} {fact.relation} {fact.tail_entity}"
    )
    return add_process_record(
        memory,
        ProcessRecord(
            record_id=f"rej:{fact.fact_key}:{len(memory.process_log) + 1:04d}",
            kind=ProcessKind.REJECTION,
            summary=summary,
            passage_ids=passage_ids,
            related_fact_key=fact.fact_key,
            applicability=_window(ttl_seconds, valid_from),
            metadata={"reason": reason},
        ),
    )


def record_negative_constraint(
    memory: GlobalMemoryState,
    constraint: str,
    *,
    passage_ids: Optional[List[str]] = None,
    ttl_seconds: Optional[float] = None,
    valid_from: Optional[datetime] = None,
) -> ProcessRecord:
    text = constraint.strip()
    return add_process_record(
        memory,
        ProcessRecord(
            record_id=f"neg:{len(memory.process_log) + 1:04d}",
            kind=ProcessKind.NEGATIVE_CONSTRAINT,
            summary=text,
            constraint=text,
            passage_ids=list(passage_ids or []),
            applicability=_window(ttl_seconds, valid_from),
        ),
    )


def expire_rejections(
    memory: GlobalMemoryState,
    fact_key: str,
    *,
    as_of: Optional[datetime] = None,
) -> None:
    """Close rejection windows once the same triple is accepted into M_fac.

    The record stays in the log. Later judgments see it as expired.
    """
    moment = ensure_aware(as_of or datetime.now(timezone.utc))
    for record in memory.process_log.values():
        if not isinstance(record, ProcessRecord):
            continue
        if record.kind != ProcessKind.REJECTION:
            continue
        if record.related_fact_key != fact_key:
            continue
        if record.applicability.judge(moment) != ApplicabilityVerdict.APPLICABLE:
            continue
        record.applicability.valid_until = moment


def applicable_process_records(
    memory: GlobalMemoryState,
    as_of: Optional[datetime] = None,
) -> List[ProcessRecord]:
    moment = ensure_aware(as_of or datetime.now(timezone.utc))
    records = [
        record
        for record in memory.process_log.values()
        if isinstance(record, ProcessRecord)
        and record.applicability.judge(moment) == ApplicabilityVerdict.APPLICABLE
    ]
    records.sort(key=lambda record: record.ordinal)
    return records
