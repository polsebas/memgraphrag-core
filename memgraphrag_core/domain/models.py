"""Pydantic V2 models for the three-layer global memory (M_ont, M_fac, M_pas)."""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Paper layers: M_pas (evidence), M_ont (schemas), M_fac (grounded facts).
LAYER_M_PAS = "M_pas"
LAYER_M_ONT = "M_ont"
LAYER_M_FAC = "M_fac"
MEMORY_NAMESPACES = (LAYER_M_PAS, LAYER_M_ONT, LAYER_M_FAC)

# Hierarchical graph vocabulary (canonical across ingestion, indexing, retrieval).
NODE_PASSAGE = "Passage"
NODE_SCHEMA = "Schema"
NODE_FACT = "Fact"
GRAPH_NODE_LABELS = (NODE_PASSAGE, NODE_SCHEMA, NODE_FACT)

# Schema -INSTANTIATES-> Fact -EVIDENCE_IN-> Passage
EDGE_INSTANTIATES = "INSTANTIATES"
EDGE_EVIDENCE_IN = "EVIDENCE_IN"
# Pre-unification alias; weighted like EVIDENCE_IN for old graphs only.
EDGE_SUPPORTED_BY = "SUPPORTED_BY"


class FactCategory(str, Enum):
    """Process-aware label for a fact stored in M_fac.

    Declarative facts are the usual domain triples. The other three values
    keep process information that a summary would drop: hard limits, explicit
    rejections, and tool failures observed as evidence.
    """

    DECLARATIVE = "declarative"
    HARD_CONSTRAINT = "hard_constraint"
    REJECTED_OPTION = "rejected_option"
    TOOL_FAILURE = "tool_failure"


def coerce_fact_category(value: Any) -> Any:
    """Accept enum names or values. Empty input stays the declarative default."""
    if value is None or value == "":
        return FactCategory.DECLARATIVE
    if isinstance(value, FactCategory):
        return value
    text = str(value).strip()
    member = text.upper().replace("-", "_")
    if member in FactCategory.__members__:
        return FactCategory[member]
    lowered = text.lower().replace("-", "_")
    for item in FactCategory:
        if item.value == lowered:
            return item
    return value


class ConflictType(str, Enum):
    """Conflict types detectable among facts in M_fac."""

    REDUNDANCY = "redundancy"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    TEMPORAL_CONFLICT = "temporal_conflict"
    GRANULARITY_MISMATCH = "granularity_mismatch"


class Schema(BaseModel):
    """Ontological layer (M_ont): conceptual schemas gated by stability threshold tau."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    head_type: str = Field(..., description="Head entity type (e.g. Organization, Person)")
    relation: str = Field(..., description="Semantic relation (e.g. LOCATED_IN, WORKS_AT)")
    tail_type: str = Field(..., description="Tail entity type (e.g. City, Product)")
    frequency: int = Field(default=1, ge=0, description="Observed frequency in the corpus")

    @property
    def schema_key(self) -> str:
        return (
            f"({self.head_type.strip().lower()})"
            f"-[{self.relation.strip().lower()}]->"
            f"({self.tail_type.strip().lower()})"
        )

    def is_stable(self, tau: int) -> bool:
        return self.frequency >= tau


class Fact(BaseModel):
    """Fact layer (M_fac): concrete triples linked to stable schemas and passage evidence."""

    model_config = ConfigDict(frozen=False)

    fact_id: Optional[str] = Field(default=None)
    head_entity: str = Field(..., description="Head entity surface form (e.g. Acme Corp)")
    head_type: str = Field(..., description="Head type (e.g. Organization)")
    relation: str = Field(..., description="Relation name")
    tail_entity: str = Field(..., description="Tail entity surface form (e.g. Berlin)")
    tail_type: str = Field(..., description="Tail type (e.g. City)")
    passage_ids: List[str] = Field(
        default_factory=list, description="Evidence IDs in M_pas"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    category: FactCategory = Field(
        default=FactCategory.DECLARATIVE,
        description="DECLARATIVE, HARD_CONSTRAINT, REJECTED_OPTION, or TOOL_FAILURE",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Lifetime in seconds. None means the fact does not expire.",
    )
    is_invalidated: bool = Field(
        default=False,
        description="Explicitly retired fact. PPR assigns it weight 0.",
    )
    source_passage_id: str = Field(
        default="",
        description="Immutable M_pas id this fact was compressed from.",
    )

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: Any) -> Any:
        return coerce_fact_category(value)

    @model_validator(mode="after")
    def _bind_source_passage(self) -> "Fact":
        if not self.source_passage_id and self.passage_ids:
            self.source_passage_id = self.passage_ids[0]
        return self

    @property
    def fact_key(self) -> str:
        return (
            f"{self.head_entity.strip().lower()}"
            f"--{self.relation.strip().lower()}"
            f"--{self.tail_entity.strip().lower()}"
        )

    @property
    def schema_key(self) -> str:
        return (
            f"({self.head_type.strip().lower()})"
            f"-[{self.relation.strip().lower()}]->"
            f"({self.tail_type.strip().lower()})"
        )

    def is_expired(self, as_of: datetime) -> bool:
        """True when ``as_of`` is at or past ``created_at + ttl_seconds``."""
        if self.ttl_seconds is None:
            return False
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        moment = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
        deadline = created.astimezone(timezone.utc) + timedelta(seconds=int(self.ttl_seconds))
        return moment >= deadline

    def is_suppressed(self, as_of: datetime) -> bool:
        return self.is_invalidated or self.is_expired(as_of)

    def graph_properties(self) -> Dict[str, Any]:
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return {
            "head": self.head_entity,
            "relation": self.relation,
            "tail": self.tail_entity,
            "category": self.category.value,
            "created_at": created.astimezone(timezone.utc).isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "is_invalidated": self.is_invalidated,
            "source_passage_id": self.source_passage_id,
        }


class Passage(BaseModel):
    """Passage layer (M_pas): source text for grounding and traceability."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Original source text")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GlobalMemoryState(BaseModel):
    """Three-layer global memory aggregate with stability threshold (tau)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tau: int = Field(default=2, ge=1, description="Ontological stability threshold")
    ontologies: Dict[str, Schema] = Field(default_factory=dict)
    facts: Dict[str, Fact] = Field(default_factory=dict)
    passages: Dict[str, Passage] = Field(default_factory=dict)
    # Process log is not a fourth paper layer. Rejected triples stay out of M_fac.
    process_log: Dict[str, Any] = Field(default_factory=dict)

    def register_schema_candidate(
        self, head_type: str, relation: str, tail_type: str
    ) -> Schema:
        temp = Schema(head_type=head_type, relation=relation, tail_type=tail_type)
        key = temp.schema_key
        if key in self.ontologies:
            self.ontologies[key].frequency += 1
        else:
            self.ontologies[key] = temp
        return self.ontologies[key]

    def get_stable_schemas(self) -> List[Schema]:
        return [s for s in self.ontologies.values() if s.is_stable(self.tau)]

    def add_fact(self, fact: Fact) -> Optional[Fact]:
        if not fact.fact_id:
            fact.fact_id = fact.fact_key

        schema = self.ontologies.get(fact.schema_key)
        # Stability rule: only instantiate facts whose schema is stable
        if not schema or not schema.is_stable(self.tau):
            return None

        if fact.fact_key in self.facts:
            existing = self.facts[fact.fact_key]
            existing.passage_ids = list(set(existing.passage_ids + fact.passage_ids))
            existing.confidence = max(existing.confidence, fact.confidence)
            if not existing.source_passage_id:
                existing.source_passage_id = fact.source_passage_id
            self.expire_rejections(existing.fact_key)
            return existing

        self.facts[fact.fact_key] = fact
        self.expire_rejections(fact.fact_key)
        return fact

    def add_passage(self, passage: Passage) -> Passage:
        self.passages[passage.id] = passage
        return passage

    def get_fact(self, fact_id: str) -> Fact:
        """Resolve a fact by ``fact_key`` or ``fact_id``."""
        if fact_id in self.facts:
            return self.facts[fact_id]
        for fact in self.facts.values():
            if fact.fact_id == fact_id or fact.fact_key == fact_id:
                return fact
        raise KeyError(fact_id)

    def get_raw_passage(self, passage_id: str) -> Passage:
        """Return the immutable M_pas passage. Raises KeyError when it is missing."""
        try:
            return self.passages[passage_id]
        except KeyError as exc:
            raise KeyError(f"M_pas has no passage {passage_id}") from exc

    def expand_fact_evidence(self, fact_id: str) -> str:
        """Drill from an M_fac fact back to the exact M_pas text it was written from."""
        fact = self.get_fact(fact_id)
        passage_id = fact.source_passage_id or (fact.passage_ids[0] if fact.passage_ids else "")
        if not passage_id:
            raise KeyError(f"fact {fact_id} has no source passage in M_pas")
        return self.get_raw_passage(passage_id).text

    def add_process_record(self, record: Any) -> Any:
        from memgraphrag_core.domain.process_memory import add_process_record

        return add_process_record(self, record)

    def record_failed_attempt(
        self,
        summary: str,
        *,
        passage_id: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        valid_from: Optional[datetime] = None,
    ) -> Any:
        from memgraphrag_core.domain.process_memory import record_failed_attempt

        return record_failed_attempt(
            self,
            summary,
            passage_id=passage_id,
            ttl_seconds=ttl_seconds,
            valid_from=valid_from,
        )

    def record_rejection(
        self,
        fact: Fact,
        passage_id: str,
        *,
        reason: str,
        ttl_seconds: Optional[float] = None,
        valid_from: Optional[datetime] = None,
    ) -> Any:
        from memgraphrag_core.domain.process_memory import record_rejection

        return record_rejection(
            self,
            fact,
            passage_id,
            reason=reason,
            ttl_seconds=ttl_seconds,
            valid_from=valid_from,
        )

    def record_negative_constraint(
        self,
        constraint: str,
        *,
        passage_ids: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
        valid_from: Optional[datetime] = None,
    ) -> Any:
        from memgraphrag_core.domain.process_memory import record_negative_constraint

        return record_negative_constraint(
            self,
            constraint,
            passage_ids=passage_ids,
            ttl_seconds=ttl_seconds,
            valid_from=valid_from,
        )

    def expire_rejections(
        self, fact_key: str, *, as_of: Optional[datetime] = None
    ) -> None:
        from memgraphrag_core.domain.process_memory import expire_rejections

        expire_rejections(self, fact_key, as_of=as_of)

    def applicable_process_records(self, as_of: Optional[datetime] = None) -> List[Any]:
        from memgraphrag_core.domain.process_memory import applicable_process_records

        return applicable_process_records(self, as_of)
