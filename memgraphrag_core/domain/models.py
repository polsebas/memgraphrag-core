"""Pydantic V2 models for the three-layer global memory (M_ont, M_fac, M_pas)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

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
            self.expire_rejections(existing.fact_key)
            return existing

        self.facts[fact.fact_key] = fact
        self.expire_rejections(fact.fact_key)
        return fact

    def add_passage(self, passage: Passage) -> Passage:
        self.passages[passage.id] = passage
        return passage

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
