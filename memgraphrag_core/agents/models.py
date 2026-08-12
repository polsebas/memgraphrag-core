"""Shared I/O models for indexing agents."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from memgraphrag_core.domain.models import ConflictType


class ExtractedTriple(BaseModel):
    """Candidate triple extracted by A_ext from a passage."""

    head_entity: str
    head_type: str
    relation: str
    tail_entity: str
    tail_type: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_span: Optional[str] = Field(
        default=None, description="Text span that supports the extraction"
    )


class ExtractionResult(BaseModel):
    """Structured output of A_ext."""

    triples: List[ExtractedTriple] = Field(default_factory=list)
    notes: Optional[str] = None


class ConflictRecord(BaseModel):
    """Conflict detected between two facts in M_fac."""

    conflict_type: ConflictType
    fact_a_key: str
    fact_b_key: str
    explanation: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class DetectionResult(BaseModel):
    """Structured output of A_det."""

    conflicts: List[ConflictRecord] = Field(default_factory=list)


class ResolutionAction(str, Enum):
    """Possible resolution actions for a conflicting fact pair."""

    KEEP_A = "keep_a"
    KEEP_B = "keep_b"
    MERGE = "merge"
    KEEP_BOTH = "keep_both"
    DISCARD_BOTH = "discard_both"


class ResolutionDecision(BaseModel):
    """A_res decision for a ConflictRecord."""

    conflict_type: ConflictType
    fact_a_key: str
    fact_b_key: str
    action: ResolutionAction
    rationale: str
    merged_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ResolutionResult(BaseModel):
    """Structured output of A_res."""

    decisions: List[ResolutionDecision] = Field(default_factory=list)
