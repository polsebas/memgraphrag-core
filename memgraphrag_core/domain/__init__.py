"""Domain models for the three-layer global memory."""

from memgraphrag_core.domain.models import (
    EDGE_EVIDENCE_IN,
    EDGE_INSTANTIATES,
    EDGE_SUPPORTED_BY,
    GRAPH_NODE_LABELS,
    LAYER_M_FAC,
    LAYER_M_ONT,
    LAYER_M_PAS,
    MEMORY_NAMESPACES,
    NODE_FACT,
    NODE_PASSAGE,
    NODE_SCHEMA,
    ConflictType,
    Fact,
    GlobalMemoryState,
    Passage,
    Schema,
)
from memgraphrag_core.domain.process_memory import (
    ApplicabilityVerdict,
    ApplicabilityWindow,
    ProcessKind,
    ProcessRecord,
)

__all__ = [
    "EDGE_EVIDENCE_IN",
    "EDGE_INSTANTIATES",
    "EDGE_SUPPORTED_BY",
    "GRAPH_NODE_LABELS",
    "LAYER_M_FAC",
    "LAYER_M_ONT",
    "LAYER_M_PAS",
    "MEMORY_NAMESPACES",
    "NODE_FACT",
    "NODE_PASSAGE",
    "NODE_SCHEMA",
    "ApplicabilityVerdict",
    "ApplicabilityWindow",
    "ConflictType",
    "Fact",
    "GlobalMemoryState",
    "Passage",
    "ProcessKind",
    "ProcessRecord",
    "Schema",
]
