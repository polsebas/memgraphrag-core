"""MemGraphRAG core — three-layer global memory + memory-guided PPR retrieval.

Domain models and retrieval are exported here. Agents and indexing pull in Agno;
import them explicitly from ``memgraphrag_core.agents`` / ``memgraphrag_core.indexing``.
"""

from memgraphrag_core.domain.models import (
    ConflictType,
    Fact,
    GlobalMemoryState,
    Passage,
    Schema,
)
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever, RetrievalResult

__all__ = [
    "ConflictType",
    "Fact",
    "GlobalMemoryState",
    "MemoryGuidedRetriever",
    "Passage",
    "RetrievalResult",
    "Schema",
]

__version__ = "0.1.0"
