"""Abstract port for vector stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseVectorStore(ABC):
    @abstractmethod
    async def add_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> List[str]:
        pass

    @abstractmethod
    async def search_similarity(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str], namespace: str = "default") -> None:
        pass
