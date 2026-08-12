"""ChromaDB adapter implementing BaseVectorStore with per-namespace collections."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from memgraphrag_core.interfaces.vector_store import BaseVectorStore, VectorSearchResult

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_collection_name(name: str) -> str:
    """Ensure a Chroma-compatible collection name (3-512 chars, [a-zA-Z0-9._-])."""
    cleaned = _SAFE_NAME_RE.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    if len(cleaned) < 3:
        cleaned = f"ns_{cleaned}" if cleaned else "ns_default"
    return cleaned[:512]


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma only accepts str | int | float | bool metadata values."""
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def _distance_to_score(distance: float) -> float:
    """Convert a non-negative distance into a similarity-like score in (0, 1]."""
    return 1.0 / (1.0 + max(float(distance), 0.0))


class ChromaVectorStore(BaseVectorStore):
    """BaseVectorStore backed by ChromaDB.

    Each ``namespace`` maps to its own Chroma collection
    (``{collection_prefix}{namespace}``), e.g. ``app_memgraph_M_pas``.

    Client modes:
    - Ephemeral (default): in-process, non-persistent
    - Persistent: ``persist_directory=...``
    - Remote HTTP: ``host=...`` (optional ``port``, ``ssl``)

    Sync Chroma client calls run in ``asyncio.to_thread`` so they do not block
    the event loop under concurrent FastAPI / AgentOS load.
    """

    def __init__(
        self,
        *,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        ssl: bool = False,
        collection_prefix: str = "memgraphrag_",
        client: Any = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ChromaVectorStore requires chromadb. "
                "Install with: pip install 'memgraphrag-core[chroma]'"
            ) from exc

        self.collection_prefix = collection_prefix
        self._collections: Dict[str, Any] = {}

        if client is not None:
            self._client = client
        elif host is not None:
            self._client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

    def _collection_name(self, namespace: str) -> str:
        return _sanitize_collection_name(f"{self.collection_prefix}{namespace}")

    def _get_collection(self, namespace: str) -> Any:
        name = self._collection_name(namespace)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name=name)
        return self._collections[name]

    def _upsert_sync(
        self,
        namespace: str,
        texts: List[str],
        ids: List[str],
        clean_metas: List[Dict[str, Any]],
    ) -> None:
        collection = self._get_collection(namespace)
        collection.upsert(
            documents=list(texts),
            ids=list(ids),
            metadatas=clean_metas,
        )

    def _query_sync(
        self,
        namespace: str,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[VectorSearchResult]:
        collection = self._get_collection(namespace)
        count = collection.count()
        if count == 0:
            return []

        n_results = min(k, count)
        where = _sanitize_metadata(filter_metadata) if filter_metadata else None
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        raw = collection.query(**kwargs)
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: List[VectorSearchResult] = []
        for doc_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            if doc_id is None:
                continue
            results.append(
                VectorSearchResult(
                    id=str(doc_id),
                    text=text or "",
                    score=_distance_to_score(distance if distance is not None else 1e9),
                    metadata=dict(meta or {}),
                )
            )
        return results

    def _delete_sync(self, namespace: str, ids: List[str]) -> None:
        collection = self._get_collection(namespace)
        collection.delete(ids=list(ids))

    async def add_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> List[str]:
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")
        metas = metadatas or [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")

        clean_metas: List[Dict[str, Any]] = []
        for m in metas:
            cleaned = _sanitize_metadata(m)
            cleaned.setdefault("namespace", namespace)
            clean_metas.append(cleaned)

        await asyncio.to_thread(self._upsert_sync, namespace, texts, ids, clean_metas)
        return list(ids)

    async def search_similarity(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        return await asyncio.to_thread(
            self._query_sync, namespace, query, k, filter_metadata
        )

    async def delete(self, ids: List[str], namespace: str = "default") -> None:
        if not ids:
            return
        await asyncio.to_thread(self._delete_sync, namespace, ids)
