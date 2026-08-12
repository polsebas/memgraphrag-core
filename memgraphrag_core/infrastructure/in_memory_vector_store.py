"""In-memory vector store with bag-of-tokens cosine similarity (PoC / tests)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from memgraphrag_core.interfaces.vector_store import BaseVectorStore, VectorSearchResult

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> Counter[str]:
    return Counter(t.lower() for t in _TOKEN_RE.findall(text))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(BaseVectorStore):
    """Namespace-scoped collections; cosine similarity over bag-of-tokens."""

    def __init__(self) -> None:
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _ns(self, namespace: str) -> Dict[str, Dict[str, Any]]:
        return self._collections.setdefault(namespace, {})

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

        store = self._ns(namespace)
        for text, doc_id, meta in zip(texts, ids, metas):
            store[doc_id] = {
                "text": text,
                "metadata": dict(meta),
                "tokens": _tokenize(text),
            }
        return list(ids)

    async def search_similarity(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        store = self._ns(namespace)
        q_tokens = _tokenize(query)
        results: List[VectorSearchResult] = []

        for doc_id, doc in store.items():
            meta = doc["metadata"]
            if filter_metadata and any(meta.get(k_) != v for k_, v in filter_metadata.items()):
                continue
            score = _cosine(q_tokens, doc["tokens"])
            if score > 0.0:
                results.append(
                    VectorSearchResult(
                        id=doc_id,
                        text=doc["text"],
                        score=score,
                        metadata=dict(meta),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def delete(self, ids: List[str], namespace: str = "default") -> None:
        store = self._ns(namespace)
        for doc_id in ids:
            store.pop(doc_id, None)
