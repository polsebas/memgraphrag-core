"""Entity resolution / canonicalization against the vectorized entity registry."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from memgraphrag_core.ingestion.schemas import ExtractedFact
from memgraphrag_core.interfaces.vector_store import BaseVectorStore

_ENTITY_NAMESPACE = "M_ent"

# Domain-agnostic legal / boilerplate tokens only.
# Domain demos (e.g. wine) may pass extra noise via EntityResolver(noise_tokens=...).
_DEFAULT_NOISE_TOKENS: Set[str] = {
    "company",
    "corp",
    "corporation",
    "sa",
    "s.a",
    "s.a.",
    "inc",
    "ltd",
    "llc",
    "the",
}


def _normalize_surface(name: str, noise_tokens: Optional[Set[str]] = None) -> str:
    noise = noise_tokens if noise_tokens is not None else _DEFAULT_NOISE_TOKENS
    s = name.strip().lower()
    s = re.sub(r"[^\w\s\.]", " ", s)
    tokens = [t for t in s.split() if t and t not in noise]
    return " ".join(tokens)


class EntityResolver:
    """Canonicalize entity mentions via vector similarity over namespace M_ent."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        similarity_threshold: float = 0.85,
        noise_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        base = set(_DEFAULT_NOISE_TOKENS)
        if noise_tokens is not None:
            base |= {t.strip().lower() for t in noise_tokens if t and t.strip()}
        self.noise_tokens: Set[str] = base
        # Local alias map: (type_lower, normalized_surface) -> canonical name
        self._alias_index: Dict[Tuple[str, str], str] = {}

    def _norm(self, name: str) -> str:
        return _normalize_surface(name, self.noise_tokens)

    async def register_entity(self, entity_name: str, entity_type: str) -> str:
        """Index a canonical entity so future mentions can resolve to it."""
        canonical = entity_name.strip()
        etype = entity_type.strip()
        norm = self._norm(canonical)
        self._alias_index[(etype.lower(), norm)] = canonical
        await self.vector_store.add_texts(
            texts=[f"{etype}: {canonical}"],
            ids=[f"{etype.lower()}::{norm}"],
            metadatas=[
                {
                    "entity_type": etype,
                    "canonical_name": canonical,
                    "normalized": norm,
                }
            ],
            namespace=_ENTITY_NAMESPACE,
        )
        return canonical

    async def canonicalize_entity(self, entity_name: str, entity_type: str) -> str:
        """Return the canonical entity name for a mention of the given type."""
        raw = entity_name.strip()
        etype = entity_type.strip()
        if not raw:
            return raw

        norm = self._norm(raw)
        cached = self._alias_index.get((etype.lower(), norm))
        if cached:
            return cached

        # Exact normalized match against known aliases of any punctuation variant.
        for (t, n), canonical in self._alias_index.items():
            if t == etype.lower() and (n == norm or n in norm or norm in n):
                if n and norm and (n == norm or len(n) >= 4 and (n in norm or norm in n)):
                    self._alias_index[(etype.lower(), norm)] = canonical
                    return canonical

        hits = await self.vector_store.search_similarity(
            query=f"{etype}: {raw}",
            k=5,
            namespace=_ENTITY_NAMESPACE,
            filter_metadata={"entity_type": etype},
        )
        for hit in hits:
            if hit.score < self.similarity_threshold:
                continue
            canonical = str(hit.metadata.get("canonical_name") or hit.text)
            # Prefer metadata canonical_name; fall back carefully.
            if ":" in canonical and hit.metadata.get("canonical_name") is None:
                canonical = canonical.split(":", 1)[-1].strip()
            self._alias_index[(etype.lower(), norm)] = canonical
            await self.register_entity(canonical, etype)
            return canonical

        # Soft phonetic/containment fallback among local index (same type).
        soft = self._soft_match(norm, etype)
        if soft:
            self._alias_index[(etype.lower(), norm)] = soft
            return soft

        # First sighting becomes canonical.
        await self.register_entity(raw, etype)
        return raw

    def _soft_match(self, norm: str, entity_type: str) -> Optional[str]:
        if not norm:
            return None
        et = entity_type.lower()
        best: Optional[str] = None
        best_score = 0.0
        for (t, n), canonical in self._alias_index.items():
            if t != et or not n:
                continue
            if n == norm:
                return canonical
            # Token Jaccard as a cheap phonetic/surface proxy.
            a, b = set(n.split()), set(norm.split())
            if not a or not b:
                continue
            jacc = len(a & b) / len(a | b)
            containment = 1.0 if (n in norm or norm in n) else 0.0
            score = max(jacc, containment)
            if score >= 0.6 and score > best_score:
                best = canonical
                best_score = score
        return best

    async def resolve_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Canonicalize head/tail entities for each extracted fact."""
        resolved: List[ExtractedFact] = []
        for fact in facts:
            head = await self.canonicalize_entity(fact.head_entity, fact.head_type)
            tail = await self.canonicalize_entity(fact.tail_entity, fact.tail_type)
            resolved.append(
                ExtractedFact(
                    head_entity=head,
                    head_type=fact.head_type,
                    relation=fact.relation,
                    tail_entity=tail,
                    tail_type=fact.tail_type,
                )
            )
        return resolved
