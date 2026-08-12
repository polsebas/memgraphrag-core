"""Factory for pluggable BaseVectorStore implementations."""

from __future__ import annotations

from typing import Callable, Dict, Type

from memgraphrag_core.interfaces.vector_store import BaseVectorStore

StoreConstructor = Callable[..., BaseVectorStore]


class VectorStoreFactory:
    """Create vector stores by provider name with runtime registration support."""

    _providers: Dict[str, StoreConstructor] = {}
    _defaults_registered: bool = False

    @classmethod
    def _ensure_defaults(cls) -> None:
        if cls._defaults_registered:
            return
        # Lazy imports keep optional backends (e.g. chromadb) out of the critical path.
        from memgraphrag_core.adapters.chroma_vector_store import ChromaVectorStore
        from memgraphrag_core.infrastructure.in_memory_vector_store import (
            InMemoryVectorStore,
        )

        cls._providers.setdefault("in_memory", InMemoryVectorStore)
        cls._providers.setdefault("chroma", ChromaVectorStore)
        cls._defaults_registered = True

    @classmethod
    def register_provider(cls, name: str, store_class: Type[BaseVectorStore] | StoreConstructor) -> None:
        """Register or override a vector-store provider.

        Example::

            VectorStoreFactory.register_provider("qdrant", QdrantVectorStore)
        """
        cls._ensure_defaults()
        key = name.strip().lower()
        if not key:
            raise ValueError("provider name must be a non-empty string")
        cls._providers[key] = store_class

    @classmethod
    def available_providers(cls) -> list[str]:
        cls._ensure_defaults()
        return sorted(cls._providers.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseVectorStore:
        """Instantiate a vector store by registered name.

        Examples::

            VectorStoreFactory.create("in_memory")
            VectorStoreFactory.create(
                "chroma",
                persist_directory="./chroma_db",
                collection_prefix="app_memgraph_",
            )
        """
        cls._ensure_defaults()
        key = name.strip().lower()
        if key not in cls._providers:
            known = ", ".join(cls.available_providers())
            raise ValueError(f"Unknown vector store provider {name!r}. Known: {known}")
        return cls._providers[key](**kwargs)
