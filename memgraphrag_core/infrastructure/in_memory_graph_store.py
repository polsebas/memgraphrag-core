"""In-memory graph store alias (NetworkX-backed)."""

from memgraphrag_core.infrastructure.networkx_graph_store import NetworkXGraphStore


class InMemoryGraphStore(NetworkXGraphStore):
    """Alias for the ephemeral NetworkX graph used in tests and local PoCs."""

    pass
