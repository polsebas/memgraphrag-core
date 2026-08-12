# memgraphrag-core

Core library for MemGraphRAG: a three-layer global memory ($M_{ont}$, $M_{fac}$, $M_{pas}$) with memory-guided Personalized PageRank retrieval. Built with Clean Architecture / DDD on Python 3.11+.

Indexing agents (`A_ext`, `A_det`, `A_res`) are implemented with the [Agno](https://docs.agno.com) framework (`agno.Agent` + structured `output_schema`).

## Layers

- **Domain** (`domain/`): `Schema`, `Fact`, `Passage`, `GlobalMemoryState` with stability threshold $\tau$
- **Interfaces** (`interfaces/`): abstract ports for LLM, vector store, and graph store
- **Adapters** (`adapters/`): pluggable backends — `VectorStoreFactory`, `ChromaVectorStore`
- **Infrastructure** (`infrastructure/`): in-memory PoC adapters — NetworkX, token-based vector store, `ScriptedModel` (Agno)
- **Agents** (`agents/`): Agno-backed `A_ext` (extractor), `A_det` (detector), `A_res` (resolver)
- **Indexing** (`indexing/`): `IndexingOrchestrator` (Agno agents → memory)
- **Ingestion** (`ingestion/`): `IngestionPipeline` (LLM extract → entity resolve → τ-gated write)
- **Retrieval** (`retrieval/`): `MemoryGuidedRetriever` with layered PPR (`LayeredPPRCalculator`) + hybrid reranking (`HybridReranker`)

## Requirements

- Python 3.11+

## Install

**Recommended (pip + venv):**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

**Optional (uv):**

```bash
uv sync --extra dev
# or: uv pip install -e ".[dev]"
```

The project is standard PEP 621 (`pyproject.toml`). You do not need `uv` to contribute.

## Agno model wiring

Pass any Agno `Model` (or model string) into the orchestrator / agents:

```python
from agno.models.openai import OpenAIChat
from memgraphrag_core.domain.models import GlobalMemoryState
from memgraphrag_core.indexing import IndexingOrchestrator

memory = GlobalMemoryState(tau=2)
orch = IndexingOrchestrator(memory, model=OpenAIChat(id="gpt-4o-mini"))
# await orch.index_passages(passages)
```

**Google Gemini** (indexing agents):

```bash
pip install google-genai
export GOOGLE_API_KEY=...
```

```python
from agno.models.google import Gemini
from memgraphrag_core.domain.models import GlobalMemoryState
from memgraphrag_core.indexing import IndexingOrchestrator

memory = GlobalMemoryState(tau=1)
orch = IndexingOrchestrator(memory, model=Gemini(id="gemini-3.5-flash"))
```

New Gemini API keys often cannot use retired/legacy IDs (`gemini-2.0-flash`,
`gemini-2.5-flash`). Prefer `gemini-3.5-flash` (or list models via the Google GenAI SDK).

For offline tests/demos, use `ScriptedModel` from `memgraphrag_core.infrastructure`.

## Pluggable vector stores

`MemoryGuidedRetriever` and `IndexingOrchestrator` depend only on `BaseVectorStore`. Swap backends via `VectorStoreFactory`:

```python
from memgraphrag_core.adapters import VectorStoreFactory

# Dev / tests
vector_store = VectorStoreFactory.create("in_memory")

# ChromaDB (persistent or remote)
vector_store = VectorStoreFactory.create(
    "chroma",
    persist_directory="./chroma_db",
    collection_prefix="app_memgraph_",
)

# Register a custom provider at runtime (e.g. Qdrant / PgVector)
# VectorStoreFactory.register_provider("qdrant", QdrantVectorStore)
# vector_store = VectorStoreFactory.create("qdrant", host="localhost", port=6333)
```

Install Chroma support with:

```bash
pip install -e ".[chroma]"
```

Namespaces (`M_pas`, `M_ont`, `M_fac`) map to separate Chroma collections.

## Examples

```bash
python examples/mre_memgraphrag.py
python examples/mre_agents_indexing.py          # offline ScriptedModel
export GOOGLE_API_KEY=... && python examples/mre_gemini_indexing.py
# optional: MEMGRAPHRAG_GEMINI_MODEL=gemini-3-flash-preview
```

## Tests

```bash
pytest tests/
# or: pytest -q
```

Layout:
- `tests/unit/` — domain, adapters, agents, PPR/reranker
- `tests/integration/` — ingestion + retrieval flows

## Multihop benchmark

```bash
python -m benchmarks.benchmark_multihop
# optional: --k 5 --output /tmp/memgraphrag_bench.json
```

With uv:

```bash
uv run python examples/mre_memgraphrag.py
uv run python examples/mre_agents_indexing.py
uv run pytest tests/ -q
uv run python -m benchmarks.benchmark_multihop
```
