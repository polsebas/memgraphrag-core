# MemGraphRAG Core — Technical Audit Report

**Date:** 2026-08-12  
**Scope:** `memgraphrag_core/`, `tests/`, `benchmarks/`, `examples/`  
**Goal:** Pre-stable packaging review (architecture, typing/async, edge cases, naming)

---

## Executive Summary

The codebase is broadly aligned with Clean Architecture: `domain/` stays free of concrete adapters, and `retrieval/` / `ingestion/` depend on `interfaces/`. Automated tests (**35 passed**) and the multihop benchmark run successfully.

However, there are **real consistency bugs** when mixing indexing vs ingestion graph writers, **async I/O that blocks the event loop** in Chroma/NetworkX adapters, and **naming collisions** (`ExtractionResult` duplicated). None of these currently break the happy-path tests, but they will cause subtle production failures if pipelines are composed.

Severity legend: **P0** = must fix before stable · **P1** = should fix soon · **P2** = improvement.

---

## 1. Critical Errors / Bugs Detected

### 1.1 [P0] Divergent graph edge vocabulary between `IndexingOrchestrator` and `IngestionPipeline`

| Writer | Fact ↔ Ontology | Fact ↔ Passage |
|--------|-----------------|----------------|
| `indexing/orchestrator.py` | `ont → fact` via `INSTANTIATES` | `fact → pas` via **`SUPPORTED_BY`** |
| `ingestion/pipeline.py` | `fact → ont` via `INSTANTIATES` | `fact → pas` via **`EVIDENCE_IN`** |

**Impact:**
- `LayeredPPRCalculator` defaults weight `EVIDENCE_IN` and `SUPPORTED_BY` separately; mixed corpora get uneven transfer.
- `get_neighbors(..., relation_type="EVIDENCE_IN")` misses edges written by indexing.
- Direction of `INSTANTIATES` differs; only works reliably because PPR often runs with `bidirectional=True`.

**Evidence:** `indexing/orchestrator.py` (~L157–172) vs `ingestion/pipeline.py` (~L201–214).

### 1.2 [P0] Synchronous Chroma / NetworkX work inside `async def` (event-loop blocking)

- `ChromaVectorStore.add_texts/search_similarity/delete` call blocking `chromadb` client APIs directly inside async methods.
- `NetworkXGraphStore.run_ppr` runs CPU-bound NumPy power iteration on the event loop.

**Impact:** Under concurrent `asyncio` load (AgentOS, FastAPI, batch ingest), latency spikes and starvation. Signatures look async-compliant but semantics are sync.

### 1.3 [P1] Dual `ExtractionResult` models with the same class name

- `memgraphrag_core.agents.models.ExtractionResult` → `{triples, notes}`
- `memgraphrag_core.ingestion.schemas.ExtractionResult` → `{schemas, facts}`

**Impact:**
- `MockLLMProvider` / `ScriptedModel` dispatch by `__name__ == "ExtractionResult"` → ambiguous if both paths share one registry.
- Public exports: `ingestion.__init__` and `agents.__init__` both export `ExtractionResult`.

### 1.4 [P1] Package root eagerly imports Agno agents

`memgraphrag_core/__init__.py` imports `DetectorAgent` / `ExtractorAgent` / `ResolverAgent`, which import `agno`.

**Impact:** `import memgraphrag_core` fails or pulls heavy deps even for consumers that only need domain + retrieval adapters. Breaks “optional Agno” mental model.

### 1.5 [P1] Ingestion LLM failures are unhandled

`IngestionPipeline.process_document` awaits `LLMExtractor.extract_from_text` with no `try/except`. A malformed LLM payload or provider outage aborts the whole document mid-passage (partial writes already flushed to memory/vector/graph).

### 1.6 [P2] Empty document / empty retrieve degrade gracefully (OK)

Probed:
- `process_document("")` → empty report, no crash.
- `run_ppr` on empty graph → `{}`.
- `retrieve` with empty memory → `vector_fallback`, 0 passages.

No bug here; listed as verified positive behavior.

---

## 2. Architectural Incongruities

### 2.1 Clean Architecture — what holds

| Layer | Depends on | Status |
|-------|------------|--------|
| `domain/` | `pydantic` only | **OK** |
| `interfaces/` | `pydantic` / stdlib | **OK** |
| `retrieval/` | `domain` + `interfaces` | **OK** |
| `ingestion/` | `domain` + `interfaces` | **OK** |
| `adapters/` | `interfaces` (+ optional chromadb) | **OK** |
| `infrastructure/` | `interfaces` (+ networkx/numpy/agno for scripts) | **OK** |

### 2.2 Abstraction leaks / dual stacks

1. **Two ingestion stacks:**
   - Agno agents (`agents/` + `IndexingOrchestrator`) — model via Agno `Model`
   - Port-based LLM (`ingestion/` + `BaseLLMProvider`)
   
   Not wrong per se, but undocumented overlap; risk of diverging prompts/normalization.

2. **Split adapter homes:** vector Chroma lives in `adapters/`, in-memory vector/graph live in `infrastructure/`. Factory bridges them, but package story is inconsistent for contributors.

3. **`BaseLLMProvider` is unused by Agno agents.** Interface exists, but A_ext/A_det/A_res bypass it. Clean Architecture “port” is only honored by `ingestion/`.

4. **Hard dependency on `networkx`/`agno` in `pyproject.toml`**, while Chroma is optional. Audit criterion “deferred import for networkx” is **not** met (only Chroma has friendly `ImportError`).

### 2.3 Signature compliance

Abstract vs concrete signatures for `BaseVectorStore` and `BaseGraphStore` (**including** `run_ppr(..., relation_weights, bidirectional)`) **match**. No Liskov signature bugs found.

---

## 3. Refactoring & Improvement Proposals

### 3.1 Normalize graph ontology (P0)

Adopt a single canonical edge convention project-wide:

```text
Schema --INSTANTIATES--> Fact --EVIDENCE_IN--> Passage
```

(or the reverse, but **one** direction only). Update indexing + ingestion + examples + tests together. Keep `SUPPORTED_BY` as a deprecated alias in `DEFAULT_RELATION_WEIGHTS` during migration.

### 3.2 Offload blocking I/O / CPU (P0/P1)

```python
await asyncio.to_thread(collection.upsert, ...)
await asyncio.to_thread(self._run_ppr_sync, ...)
```

### 3.3 Rename colliding schemas (P1)

- Agents: `AgentExtractionResult` (or `TripleExtractionResult`)
- Ingestion: `DocumentExtractionResult`

Update ScriptedModel heuristics accordingly.

### 3.4 Lazy public exports (P1)

Make `memgraphrag_core/__init__.py` export domain + retrieval only; document `from memgraphrag_core.agents import ...` separately so Agno is not required for core import.

### 3.5 Ingestion resilience (P1)

- Skip empty text early with an explicit report flag (`skipped_empty_document=True`).
- Per-passage `try/except` around LLM extract; continue with remaining passages; collect `errors[]` in the report.
- Optional transactional staging (write memory only after full extract) — larger change.

### 3.6 Dead / unused knobs (P2)

- `ExtractorAgent.temperature` / `LLMExtractor.temperature` stored but not applied to Agno `Agent` construction.
- `ConflictType` defined in domain but only used by agents — fine, keep.
- `BaseLLMProvider` without a production adapter (only `MockLLMProvider`) — add LiteLLM/Ollama adapter or document as intentional port.

### 3.7 Type hygiene (P2)

- Prefer `list[str]` / `dict[str, float]` consistently (3.11+); mix of `List`/`Dict` and builtins is harmless but uneven.
- `GlobalMemoryState` mutates nested Pydantic models in-place (`frequency += 1`) — document as intentionally mutable aggregate; consider `model_copy` if immutability is desired later.

### 3.8 Factory global registry (P2)

`VectorStoreFactory._providers` is process-global. Tests calling `register_provider` can leak. Provide `VectorStoreFactory.clear_custom_providers()` or instance-scoped factories for tests.

### 3.9 Performance

- Ingestion indexes schemas on every candidate registration (repeated `add_texts` for same schema key) — upsert OK for Chroma, but noisy for in-memory.
- Entity resolver may `register_entity` again after a vector hit (redundant upsert).

---

## 4. Suggested Patches

### Patch A — Unify evidence edge label in indexing (P0)

```python
# memgraphrag_core/indexing/orchestrator.py  (_sync_fact)
# BEFORE:
#   relation="SUPPORTED_BY",
# AFTER:
                await self.graph_store.add_edge(
                    GraphEdge(
                        source_id=fid,
                        target_id=f"pas_{pid}",
                        relation="EVIDENCE_IN",
                    )
                )
```

Also align `INSTANTIATES` direction with ingestion (`fact → ont`) **or** change ingestion to `ont → fact`. Recommended single convention:

```python
# Prefer Schema → Fact (indexing style), update ingestion:
await self.graph_store.add_edge(
    GraphEdge(
        source_id=f"ont_{fact.schema_key}",
        target_id=fact_node,
        relation="INSTANTIATES",
    )
)
```

### Patch B — Non-blocking Chroma upsert/query (P0)

```python
# memgraphrag_core/adapters/chroma_vector_store.py
import asyncio

async def add_texts(...):
    ...
    await asyncio.to_thread(
        collection.upsert,
        documents=list(texts),
        ids=list(ids),
        metadatas=clean_metas,
    )
    return list(ids)

async def search_similarity(...):
    ...
    raw = await asyncio.to_thread(collection.query, **kwargs)
    ...
```

### Patch C — Non-blocking PPR (P1)

```python
# memgraphrag_core/infrastructure/networkx_graph_store.py
async def run_ppr(self, ...):
    return await asyncio.to_thread(
        self._run_ppr_sync,
        seed_node_weights,
        alpha,
        max_iter,
        tol,
        relation_weights,
        bidirectional,
    )
```

(Move current body into `_run_ppr_sync`.)

### Patch D — Rename ingestion extraction model (P1)

```python
# memgraphrag_core/ingestion/schemas.py
class DocumentExtractionResult(BaseModel):
    schemas: List[ExtractedSchema] = Field(default_factory=list)
    facts: List[ExtractedFact] = Field(default_factory=list)

# Keep temporary alias for migration:
ExtractionResult = DocumentExtractionResult  # deprecate
```

### Patch E — Lazy package exports (P1)

```python
# memgraphrag_core/__init__.py
"""MemGraphRAG core — three-layer global memory + memory-guided PPR retrieval."""

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

# Agents / indexing: import from memgraphrag_core.agents / .indexing explicitly.
```

### Patch F — Guard empty ingest + LLM errors (P1)

```python
# memgraphrag_core/ingestion/pipeline.py  (inside process_document)
if not text or not text.strip():
    report["skipped_empty_document"] = True
    return report

for passage in passages:
    await self._index_passage(passage)
    report["passages"].append(passage.id)
    try:
        extraction = await self.extractor.extract_from_text(passage.text)
    except Exception as exc:  # noqa: BLE001 — surfaced in report for batch resilience
        report.setdefault("errors", []).append(
            {"passage_id": passage.id, "error": str(exc)}
        )
        continue
    ...
```

---

## 5. Naming & Standards Checklist

| Item | Expected | Actual | Verdict |
|------|----------|--------|---------|
| Vector namespaces | `M_pas`, `M_ont`, `M_fac` | Same + `M_ent` for entities | **OK** (document `M_ent`) |
| Domain relations | `LOCATED_IN`, `PRODUCES` | English UPPER_SNAKE | **OK** |
| Graph structural relations | one vocabulary | `EVIDENCE_IN` vs `SUPPORTED_BY` | **FAIL** |
| `INSTANTIATES` direction | single convention | indexing ≠ ingestion | **FAIL** |
| PEP 8 modules/classes | snake/Pascal | Generally consistent | **OK** |
| Spanish leftovers in core | none | none in `memgraphrag_core/` | **OK** |
| Optional chromadb error message | clear | Present | **OK** |
| Optional networkx error message | clear | Hard dependency | **GAP** |

---

## 6. Recommended Fix Order Before v0.1.0 Stable

1. **Unify graph edges** (Patch A) + update tests/examples.  
2. **`asyncio.to_thread` for Chroma + PPR** (Patches B/C).  
3. **Lazy `__init__` exports** (Patch E).  
4. **Rename `ExtractionResult`** (Patch D).  
5. **Ingestion error isolation** (Patch F).  
6. Docs: clarify Agno agents vs `BaseLLMProvider` ingestion paths.

---

## 7. Verification Snapshot (at audit time)

```text
pytest tests/     → 35 passed
python -m benchmarks.benchmark_multihop → runs; R@5 facts/passages = 1.0 on sample set
```

Edge probes: empty ingest / empty PPR / empty retrieve → no crashes.

---

*End of audit report.*
