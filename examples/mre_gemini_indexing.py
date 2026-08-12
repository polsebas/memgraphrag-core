#!/usr/bin/env python3
"""Live smoke test: Agno indexing agents with Google Gemini.

Requires:
  - ``GOOGLE_API_KEY`` in the environment
  - ``pip install google-genai`` (Agno Gemini backend)

Usage:
  export GOOGLE_API_KEY=...
  python examples/mre_gemini_indexing.py
  # optional: MEMGRAPHRAG_GEMINI_MODEL=gemini-3-flash-preview
"""

from __future__ import annotations

import asyncio
import os
import sys

from agno.models.google import Gemini

from memgraphrag_core.domain.models import GlobalMemoryState, Passage
from memgraphrag_core.indexing.orchestrator import IndexingOrchestrator

# New Gemini API keys typically need 3.x models (2.0 / 2.5 may return 404).
DEFAULT_MODEL = "gemini-3.5-flash"


def _require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "GOOGLE_API_KEY is not set. Export it before running this example.",
            file=sys.stderr,
        )
        sys.exit(1)


async def main() -> None:
    _require_api_key()
    model_id = os.getenv("MEMGRAPHRAG_GEMINI_MODEL", DEFAULT_MODEL)

    memory = GlobalMemoryState(tau=1)
    orch = IndexingOrchestrator(
        memory,
        model=Gemini(id=model_id),
        run_conflict_pipeline=False,
    )

    print("=" * 65)
    print(f"GEMINI INDEXING SMOKE TEST (model={model_id})")
    print("=" * 65)

    passages = [
        Passage(
            id="pas_gemini_001",
            text=(
                "Catena Zapata winery is located in Mendoza and produces Gran Enemigo."
            ),
        ),
        Passage(
            id="pas_gemini_002",
            text="Gran Enemigo is a Malbec-blend wine from the Uco Valley.",
        ),
    ]

    report = await orch.index_passages(passages)

    print(f"Passages indexed: {report.passages_indexed}")
    print(f"Schema registrations (increments): {report.schemas_registered}")
    print(f"Facts accepted: {report.facts_accepted}")
    print(f"Facts rejected by tau: {report.facts_rejected_by_tau}")
    print(f"Stable schemas: {len(memory.get_stable_schemas())}")
    print(f"Final facts in M_fac: {len(memory.facts)}")
    for f in memory.facts.values():
        print(
            f"   • {f.head_entity} -[{f.relation}]-> {f.tail_entity} "
            f"(conf={f.confidence}, passages={f.passage_ids})"
        )
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
