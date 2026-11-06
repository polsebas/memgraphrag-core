"""Locks the deterministic MemoryTax benchmark numbers."""

from __future__ import annotations

import pytest

from benchmarks.verify_memory_tax import evaluate, load_scenario, main, render_markdown


@pytest.mark.asyncio
async def test_deterministic_benchmark_separates_summary_from_three_layer_memory() -> None:
    rows = await evaluate(load_scenario(), mode="deterministic")
    by_engine = {row.engine: row for row in rows}
    baseline = by_engine["BaselineSummarizerMemory"]
    core = by_engine["MemGraphRAGCore"]
    history = by_engine["FullRawHistory"]

    assert baseline.process_recall == 0.0
    assert baseline.constraint_violation == 100.0
    assert core.process_recall == 100.0
    assert core.constraint_violation == 0.0
    assert history.process_recall == 100.0
    assert history.constraint_violation == 0.0
    assert core.tokens < history.tokens
    assert "MemGraphRAGCore" in render_markdown(rows)


def test_cli_default_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mode", "deterministic"]) == 0
    captured = capsys.readouterr().out
    assert "BaselineSummarizerMemory" in captured
    assert "MemGraphRAGCore" in captured
