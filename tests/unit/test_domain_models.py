"""Unit tests for domain models and GlobalMemoryState invariants."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from memgraphrag_core.domain.models import (
    ConflictType,
    Fact,
    GlobalMemoryState,
    Passage,
    Schema,
)


def test_schema_key_and_stability() -> None:
    schema = Schema(head_type="Winery", relation="PRODUCES", tail_type="Wine", frequency=1)
    assert schema.schema_key == "(winery)-[produces]->(wine)"
    assert not schema.is_stable(tau=2)
    schema.frequency = 2
    assert schema.is_stable(tau=2)


def test_fact_keys_are_normalized() -> None:
    fact = Fact(
        head_entity=" Catena Zapata ",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity=" Mendoza ",
        tail_type="Region",
    )
    assert fact.fact_key == "catena zapata--located_in--mendoza"
    assert fact.schema_key == "(winery)-[located_in]->(region)"


def test_passage_defaults() -> None:
    p = Passage(id="pas_001", text="hello")
    assert p.metadata == {}
    assert isinstance(p.created_at, datetime)
    assert p.created_at.tzinfo == timezone.utc


def test_conflict_type_enum_values() -> None:
    assert ConflictType.LOGICAL_CONTRADICTION.value == "logical_contradiction"
    assert ConflictType.REDUNDANCY in ConflictType


def test_fact_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Fact(
            head_entity="A",
            head_type="Winery",
            relation="PRODUCES",
            tail_entity="B",
            tail_type="Wine",
            confidence=1.5,
        )


def test_tau_filters_unstable_schema() -> None:
    memory = GlobalMemoryState(tau=2)
    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    assert memory.get_stable_schemas() == []

    memory.register_schema_candidate("Winery", "PRODUCES", "Wine")
    stable = memory.get_stable_schemas()
    assert len(stable) == 1
    assert stable[0].frequency == 2


def test_add_fact_rejects_unstable_schema() -> None:
    memory = GlobalMemoryState(tau=2)
    memory.register_schema_candidate("Winery", "PRODUCES", "Vermouth")  # freq=1

    noisy = Fact(
        head_entity="X",
        head_type="Winery",
        relation="PRODUCES",
        tail_entity="Y",
        tail_type="Vermouth",
        passage_ids=["p1"],
    )
    assert memory.add_fact(noisy) is None
    assert memory.facts == {}


def test_add_fact_requires_registered_schema() -> None:
    memory = GlobalMemoryState(tau=1)
    fact = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
    )
    # Schema never registered → reject even with tau=1
    assert memory.add_fact(fact) is None


def test_add_fact_merges_passage_ids() -> None:
    memory = GlobalMemoryState(tau=2)
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")
    memory.register_schema_candidate("Winery", "LOCATED_IN", "Region")

    f1 = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
        passage_ids=["pas_001"],
        confidence=0.8,
    )
    f2 = Fact(
        head_entity="Catena Zapata",
        head_type="Winery",
        relation="LOCATED_IN",
        tail_entity="Mendoza",
        tail_type="Region",
        passage_ids=["pas_002"],
        confidence=0.9,
    )
    assert memory.add_fact(f1) is not None
    merged = memory.add_fact(f2)
    assert merged is not None
    assert set(merged.passage_ids) == {"pas_001", "pas_002"}
    assert merged.confidence == 0.9


def test_add_passage() -> None:
    memory = GlobalMemoryState(tau=2)
    p = Passage(id="pas_001", text="hello")
    memory.add_passage(p)
    assert memory.passages["pas_001"].text == "hello"
