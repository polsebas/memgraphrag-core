"""Entrypoint so `pytest test_paper_compliance.py` and `python test_paper_compliance.py` work.

The canonical suite lives in ``tests/test_paper_compliance.py``.
"""

from __future__ import annotations

import sys

import pytest

from tests.test_paper_compliance import (  # noqa: F401
    TestAxis1ThreeLayerMemory,
    TestAxis2OntologicalStability,
    TestAxis3HierarchicalGraph,
    TestAxis4PersonalizedPageRank,
    TestAxis5HybridReranking,
)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
