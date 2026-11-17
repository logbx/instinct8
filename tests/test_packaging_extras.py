"""
data/A-mem/memory_layer.py imports rank_bm25 at module level, so the extras
used to run A-MEM (Strategy D/I, AMemAgent) must declare rank-bm25. Otherwise
Strategy D silently falls back to non-A-MEM compression.
"""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.mark.parametrize("extra", ["eval", "all"])
def test_amem_extras_declare_rank_bm25(extra):
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]

    assert any(req.startswith("rank-bm25") for req in extras[extra])
