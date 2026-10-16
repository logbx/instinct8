"""
Regression tests for locating the vendored A-mem code at data/A-mem.

AMemAgent, LoCoMoDataset and StrategyD_AMemStyle lazily import A-mem modules
(test_advanced, load_dataset, memory_layer) by bare name. These tests check
that those imports resolve to the files under data/A-mem regardless of the
working directory. A-mem's third-party dependencies are replaced with stub
modules, so no models, NLTK data, network, or API keys are needed.
"""

import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.agents import amem_agent
from evaluation.agents.base import AgentConfig
from evaluation.datasets import locomo_dataset
from strategies import strategy_d_amem

REPO_ROOT = Path(__file__).resolve().parent.parent
AMEM_DIR = REPO_ROOT / "data" / "A-mem"
LOCOMO_JSON = AMEM_DIR / "data" / "locomo10.json"

AMEM_MODULES = ("load_dataset", "memory_layer", "test_advanced", "utils")

# Third-party imports made at module level by data/A-mem/{memory_layer,utils,test_advanced}.py
STUBBED_DEPS = (
    "bert_score",
    "litellm",
    "nltk",
    "nltk.tokenize",
    "nltk.translate",
    "nltk.translate.bleu_score",
    "nltk.translate.meteor_score",
    "openai",
    "rank_bm25",
    "requests",
    "rouge_score",
    "sentence_transformers",
    "sentence_transformers.util",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "tqdm",
    "transformers",
)


def _stub_module(name):
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: MagicMock(name=f"{name}.{attr}")
    return module


@pytest.fixture
def clean_amem_imports(monkeypatch, tmp_path):
    """Run from an unrelated cwd with fresh A-mem imports, undoing all changes afterwards."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: sys.modules.pop(name) for name in AMEM_MODULES if name in sys.modules}
    yield
    for name in AMEM_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def stub_amem_deps(monkeypatch, clean_amem_imports):
    for name in STUBBED_DEPS:
        monkeypatch.setitem(sys.modules, name, _stub_module(name))


def assert_loaded_from_amem(*names):
    for name in names:
        assert Path(sys.modules[name].__file__) == AMEM_DIR / f"{name}.py"


@pytest.mark.parametrize(
    "module,entry_file",
    [
        (amem_agent, "test_advanced.py"),
        (locomo_dataset, "load_dataset.py"),
        (strategy_d_amem, "memory_layer.py"),
    ],
    ids=["amem_agent", "locomo_dataset", "strategy_d_amem"],
)
def test_amem_path_points_at_vendored_code(module, entry_file):
    assert module._amem_path == AMEM_DIR
    assert module._amem_path.is_absolute()
    assert (module._amem_path / entry_file).is_file()


def test_importing_modules_does_not_put_amem_on_sys_path(tmp_path):
    # data/A-mem contains a generic utils.py, so it must only be added when
    # A-mem is actually loaded, never as a side effect of importing a package.
    code = (
        "import sys\n"
        "import evaluation.agents.amem_agent, evaluation.datasets.locomo_dataset\n"
        "import strategies.strategy_d_amem\n"
        f"assert {str(AMEM_DIR)!r} not in sys.path, sys.path\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_locomo_dataset_loads_from_amem(clean_amem_imports):
    dataset = locomo_dataset.LoCoMoDataset(LOCOMO_JSON, ratio=0.1)

    assert_loaded_from_amem("load_dataset")
    assert sys.path[-1] == str(AMEM_DIR)
    assert len(dataset) == 1
    assert dataset[0].turns and dataset[0].questions


def test_strategy_d_uses_real_memory_system(stub_amem_deps):
    strategy = strategy_d_amem.StrategyD_AMemStyle(llm_backend="sglang")

    assert_loaded_from_amem("memory_layer")
    assert sys.path[-1] == str(AMEM_DIR)
    assert isinstance(strategy._memory_system, sys.modules["memory_layer"].AgenticMemorySystem)


def test_strategy_i_gets_real_memory_system(stub_amem_deps):
    from strategies.strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore

    strategy = StrategyI_AMemProtectedCore(backend="openai", llm_backend="sglang")

    memory_layer = sys.modules["memory_layer"]
    assert isinstance(strategy._amem_strategy._memory_system, memory_layer.AgenticMemorySystem)


def test_amem_agent_loads_test_advanced_and_siblings(stub_amem_deps):
    agent = amem_agent.AMemAgent(AgentConfig(model="m", backend="sglang"))

    assert_loaded_from_amem(*AMEM_MODULES)
    assert sys.path[-1] == str(AMEM_DIR)
    assert isinstance(agent._agent, sys.modules["test_advanced"].advancedMemAgent)
    assert isinstance(agent.memory_system, sys.modules["memory_layer"].AgenticMemorySystem)
