"""
Regression tests for AMemAgent backend handling.

AgentConfig defaults to the "anthropic" backend, which A-mem's LLMController
does not support. AMemAgent must reject it up front with an actionable error
instead of failing deep inside data/A-mem after loading its dependencies.
A-mem's third-party dependencies are replaced with stub modules, so no models,
NLTK data, network, or API keys are needed.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from evaluation.agents.amem_agent import SUPPORTED_BACKENDS, AMemAgent
from evaluation.agents.base import AgentConfig

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
def isolated_amem(monkeypatch):
    """Fresh, stubbed A-mem imports, undoing all changes afterwards."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {
        name: sys.modules.pop(name) for name in AMEM_MODULES if name in sys.modules
    }
    for name in STUBBED_DEPS:
        monkeypatch.setitem(sys.modules, name, _stub_module(name))
    yield
    for name in AMEM_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_default_config_fails_fast_with_actionable_error(isolated_amem):
    assert AgentConfig().backend not in SUPPORTED_BACKENDS

    with pytest.raises(
        ValueError, match="does not support backend 'anthropic'"
    ) as excinfo:
        AMemAgent(AgentConfig())

    assert "backend='openai'" in str(excinfo.value)
    assert "test_advanced" not in sys.modules


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_supported_backends_reach_amem(isolated_amem, monkeypatch, backend):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    agent = AMemAgent(AgentConfig(model="gpt-4o-mini", backend=backend))

    memory_layer = sys.modules["memory_layer"]
    assert isinstance(agent.memory_system, memory_layer.AgenticMemorySystem)
    assert isinstance(agent._agent.retriever_llm, memory_layer.LLMController)
