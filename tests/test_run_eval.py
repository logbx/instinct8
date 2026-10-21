"""Offline regression tests for scripts/run_eval.py (no API keys, no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

from evaluation.agents.base import AgentConfig

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_eval.py"


@pytest.fixture
def run_eval(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_eval_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubAMemAgent:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(module, "AMemAgent", StubAMemAgent)
    return module


@pytest.mark.parametrize("strategy", ["amem", "recency", "first_last"])
def test_create_agent_builds_valid_agent_config(run_eval, strategy):
    agent = run_eval.create_agent(strategy, model="gpt-4o-mini")

    assert isinstance(agent.config, AgentConfig)
    assert agent.config.model == "gpt-4o-mini"


def test_locomo_path_points_at_bundled_dataset(run_eval):
    assert run_eval.LOCOMO_PATH.is_file()
    assert json.loads(run_eval.LOCOMO_PATH.read_text())


def test_run_evaluation_finds_dataset_from_any_cwd(run_eval, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loaded_paths = []

    class StubDataset:
        def __init__(self, path, ratio=1.0):
            loaded_paths.append(Path(path))

        def __iter__(self):
            return iter([object()])

    class StubResults:
        def to_dict(self):
            return {}

    class StubHarness:
        def __init__(self, agent, dataset):
            self.agent = agent

        def run_evaluation(self, **kwargs):
            return StubResults()

    monkeypatch.setattr(run_eval, "LoCoMoDataset", StubDataset)
    monkeypatch.setattr(run_eval, "UnifiedHarness", StubHarness)

    output = run_eval.run_evaluation(samples=1, output_dir=str(tmp_path / "out"), verbose=False)

    assert output is not None
    assert loaded_paths and all(p.is_file() for p in loaded_paths)
    _, filepath = output
    assert json.loads(Path(filepath).read_text())["metadata"]["strategy"] == "amem"
