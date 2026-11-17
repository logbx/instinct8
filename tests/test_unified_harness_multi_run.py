"""Offline tests for UnifiedHarness multi-run evaluation (n_runs > 1)."""

from typing import Any, Dict, Iterator, List, Optional

import pytest

from evaluation.agents.base import BaseAgent
from evaluation.datasets.base import BaseDataset, EvalQuestion, EvalSample, EvalTurn
from evaluation.metric_interfaces import MetricResult, MetricType
from evaluation.unified_harness import EvaluationResults, UnifiedHarness


class FakeAgent(BaseAgent):
    """Answers correctly on odd-numbered resets so metrics vary across runs."""

    def __init__(self):
        self.resets = 0
        self.turns: List[Dict[str, Any]] = []

    def ingest_turn(self, turn: Dict[str, Any]) -> None:
        self.turns.append(turn)

    def answer_question(
        self,
        question: str,
        category: Optional[int] = None,
        reference_answer: Optional[str] = None,
    ) -> str:
        return "Paris" if self.resets % 2 == 1 else "London"

    def get_context_size(self) -> int:
        return len(self.turns)

    def compress(self, trigger_point: Optional[int] = None) -> None:
        pass

    def reset(self) -> None:
        self.resets += 1
        self.turns = []

    @property
    def name(self) -> str:
        return "fake-agent"


class FakeQADataset(BaseDataset):
    def __init__(self, n_samples: int = 2):
        self._samples = [
            EvalSample(
                sample_id=f"s{i}",
                turns=[EvalTurn(id=0, role="user", content="The capital of France is Paris.")],
                questions=[EvalQuestion(question="Capital of France?", reference_answer="Paris", category=1)],
            )
            for i in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]

    @property
    def name(self) -> str:
        return "fake-qa"

    @property
    def evaluation_type(self) -> str:
        return "qa"


class ExactMatchCalculator:
    """Stands in for QAMetricCalculator, which needs A-mem utils."""

    def calculate(self, prediction: str, reference: str, category: Optional[int] = None, **kwargs):
        return [
            MetricResult(
                name="exact_match",
                value=float(prediction.strip().lower() == reference.strip().lower()),
                metric_type=MetricType.SCORE_0_1,
                category=str(category) if category is not None else None,
            )
        ]


@pytest.fixture
def harness():
    h = UnifiedHarness(FakeAgent(), FakeQADataset(n_samples=2))
    h._qa_calc = ExactMatchCalculator()
    return h


def test_single_run_skips_multi_run_branch(harness, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("multi-run branch entered for n_runs=1")

    monkeypatch.setattr(harness, "_run_multi_run_evaluation", fail)

    results = harness.run_evaluation(n_runs=1, verbose=False)

    assert results.n_runs == 1
    assert results.statistical_summary == {}
    assert results.num_samples == 2


def test_multi_run_returns_statistical_summary(harness):
    results = harness.run_evaluation(n_runs=3, confidence=0.9, verbose=False)

    assert isinstance(results, EvaluationResults)
    assert results.n_runs == 3
    assert results.confidence == 0.9
    assert results.num_samples == 2
    assert harness.agent.resets == 2 * 3

    stats = results.statistical_summary["exact_match"]
    assert set(stats) == {"mean", "std", "ci_lower", "ci_upper", "n"}
    assert stats["n"] == 2
    # Per-sample means are 2/3 (resets 1-3) and 1/3 (resets 4-6).
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] > 0
    assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]
    assert results.aggregate_metrics["overall"] == results.statistical_summary


def test_multi_run_to_dict_includes_run_metadata(harness):
    data = harness.run_evaluation(n_runs=2, verbose=False).to_dict()

    assert data["n_runs"] == 2
    assert data["confidence"] == 0.95
    assert "exact_match" in data["statistical_summary"]
