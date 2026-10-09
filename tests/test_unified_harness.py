"""
Unit tests for evaluation/unified_harness.py.

Covers the result dataclasses, harness construction (with and without a cache
dir), routing by dataset.evaluation_type, the qa / compression / coding sample
paths, the A-mem cache hit/miss branches, and the multi-run aggregation path.

Everything is offline: the agent and dataset are in-file fakes, the QA metric
calculator is a deterministic stand-in (the real one needs A-mem utils), the
coding calculator runs with embeddings disabled, and the LLM judges in
evaluation.metrics are stubbed. An autouse fixture removes API keys and makes
any attempt to build a real LLM client or embedding model fail loudly.
"""

import json
import sys
import types

import pytest

import evaluation.metrics as metrics_module
from evaluation.agents.base import AgentConfig, BaseAgent
from evaluation.cache_manager import CacheManager
from evaluation.coding_metrics import CodingMetricCalculator
from evaluation.datasets.base import BaseDataset, EvalQuestion, EvalSample, EvalTurn
from evaluation.metric_interfaces import MetricResult, MetricType
from evaluation.qa_metrics import QAMetricCalculator
from evaluation.unified_harness import (
    CodingResult,
    EvaluationResults,
    QAResult,
    SampleResult,
    UnifiedHarness,
)


GOAL = "Build a REST API with FastAPI"
CONSTRAINTS = ["Use FastAPI", "JWT auth"]
DEFAULT_GOAL_PROBE = "What is your current goal?"
DEFAULT_CONSTRAINT_PROBE = "What constraints apply?"
DEFAULT_BEHAVIOR_PROBE = "What should we do next?"
ADD_SRC = "def add(a, b):\n    return a + b\n"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent(BaseAgent):
    """Records every harness call; context grows per turn and shrinks on compress."""

    def __init__(self, name="fake-agent", tokens_per_turn=100, answer_fn=None):
        self._name = name
        self.tokens_per_turn = tokens_per_turn
        self.answer_fn = answer_fn or (lambda agent, q: f"answer to {q}")
        self.events = []
        self.ingested = []
        self.asked = []
        self.goals = []
        self.reset_count = 0
        self.tokens = 0
        self.compressed = False

    @property
    def name(self):
        return self._name

    def ingest_turn(self, turn):
        self.events.append(("ingest", turn["id"]))
        self.ingested.append(turn)
        self.tokens += self.tokens_per_turn

    def answer_question(self, question, category=None, reference_answer=None):
        self.events.append(("ask", question))
        self.asked.append((question, category, reference_answer))
        return self.answer_fn(self, question)

    def get_context_size(self):
        return self.tokens

    def compress(self, trigger_point=None):
        self.events.append(("compress", trigger_point))
        self.tokens //= 4
        self.compressed = True

    def reset(self):
        self.events.append(("reset",))
        self.reset_count += 1
        self.ingested = []
        self.tokens = 0
        self.compressed = False

    def initialize_goal(self, goal, constraints):
        self.events.append(("goal", goal))
        self.goals.append((goal, list(constraints)))


class FakeCodingAgent(FakeAgent):
    def __init__(self, generated=None, **kwargs):
        super().__init__(**kwargs)
        self.generated = {"main.py": ADD_SRC} if generated is None else generated
        self.initial_files = []

    def set_initial_files(self, files):
        self.initial_files.append(files)

    def get_generated_code(self):
        return dict(self.generated)


class FakeRetriever:
    def __init__(self):
        self.saved_to = None
        self.loaded_from = None

    def save(self, retriever_path, embeddings_path):
        self.saved_to = (retriever_path, embeddings_path)
        with open(retriever_path, "wb") as f:
            f.write(b"retriever")

    def load(self, retriever_path, embeddings_path):
        self.loaded_from = (retriever_path, embeddings_path)
        return self


class FakeMemorySystem:
    def __init__(self):
        self.memories = []
        self.retriever = FakeRetriever()


class FakeMemoryAgent(FakeAgent):
    """A-mem-shaped agent: exposes memory_system so the harness caches it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_system = FakeMemorySystem()

    def ingest_turn(self, turn):
        super().ingest_turn(turn)
        self.memory_system.memories.append(turn["content"])

    def get_context_size(self):
        return len(self.memory_system.memories)


class FakeDataset(BaseDataset):
    def __init__(self, samples, evaluation_type="qa", name="fake-dataset"):
        self._samples = list(samples)
        self._evaluation_type = evaluation_type
        self._name = name

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    @property
    def name(self):
        return self._name

    @property
    def evaluation_type(self):
        return self._evaluation_type


class FakeQACalculator:
    """Exact-match + token-F1 only, mirroring QAMetricCalculator's MetricResult shape."""

    def __init__(self):
        self.calls = []

    def calculate(self, prediction, reference, category=None, **kwargs):
        self.calls.append((prediction, reference, category))
        cat = str(category) if category is not None else None
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        common = pred_tokens & ref_tokens
        if common:
            p = len(common) / len(pred_tokens)
            r = len(common) / len(ref_tokens)
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        return [
            MetricResult(
                "exact_match",
                float(prediction.strip().lower() == reference.strip().lower()),
                MetricType.SCORE_0_1,
                category=cat,
            ),
            MetricResult("f1", f1, MetricType.SCORE_0_1, category=cat),
        ]


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No keys, no LLM client, no embedding model; judges are deterministic."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _no_client():
        raise AssertionError("LLM client must not be created in unit tests")

    def _no_embeddings():
        raise AssertionError("embedding model must not be loaded in unit tests")

    monkeypatch.setattr(metrics_module, "_get_client", _no_client)
    monkeypatch.setattr(metrics_module, "_get_embedding_model", _no_embeddings)

    judge_calls = []

    def goal_coherence(original_goal, stated_goal, model="gpt-4o"):
        if not original_goal:
            return 0.0
        return 1.0 if original_goal in stated_goal else 0.25

    def constraint_recall(known_constraints, stated_constraints, model="gpt-4o"):
        if not known_constraints:
            return 1.0
        hits = sum(1 for c in known_constraints if c in stated_constraints)
        return hits / len(known_constraints)

    def behavioral_alignment(
        original_goal, constraints, agent_response, test_context="", model="gpt-4o"
    ):
        judge_calls.append(
            {
                "goal": original_goal,
                "constraints": list(constraints),
                "response": agent_response,
                "test_context": test_context,
            }
        )
        return 4

    monkeypatch.setattr(metrics_module, "measure_goal_coherence", goal_coherence)
    monkeypatch.setattr(metrics_module, "measure_constraint_recall", constraint_recall)
    monkeypatch.setattr(
        metrics_module, "measure_behavioral_alignment", behavioral_alignment
    )
    return judge_calls


def make_harness(agent, dataset, cache_dir=None):
    harness = UnifiedHarness(agent, dataset, cache_dir=cache_dir)
    harness._qa_calc = FakeQACalculator()
    harness._coding_calc = CodingMetricCalculator(use_embeddings=False)
    return harness


def make_turns(n, compression_at=()):
    return [
        EvalTurn(
            id=i,
            role="user" if i % 2 else "assistant",
            content=f"turn {i}",
            speaker="A" if i % 2 else "B",
            is_compression_point=i in compression_at,
        )
        for i in range(1, n + 1)
    ]


def qa_sample(sample_id="qa-1", questions=None, n_turns=3):
    if questions is None:
        questions = [
            EvalQuestion("Where is the meeting?", "Paris", category=1),
            EvalQuestion("Who hosts it?", "Alice Smith", category=2),
        ]
    return EvalSample(sample_id=sample_id, turns=make_turns(n_turns), questions=questions)


def compression_sample(
    sample_id="cmp-1",
    n_turns=4,
    compression_at=(2,),
    probing_tasks=None,
    goal=GOAL,
    constraints=CONSTRAINTS,
):
    return EvalSample(
        sample_id=sample_id,
        turns=make_turns(n_turns, compression_at),
        questions=[],
        initial_goal=goal,
        constraints=constraints,
        probing_tasks=probing_tasks,
    )


def coding_sample(
    sample_id="code-1",
    n_turns=3,
    compression_at=(2,),
    metadata=None,
    ground_truth=None,
    probing_tasks=None,
):
    return EvalSample(
        sample_id=sample_id,
        turns=make_turns(n_turns, compression_at),
        questions=[],
        metadata={} if metadata is None else metadata,
        initial_goal=GOAL,
        constraints=CONSTRAINTS,
        probing_tasks=probing_tasks,
        ground_truth={"expected_code": ADD_SRC} if ground_truth is None else ground_truth,
    )


def drifting_answers(agent, question):
    """States goal and constraints until compressed, then forgets both."""
    if agent.compressed:
        return "I am writing documentation"
    if "constraint" in question.lower():
        return "Use FastAPI; JWT auth"
    return f"My goal: {GOAL}"


def echo_reference(agent, question):
    return {"Where is the meeting?": "Paris", "Who hosts it?": "Bob"}.get(question, "")


@pytest.fixture
def stub_multi_run_module(monkeypatch):
    """Stand-in for the missing evaluation.multi_run_evaluator (names are imported, unused)."""
    module = types.ModuleType("evaluation.multi_run_evaluator")
    module.MultiRunEvaluator = type("MultiRunEvaluator", (), {})
    module.MultiRunResult = type("MultiRunResult", (), {})
    monkeypatch.setitem(sys.modules, "evaluation.multi_run_evaluator", module)
    return module


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestQAResult:
    def test_to_dict_shape(self):
        r = QAResult("q?", "pred", "ref", 3, {"f1": 0.5})
        assert r.to_dict() == {
            "question": "q?",
            "prediction": "pred",
            "reference": "ref",
            "category": 3,
            "metrics": {"f1": 0.5},
        }

    def test_category_may_be_none(self):
        assert QAResult("q", "p", "r", None, {}).to_dict()["category"] is None


class TestCodingResult:
    def test_to_dict_shape_and_default_compression_points(self):
        r = CodingResult("bug_fixing", {"a.py": "x = 1"}, {"syntax_validity": 1.0})
        assert r.to_dict() == {
            "task_type": "bug_fixing",
            "generated_code": {"a.py": "x = 1"},
            "metrics": {"syntax_validity": 1.0},
            "compression_points": [],
        }

    def test_default_compression_points_not_shared(self):
        a = CodingResult("t", {}, {})
        b = CodingResult("t", {}, {})
        a.compression_points.append({"id": 1})
        assert b.compression_points == []


class TestSampleResult:
    def test_minimal_to_dict_defaults_and_no_coding_result_key(self):
        d = SampleResult("s1", "agent", "qa").to_dict()
        assert set(d) == {
            "sample_id",
            "agent_name",
            "evaluation_type",
            "qa_results",
            "aggregate_qa_metrics",
            "compression_points",
            "compression_summary",
            "coding_metrics",
            "context_sizes",
            "timestamp",
        }
        assert d["qa_results"] == []
        assert d["aggregate_qa_metrics"] == {}
        assert d["compression_points"] == []
        assert d["compression_summary"] == {}
        assert d["coding_metrics"] == {}
        assert d["context_sizes"] == []
        assert isinstance(d["timestamp"], str) and "T" in d["timestamp"]

    def test_nested_results_are_serialized(self):
        qa = QAResult("q", "p", "r", 1, {"f1": 1.0})
        coding = CodingResult("code_generation", {"m.py": "pass"}, {"x": 1.0})
        d = SampleResult(
            "s1", "agent", "coding", qa_results=[qa], coding_result=coding
        ).to_dict()
        assert d["qa_results"] == [qa.to_dict()]
        assert d["coding_result"] == coding.to_dict()


class TestEvaluationResults:
    def _results(self, **kwargs):
        sample = SampleResult("s1", "agent", "qa", context_sizes=[3])
        return EvaluationResults(
            agent_name="agent",
            dataset_name="ds",
            evaluation_type="qa",
            num_samples=1,
            sample_results=[sample],
            aggregate_metrics={"overall": {}},
            **kwargs,
        )

    def test_single_run_to_dict_omits_multi_run_fields(self):
        d = self._results().to_dict()
        assert set(d) == {
            "agent_name",
            "dataset_name",
            "evaluation_type",
            "num_samples",
            "sample_results",
            "aggregate_metrics",
            "timestamp",
        }
        assert d["sample_results"][0]["sample_id"] == "s1"

    def test_multi_run_to_dict_includes_statistics(self):
        summary = {"f1": {"mean": 0.5, "std": 0.1, "ci_lower": 0.4, "ci_upper": 0.6, "n": 2}}
        d = self._results(n_runs=3, confidence=0.9, statistical_summary=summary).to_dict()
        assert d["n_runs"] == 3
        assert d["confidence"] == 0.9
        assert d["statistical_summary"] == summary

    def test_save_creates_parent_dirs_and_round_trips(self, tmp_path):
        results = self._results()
        path = tmp_path / "nested" / "out" / "results.json"
        results.save(str(path))
        assert json.loads(path.read_text()) == results.to_dict()

    @pytest.mark.xfail(
        strict=True,
        raises=FileNotFoundError,
        reason="save() calls os.makedirs('') when the path has no directory component",
    )
    def test_save_to_bare_filename_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._results().save("results.json")
        assert (tmp_path / "results.json").exists()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_without_cache_dir(self):
        agent, dataset = FakeAgent(), FakeDataset([])
        harness = UnifiedHarness(agent, dataset)
        assert harness.agent is agent
        assert harness.dataset is dataset
        assert harness.cache_dir is None
        assert harness._cache is None
        assert isinstance(harness._qa_calc, QAMetricCalculator)
        assert isinstance(harness._coding_calc, CodingMetricCalculator)

    def test_cache_dir_without_agent_config_uses_agent_name_only(self, tmp_path):
        harness = UnifiedHarness(FakeAgent(name="amem"), FakeDataset([]), str(tmp_path))
        assert isinstance(harness._cache, CacheManager)
        assert harness._cache.cache_dir == tmp_path / "amem"
        assert harness._cache.cache_dir.is_dir()

    def test_cache_dir_uses_agent_config_backend(self, tmp_path):
        agent = FakeAgent(name="amem")
        agent._config = AgentConfig(backend="openai")
        harness = UnifiedHarness(agent, FakeDataset([]), str(tmp_path))
        assert harness._cache.cache_dir == tmp_path / "amem_openai"


# ---------------------------------------------------------------------------
# run_evaluation: routing, sample limits, reset, result envelope
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.parametrize(
        "evaluation_type, expected",
        [
            ("qa", "_evaluate_qa_sample"),
            ("coding", "_evaluate_coding_sample"),
            ("compression", "_evaluate_compression_sample"),
            ("templates", "_evaluate_compression_sample"),
        ],
    )
    def test_dispatch_by_evaluation_type(self, monkeypatch, evaluation_type, expected):
        samples = [qa_sample("a"), qa_sample("b")]
        harness = make_harness(FakeAgent(), FakeDataset(samples, evaluation_type))
        calls = []

        def recorder(method):
            def _record(sample, aggregator, verbose):
                calls.append((method, sample.sample_id, aggregator, verbose))
                return SampleResult(sample.sample_id, "fake-agent", method)

            return _record

        for method in (
            "_evaluate_qa_sample",
            "_evaluate_coding_sample",
            "_evaluate_compression_sample",
        ):
            monkeypatch.setattr(harness, method, recorder(method))

        results = harness.run_evaluation(verbose=False)

        assert [(c[0], c[1]) for c in calls] == [(expected, "a"), (expected, "b")]
        assert calls[0][2] is calls[1][2], "one aggregator is shared across samples"
        assert all(c[3] is False for c in calls)
        assert results.evaluation_type == evaluation_type


class TestRunEvaluation:
    def _three_samples(self):
        return [qa_sample("s1"), qa_sample("s2"), qa_sample("s3")]

    def test_num_samples_trims_in_dataset_order(self):
        harness = make_harness(FakeAgent(), FakeDataset(self._three_samples()))
        results = harness.run_evaluation(num_samples=2, verbose=False)
        assert results.num_samples == 2
        assert [r.sample_id for r in results.sample_results] == ["s1", "s2"]

    @pytest.mark.parametrize("num_samples", [None, 0, 10])
    def test_no_effective_limit_evaluates_all(self, num_samples):
        # num_samples is checked for truthiness, so 0 means "no limit".
        harness = make_harness(FakeAgent(), FakeDataset(self._three_samples()))
        results = harness.run_evaluation(num_samples=num_samples, verbose=False)
        assert results.num_samples == 3

    def test_reset_after_each_sample(self):
        agent = FakeAgent()
        harness = make_harness(agent, FakeDataset(self._three_samples()))
        harness.run_evaluation(verbose=False)

        assert agent.reset_count == 3
        assert agent.events[-1] == ("reset",)
        segments, current = [], []
        for event in agent.events:
            if event == ("reset",):
                segments.append(current)
                current = []
            else:
                current.append(event)
        assert len(segments) == 3
        for segment in segments:
            assert [e for e in segment if e[0] == "ingest"] == [
                ("ingest", 1),
                ("ingest", 2),
                ("ingest", 3),
            ]

    def test_result_envelope(self):
        harness = make_harness(
            FakeAgent(name="agent-x"),
            FakeDataset([qa_sample()], name="locomo-mini"),
        )
        results = harness.run_evaluation(verbose=False)
        assert isinstance(results, EvaluationResults)
        assert results.agent_name == "agent-x"
        assert results.dataset_name == "locomo-mini"
        assert results.evaluation_type == "qa"
        assert results.n_runs == 1
        assert results.statistical_summary == {}
        assert "n_runs" not in results.to_dict()

    def test_empty_dataset(self):
        agent = FakeAgent()
        results = make_harness(agent, FakeDataset([])).run_evaluation(verbose=False)
        assert results.num_samples == 0
        assert results.sample_results == []
        assert results.aggregate_metrics == {"overall": {}}
        assert agent.reset_count == 0

    def test_results_are_json_serializable(self, tmp_path):
        samples = [qa_sample()]
        results = make_harness(FakeAgent(), FakeDataset(samples)).run_evaluation(
            verbose=False
        )
        path = tmp_path / "r.json"
        results.save(str(path))
        assert json.loads(path.read_text())["sample_results"][0]["sample_id"] == "qa-1"


# ---------------------------------------------------------------------------
# QA path
# ---------------------------------------------------------------------------


class TestQAPath:
    def test_ingests_every_turn_as_dict(self):
        agent = FakeAgent()
        sample = qa_sample(n_turns=2)
        make_harness(agent, FakeDataset([sample])).run_evaluation(verbose=False)
        ingest_ids = [e[1] for e in agent.events if e[0] == "ingest"]
        assert ingest_ids == [1, 2]
        assert agent.goals == [], "QA path never initializes a goal"

    def test_turn_payload_is_full_eval_turn_dict(self):
        captured = []

        class CapturingAgent(FakeAgent):
            def ingest_turn(self, turn):
                captured.append(dict(turn))
                super().ingest_turn(turn)

        sample = qa_sample(n_turns=1)
        make_harness(CapturingAgent(), FakeDataset([sample])).run_evaluation(
            verbose=False
        )
        assert captured == [sample.turns[0].__dict__]

    def test_agent_receives_adversarial_or_reference_answer(self):
        # The gold answer is forwarded to the agent for every category, matching
        # A-mem's answer_question(question, category, answer) contract.
        questions = [
            EvalQuestion("Q1", "ref-1", category=1),
            EvalQuestion("Q5", "ref-5", category=5, adversarial_answer="adv-5"),
        ]
        agent = FakeAgent()
        make_harness(agent, FakeDataset([qa_sample(questions=questions)])).run_evaluation(
            verbose=False
        )
        assert agent.asked == [("Q1", 1, "ref-1"), ("Q5", 5, "adv-5")]

    def test_metrics_scored_against_final_answer(self):
        questions = [
            EvalQuestion("Q1", "ref-1", category=1),
            EvalQuestion("Q4", "ref-4", category=4, adversarial_answer="adv-4"),
            EvalQuestion("Q5", "ref-5", category=5, adversarial_answer="adv-5"),
        ]
        harness = make_harness(
            FakeAgent(answer_fn=lambda a, q: f"pred {q}"),
            FakeDataset([qa_sample(questions=questions)]),
        )
        harness.run_evaluation(verbose=False)
        assert harness._qa_calc.calls == [
            ("pred Q1", "ref-1", 1),
            ("pred Q4", "ref-4", 4),
            ("pred Q5", "adv-5", 5),
        ]

    def test_sample_result_contents(self):
        harness = make_harness(
            FakeAgent(answer_fn=echo_reference), FakeDataset([qa_sample(n_turns=3)])
        )
        result = harness.run_evaluation(verbose=False).sample_results[0]

        assert result.evaluation_type == "qa"
        assert result.agent_name == "fake-agent"
        assert [r.to_dict() for r in result.qa_results] == [
            {
                "question": "Where is the meeting?",
                "prediction": "Paris",
                "reference": "Paris",
                "category": 1,
                "metrics": {"exact_match": 1.0, "f1": 1.0},
            },
            {
                "question": "Who hosts it?",
                "prediction": "Bob",
                "reference": "Alice Smith",
                "category": 2,
                "metrics": {"exact_match": 0.0, "f1": 0.0},
            },
        ]
        assert result.context_sizes == [300]
        assert result.compression_points == []
        assert result.coding_result is None

    def test_per_sample_aggregate_by_category(self):
        harness = make_harness(
            FakeAgent(answer_fn=echo_reference), FakeDataset([qa_sample()])
        )
        agg = harness.run_evaluation(verbose=False).sample_results[0].aggregate_qa_metrics
        assert agg["overall"]["exact_match"]["mean"] == pytest.approx(0.5)
        assert agg["overall"]["exact_match"]["count"] == 2
        assert set(agg["by_category"]) == {"category_1", "category_2"}
        assert agg["by_category"]["category_1"]["exact_match"]["mean"] == 1.0
        assert agg["by_category"]["category_2"]["exact_match"]["mean"] == 0.0
        assert agg["qa_summary"] == {"accuracy": 0.5, "f1_score": 0.5}

    def test_uncategorized_questions_have_no_category_breakdown(self):
        questions = [EvalQuestion("Q", "A")]
        harness = make_harness(
            FakeAgent(answer_fn=lambda a, q: "A"),
            FakeDataset([qa_sample(questions=questions)]),
        )
        results = harness.run_evaluation(verbose=False)
        assert "by_category" not in results.sample_results[0].aggregate_qa_metrics
        assert "by_category" not in results.aggregate_metrics

    def test_run_aggregate_spans_all_samples(self):
        samples = [qa_sample("s1"), qa_sample("s2")]
        harness = make_harness(FakeAgent(answer_fn=echo_reference), FakeDataset(samples))
        agg = harness.run_evaluation(verbose=False).aggregate_metrics
        assert agg["overall"]["exact_match"]["count"] == 4
        assert agg["qa_summary"]["accuracy"] == pytest.approx(0.5)
        assert agg["by_category"]["category_1"]["exact_match"]["count"] == 2
        assert "compression_summary" not in agg


# ---------------------------------------------------------------------------
# Compression path
# ---------------------------------------------------------------------------


class TestCompressionPath:
    def _run(self, sample, agent=None, verbose=False):
        agent = agent or FakeAgent(answer_fn=drifting_answers)
        harness = make_harness(agent, FakeDataset([sample], "compression"))
        return agent, harness.run_evaluation(verbose=verbose)

    def test_initializes_goal_and_constraints(self):
        agent, _ = self._run(compression_sample())
        assert agent.goals == [(GOAL, CONSTRAINTS)]
        assert agent.events[0] == ("goal", GOAL)

    def test_skips_goal_initialization_without_initial_goal(self):
        agent, _ = self._run(compression_sample(goal=None, constraints=None))
        assert agent.goals == []

    def test_default_probe_sequence_around_compress(self):
        agent, _ = self._run(compression_sample(n_turns=3, compression_at=(2,)))
        assert agent.events == [
            ("goal", GOAL),
            ("ingest", 1),
            ("ingest", 2),
            ("ask", DEFAULT_GOAL_PROBE),
            ("ask", DEFAULT_CONSTRAINT_PROBE),
            ("compress", 2),
            ("ask", DEFAULT_GOAL_PROBE),
            ("ask", DEFAULT_CONSTRAINT_PROBE),
            ("ask", DEFAULT_BEHAVIOR_PROBE),
            ("ingest", 3),
            ("reset",),
        ]

    def test_custom_probing_tasks(self, offline):
        probing = {
            "goal_probe": "G?",
            "constraint_probe": "C constraint?",
            "behavioral_test": {"prompt": "B?"},
        }
        agent, _ = self._run(compression_sample(n_turns=2, probing_tasks=probing))
        asked = [e[1] for e in agent.events if e[0] == "ask"]
        assert asked == ["G?", "C constraint?", "G?", "C constraint?", "B?"]
        assert offline == [
            {
                "goal": GOAL,
                "constraints": CONSTRAINTS,
                "response": "I am writing documentation",
                "test_context": "B?",
            }
        ]

    def test_compression_point_metrics(self):
        _, results = self._run(compression_sample(n_turns=4, compression_at=(2,)))
        sample = results.sample_results[0]

        assert sample.evaluation_type == "compression"
        assert sample.context_sizes == [100, 200, 150, 250]
        assert len(sample.compression_points) == 1
        cp = sample.compression_points[0]
        assert cp["compression_point_id"] == 1
        assert cp["turn_id"] == 2
        assert cp["metrics_before"] == {
            "goal_coherence": 1.0,
            "constraint_recall": 1.0,
            "behavioral_alignment": 5,
            "tokens": 200,
        }
        assert cp["metrics_after"] == {
            "goal_coherence": 0.25,
            "constraint_recall": 0.0,
            "behavioral_alignment": 4,
            "tokens": 50,
        }
        assert cp["drift"] == {
            "goal_drift": 0.75,
            "constraint_loss": 1.0,
            "drift_detected": True,
        }
        assert cp["compression_ratio"] == 0.25

    def test_sample_and_run_summaries(self):
        _, results = self._run(compression_sample(n_turns=4, compression_at=(2,)))
        summary = results.sample_results[0].compression_summary
        assert summary["num_compression_points"] == 1
        assert summary["avg_goal_drift"] == 0.75
        assert summary["drift_events_detected"] == 1
        assert summary["avg_compression_ratio"] == 0.25

        agg = results.aggregate_metrics
        assert set(agg["overall"]) == {
            "goal_coherence_before",
            "goal_coherence_after",
            "goal_drift",
            "constraint_recall_before",
            "constraint_recall_after",
            "constraint_loss",
            "behavioral_alignment_after",
            "compression_ratio",
        }
        assert agg["compression_summary"] == {
            "avg_goal_drift": 0.75,
            "max_goal_drift": 0.75,
            "final_goal_coherence": 0.25,
            "avg_constraint_loss": 1.0,
            "final_constraint_recall": 0.0,
            "avg_behavioral_alignment": 4,
            "avg_compression_ratio": 0.25,
            "drift_events_detected": 1,
        }
        assert "qa_summary" not in agg

    def test_multiple_compression_points_are_numbered(self):
        agent, results = self._run(compression_sample(n_turns=4, compression_at=(2, 4)))
        cps = results.sample_results[0].compression_points
        assert [(cp["compression_point_id"], cp["turn_id"]) for cp in cps] == [(1, 2), (2, 4)]
        assert [e for e in agent.events if e[0] == "compress"] == [
            ("compress", 2),
            ("compress", 4),
        ]
        assert results.aggregate_metrics["overall"]["goal_drift"]["count"] == 2

    def test_no_compression_points(self):
        agent, results = self._run(compression_sample(n_turns=3, compression_at=()))
        sample = results.sample_results[0]
        assert sample.compression_points == []
        assert sample.compression_summary["num_compression_points"] == 0
        assert sample.context_sizes == [100, 200, 300]
        assert results.aggregate_metrics == {"overall": {}}
        assert not any(e[0] in ("ask", "compress") for e in agent.events)


# ---------------------------------------------------------------------------
# Coding path
# ---------------------------------------------------------------------------


class TestCodingPath:
    def _run(self, sample, agent=None, verbose=False):
        agent = agent or FakeCodingAgent(answer_fn=drifting_answers)
        harness = make_harness(agent, FakeDataset([sample], "coding"))
        return agent, harness.run_evaluation(verbose=verbose)

    def test_initial_files_forwarded_when_supported(self):
        files = {"app.py": "x = 1\n"}
        agent, _ = self._run(coding_sample(metadata={"initial_files": files}))
        assert agent.initial_files == [files]
        assert agent.goals == [(GOAL, CONSTRAINTS)]

    def test_initial_files_ignored_for_agents_without_hook(self):
        agent = FakeAgent(answer_fn=drifting_answers)
        _, results = self._run(
            coding_sample(metadata={"initial_files": {"app.py": "x = 1\n"}}), agent=agent
        )
        assert results.sample_results[0].coding_result.generated_code == {}

    def test_coding_default_probes(self):
        agent, _ = self._run(coding_sample(n_turns=2, compression_at=(2,)))
        asked = [e[1] for e in agent.events if e[0] == "ask"]
        assert asked == [
            "What is the current coding task you're working on?",
            "What constraints or requirements must the solution meet?",
            "What is the current coding task you're working on?",
            "What constraints or requirements must the solution meet?",
            "What would be your next step to complete this task?",
        ]

    def test_coding_result_and_metrics(self):
        _, results = self._run(coding_sample(n_turns=3, compression_at=(2,)))
        sample = results.sample_results[0]

        assert sample.evaluation_type == "coding"
        assert sample.context_sizes == [100, 200, 150]
        assert sample.coding_result.task_type == "code_generation"
        assert sample.coding_result.generated_code == {"main.py": ADD_SRC}
        assert sample.coding_result.compression_points == sample.compression_points
        assert sample.coding_result.metrics == sample.coding_metrics
        assert set(sample.coding_metrics) == {
            "syntax_validity",
            "ast_similarity",
            "token_efficiency",
            "compression_retention",
        }
        assert sample.coding_metrics["syntax_validity"] == 1.0
        assert sample.coding_metrics["ast_similarity"] == 1.0
        assert sample.compression_summary["num_compression_points"] == 1
        assert "coding_result" in sample.to_dict()

    def test_multi_file_output_is_joined_and_diff_metric_for_bug_fixing(self):
        generated = {"a.py": "x = 1", "b.py": "y = 2"}
        agent = FakeCodingAgent(generated=generated, answer_fn=drifting_answers)
        sample = coding_sample(
            metadata={"task_type": "bug_fixing"},
            ground_truth={"expected_files": {"a.py": "x = 1\ny = 2"}},
        )
        _, results = self._run(sample, agent=agent)
        metrics = results.sample_results[0].coding_metrics
        assert results.sample_results[0].coding_result.task_type == "bug_fixing"
        assert metrics["diff_accuracy"] == 1.0
        assert "ast_similarity" not in metrics

    def test_invalid_generated_code(self):
        agent = FakeCodingAgent(generated={"m.py": "def broken(:\n"})
        _, results = self._run(coding_sample(), agent=agent)
        assert results.sample_results[0].coding_metrics["syntax_validity"] == 0.0

    def test_aggregator_gets_compression_subset_plus_coding_metrics(self):
        _, results = self._run(coding_sample(n_turns=3, compression_at=(2,)))
        overall = results.aggregate_metrics["overall"]
        assert {
            "goal_coherence_after",
            "goal_drift",
            "constraint_recall_after",
            "compression_ratio",
        } <= set(overall)
        assert "goal_coherence_before" not in overall
        assert "behavioral_alignment_after" not in overall
        assert overall["syntax_validity"]["mean"] == 1.0

    def test_no_compression_points_gives_empty_summary(self):
        agent, results = self._run(coding_sample(compression_at=()))
        sample = results.sample_results[0]
        assert sample.compression_points == []
        assert sample.compression_summary == {}
        assert not any(e[0] == "compress" for e in agent.events)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "token_efficiency is computed from the first and last turn context "
            "sizes rather than pre/post-compression tokens, so it exceeds 1.0 "
            "when context grows"
        ),
    )
    def test_token_efficiency_reflects_compression(self):
        _, results = self._run(coding_sample(n_turns=3, compression_at=(2,)))
        sample = results.sample_results[0]
        assert sample.coding_metrics["token_efficiency"] == pytest.approx(
            sample.compression_points[0]["compression_ratio"]
        )


# ---------------------------------------------------------------------------
# A-mem cache branches (QA path)
# ---------------------------------------------------------------------------


class TestQACache:
    def test_miss_ingests_and_saves_state(self, tmp_path):
        agent = FakeMemoryAgent(name="amem")
        harness = make_harness(agent, FakeDataset([qa_sample("s1")]), str(tmp_path))
        harness.run_evaluation(verbose=False)

        assert [e for e in agent.events if e[0] == "ingest"] == [
            ("ingest", 1),
            ("ingest", 2),
            ("ingest", 3),
        ]
        assert harness._cache.has_amem_cache("s1")
        assert agent.memory_system.retriever.saved_to is not None

    def test_hit_skips_ingest_and_restores_memories(self, tmp_path, capsys):
        first = FakeMemoryAgent(name="amem")
        make_harness(first, FakeDataset([qa_sample("s1")]), str(tmp_path)).run_evaluation(
            verbose=False
        )

        second = FakeMemoryAgent(name="amem", answer_fn=echo_reference)
        results = make_harness(
            second, FakeDataset([qa_sample("s1")]), str(tmp_path)
        ).run_evaluation(verbose=True)

        assert not any(e[0] == "ingest" for e in second.events)
        assert second.memory_system.memories == ["turn 1", "turn 2", "turn 3"]
        assert second.memory_system.retriever.loaded_from is not None
        assert [q for q, _, _ in second.asked] == ["Where is the meeting?", "Who hosts it?"]
        assert results.sample_results[0].context_sizes == [3]
        assert "Loaded cached memories" in capsys.readouterr().out

    def test_agent_without_memory_system_writes_no_cache(self, tmp_path):
        harness = make_harness(FakeAgent(name="plain"), FakeDataset([qa_sample("s1")]), str(tmp_path))
        harness.run_evaluation(verbose=False)
        assert harness._cache.list_cached_samples() == []

    def test_compression_path_does_not_use_cache(self, tmp_path):
        agent = FakeMemoryAgent(name="amem", answer_fn=drifting_answers)
        harness = make_harness(
            agent, FakeDataset([compression_sample()], "compression"), str(tmp_path)
        )
        harness.run_evaluation(verbose=False)
        assert harness._cache.list_cached_samples() == []


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------


class TestVerbose:
    def test_qa_banner_progress_and_summary(self, capsys):
        harness = make_harness(
            FakeAgent(name="agent-x", answer_fn=echo_reference),
            FakeDataset([qa_sample("s1")], name="ds-x"),
        )
        harness.run_evaluation(verbose=True)
        out = capsys.readouterr().out
        for needle in (
            "UNIFIED EVALUATION",
            "Agent: agent-x",
            "Dataset: ds-x",
            "Type: qa",
            "Samples: 1",
            "Processing sample 1/1: s1",
            "EVALUATION COMPLETE",
            "QA Summary:",
            "accuracy: 0.5000",
        ):
            assert needle in out
        assert "Runs per sample" not in out

    def test_compression_point_and_summary(self, capsys):
        harness = make_harness(
            FakeAgent(answer_fn=drifting_answers),
            FakeDataset([compression_sample()], "compression"),
        )
        harness.run_evaluation(verbose=True)
        out = capsys.readouterr().out
        assert "Compression point 1" in out
        assert "Compression Summary:" in out
        assert "drift_events_detected: 1" in out

    def test_coding_sample_details(self, capsys):
        harness = make_harness(
            FakeCodingAgent(answer_fn=drifting_answers),
            FakeDataset([coding_sample()], "coding"),
        )
        harness.run_evaluation(verbose=True)
        out = capsys.readouterr().out
        assert "Task type: code_generation" in out
        assert "Files generated: 1" in out
        assert "Syntax valid: 1.00" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_print_summary expects a 'coding_summary' key that "
            "UnifiedMetricAggregator.aggregate() never produces"
        ),
    )
    def test_coding_run_summary_is_printed(self, capsys):
        harness = make_harness(
            FakeCodingAgent(answer_fn=drifting_answers),
            FakeDataset([coding_sample()], "coding"),
        )
        harness.run_evaluation(verbose=True)
        assert "Coding Summary:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Multi-run (n_runs > 1)
# ---------------------------------------------------------------------------


def alternating_qa_answers(agent, question):
    """Correct on odd-numbered runs, wrong on even ones (run index = resets so far)."""
    return "Paris" if agent.reset_count % 2 else "London"


class TestMultiRun:
    @pytest.mark.xfail(
        strict=True,
        raises=ModuleNotFoundError,
        reason="evaluation/multi_run_evaluator.py is imported but not present on main",
    )
    def test_multi_run_without_stub(self):
        harness = make_harness(FakeAgent(), FakeDataset([qa_sample()]))
        results = harness.run_evaluation(n_runs=2, verbose=False)
        assert results.n_runs == 2

    def test_qa_runs_reset_per_run_and_keep_first_run_result(self, stub_multi_run_module):
        questions = [EvalQuestion("Where is the meeting?", "Paris", category=1)]
        samples = [qa_sample("s1", questions=questions), qa_sample("s2", questions=questions)]
        agent = FakeAgent(answer_fn=alternating_qa_answers)
        harness = make_harness(agent, FakeDataset(samples))

        results = harness.run_evaluation(n_runs=2, confidence=0.9, verbose=False)

        assert agent.reset_count == 4
        assert [e for e in agent.events if e[0] != "ask"][:5] == [
            ("reset",),
            ("ingest", 1),
            ("ingest", 2),
            ("ingest", 3),
            ("reset",),
        ]
        assert results.num_samples == 2
        assert [r.sample_id for r in results.sample_results] == ["s1", "s2"]
        assert [r.qa_results[0].prediction for r in results.sample_results] == [
            "Paris",
            "Paris",
        ]

        em = results.statistical_summary["exact_match"]
        assert em["mean"] == pytest.approx(0.5)
        assert em["std"] == 0.0
        assert em["n"] == 2
        assert em["ci_lower"] == pytest.approx(0.5)
        assert em["ci_upper"] == pytest.approx(0.5)

        assert results.n_runs == 2
        assert results.confidence == 0.9
        assert results.aggregate_metrics == {"overall": results.statistical_summary}
        d = results.to_dict()
        assert d["n_runs"] == 2
        assert d["statistical_summary"] == results.statistical_summary

    def test_compression_uses_numeric_summary_fields(self, stub_multi_run_module):
        harness = make_harness(
            FakeAgent(answer_fn=drifting_answers),
            FakeDataset([compression_sample()], "compression"),
        )
        results = harness.run_evaluation(n_runs=3, verbose=False)
        summary = results.statistical_summary
        assert summary["avg_goal_drift"]["mean"] == pytest.approx(0.75)
        assert summary["drift_events_detected"]["mean"] == 1.0
        assert summary["num_compression_points"]["n"] == 1
        assert harness.agent.reset_count == 3

    def test_coding_uses_coding_metrics(self, stub_multi_run_module):
        samples = [coding_sample("c1"), coding_sample("c2")]
        harness = make_harness(
            FakeCodingAgent(answer_fn=drifting_answers), FakeDataset(samples, "coding")
        )
        results = harness.run_evaluation(n_runs=2, verbose=False)
        summary = results.statistical_summary
        assert set(summary) == {
            "syntax_validity",
            "ast_similarity",
            "token_efficiency",
            "compression_retention",
        }
        assert summary["syntax_validity"] == {
            "mean": 1.0,
            "std": 0.0,
            "ci_lower": 1.0,
            "ci_upper": 1.0,
            "n": 2,
        }

    def test_ci_bounds_bracket_mean_for_varying_samples(self, stub_multi_run_module):
        questions = [EvalQuestion("Where is the meeting?", "Paris", category=1)]
        samples = [qa_sample(f"s{i}", questions=questions) for i in range(4)]
        # Resets 1-3 answer correctly: per-sample run means are 1.0, 0.5, 0.0, 0.0.
        harness = make_harness(
            FakeAgent(answer_fn=lambda a, q: "Paris" if a.reset_count <= 3 else "London"),
            FakeDataset(samples),
        )
        stats = harness.run_evaluation(n_runs=2, verbose=False).statistical_summary[
            "exact_match"
        ]
        assert stats["n"] == 4
        assert stats["mean"] == pytest.approx(0.375)
        assert stats["std"] > 0
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]
        assert stats["ci_lower"] < stats["ci_upper"]

    def test_verbose_multi_run_output(self, stub_multi_run_module, capsys):
        harness = make_harness(
            FakeAgent(answer_fn=lambda a, q: "Paris"), FakeDataset([qa_sample("s1")])
        )
        harness.run_evaluation(n_runs=2, confidence=0.95, verbose=True)
        out = capsys.readouterr().out
        assert "Runs per sample: 2" in out
        assert "Processing sample 1/1: s1 (2 runs)" in out
        assert "MULTI-RUN EVALUATION COMPLETE (2 runs/sample)" in out
        assert "Statistical Summary (with CI):" in out
        assert "exact_match: 0.5000" in out

    @pytest.mark.xfail(
        strict=True,
        reason="_print_statistical_summary hardcodes '95% CI' regardless of confidence",
    )
    def test_verbose_ci_label_reflects_confidence(self, stub_multi_run_module, capsys):
        harness = make_harness(
            FakeAgent(answer_fn=lambda a, q: "Paris"), FakeDataset([qa_sample("s1")])
        )
        harness.run_evaluation(n_runs=2, confidence=0.9, verbose=True)
        out = capsys.readouterr().out
        assert "95% CI" not in out
        assert "90% CI" in out
