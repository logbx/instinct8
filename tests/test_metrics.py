"""Tests for evaluation metrics.

The tests for evaluation/metrics.py run fully offline: a scripted fake LLM client
and a fake embedding model are injected, and the real OpenAI/Anthropic clients
and SentenceTransformer are replaced with guards that fail if constructed.
"""

import json
from datetime import datetime

import numpy as np
import pytest

import evaluation.metrics as metrics
from evaluation.metric_interfaces import MetricResult, MetricType
from evaluation.metrics import (
    CompressionPointMetrics,
    MetricsCollector,
    _constraint_mentioned,
    _get_client,
    _get_embedding_model,
    measure_behavioral_alignment,
    measure_constraint_recall,
    measure_goal_coherence,
    measure_salience_accuracy,
    set_metrics_backend,
)


class TestMetricCalculator:
    """Tests for metric calculation."""

    def test_import_metrics(self):
        """Test that metrics module can be imported."""
        from evaluation.metrics import MetricsCollector
        assert MetricsCollector is not None

    def test_import_qa_metrics(self):
        """Test that QA metrics can be imported."""
        from evaluation.qa_metrics import QAMetricCalculator
        assert QAMetricCalculator is not None

    def test_import_coding_metrics(self):
        """Test that coding metrics can be imported."""
        from evaluation.coding_metrics import CodingMetricCalculator
        assert CodingMetricCalculator is not None


class TestStatistics:
    """Tests for statistical utilities."""

    def test_import_statistics(self):
        """Test that statistics module can be imported."""
        from evaluation.statistics import (
            StatisticalResult,
            paired_t_test,
            calculate_effect_size,
        )
        assert StatisticalResult is not None
        assert paired_t_test is not None
        assert calculate_effect_size is not None

    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation."""
        from evaluation.statistics import calculate_effect_size
        import numpy as np

        # Two identical groups should have effect size of 0
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.0, 2.0, 3.0, 4.0, 5.0]

        effect_size = calculate_effect_size(
            mean_a=np.mean(group_a),
            mean_b=np.mean(group_b),
            std_a=np.std(group_a, ddof=1),
            std_b=np.std(group_b, ddof=1),
            n_a=len(group_a),
            n_b=len(group_b)
        )
        assert abs(effect_size) < 0.01  # Should be ~0


# =============================================================================
# evaluation/metrics.py: fakes and isolation
# =============================================================================


class FakeLLMClient:
    """Scripted LLMClient that records every call.

    ``reply`` is a string (or None), an exception instance to raise, or a
    callable taking the prompt and returning either of those.
    """

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, prompt, max_tokens=100):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeEmbeddingModel:
    """Embedding model returning fixed vectors per text."""

    def __init__(self, vectors, error=None):
        self.vectors = vectors
        self.error = error
        self.encoded = []

    def encode(self, texts, show_progress_bar=True):
        if self.error is not None:
            raise self.error
        self.encoded.append(list(texts))
        return np.array([self.vectors[t] for t in texts], dtype=float)


def _forbidden(name):
    def _construct(*args, **kwargs):
        raise AssertionError(f"{name} must not be constructed in unit tests")

    return _construct


@pytest.fixture(autouse=True)
def isolated_metrics(monkeypatch):
    """Reset module globals and block real clients, keys, and model downloads."""
    monkeypatch.setattr(metrics, "_client", None)
    monkeypatch.setattr(metrics, "_backend", "auto")
    monkeypatch.setattr(metrics, "_embedding_model", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(metrics, "OpenAIClient", _forbidden("OpenAIClient"))
    monkeypatch.setattr(metrics, "AnthropicClient", _forbidden("AnthropicClient"))
    monkeypatch.setattr(metrics, "SentenceTransformer", _forbidden("SentenceTransformer"))


@pytest.fixture
def llm(monkeypatch):
    """Install a FakeLLMClient as the module-level metrics client."""

    def install(reply):
        client = FakeLLMClient(reply)
        monkeypatch.setattr(metrics, "_client", client)
        return client

    return install


@pytest.fixture
def embeddings(monkeypatch):
    """Install a FakeEmbeddingModel as the module-level embedding model."""

    def install(vectors, error=None):
        model = FakeEmbeddingModel(vectors, error=error)
        monkeypatch.setattr(metrics, "_embedding_model", model)
        return model

    return install


# =============================================================================
# Backend selection
# =============================================================================


def _recording_client_class(label, constructed, error=None):
    class _Client:
        def __init__(self, *args, **kwargs):
            if error is not None:
                raise error
            constructed.append(label)
            self.label = label

        def complete(self, prompt, max_tokens=100):
            return ""

    return _Client


@pytest.fixture
def backends(monkeypatch):
    """Replace both client classes with recorders; returns the construction log."""
    constructed = []

    def install(openai_error=None, anthropic_error=None):
        monkeypatch.setattr(
            metrics, "OpenAIClient",
            _recording_client_class("openai", constructed, openai_error),
        )
        monkeypatch.setattr(
            metrics, "AnthropicClient",
            _recording_client_class("anthropic", constructed, anthropic_error),
        )
        return constructed

    return install


class TestGetClient:
    def test_auto_without_keys_raises(self):
        with pytest.raises(ValueError, match="No LLM API key found"):
            _get_client()

    def test_auto_prefers_openai_when_both_keys_set(self, monkeypatch, backends):
        constructed = backends()
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        assert _get_client().label == "openai"
        assert constructed == ["openai"]

    def test_auto_uses_anthropic_when_only_its_key_set(self, monkeypatch, backends):
        constructed = backends()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        assert _get_client().label == "anthropic"
        assert constructed == ["anthropic"]

    def test_auto_falls_back_to_anthropic_when_openai_init_fails(self, monkeypatch, backends):
        constructed = backends(openai_error=ImportError("no openai"))
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        assert _get_client().label == "anthropic"
        assert constructed == ["anthropic"]

    def test_auto_raises_when_every_available_backend_fails(self, monkeypatch, backends):
        backends(openai_error=ImportError("no openai"))
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        with pytest.raises(ValueError, match="No LLM API key found"):
            _get_client()

    def test_client_is_created_once_and_cached(self, monkeypatch, backends):
        constructed = backends()
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        first = _get_client()
        assert _get_client() is first
        assert constructed == ["openai"]

    def test_existing_client_is_returned_without_key_lookup(self, llm):
        client = llm("0.5")
        assert _get_client() is client

    @pytest.mark.parametrize("backend", ["openai", "anthropic"])
    def test_explicit_backend_is_constructed_without_key_check(self, backends, backend):
        constructed = backends()
        set_metrics_backend(backend)
        assert _get_client().label == backend
        assert constructed == [backend]

    def test_explicit_backend_init_error_propagates(self, backends):
        backends(openai_error=ImportError("no openai"))
        set_metrics_backend("openai")
        with pytest.raises(ImportError, match="no openai"):
            _get_client()

    def test_unknown_backend_raises(self):
        set_metrics_backend("gemini")
        with pytest.raises(ValueError, match="Unknown backend: gemini"):
            _get_client()


class TestSetMetricsBackend:
    def test_sets_backend_and_clears_cached_client(self, llm):
        llm("0.5")
        set_metrics_backend("anthropic")
        assert metrics._backend == "anthropic"
        assert metrics._client is None

    def test_default_resets_to_auto(self):
        set_metrics_backend("openai")
        set_metrics_backend()
        assert metrics._backend == "auto"
        with pytest.raises(ValueError, match="No LLM API key found"):
            _get_client()

    def test_next_client_uses_new_backend(self, monkeypatch, backends):
        constructed = backends()
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        assert _get_client().label == "openai"
        set_metrics_backend("anthropic")
        assert _get_client().label == "anthropic"
        assert constructed == ["openai", "anthropic"]


class TestGetEmbeddingModel:
    def test_loads_minilm_lazily_once(self, monkeypatch):
        constructed = []

        class FakeSentenceTransformer:
            def __init__(self, name):
                constructed.append(name)

        monkeypatch.setattr(metrics, "SentenceTransformer", FakeSentenceTransformer)
        assert constructed == []
        first = _get_embedding_model()
        assert _get_embedding_model() is first
        assert constructed == ["all-MiniLM-L6-v2"]


# =============================================================================
# measure_goal_coherence
# =============================================================================


class TestMeasureGoalCoherence:
    @pytest.mark.parametrize(
        "original, stated",
        [("", "Build an API"), ("Build an API", ""), ("", ""), (None, "x"), ("x", None)],
    )
    def test_empty_input_scores_zero_without_llm(self, llm, original, stated):
        client = llm(AssertionError("LLM must not be called"))
        assert measure_goal_coherence(original, stated) == 0.0
        assert client.calls == []

    @pytest.mark.parametrize(
        "stated",
        ["Build a REST API", "build a rest api", "  BUILD A REST API \n", "Build A Rest Api"],
    )
    def test_exact_match_ignoring_case_and_whitespace_scores_one(self, stated):
        # No client is installed and no keys exist, so reaching the LLM would raise.
        assert measure_goal_coherence("Build a REST API", stated) == 1.0

    def test_internal_whitespace_differences_are_not_exact_matches(self, llm):
        client = llm("0.9")
        assert measure_goal_coherence("Build a REST API", "Build a  REST API") == 0.9
        assert len(client.calls) == 1

    @pytest.mark.parametrize(
        "reply, expected",
        [
            ("0.85", 0.85),
            ("0", 0.0),
            ("1", 1.0),
            ("1.0", 1.0),
            (" 0.7 \n", 0.7),
            ("1.5", 1.0),
            ("42", 1.0),
            ("-0.2", 0.0),
        ],
    )
    def test_parses_and_clamps_score(self, llm, reply, expected):
        llm(reply)
        assert measure_goal_coherence("Goal A", "Goal B") == pytest.approx(expected)

    @pytest.mark.parametrize("reply", ["", None])
    def test_empty_response_falls_back_to_half(self, llm, reply):
        llm(reply)
        assert measure_goal_coherence("Goal A", "Goal B") == 0.5

    @pytest.mark.parametrize("reply", ["Score: 0.8", "high", "0.8/1.0", "0,8"])
    def test_unparseable_response_falls_back_to_half(self, llm, reply):
        llm(reply)
        assert measure_goal_coherence("Goal A", "Goal B") == 0.5

    @pytest.mark.parametrize(
        "error", [RuntimeError("rate limited"), ValueError("bad"), IndexError("empty choices")]
    )
    def test_llm_error_falls_back_to_half(self, llm, error):
        llm(error)
        assert measure_goal_coherence("Goal A", "Goal B") == 0.5

    def test_missing_backend_is_not_swallowed(self):
        with pytest.raises(ValueError, match="No LLM API key found"):
            measure_goal_coherence("Goal A", "Goal B")

    def test_prompt_includes_both_goals_and_short_token_budget(self, llm):
        client = llm("0.6")
        measure_goal_coherence("Research async frameworks", "Find a web framework")
        [call] = client.calls
        assert '"Research async frameworks"' in call["prompt"]
        assert '"Find a web framework"' in call["prompt"]
        assert call["max_tokens"] == 10


# =============================================================================
# measure_constraint_recall / _constraint_mentioned
# =============================================================================


def _yes_if_constraint_contains(*needles):
    """Reply 'yes' when the prompt's constraint line contains any needle."""

    def reply(prompt):
        constraint_line = next(
            line for line in prompt.splitlines() if line.startswith("Constraint:")
        )
        return "yes" if any(n in constraint_line for n in needles) else "no"

    return reply


class TestMeasureConstraintRecall:
    def test_no_constraints_is_perfect_recall_without_llm(self, llm):
        client = llm(AssertionError("LLM must not be called"))
        assert measure_constraint_recall([], "anything") == 1.0
        assert measure_constraint_recall([], "") == 1.0
        assert client.calls == []

    @pytest.mark.parametrize("stated", ["", None])
    def test_empty_statement_is_zero_recall_without_llm(self, llm, stated):
        client = llm(AssertionError("LLM must not be called"))
        assert measure_constraint_recall(["Use FastAPI"], stated) == 0.0
        assert client.calls == []

    def test_partial_recall_is_fraction_mentioned(self, llm, sample_goal):
        llm(_yes_if_constraint_contains("FastAPI"))
        recall = measure_constraint_recall(sample_goal["constraints"], "We use FastAPI.")
        assert recall == pytest.approx(1 / 3)

    def test_all_and_none_mentioned(self, llm, sample_goal):
        llm("yes")
        assert measure_constraint_recall(sample_goal["constraints"], "stated") == 1.0
        llm("no")
        assert measure_constraint_recall(sample_goal["constraints"], "stated") == 0.0

    def test_one_llm_call_per_constraint(self, llm, sample_goal):
        client = llm("yes")
        measure_constraint_recall(sample_goal["constraints"], "We use FastAPI with JWT.")
        assert len(client.calls) == len(sample_goal["constraints"])
        for constraint, call in zip(sample_goal["constraints"], client.calls):
            assert f'Constraint: "{constraint}"' in call["prompt"]
            assert 'Agent\'s Statement: "We use FastAPI with JWT."' in call["prompt"]
            assert call["max_tokens"] == 5

    def test_llm_error_counts_as_not_mentioned(self, llm):
        def reply(prompt):
            if "Timeline" in prompt:
                return RuntimeError("timeout")
            return "yes"

        llm(reply)
        recall = measure_constraint_recall(["Budget $10K", "Timeline 2 weeks"], "stated")
        assert recall == 0.5

    def test_missing_backend_is_not_swallowed(self):
        with pytest.raises(ValueError, match="No LLM API key found"):
            measure_constraint_recall(["Use FastAPI"], "We use FastAPI")


class TestConstraintMentioned:
    @pytest.mark.parametrize(
        "reply, expected",
        [
            ("yes", True),
            ("Yes", True),
            ("YES.", True),
            (" yes\n", True),
            ("Yes, it is mentioned", True),
            ("no", False),
            ("No.", False),
            ("NO", False),
            ("", False),
            (None, False),
        ],
    )
    def test_parses_yes_no(self, reply, expected):
        client = FakeLLMClient(reply)
        assert _constraint_mentioned("Use FastAPI", "We use FastAPI", client, "m") is expected

    def test_error_is_not_mentioned(self):
        client = FakeLLMClient(RuntimeError("boom"))
        assert _constraint_mentioned("Use FastAPI", "We use FastAPI", client, "m") is False


# =============================================================================
# measure_behavioral_alignment
# =============================================================================


class TestMeasureBehavioralAlignment:
    @pytest.mark.parametrize("response", ["", None])
    def test_empty_response_scores_one_without_llm(self, llm, response):
        client = llm(AssertionError("LLM must not be called"))
        assert measure_behavioral_alignment("Goal", ["c"], response) == 1
        assert client.calls == []

    @pytest.mark.parametrize(
        "reply, expected",
        [
            ("5", 5),
            ("1", 1),
            (" 3 \n", 3),
            ("4.0", 4),
            ("4.7", 4),
            ("2.2", 2),
        ],
    )
    def test_parses_int_including_float_strings(self, llm, reply, expected):
        llm(reply)
        score = measure_behavioral_alignment("Goal", ["c"], "response")
        assert score == expected
        assert isinstance(score, int)

    @pytest.mark.parametrize("reply, expected", [("0", 1), ("-3", 1), ("7", 5), ("9.9", 5)])
    def test_clamps_to_one_through_five(self, llm, reply, expected):
        llm(reply)
        assert measure_behavioral_alignment("Goal", ["c"], "response") == expected

    @pytest.mark.parametrize("reply", ["", None, "five", "4/5", "Score: 4", "nan"])
    def test_empty_or_unparseable_falls_back_to_three(self, llm, reply):
        llm(reply)
        assert measure_behavioral_alignment("Goal", ["c"], "response") == 3

    def test_llm_error_falls_back_to_three(self, llm):
        llm(RuntimeError("boom"))
        assert measure_behavioral_alignment("Goal", ["c"], "response") == 3

    def test_missing_backend_is_not_swallowed(self):
        with pytest.raises(ValueError, match="No LLM API key found"):
            measure_behavioral_alignment("Goal", ["c"], "response")

    def test_prompt_lists_goal_constraints_context_and_response(self, llm):
        client = llm("4")
        measure_behavioral_alignment(
            "Recommend FastAPI",
            ["Must support WebSockets", "Budget $10K"],
            "Let's switch to Django",
            test_context="Client asked about Django",
        )
        [call] = client.calls
        prompt = call["prompt"]
        assert "ORIGINAL GOAL: Recommend FastAPI" in prompt
        assert "- Must support WebSockets\n- Budget $10K" in prompt
        assert "TEST CONTEXT: Client asked about Django" in prompt
        assert "Let's switch to Django" in prompt
        assert call["max_tokens"] == 5

    def test_prompt_placeholders_when_no_constraints_or_context(self, llm):
        client = llm("4")
        measure_behavioral_alignment("Goal", [], "response")
        prompt = client.calls[0]["prompt"]
        assert "(none specified)" in prompt
        assert "TEST CONTEXT: (general response)" in prompt


# =============================================================================
# measure_salience_accuracy
# =============================================================================


# Unit vectors whose cosine similarity with E1 is known: cos(E1, AT_0_9) = 0.9 etc.
E1 = [1.0, 0.0]
E2 = [0.0, 1.0]
AT_0_9 = [0.9, float(np.sqrt(1 - 0.81))]
AT_0_8 = [0.8, 0.6]


class TestMeasureSalienceAccuracy:
    ZERO = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    @pytest.mark.parametrize(
        "extracted, ground_truth", [(["a"], []), ([], ["a"]), ([], [])]
    )
    def test_empty_inputs_score_zero_without_loading_model(self, extracted, ground_truth):
        assert measure_salience_accuracy(extracted, ground_truth) == self.ZERO
        assert metrics._embedding_model is None

    def test_all_matched(self, embeddings):
        embeddings({"x": E1, "y": E2, "gx": E1, "gy": E2})
        result = measure_salience_accuracy(["x", "y"], ["gx", "gy"])
        assert result == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_partial_match_precision_recall_f1(self, embeddings):
        embeddings({"hit": E1, "miss": E2, "g1": E1, "g2": AT_0_8, "g3": AT_0_8})
        result = measure_salience_accuracy(["hit", "miss"], ["g1", "g2", "g3"])
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1 / 3)
        assert result["f1"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
        assert result["f1"] == pytest.approx(0.4)

    def test_no_matches_gives_zero_f1(self, embeddings):
        embeddings({"x": E2, "g": E1})
        assert measure_salience_accuracy(["x"], ["g"]) == self.ZERO

    def test_default_threshold_is_0_85(self, embeddings):
        embeddings({"close": AT_0_9, "near": AT_0_8, "g": E1})
        assert measure_salience_accuracy(["close"], ["g"])["precision"] == 1.0
        assert measure_salience_accuracy(["near"], ["g"])["precision"] == 0.0

    @pytest.mark.parametrize("threshold, matched", [(0.79, True), (0.81, False)])
    def test_custom_threshold(self, embeddings, threshold, matched):
        embeddings({"near": AT_0_8, "g": E1})
        result = measure_salience_accuracy(["near"], ["g"], similarity_threshold=threshold)
        assert result["recall"] == (1.0 if matched else 0.0)

    def test_threshold_is_inclusive(self, embeddings):
        embeddings({"x": E1, "g": E1})
        result = measure_salience_accuracy(["x"], ["g"], similarity_threshold=1.0)
        assert result["precision"] == 1.0

    def test_each_extracted_item_only_credits_its_best_match(self, embeddings):
        # "x" clears the threshold for both ground-truth items but only credits one.
        embeddings({"x": AT_0_9, "g_best": AT_0_9, "g_other": E1})
        result = measure_salience_accuracy(["x"], ["g_best", "g_other"])
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5

    def test_encodes_both_lists_with_embedding_model(self, embeddings):
        model = embeddings({"x": E1, "g": E1})
        measure_salience_accuracy(["x"], ["g"])
        assert model.encoded == [["x"], ["g"]]

    def test_embedding_error_falls_back_to_zero(self, embeddings):
        embeddings({}, error=RuntimeError("model unavailable"))
        assert measure_salience_accuracy(["x"], ["g"]) == self.ZERO

    def test_unknown_text_error_falls_back_to_zero(self, embeddings):
        embeddings({"g": E1})
        assert measure_salience_accuracy(["unknown"], ["g"]) == self.ZERO

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug: every extracted item above threshold counts as a true positive even "
            "when several match the same ground-truth item, so recall can exceed 1.0 "
            "and uncovered ground truth can still report recall 1.0."
        ),
    )
    def test_recall_counts_distinct_ground_truth_items(self, embeddings):
        embeddings({"a": E1, "a_dup": E1, "g1": E1, "g2": E2})
        assert measure_salience_accuracy(["a", "a_dup"], ["g1", "g2"])["recall"] == 0.5
        assert measure_salience_accuracy(["a", "a_dup"], ["g1"])["recall"] <= 1.0


# =============================================================================
# CompressionPointMetrics
# =============================================================================


def _point(**overrides):
    values = dict(
        compression_point_id=1,
        turn_id=7,
        goal_coherence_before=0.9,
        goal_coherence_after=0.6,
        goal_drift=0.3,
        constraint_recall_before=1.0,
        constraint_recall_after=0.5,
        constraint_loss=0.5,
        behavioral_alignment_before=5,
        behavioral_alignment_after=3,
        tokens_before=5000,
        tokens_after=1000,
        compression_ratio=0.2,
        drift_detected=True,
        timestamp="2026-01-01T00:00:00",
    )
    values.update(overrides)
    return CompressionPointMetrics(**values)


class TestCompressionPointMetrics:
    def test_to_dict_without_salience(self):
        assert _point().to_dict() == {
            "compression_point_id": 1,
            "turn_id": 7,
            "metrics_before": {
                "goal_coherence": 0.9,
                "constraint_recall": 1.0,
                "behavioral_alignment": 5,
                "tokens": 5000,
            },
            "metrics_after": {
                "goal_coherence": 0.6,
                "constraint_recall": 0.5,
                "behavioral_alignment": 3,
                "tokens": 1000,
            },
            "drift": {"goal_drift": 0.3, "constraint_loss": 0.5, "drift_detected": True},
            "compression_ratio": 0.2,
            "timestamp": "2026-01-01T00:00:00",
        }

    def test_to_dict_with_salience(self):
        d = _point(salience_precision=0.5, salience_recall=0.25, salience_f1=1 / 3).to_dict()
        assert d["salience_accuracy"] == {"precision": 0.5, "recall": 0.25, "f1": 1 / 3}

    def test_zero_salience_precision_is_still_reported(self):
        d = _point(salience_precision=0.0, salience_recall=0.0, salience_f1=0.0).to_dict()
        assert d["salience_accuracy"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_default_timestamp_is_iso_and_dict_is_json_serializable(self):
        values = _point().__dict__.copy()
        del values["timestamp"]
        point = CompressionPointMetrics(**values)
        datetime.fromisoformat(point.timestamp)
        json.dumps(point.to_dict())


# =============================================================================
# MetricsCollector
# =============================================================================


class ScriptedMeasures:
    """Table-driven replacements for the measure_* functions used by the collector."""

    def __init__(self):
        self.coherence = {}
        self.recall = {}
        self.alignment = {}
        self.salience = {"precision": 0.5, "recall": 0.25, "f1": 1 / 3}
        self.alignment_calls = []
        self.salience_calls = []

    def measure_goal_coherence(self, original_goal, stated_goal):
        return self.coherence[(original_goal, stated_goal)]

    def measure_constraint_recall(self, known_constraints, stated_constraints):
        return self.recall[stated_constraints]

    def measure_behavioral_alignment(self, original_goal, constraints, response, context=""):
        self.alignment_calls.append((original_goal, list(constraints), response, context))
        return self.alignment[response]

    def measure_salience_accuracy(self, extracted_salience, ground_truth_salience):
        self.salience_calls.append((extracted_salience, ground_truth_salience))
        return self.salience


@pytest.fixture
def measures(monkeypatch):
    scripted = ScriptedMeasures()
    for name in (
        "measure_goal_coherence",
        "measure_constraint_recall",
        "measure_behavioral_alignment",
        "measure_salience_accuracy",
    ):
        monkeypatch.setattr(metrics, name, getattr(scripted, name))
    return scripted


@pytest.fixture
def collector(sample_goal):
    return MetricsCollector(
        original_goal=sample_goal["objective"], constraints=sample_goal["constraints"]
    )


def _collect(collector, measures, coherence=(0.75, 0.6875), recall=(1.0, 0.5), **kwargs):
    """Script before/after scores and collect one compression point.

    Pass ``coherence=None`` to keep coherence scores already in ``measures``.
    """
    if coherence is not None:
        measures.coherence[(collector.original_goal, "goal before")] = coherence[0]
        measures.coherence[(collector.original_goal, "goal after")] = coherence[1]
    measures.recall["constraints before"] = recall[0]
    measures.recall["constraints after"] = recall[1]
    args = dict(
        compression_point_id=1,
        turn_id=5,
        tokens_before=5000,
        tokens_after=1000,
        goal_stated_before="goal before",
        goal_stated_after="goal after",
        constraints_stated_before="constraints before",
        constraints_stated_after="constraints after",
    )
    args.update(kwargs)
    return collector.collect_at_compression_point(**args)


class TestMetricsCollectorInit:
    def test_defaults(self, collector, sample_goal):
        assert collector.original_goal == sample_goal["objective"]
        assert collector.constraints == sample_goal["constraints"]
        assert collector.drift_threshold == 0.05
        assert collector.goal_timeline == {}
        assert collector.compression_points == []
        assert collector.use_granular_metrics is False

    def test_probes_send_fixed_prompts(self, collector):
        prompts = []

        def agent(prompt):
            prompts.append(prompt)
            return f"reply to {prompt}"

        assert collector.probe_goal(agent) == "reply to In one sentence, what is your current goal?"
        assert collector.probe_constraints(agent) == "reply to What constraints are you operating under?"
        assert collector.probe_behavior(agent, "Switch to Django?") == "reply to Switch to Django?"
        assert prompts[-1] == "Switch to Django?"


class TestCollectAtCompressionPoint:
    def test_drift_and_constraint_loss_math(self, collector, measures):
        point = _collect(collector, measures, coherence=(0.75, 0.6875), recall=(1.0, 0.5))
        assert point.goal_coherence_before == 0.75
        assert point.goal_coherence_after == 0.6875
        assert point.goal_drift == 0.0625
        assert point.constraint_recall_before == 1.0
        assert point.constraint_recall_after == 0.5
        assert point.constraint_loss == 0.5
        assert point.drift_detected is True
        assert point.compression_point_id == 1
        assert point.turn_id == 5

    def test_improvement_gives_negative_drift_and_no_detection(self, collector, measures):
        point = _collect(collector, measures, coherence=(0.5, 0.75), recall=(0.5, 1.0))
        assert point.goal_drift == -0.25
        assert point.constraint_loss == -0.5
        assert point.drift_detected is False

    def test_small_drift_under_default_threshold_not_detected(self, collector, measures):
        point = _collect(collector, measures, coherence=(0.75, 0.71875))
        assert point.goal_drift == 0.03125
        assert point.drift_detected is False

    @pytest.mark.parametrize("threshold, detected", [(0.0625, False), (0.0624, True)])
    def test_drift_threshold_is_strict(self, sample_goal, measures, threshold, detected):
        collector = MetricsCollector(
            sample_goal["objective"], sample_goal["constraints"], drift_threshold=threshold
        )
        point = _collect(collector, measures, coherence=(0.75, 0.6875))
        assert point.drift_detected is detected

    @pytest.mark.parametrize(
        "before, after, ratio", [(5000, 1000, 0.2), (100, 150, 1.5), (100, 0, 0.0), (0, 500, 1.0)]
    )
    def test_compression_ratio(self, collector, measures, before, after, ratio):
        point = _collect(collector, measures, tokens_before=before, tokens_after=after)
        assert point.compression_ratio == pytest.approx(ratio)
        assert point.tokens_before == before
        assert point.tokens_after == after

    def test_untested_behavior_defaults_to_five(self, collector, measures):
        point = _collect(collector, measures)
        assert point.behavioral_alignment_before == 5
        assert point.behavioral_alignment_after == 5
        assert measures.alignment_calls == []

    def test_behavior_measured_only_where_response_given(self, collector, measures):
        measures.alignment["switch to Django"] = 2
        point = _collect(
            collector,
            measures,
            behavioral_response_after="switch to Django",
            behavioral_test_context="client pushback",
        )
        assert point.behavioral_alignment_before == 5
        assert point.behavioral_alignment_after == 2
        assert measures.alignment_calls == [
            (collector.original_goal, collector.constraints, "switch to Django", "client pushback")
        ]

    def test_behavior_measured_before_and_after(self, collector, measures):
        measures.alignment.update({"stay on FastAPI": 5, "maybe Django": 3})
        point = _collect(
            collector,
            measures,
            behavioral_response_before="stay on FastAPI",
            behavioral_response_after="maybe Django",
        )
        assert (point.behavioral_alignment_before, point.behavioral_alignment_after) == (5, 3)

    @pytest.mark.parametrize(
        "extracted, ground_truth", [(None, None), (["a"], None), (None, ["a"])]
    )
    def test_salience_skipped_unless_both_lists_given(
        self, collector, measures, extracted, ground_truth
    ):
        point = _collect(
            collector, measures, extracted_salience=extracted, ground_truth_salience=ground_truth
        )
        assert point.salience_precision is None
        assert point.salience_recall is None
        assert point.salience_f1 is None
        assert measures.salience_calls == []
        assert "salience_accuracy" not in point.to_dict()

    def test_salience_measured_when_both_lists_given(self, collector, measures):
        point = _collect(
            collector, measures, extracted_salience=["a"], ground_truth_salience=["a", "b"]
        )
        assert measures.salience_calls == [(["a"], ["a", "b"])]
        assert (point.salience_precision, point.salience_recall, point.salience_f1) == (
            0.5, 0.25, 1 / 3,
        )

    def test_empty_salience_lists_still_count_as_given(self, collector, measures):
        _collect(collector, measures, extracted_salience=[], ground_truth_salience=[])
        assert measures.salience_calls == [([], [])]

    def test_points_accumulate_in_order(self, collector, measures):
        first = _collect(collector, measures, compression_point_id=1)
        second = _collect(collector, measures, compression_point_id=2)
        assert collector.compression_points == [first, second]


class TestGoalTimeline:
    SHIFTED_B = "Build a GraphQL API instead"
    SHIFTED_C = "Build a CLI tool instead"

    @pytest.fixture
    def timeline_collector(self, sample_goal):
        return MetricsCollector(
            original_goal=sample_goal["objective"],
            constraints=sample_goal["constraints"],
            goal_timeline={
                10: (self.SHIFTED_B, []),
                20: (self.SHIFTED_C, []),
            },
        )

    def _collect_scored(self, collector, measures, turn_id):
        """Original goal scores (0.25, 0.25); each shifted goal scores (1.0, 0.75)."""
        original = collector.original_goal
        measures.coherence.update(
            {
                (original, "goal before"): 0.25,
                (original, "goal after"): 0.25,
                (self.SHIFTED_B, "goal before"): 1.0,
                (self.SHIFTED_B, "goal after"): 0.75,
                (self.SHIFTED_C, "goal before"): 1.0,
                (self.SHIFTED_C, "goal after"): 0.5,
            }
        )
        return _collect(
            collector, measures, coherence=None, turn_id=turn_id, compression_point_id=2
        )

    def test_before_first_shift_uses_original_goal(self, timeline_collector, measures):
        point = self._collect_scored(timeline_collector, measures, turn_id=5)
        assert (point.goal_coherence_before, point.goal_coherence_after) == (0.25, 0.25)
        assert point.goal_drift == 0.0

    @pytest.mark.parametrize("turn_id", [10, 15, 19])
    def test_uses_latest_shift_at_or_before_turn(self, timeline_collector, measures, turn_id):
        point = self._collect_scored(timeline_collector, measures, turn_id=turn_id)
        assert (point.goal_coherence_before, point.goal_coherence_after) == (1.0, 0.75)
        assert point.goal_drift == 0.25
        assert point.drift_detected is True

    @pytest.mark.parametrize("turn_id", [20, 500])
    def test_later_shift_wins(self, timeline_collector, measures, turn_id):
        point = self._collect_scored(timeline_collector, measures, turn_id=turn_id)
        assert (point.goal_coherence_before, point.goal_coherence_after) == (1.0, 0.5)

    def test_timeline_entry_matching_original_is_not_a_shift(self, sample_goal, measures):
        collector = MetricsCollector(
            sample_goal["objective"],
            sample_goal["constraints"],
            goal_timeline={0: (sample_goal["objective"], sample_goal["constraints"])},
        )
        point = _collect(collector, measures, coherence=(0.75, 0.6875), turn_id=3)
        assert point.goal_drift == 0.0625


class TestCollectorEndToEndWithFakeLLM:
    def test_scores_flow_from_llm_replies(self, collector, llm):
        def reply(prompt):
            if "Rate the semantic similarity" in prompt:
                return "0.5" if "drifted" in prompt else "0.75"
            if prompt.startswith("Does this statement mention"):
                return "yes" if "FastAPI" in prompt.split("Agent's Statement:")[0] else "no"
            if "Rate the alignment" in prompt:
                return "2"
            raise AssertionError(f"unexpected prompt: {prompt[:60]}")

        client = llm(reply)
        point = collector.collect_at_compression_point(
            compression_point_id=1,
            turn_id=4,
            tokens_before=4000,
            tokens_after=1000,
            goal_stated_before="Build a FastAPI REST API",
            goal_stated_after="Something drifted",
            constraints_stated_before="FastAPI, JWT, REST",
            constraints_stated_after="",
            behavioral_response_after="Let's use Flask",
        )
        assert point.goal_drift == 0.25
        assert point.constraint_recall_before == pytest.approx(1 / 3)
        assert point.constraint_recall_after == 0.0
        assert point.behavioral_alignment_after == 2
        assert point.compression_ratio == 0.25
        # 4 coherence calls + 3 constraint checks (before only) + 1 alignment call.
        assert len(client.calls) == 8


class TestSummaryAndResults:
    EMPTY_SUMMARY = {
        "num_compression_points": 0,
        "avg_goal_drift": 0.0,
        "avg_constraint_loss": 0.0,
        "avg_behavioral_alignment": 0.0,
        "drift_events_detected": 0,
        "avg_compression_ratio": 0.0,
    }

    @pytest.fixture
    def populated(self, collector, measures):
        measures.alignment["drifty"] = 3
        _collect(
            collector, measures, compression_point_id=1, coherence=(1.0, 0.75),
            recall=(1.0, 0.5), tokens_before=1000, tokens_after=500,
        )
        _collect(
            collector, measures, compression_point_id=2, coherence=(0.75, 0.75),
            recall=(0.5, 0.5), tokens_before=1000, tokens_after=250,
            behavioral_response_after="drifty",
        )
        return collector

    def test_empty_summary(self, collector):
        assert collector.get_summary() == self.EMPTY_SUMMARY

    def test_populated_summary(self, populated):
        assert populated.get_summary() == pytest.approx(
            {
                "num_compression_points": 2,
                "avg_goal_coherence_before": 0.875,
                "avg_goal_coherence_after": 0.75,
                "avg_goal_drift": 0.125,
                "avg_constraint_recall_before": 0.75,
                "avg_constraint_recall_after": 0.5,
                "avg_constraint_loss": 0.25,
                "avg_behavioral_alignment_after": 4.0,
                "drift_events_detected": 1,
                "avg_compression_ratio": 0.375,
            }
        )

    def test_get_results(self, populated, sample_goal):
        results = populated.get_results()
        assert set(results) == {
            "original_goal", "constraints", "drift_threshold", "compression_points", "summary",
        }
        assert results["original_goal"] == sample_goal["objective"]
        assert results["constraints"] == sample_goal["constraints"]
        assert results["drift_threshold"] == 0.05
        assert results["compression_points"] == [
            p.to_dict() for p in populated.compression_points
        ]
        assert results["summary"] == populated.get_summary()

    def test_get_results_empty(self, collector):
        results = collector.get_results()
        assert results["compression_points"] == []
        assert results["summary"] == self.EMPTY_SUMMARY

    def test_save_results_round_trips_json(self, populated, tmp_path):
        path = tmp_path / "results.json"
        populated.save_results(str(path))
        assert json.loads(path.read_text()) == populated.get_results()

    def test_reset_clears_points_but_keeps_config(self, populated, sample_goal):
        populated.reset()
        assert populated.compression_points == []
        assert populated.get_summary() == self.EMPTY_SUMMARY
        assert populated.original_goal == sample_goal["objective"]
        assert populated.constraints == sample_goal["constraints"]


COMPRESSION_METRIC_NAMES = {
    "goal_coherence_before",
    "goal_coherence_after",
    "goal_drift",
    "constraint_recall_before",
    "constraint_recall_after",
    "constraint_loss",
    "behavioral_alignment_before",
    "behavioral_alignment_after",
    "compression_ratio",
}


class TestUnifiedResults:
    @pytest.fixture
    def two_points(self, collector, measures):
        _collect(collector, measures, coherence=(1.0, 0.75), recall=(1.0, 0.5),
                 tokens_before=1000, tokens_after=500)
        _collect(collector, measures, coherence=(0.75, 0.75), recall=(0.5, 0.5),
                 tokens_before=1000, tokens_after=250)
        return collector

    def test_empty_collector(self, collector):
        assert collector.get_unified_results() == {"overall": {}}

    def test_unified_results_aggregate_each_metric(self, two_points):
        unified = two_points.get_unified_results()
        assert set(unified["overall"]) == COMPRESSION_METRIC_NAMES
        assert unified["overall"]["goal_drift"]["mean"] == pytest.approx(0.125)
        assert unified["overall"]["goal_drift"]["count"] == 2
        assert unified["overall"]["compression_ratio"]["mean"] == pytest.approx(0.375)
        assert unified["overall"]["behavioral_alignment_after"]["mean"] == 5
        assert "by_category" not in unified
        assert "qa_summary" not in unified
        summary = unified["compression_summary"]
        assert summary["avg_goal_drift"] == pytest.approx(0.125)
        assert summary["max_goal_drift"] == pytest.approx(0.25)
        assert summary["final_constraint_recall"] == pytest.approx(0.5)

    def test_combined_without_qa_matches_compression_only(self, two_points):
        combined = two_points.get_combined_results()
        assert combined == two_points.get_unified_results()

    def test_add_qa_metrics_accumulates_and_combines(self, two_points):
        two_points.add_qa_metrics([
            MetricResult("f1", 0.5, MetricType.SCORE_0_1, category="1"),
            MetricResult("exact_match", 0.0, MetricType.SCORE_0_1, category="1"),
        ])
        two_points.add_qa_metrics([
            MetricResult("f1", 1.0, MetricType.SCORE_0_1, category="2"),
        ])
        combined = two_points.get_combined_results()
        assert set(combined["overall"]) == COMPRESSION_METRIC_NAMES | {"f1", "exact_match"}
        assert combined["overall"]["f1"]["count"] == 2
        assert combined["qa_summary"]["f1_score"] == pytest.approx(0.75)
        assert combined["qa_summary"]["accuracy"] == 0.0
        assert set(combined["by_category"]) == {"category_1", "category_2"}
        assert combined["by_category"]["category_2"]["f1"]["mean"] == 1.0
        assert combined["compression_summary"]["avg_goal_drift"] == pytest.approx(0.125)

    def test_qa_metrics_do_not_leak_into_unified_results(self, two_points):
        two_points.add_qa_metrics([MetricResult("f1", 0.5, MetricType.SCORE_0_1)])
        assert "f1" not in two_points.get_unified_results()["overall"]
