"""
Unit tests for evaluation/qa_metrics.py.

Covers QAMetricCalculator's adapter logic (metric naming, empty-input handling,
exact match, token F1, mapping of helper outputs into MetricResults, batch
behavior) and the lazy A-mem utils loader. The heavy A-mem helpers
(ROUGE/BLEU/BERTScore/METEOR/SBERT) are replaced with deterministic stubs, and
loader tests point at tiny fake A-mem directories under tmp_path, so no models,
NLTK data, network, or API keys are needed.
"""

import sys
import types

import pytest

from evaluation import qa_metrics
from evaluation.metric_interfaces import MetricResult, MetricType
from evaluation.qa_metrics import QAMetricCalculator

EXPECTED_NAMES = [
    "exact_match",
    "f1",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "meteor",
    "sbert_similarity",
]

STUB_VALUES = {
    "rouge1_f": 0.11,
    "rouge2_f": 0.12,
    "rougeL_f": 0.13,
    "bleu1": 0.21,
    "bleu2": 0.22,
    "bleu3": 0.23,
    "bleu4": 0.24,
    "bert_precision": 0.31,
    "bert_recall": 0.32,
    "bert_f1": 0.33,
    "meteor": 0.41,
    "sbert_similarity": 0.51,
}

LOADER_GLOBALS = [
    "_utils_loaded",
    "_calculate_rouge_scores",
    "_calculate_bleu_scores",
    "_calculate_bert_scores",
    "_calculate_meteor_score",
    "_calculate_sentence_similarity",
    "_simple_tokenize",
]

FAKE_UTILS_SRC = '''
from load_dataset import record_load

record_load()

def simple_tokenize(text):
    return str(text).lower().split()

def calculate_rouge_scores(prediction, reference):
    return {"rouge1_f": 0.1, "rouge2_f": 0.2, "rougeL_f": 0.3}

def calculate_bleu_scores(prediction, reference):
    return {"bleu1": 0.4, "bleu2": 0.5, "bleu3": 0.6, "bleu4": 0.7}

def calculate_bert_scores(prediction, reference):
    return {"bert_precision": 0.8, "bert_recall": 0.9, "bert_f1": 0.85}

def calculate_meteor_score(prediction, reference):
    return 0.65

def calculate_sentence_similarity(prediction, reference):
    return 0.75
'''

FAKE_LOAD_DATASET_SRC = '''
LOADS = []

def record_load():
    LOADS.append(1)
'''


class HelperStubs:
    """Deterministic stand-ins for the A-mem metric helpers that record calls."""

    def __init__(self):
        self.calls = []

    def simple_tokenize(self, text):
        text = str(text).lower()
        for ch in ".,!?":
            text = text.replace(ch, " ")
        return text.split()

    def rouge(self, prediction, reference):
        self.calls.append(("rouge", prediction, reference))
        return {k: STUB_VALUES[k] for k in ("rouge1_f", "rouge2_f", "rougeL_f")}

    def bleu(self, prediction, reference):
        self.calls.append(("bleu", prediction, reference))
        return {k: STUB_VALUES[k] for k in ("bleu1", "bleu2", "bleu3", "bleu4")}

    def bert(self, prediction, reference):
        self.calls.append(("bert", prediction, reference))
        return {k: STUB_VALUES[k] for k in ("bert_precision", "bert_recall", "bert_f1")}

    def meteor(self, prediction, reference):
        self.calls.append(("meteor", prediction, reference))
        return STUB_VALUES["meteor"]

    def sbert(self, prediction, reference):
        self.calls.append(("sbert", prediction, reference))
        return STUB_VALUES["sbert_similarity"]


@pytest.fixture
def stubs(monkeypatch):
    s = HelperStubs()
    monkeypatch.setattr(qa_metrics, "_utils_loaded", True)
    monkeypatch.setattr(qa_metrics, "_simple_tokenize", s.simple_tokenize)
    monkeypatch.setattr(qa_metrics, "_calculate_rouge_scores", s.rouge)
    monkeypatch.setattr(qa_metrics, "_calculate_bleu_scores", s.bleu)
    monkeypatch.setattr(qa_metrics, "_calculate_bert_scores", s.bert)
    monkeypatch.setattr(qa_metrics, "_calculate_meteor_score", s.meteor)
    monkeypatch.setattr(qa_metrics, "_calculate_sentence_similarity", s.sbert)
    return s


@pytest.fixture
def calc():
    return QAMetricCalculator()


@pytest.fixture
def fresh_loader(monkeypatch):
    """Reset loader state and undo any sys.path/sys.modules changes afterwards."""
    for name in LOADER_GLOBALS:
        monkeypatch.setattr(qa_metrics, name, False if name == "_utils_loaded" else None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    tracked = (qa_metrics._AMEM_UTILS_MODULE, "load_dataset", "utils")
    saved = {mod: sys.modules.pop(mod) for mod in tracked if mod in sys.modules}
    yield
    for mod in tracked:
        sys.modules.pop(mod, None)
    sys.modules.update(saved)


@pytest.fixture
def fake_amem(tmp_path, monkeypatch, fresh_loader):
    amem_dir = tmp_path / "A-mem"
    amem_dir.mkdir()
    (amem_dir / "utils.py").write_text(FAKE_UTILS_SRC)
    (amem_dir / "load_dataset.py").write_text(FAKE_LOAD_DATASET_SRC)
    monkeypatch.setattr(qa_metrics, "_amem_path", amem_dir)
    return amem_dir


def by_name(results):
    return {r.name: r for r in results}


class TestMetricNames:
    def test_order_and_completeness(self, calc):
        assert calc.metric_names == EXPECTED_NAMES
        assert len(calc.metric_names) == 14
        assert len(set(calc.metric_names)) == 14

    def test_matches_class_constant(self, calc):
        assert calc.metric_names is QAMetricCalculator.QA_METRIC_NAMES


class TestEmptyInputs:
    @pytest.mark.parametrize(
        "prediction,reference",
        [("", "Paris"), ("Paris", ""), ("", ""), (None, "Paris"), ("Paris", None)],
    )
    def test_all_metrics_zero(self, stubs, calc, prediction, reference):
        results = calc.calculate(prediction, reference)
        assert [r.name for r in results] == EXPECTED_NAMES
        assert all(r.value == 0.0 for r in results)
        assert all(r.metric_type is MetricType.SCORE_0_1 for r in results)
        assert all(r.category is None for r in results)
        assert stubs.calls == []

    def test_category_propagated_as_string(self, stubs, calc):
        results = calc.calculate("", "Paris", category=3)
        assert len(results) == 14
        assert all(r.category == "3" for r in results)

    def test_category_zero_is_kept(self, stubs, calc):
        results = calc.calculate("", "Paris", category=0)
        assert all(r.category == "0" for r in results)

    def test_whitespace_only_is_not_treated_as_empty(self, stubs, calc):
        # The emptiness check runs before strip(), so "  " goes through the
        # full metric path rather than the all-zero short-circuit.
        results = by_name(calc.calculate("   ", "Paris"))
        assert results["exact_match"].value == 0
        assert results["f1"].value == 0.0
        assert results["rouge1_f"].value == STUB_VALUES["rouge1_f"]
        assert stubs.calls[0] == ("rouge", "", "Paris")


class TestExactMatch:
    @pytest.mark.parametrize(
        "prediction,reference",
        [
            ("Paris", "Paris"),
            ("paris", "PARIS"),
            ("  Paris\n", "paris"),
            ("\tThe Eiffel Tower ", "the eiffel tower"),
        ],
    )
    def test_match_ignores_case_and_outer_whitespace(self, stubs, calc, prediction, reference):
        assert by_name(calc.calculate(prediction, reference))["exact_match"].value == 1

    @pytest.mark.parametrize(
        "prediction,reference",
        [("Paris", "London"), ("Paris France", "Paris"), ("Par is", "Paris")],
    )
    def test_mismatch(self, stubs, calc, prediction, reference):
        assert by_name(calc.calculate(prediction, reference))["exact_match"].value == 0

    def test_value_is_int(self, stubs, calc):
        value = by_name(calc.calculate("a", "a"))["exact_match"].value
        assert type(value) is int

    def test_inputs_are_stripped_before_helpers(self, stubs, calc):
        calc.calculate("  Paris  ", "\tparis\n")
        assert {(p, r) for _, p, r in stubs.calls} == {("Paris", "paris")}


class TestTokenF1:
    def f1(self, calc, prediction, reference):
        return by_name(calc.calculate(prediction, reference))["f1"].value

    def test_full_overlap(self, stubs, calc):
        assert self.f1(calc, "red blue", "Blue red") == pytest.approx(1.0)

    def test_partial_overlap(self, stubs, calc):
        # pred {the, cat, sat}, ref {the, cat, ran, away}: P=2/3, R=2/4.
        expected = 2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))
        assert self.f1(calc, "the cat sat", "the cat ran away") == pytest.approx(expected)

    def test_no_overlap(self, stubs, calc):
        assert self.f1(calc, "alpha beta", "gamma delta") == 0.0

    def test_tokens_are_deduplicated(self, stubs, calc):
        # Sets, not multisets: repeated tokens don't change precision.
        assert self.f1(calc, "yes yes yes", "yes") == pytest.approx(1.0)

    @pytest.mark.parametrize("prediction,reference", [("...", "Paris"), ("Paris", "?!"), (".", ",")])
    def test_empty_token_set_gives_zero(self, stubs, calc, prediction, reference):
        assert self.f1(calc, prediction, reference) == 0.0

    def test_uses_injected_tokenizer(self, stubs, calc, monkeypatch):
        monkeypatch.setattr(qa_metrics, "_simple_tokenize", lambda text: ["same"])
        assert self.f1(calc, "anything", "different") == pytest.approx(1.0)


class TestFullResults:
    def test_names_and_order(self, stubs, calc):
        results = calc.calculate("Paris", "Paris, France")
        assert [r.name for r in results] == EXPECTED_NAMES

    @pytest.mark.parametrize("category,expected", [(None, None), (1, "1"), (5, "5")])
    def test_type_and_category(self, stubs, calc, category, expected):
        results = calc.calculate("Paris", "Paris, France", category=category)
        assert all(isinstance(r, MetricResult) for r in results)
        assert all(r.metric_type is MetricType.SCORE_0_1 for r in results)
        assert all(r.category == expected for r in results)

    def test_helper_outputs_mapped_by_key(self, stubs, calc):
        results = by_name(calc.calculate("Paris", "Paris, France"))
        for name, value in STUB_VALUES.items():
            assert results[name].value == value

    def test_each_helper_called_once_with_stripped_inputs(self, stubs, calc):
        calc.calculate(" Paris ", " Paris, France ")
        assert stubs.calls == [
            (kind, "Paris", "Paris, France")
            for kind in ("rouge", "bleu", "bert", "meteor", "sbert")
        ]

    def test_helper_dict_keys_pass_through_verbatim(self, stubs, calc, monkeypatch):
        monkeypatch.setattr(
            qa_metrics, "_calculate_rouge_scores", lambda p, r: {"rougeLsum_f": 0.9}
        )
        names = [r.name for r in calc.calculate("a", "a")]
        assert "rougeLsum_f" in names
        assert "rouge1_f" not in names

    def test_non_string_inputs_are_coerced(self, stubs, calc):
        results = by_name(calc.calculate(42, "42"))
        assert results["exact_match"].value == 1
        assert stubs.calls[0] == ("rouge", "42", "42")

    def test_kwargs_are_ignored(self, stubs, calc):
        results = calc.calculate("a", "a", question="q?", extra=1)
        assert len(results) == 14


class TestCalculateBatch:
    def test_without_categories(self, stubs, calc):
        batch = calc.calculate_batch(["Paris", "Rome"], ["paris", "Madrid"])
        assert len(batch) == 2
        assert [by_name(r)["exact_match"].value for r in batch] == [1, 0]
        assert all(r.category is None for results in batch for r in results)

    def test_with_categories(self, stubs, calc):
        batch = calc.calculate_batch(["a", "b", ""], ["a", "c", "d"], categories=[1, 2, 4])
        assert [{r.category for r in results} for results in batch] == [{"1"}, {"2"}, {"4"}]
        assert all(len(results) == 14 for results in batch)

    def test_matches_individual_calculate(self, stubs, calc):
        preds, refs, cats = ["the cat", "dog"], ["the cat sat", "dog"], [2, 3]
        batch = calc.calculate_batch(preds, refs, categories=cats)
        single = [calc.calculate(p, r, category=c) for p, r, c in zip(preds, refs, cats)]
        assert batch == single

    def test_empty_batch(self, stubs, calc):
        assert calc.calculate_batch([], []) == []

    # zip() truncates to the shortest input, so mismatched lengths silently
    # drop the trailing pairs instead of raising.
    def test_mismatched_references_truncate(self, stubs, calc):
        batch = calc.calculate_batch(["a", "b", "c"], ["a", "b"])
        assert len(batch) == 2

    def test_mismatched_predictions_truncate(self, stubs, calc):
        batch = calc.calculate_batch(["a"], ["a", "b", "c"])
        assert len(batch) == 1

    def test_short_categories_truncate(self, stubs, calc):
        batch = calc.calculate_batch(["a", "b", "c"], ["a", "b", "c"], categories=[1])
        assert len(batch) == 1
        assert {r.category for r in batch[0]} == {"1"}


class TestLoader:
    def test_default_path_points_at_vendored_amem(self):
        assert qa_metrics._amem_path.parts[-2:] == ("data", "A-mem")
        assert (qa_metrics._amem_path / "utils.py").is_file()

    def test_importing_module_does_not_touch_sys_path(self):
        assert str(qa_metrics._amem_path) not in sys.path

    def test_loads_helpers_from_amem_dir(self, fake_amem):
        qa_metrics._load_amem_utils()
        assert qa_metrics._utils_loaded is True
        assert qa_metrics._calculate_meteor_score("a", "b") == 0.65
        assert qa_metrics._simple_tokenize("A b") == ["a", "b"]
        assert sys.modules[qa_metrics._AMEM_UTILS_MODULE].__file__ == str(fake_amem / "utils.py")

    def test_amem_dir_appended_for_sibling_imports(self, fake_amem):
        qa_metrics._load_amem_utils()
        assert sys.path[-1] == str(fake_amem)
        assert sys.path.count(str(fake_amem)) == 1

    def test_ignores_unrelated_utils_module(self, fake_amem, monkeypatch):
        decoy = types.ModuleType("utils")
        monkeypatch.setitem(sys.modules, "utils", decoy)
        qa_metrics._load_amem_utils()
        assert qa_metrics._calculate_sentence_similarity("a", "b") == 0.75
        assert sys.modules["utils"] is decoy

    def test_loads_only_once(self, fake_amem, calc):
        qa_metrics._load_amem_utils()
        qa_metrics._load_amem_utils()
        calc.calculate("a", "a")
        calc.calculate_batch(["a", "b"], ["a", "b"])
        assert sys.modules["load_dataset"].LOADS == [1]

    def test_end_to_end_with_fake_amem(self, fake_amem, calc):
        results = by_name(calc.calculate("Paris", "paris", category=2))
        assert results["exact_match"].value == 1
        assert results["f1"].value == pytest.approx(1.0)
        assert results["bleu4"].value == 0.7
        assert results["sbert_similarity"].value == 0.75
        assert {r.category for r in results.values()} == {"2"}

    def test_missing_utils_file(self, tmp_path, monkeypatch, fresh_loader, calc):
        monkeypatch.setattr(qa_metrics, "_amem_path", tmp_path / "nope")
        with pytest.raises(ImportError, match=r"Could not import A-mem utils.*utils\.py not found"):
            calc.calculate("a", "a")
        assert qa_metrics._utils_loaded is False

    def test_missing_dependency(self, fake_amem, calc):
        (fake_amem / "utils.py").write_text("import _definitely_not_installed_dep\n")
        with pytest.raises(ImportError, match="Could not import A-mem utils") as exc:
            calc.calculate("a", "a")
        assert "_definitely_not_installed_dep" in str(exc.value)
        assert isinstance(exc.value.__cause__, ModuleNotFoundError)
        assert qa_metrics._AMEM_UTILS_MODULE not in sys.modules
        assert qa_metrics._utils_loaded is False

    def test_missing_helper_function(self, fake_amem, calc):
        (fake_amem / "utils.py").write_text("def simple_tokenize(text):\n    return []\n")
        with pytest.raises(ImportError, match="calculate_rouge_scores"):
            calc.calculate("a", "a")
        assert qa_metrics._utils_loaded is False

    def test_retries_after_failure(self, fake_amem, calc):
        (fake_amem / "utils.py").write_text("import _definitely_not_installed_dep\n")
        with pytest.raises(ImportError):
            qa_metrics._load_amem_utils()
        (fake_amem / "utils.py").write_text(FAKE_UTILS_SRC)
        qa_metrics._load_amem_utils()
        assert qa_metrics._utils_loaded is True

    def test_empty_inputs_still_require_loader(self, tmp_path, monkeypatch, fresh_loader, calc):
        monkeypatch.setattr(qa_metrics, "_amem_path", tmp_path / "nope")
        with pytest.raises(ImportError):
            calc.calculate("", "")
