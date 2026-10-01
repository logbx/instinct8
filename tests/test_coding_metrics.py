"""
Unit tests for evaluation/coding_metrics.py.

Covers the deterministic helpers on CodingMetricCalculator (syntax checks, AST
and text similarity, token efficiency, requirements/test heuristics,
compression retention, diff accuracy) and the calculate() orchestration.
Every calculator is built with use_embeddings=False; the embedding helper is
exercised only through a stub `sentence_transformers` module or a forced
ImportError, so no model download, network, API keys, or LLM calls happen.
Node is never invoked: the JavaScript branch runs against a stubbed
subprocess.run.
"""

import ast
import difflib
import json
import subprocess
import sys
import types

import pytest

from evaluation import coding_metrics
from evaluation.coding_metrics import (
    CODING_METRICS,
    CodingEvalInput,
    CodingMetricCalculator,
)
from evaluation.coding_metrics import TestCase as CodingTestCase
from evaluation.metric_interfaces import MetricResult, MetricType


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _names(results):
    return [r.name for r in results]


def _by_name(results):
    return {r.name: r for r in results}


def _stub_module_ast_parse(monkeypatch, parse):
    # Replace only coding_metrics' `ast` reference; patching ast.parse globally breaks pytest.
    stub = types.SimpleNamespace(parse=parse, walk=ast.walk, AST=ast.AST)
    monkeypatch.setattr(coding_metrics, "ast", stub)


@pytest.fixture
def calc():
    return CodingMetricCalculator(use_embeddings=False)


VALID_PY = "def add(a, b):\n    return a + b\n"
INVALID_PY = "def add(a, b)\n    return a + b\n"


class TestContract:
    def test_coding_metrics_list(self):
        assert CODING_METRICS == [
            "syntax_validity",
            "test_pass_rate",
            "ast_similarity",
            "code_embedding_similarity",
            "requirements_met",
            "token_efficiency",
            "compression_retention",
            "diff_accuracy",
        ]
        assert len(set(CODING_METRICS)) == len(CODING_METRICS)

    def test_metric_names_matches_module_list(self, calc):
        assert calc.metric_names == CODING_METRICS

    def test_every_emitted_metric_is_declared(self, calc):
        results = calc.calculate(
            generated_code=VALID_PY,
            expected_code=VALID_PY,
            expected_files={"a.py": VALID_PY},
            test_cases=[{"name": "t", "expected_output": "add return"}],
            acceptance_criteria=["add numbers"],
            task_type="refactoring",
            tokens_before_compression=100,
            tokens_after_compression=50,
        )
        assert set(_names(results)) <= set(calc.metric_names)

    def test_constructor_defaults_and_lazy_encoder(self):
        default = CodingMetricCalculator()
        assert default._use_embeddings is True
        assert default._embedding_model == "all-MiniLM-L6-v2"
        assert default._encoder is None

    def test_constructor_stores_options(self):
        c = CodingMetricCalculator(use_embeddings=False, embedding_model="custom-model")
        assert c._use_embeddings is False
        assert c._embedding_model == "custom-model"
        assert c._encoder is None


class TestDataclasses:
    def test_test_case_defaults(self):
        tc = CodingTestCase(name="n", input="i", expected_output="o")
        assert (tc.name, tc.input, tc.expected_output, tc.description) == ("n", "i", "o", None)

    def test_coding_eval_input_defaults(self):
        inp = CodingEvalInput(generated_code="x = 1")
        assert inp.generated_code == "x = 1"
        assert inp.expected_code is None
        assert inp.expected_files is None
        assert inp.test_cases is None
        assert inp.acceptance_criteria is None
        assert inp.language == "python"
        assert inp.task_type == "code_generation"


class TestCheckSyntaxValidity:
    def test_valid_python(self, calc):
        assert calc.check_syntax_validity(VALID_PY) == (True, [])

    def test_empty_python_is_valid(self, calc):
        assert calc.check_syntax_validity("") == (True, [])

    def test_default_language_is_python(self, calc):
        # Balanced brackets but not Python: only the Python path rejects it.
        go_code = "func main() { x := 1 }"
        assert calc.check_syntax_validity(go_code)[0] is False
        assert calc.check_syntax_validity(go_code, "go") == (True, [])

    def test_invalid_python_reports_line_and_message(self, calc):
        valid, errors = calc.check_syntax_validity("x = 1\nif True\n    pass\n")
        assert valid is False
        assert len(errors) == 1
        assert errors[0].startswith("Line 2: ")
        assert len(errors[0]) > len("Line 2: ")

    def test_indentation_error_is_invalid(self, calc):
        valid, errors = calc.check_syntax_validity("def f():\nreturn 1\n")
        assert valid is False
        assert errors and errors[0].startswith("Line ")

    def test_null_bytes_are_invalid_not_raised(self, calc):
        # Python < 3.11.4 raises ValueError here instead of SyntaxError.
        valid, errors = calc.check_syntax_validity("x = 1\x00")
        assert valid is False
        assert len(errors) == 1
        assert "null bytes" in errors[0]

    def test_ast_value_error_is_invalid_on_any_python(self, calc, monkeypatch):
        def raise_value_error(source):
            raise ValueError("source code string cannot contain null bytes")

        _stub_module_ast_parse(monkeypatch, raise_value_error)
        assert calc.check_syntax_validity("x = 1") == (
            False,
            ["source code string cannot contain null bytes"],
        )

    @pytest.mark.parametrize("language", ["go", "rust", "java", "unknown"])
    def test_other_languages_use_bracket_heuristic(self, calc, language):
        assert calc.check_syntax_validity("fn() { a[0] }", language) == (True, [])
        assert calc.check_syntax_validity("fn() { a[0 }", language) == (False, [])


class TestJavaScriptSyntaxPath:
    """The JS/TS branch shells out to node; stub subprocess.run so node never runs."""

    @pytest.fixture
    def fake_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(coding_metrics.tempfile, "tempdir", str(tmp_path))
        calls = []

        def install(behavior):
            def _run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                return behavior(cmd)

            monkeypatch.setattr(coding_metrics.subprocess, "run", _run)
            return calls

        return install

    @pytest.mark.parametrize("language", ["javascript", "typescript"])
    def test_node_success(self, calc, fake_run, language):
        calls = fake_run(lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""))
        assert calc.check_syntax_validity("const x = 1;", language) == (True, [])
        cmd, kwargs = calls[0]
        assert cmd[:2] == ["node", "--check"]
        assert cmd[2].endswith(".js")
        assert kwargs["timeout"] == 5

    def test_node_failure_returns_stderr(self, calc, fake_run):
        fake_run(lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "SyntaxError: boom"))
        assert calc.check_syntax_validity("const = ;", "javascript") == (
            False,
            ["SyntaxError: boom"],
        )

    def test_code_is_written_to_checked_file(self, calc, fake_run):
        seen = {}

        def behavior(cmd):
            with open(cmd[2]) as fh:
                seen["content"] = fh.read()
            return subprocess.CompletedProcess(cmd, 0, "", "")

        fake_run(behavior)
        calc.check_syntax_validity("let y = [1, 2];", "javascript")
        assert seen["content"] == "let y = [1, 2];"

    @pytest.mark.parametrize(
        "exc",
        [FileNotFoundError("node"), subprocess.TimeoutExpired(cmd="node", timeout=5)],
    )
    def test_missing_or_slow_node_falls_back_to_heuristic(self, calc, fake_run, exc):
        def behavior(cmd):
            raise exc

        fake_run(behavior)
        assert calc.check_syntax_validity("f({a: [1]})", "javascript") == (True, [])
        assert calc.check_syntax_validity("f({a: [1)})", "javascript") == (False, [])


class TestBasicSyntaxCheck:
    @pytest.mark.parametrize(
        "code",
        [
            "",
            "no brackets at all",
            "()[]{}",
            "f(a[1], {b: 2})",
            "{[()()]}",
            'print(")")',
            "s = '(['",
            'x = "(" + ")"',
            'f("it\'s")',
            "d = {'k': \"v]\"}",
        ],
    )
    def test_balanced(self, calc, code):
        assert calc._basic_syntax_check(code) is True

    @pytest.mark.parametrize(
        "code",
        [
            "(",
            ")",
            "(]",
            "([)]",
            "{",
            "f(a[1)]",
            "())",
            '"a" (',
        ],
    )
    def test_unbalanced_or_mismatched(self, calc, code):
        assert calc._basic_syntax_check(code) is False


class TestAstSimilarity:
    def test_identical_code(self, calc):
        assert calc.calculate_ast_similarity(VALID_PY, VALID_PY) == 1.0

    def test_same_structure_different_identifiers(self, calc):
        # Only node *types* are compared, so renames and literal changes don't count.
        assert calc.calculate_ast_similarity("x = 1", "y = 2") == 1.0

    def test_jaccard_of_node_types(self, calc):
        # "x = 1" -> {Module, Assign, Name, Store, Constant}
        # "x"     -> {Module, Expr, Name, Load}
        assert calc.calculate_ast_similarity("x = 1", "x") == pytest.approx(2 / 7)

    def test_symmetric(self, calc):
        a = "x = 1"
        b = "def f():\n    return [i for i in range(3)]\n"
        assert calc.calculate_ast_similarity(a, b) == calc.calculate_ast_similarity(b, a)

    def test_structurally_different_is_strictly_between(self, calc):
        a = "x = 1"
        b = "class C:\n    def m(self):\n        while True:\n            break\n"
        sim = calc.calculate_ast_similarity(a, b)
        assert 0.0 < sim < 0.5

    def test_empty_modules(self, calc):
        assert calc.calculate_ast_similarity("", "") == 1.0
        assert calc.calculate_ast_similarity("", "x = 1") == pytest.approx(1 / 5)

    @pytest.mark.parametrize(
        "generated, expected",
        [(INVALID_PY, VALID_PY), (VALID_PY, INVALID_PY), (INVALID_PY, INVALID_PY + "x")],
    )
    def test_invalid_syntax_falls_back_to_text_similarity(self, calc, generated, expected):
        assert calc.calculate_ast_similarity(generated, expected) == pytest.approx(
            _ratio(generated, expected)
        )

    def test_null_bytes_fall_back_to_text_similarity(self, calc):
        a, b = "x = 1\x00", "x = 1"
        assert calc.calculate_ast_similarity(a, b) == pytest.approx(_ratio(a, b))

    def test_ast_value_error_falls_back_on_any_python(self, calc, monkeypatch):
        def raise_value_error(source):
            raise ValueError("source code string cannot contain null bytes")

        _stub_module_ast_parse(monkeypatch, raise_value_error)
        assert calc.calculate_ast_similarity("x = 1", "y = 2") == pytest.approx(0.6)

    def test_non_python_uses_text_similarity(self, calc):
        assert calc.calculate_ast_similarity("x = 1", "y = 2", "javascript") == pytest.approx(0.6)
        assert calc.calculate_ast_similarity("same", "same", "go") == 1.0

    def test_text_similarity_helper(self, calc):
        assert calc._text_similarity("abc", "abc") == 1.0
        assert calc._text_similarity("abc", "xyz") == 0.0
        assert calc._text_similarity("", "") == 1.0


class TestTokenEfficiency:
    def test_normal_ratio(self, calc):
        assert calc.calculate_token_efficiency(1000, 250) == pytest.approx(0.25)

    def test_no_compression(self, calc):
        assert calc.calculate_token_efficiency(400, 400) == 1.0

    def test_full_compression(self, calc):
        assert calc.calculate_token_efficiency(400, 0) == 0.0

    @pytest.mark.parametrize("after", [0, 5])
    def test_zero_before_returns_one(self, calc, after):
        assert calc.calculate_token_efficiency(0, after) == 1.0

    def test_expansion_is_not_clamped(self, calc):
        assert calc.calculate_token_efficiency(100, 150) == pytest.approx(1.5)


class TestRequirementsMet:
    CODE = (
        "def parse_config(path):\n"
        "    validate(path)\n"
        "    return load_json(path)\n"
    )

    def test_empty_criteria(self, calc):
        assert calc.calculate_requirements_met(self.CODE, []) == 1.0

    def test_all_met(self, calc):
        assert calc.calculate_requirements_met(self.CODE, ["validate path", "load json"]) == 1.0

    def test_none_met(self, calc):
        assert calc.calculate_requirements_met(self.CODE, ["retry network", "cache results"]) == 0.0

    def test_fraction_met(self, calc):
        criteria = ["Must validate the path", "Should be fast", "load json", "uses redis"]
        assert calc.calculate_requirements_met(self.CODE, criteria) == pytest.approx(0.5)

    def test_stopwords_are_ignored(self, calc):
        # Only "path" survives stopword filtering; without the filter this would be 1/9.
        assert calc.calculate_requirements_met(self.CODE, ["the path is a must be should are an"]) == 1.0

    def test_half_of_keywords_is_enough(self, calc):
        assert calc.calculate_requirements_met(self.CODE, ["validate cache"]) == 1.0

    def test_below_half_of_keywords_fails(self, calc):
        assert calc.calculate_requirements_met(self.CODE, ["validate cache retry"]) == 0.0

    def test_case_insensitive(self, calc):
        assert calc.calculate_requirements_met(self.CODE.upper(), ["Validate PATH"]) == 1.0

    def test_keywords_match_as_substrings(self, calc):
        assert calc.calculate_requirements_met("width = 3", ["id"]) == 1.0

    def test_stopword_only_criterion_counts_as_unmet(self, calc):
        assert calc.calculate_requirements_met(self.CODE, ["should be", "path"]) == pytest.approx(0.5)


class TestTestPassRate:
    def test_no_test_cases(self, calc):
        assert calc.calculate_test_pass_rate("x = 1", []) == (1.0, [])

    def test_mixed_results(self, calc):
        code = "total = sum(xs)\n"
        cases = [
            CodingTestCase(name="sums", input="[1, 2]", expected_output="sum total"),
            CodingTestCase(name="raises", input="None", expected_output="raises ValueError"),
            CodingTestCase(name="half", input="", expected_output="sum missing"),
            CodingTestCase(name="empty", input="", expected_output=""),
        ]
        rate, results = calc.calculate_test_pass_rate(code, cases)
        assert rate == pytest.approx(0.25)
        assert [r["name"] for r in results] == ["sums", "raises", "half", "empty"]
        assert [r["passed"] for r in results] == [True, False, False, False]
        assert results[0]["reason"] == "Code contains expected patterns"
        assert results[1]["reason"] == "Missing expected patterns"
        assert results[3]["reason"] == ""


class TestCompressionRetention:
    def test_with_expected_identical_code(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 50) == pytest.approx(0.75)

    def test_no_compression_halves_quality(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 100) == pytest.approx(0.5)

    def test_full_compression_returns_quality(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 0) == 1.0
        assert calc.calculate_compression_retention(INVALID_PY, None, 100, 0) == 0.5

    def test_zero_tokens_before_treated_as_no_compression(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, None, 0, 10) == pytest.approx(0.5)

    def test_quality_uses_ast_similarity_when_expected_given(self, calc):
        quality = calc.calculate_ast_similarity("x = 1", "x")
        assert calc.calculate_compression_retention("x = 1", "x", 100, 50) == pytest.approx(
            quality * 1.5 / 2
        )

    def test_invalid_generated_with_expected_uses_text_fallback(self, calc):
        quality = _ratio(INVALID_PY, VALID_PY)
        assert calc.calculate_compression_retention(INVALID_PY, VALID_PY, 100, 50) == pytest.approx(
            quality * 1.5 / 2
        )

    def test_without_expected_valid_code(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, None, 100, 50) == pytest.approx(0.75)

    def test_without_expected_invalid_code_uses_half_quality(self, calc):
        assert calc.calculate_compression_retention(INVALID_PY, None, 100, 50) == pytest.approx(0.375)

    def test_empty_expected_treated_as_missing(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, "", 100, 50) == pytest.approx(0.75)

    def test_large_expansion_clamps_to_zero(self, calc):
        assert calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 300) == 0.0

    def test_more_compression_scores_higher(self, calc):
        light = calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 90)
        heavy = calc.calculate_compression_retention(VALID_PY, VALID_PY, 100, 10)
        assert heavy > light


class TestDiffAccuracy:
    def test_empty_expected_files(self, calc):
        assert calc.calculate_diff_accuracy("anything", {}) == 1.0

    def test_single_identical_file(self, calc):
        assert calc.calculate_diff_accuracy(VALID_PY, {"a.py": VALID_PY}) == 1.0

    def test_average_across_files(self, calc):
        assert calc.calculate_diff_accuracy("abc", {"a.py": "abc", "b.py": "xyz"}) == pytest.approx(0.5)

    def test_matches_mean_of_text_ratios(self, calc):
        generated = "def f():\n    return 2\n"
        files = {"a.py": "def f():\n    return 1\n", "b.py": "class C: pass\n", "c.py": ""}
        expected = sum(_ratio(generated, v) for v in files.values()) / len(files)
        assert calc.calculate_diff_accuracy(generated, files) == pytest.approx(expected)


def _stub_sentence_transformers(monkeypatch, similarity):
    """Install a fake sentence_transformers package; returns a record of constructions."""
    constructed = []

    class FakeEncoder:
        def __init__(self, model_name):
            constructed.append(model_name)

        def encode(self, text, convert_to_tensor=False):
            return text

    class FakeScore:
        def item(self):
            return similarity

    pkg = types.ModuleType("sentence_transformers")
    pkg.SentenceTransformer = FakeEncoder
    util = types.ModuleType("sentence_transformers.util")
    util.cos_sim = lambda a, b: FakeScore()
    pkg.util = util
    monkeypatch.setitem(sys.modules, "sentence_transformers", pkg)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util)
    return constructed


class TestCodeEmbeddingSimilarity:
    def test_import_error_falls_back_to_text_similarity(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        c = CodingMetricCalculator()
        assert c.calculate_code_embedding_similarity("x = 1", "y = 2") == pytest.approx(0.6)
        assert c._encoder is None

    @pytest.mark.parametrize("raw, clamped", [(0.42, 0.42), (-0.3, 0.0), (1.7, 1.0)])
    def test_cosine_similarity_is_clamped(self, monkeypatch, raw, clamped):
        _stub_sentence_transformers(monkeypatch, raw)
        c = CodingMetricCalculator()
        assert c.calculate_code_embedding_similarity("a", "b") == pytest.approx(clamped)

    def test_encoder_is_loaded_once_with_configured_model(self, monkeypatch):
        constructed = _stub_sentence_transformers(monkeypatch, 0.5)
        c = CodingMetricCalculator(embedding_model="tiny-model")
        c.calculate_code_embedding_similarity("a", "b")
        c.calculate_code_embedding_similarity("c", "d")
        assert constructed == ["tiny-model"]


class TestCalculate:
    def test_minimal_input_only_syntax_validity(self, calc):
        results = calc.calculate(generated_code=VALID_PY)
        assert _names(results) == ["syntax_validity"]
        r = results[0]
        assert isinstance(r, MetricResult)
        assert r.value == 1.0
        assert r.metric_type == MetricType.SCORE_0_1
        assert r.metadata is None

    def test_invalid_code_records_errors(self, calc):
        (r,) = calc.calculate(generated_code=INVALID_PY)
        assert r.value == 0.0
        assert len(r.metadata["errors"]) == 1
        assert r.metadata["errors"][0].startswith("Line 1: ")

    def test_language_is_forwarded_to_syntax_check(self, calc):
        (r,) = calc.calculate(generated_code="func main() { x := 1 }", language="go")
        assert r.value == 1.0

    def test_expected_code_adds_ast_similarity_only_when_embeddings_disabled(self, calc):
        results = calc.calculate(generated_code="x = 1", expected_code="x")
        assert _names(results) == ["syntax_validity", "ast_similarity"]
        assert _by_name(results)["ast_similarity"].value == pytest.approx(2 / 7)

    def test_embeddings_disabled_never_touches_sentence_transformers(self, calc, monkeypatch):
        class Boom:
            def __init__(self, *a, **k):
                raise AssertionError("encoder must not load when use_embeddings=False")

        pkg = types.ModuleType("sentence_transformers")
        pkg.SentenceTransformer = Boom
        monkeypatch.setitem(sys.modules, "sentence_transformers", pkg)
        results = calc.calculate(generated_code=VALID_PY, expected_code=VALID_PY)
        assert "code_embedding_similarity" not in _names(results)
        assert calc._encoder is None

    def test_embeddings_enabled_adds_embedding_metric(self, monkeypatch):
        _stub_sentence_transformers(monkeypatch, 0.8)
        c = CodingMetricCalculator(use_embeddings=True)
        results = c.calculate(generated_code=VALID_PY, expected_code=VALID_PY)
        assert _names(results) == ["syntax_validity", "ast_similarity", "code_embedding_similarity"]
        assert _by_name(results)["code_embedding_similarity"].value == pytest.approx(0.8)

    def test_embeddings_enabled_import_error_uses_text_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        c = CodingMetricCalculator(use_embeddings=True)
        results = _by_name(c.calculate(generated_code="x = 1", expected_code="y = 2"))
        assert results["code_embedding_similarity"].value == pytest.approx(0.6)

    def test_embeddings_enabled_without_expected_code_skips_embedding(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        c = CodingMetricCalculator(use_embeddings=True)
        assert _names(c.calculate(generated_code=VALID_PY)) == ["syntax_validity"]

    def test_test_cases_dicts_are_converted(self, calc):
        results = calc.calculate(
            generated_code="total = sum(xs)",
            test_cases=[
                {"name": "via expected_output", "expected_output": "sum total"},
                {"name": "via expected", "expected": "total sum"},
                {"expected_output": "raises ValueError"},
            ],
        )
        r = _by_name(results)["test_pass_rate"]
        assert r.value == pytest.approx(2 / 3)
        assert r.metric_type == MetricType.SCORE_0_1
        assert [t["name"] for t in r.metadata["test_results"]] == [
            "via expected_output",
            "via expected",
            "",
        ]
        assert [t["passed"] for t in r.metadata["test_results"]] == [True, True, False]

    def test_acceptance_criteria_adds_requirements_met(self, calc):
        results = calc.calculate(
            generated_code=TestRequirementsMet.CODE,
            acceptance_criteria=["validate path", "uses redis"],
        )
        assert _names(results) == ["syntax_validity", "requirements_met"]
        assert _by_name(results)["requirements_met"].value == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"test_cases": []},
            {"acceptance_criteria": []},
            {"expected_code": ""},
            {"tokens_before_compression": 100},
            {"tokens_after_compression": 50},
            {"expected_files": {"a.py": VALID_PY}},
            {"expected_files": {"a.py": VALID_PY}, "task_type": "code_generation"},
            {"expected_files": {"a.py": VALID_PY}, "task_type": "research_synthesis"},
            {"expected_files": {}, "task_type": "refactoring"},
            {"task_type": "bug_fixing"},
        ],
    )
    def test_optional_metrics_absent_without_their_inputs(self, calc, kwargs):
        assert _names(calc.calculate(generated_code=VALID_PY, **kwargs)) == ["syntax_validity"]

    def test_token_counts_add_efficiency_and_retention(self, calc):
        results = calc.calculate(
            generated_code=VALID_PY,
            tokens_before_compression=200,
            tokens_after_compression=50,
        )
        assert _names(results) == ["syntax_validity", "token_efficiency", "compression_retention"]
        by = _by_name(results)
        assert by["token_efficiency"].value == pytest.approx(0.25)
        assert by["token_efficiency"].metric_type == MetricType.PERCENTAGE
        assert by["compression_retention"].value == pytest.approx(1.75 / 2)
        assert by["compression_retention"].metric_type == MetricType.SCORE_0_1

    def test_zero_token_counts_still_emit_token_metrics(self, calc):
        by = _by_name(
            calc.calculate(
                generated_code=VALID_PY,
                tokens_before_compression=0,
                tokens_after_compression=0,
            )
        )
        assert by["token_efficiency"].value == 1.0
        assert by["compression_retention"].value == pytest.approx(0.5)

    def test_retention_uses_expected_code_when_given(self, calc):
        by = _by_name(
            calc.calculate(
                generated_code="x = 1",
                expected_code="x",
                tokens_before_compression=100,
                tokens_after_compression=50,
            )
        )
        assert by["compression_retention"].value == pytest.approx(
            calc.calculate_compression_retention("x = 1", "x", 100, 50)
        )

    @pytest.mark.parametrize("task_type", ["refactoring", "bug_fixing"])
    def test_diff_accuracy_for_refactoring_and_bug_fixing(self, calc, task_type):
        results = calc.calculate(
            generated_code="abc",
            expected_files={"a.py": "abc", "b.py": "xyz"},
            task_type=task_type,
        )
        assert _names(results) == ["syntax_validity", "diff_accuracy"]
        assert _by_name(results)["diff_accuracy"].value == pytest.approx(0.5)

    def test_full_input_order_ranges_and_serialization(self, calc):
        results = calc.calculate(
            generated_code=VALID_PY,
            expected_code=VALID_PY,
            expected_files={"math.py": VALID_PY},
            test_cases=[{"name": "adds", "expected_output": "return a + b"}],
            acceptance_criteria=["add a and b"],
            language="python",
            task_type="bug_fixing",
            tokens_before_compression=1000,
            tokens_after_compression=400,
        )
        assert _names(results) == [
            "syntax_validity",
            "test_pass_rate",
            "ast_similarity",
            "requirements_met",
            "token_efficiency",
            "compression_retention",
            "diff_accuracy",
        ]
        for r in results:
            assert 0.0 <= r.value <= 1.0
            assert r.category is None
            json.dumps(r.to_dict())
        by = _by_name(results)
        assert by["ast_similarity"].value == 1.0
        assert by["diff_accuracy"].value == 1.0
        assert by["token_efficiency"].value == pytest.approx(0.4)
        assert by["compression_retention"].value == pytest.approx(0.8)

    def test_unknown_kwargs_are_ignored(self, calc):
        results = calc.calculate(generated_code=VALID_PY, sample_id="s1", strategy="F")
        assert _names(results) == ["syntax_validity"]
