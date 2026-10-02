"""
Unit tests for the pure dataset adapters in evaluation/datasets/.

Covers the shared dataclasses and BaseDataset.get_statistics() in base.py,
TemplateDataset (template_dataset.py), and CodingDataset plus its from_dict
helpers (coding_dataset.py). Every fixture is a JSON file written to pytest's
tmp_path, so no repository data, A-mem/LoCoMo loaders, network, or API keys
are involved.
"""

import json

import pytest

from evaluation.datasets.base import EvalQuestion, EvalSample, EvalTurn
from evaluation.datasets.coding_dataset import (
    CodingDataset,
    CodingGroundTruth,
    CodingSpecification,
    CodingTask,
    ToolCall,
)
from evaluation.datasets.coding_dataset import TestCase as CodingTestCase
from evaluation.datasets.template_dataset import TemplateDataset


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def _template(template_id="tmpl-1", **overrides):
    template = {
        "template_id": template_id,
        "initial_setup": {
            "original_goal": "Build a REST API",
            "hard_constraints": ["Use FastAPI", "No external DB"],
            "system_prompt": "You are a helpful assistant.",
        },
        "turns": [
            {"turn_id": 1, "role": "user", "content": "Start the API"},
            {
                "turn_id": 2,
                "role": "assistant",
                "content": "Running scaffold",
                "is_compression_point": True,
                "tool_call": {"name": "shell", "input": "mkdir api"},
            },
        ],
        "probing_tasks": {
            "goal_probe": "What is the goal?",
            "constraint_probe": "What are the constraints?",
            "behavioral_test": {
                "prompt": "Should we add Postgres?",
                "expected_behavior": "Decline: no external DB",
            },
        },
        "ground_truth": {"goal_coherence": 1.0},
        "metadata": {"difficulty": "easy"},
    }
    template.update(overrides)
    return template


def _coding_task(task_id="task-1", task_type="code_generation", **overrides):
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "specification": {
            "goal": "Implement add()",
            "requirements": ["Return the sum"],
            "constraints": ["No imports"],
            "context": "math utils",
        },
        "ground_truth": {
            "expected_code": "def add(a, b):\n    return a + b\n",
            "test_cases": [{"name": "t1", "input": "add(1, 2)", "expected_output": "3"}],
            "acceptance_criteria": ["Passes tests"],
        },
        "conversation": [
            {"turn_id": 1, "role": "user", "content": "Write add()"},
            {
                "turn_id": 2,
                "role": "assistant",
                "content": "Writing file",
                "timestamp": "2026-01-01T00:00:00Z",
                "is_compression_point": True,
                "tool_calls": [{"name": "write_file", "input": "add.py", "output": "ok"}],
            },
        ],
        "initial_files": {"README.md": "# utils"},
        "compression_triggers": {"turn_count": 2},
        "metadata": {"difficulty": "easy"},
    }
    task.update(overrides)
    return task


class TestEvalQuestionFinalAnswer:
    def test_category_5_with_adversarial_returns_adversarial(self):
        q = EvalQuestion(
            question="q", reference_answer="ref", category=5, adversarial_answer="adv"
        )
        assert q.final_answer == "adv"

    def test_category_5_without_adversarial_returns_reference(self):
        q = EvalQuestion(question="q", reference_answer="ref", category=5)
        assert q.final_answer == "ref"

    def test_category_5_with_empty_adversarial_returns_reference(self):
        q = EvalQuestion(
            question="q", reference_answer="ref", category=5, adversarial_answer=""
        )
        assert q.final_answer == "ref"

    @pytest.mark.parametrize("category", [None, 1, 2, 3, 4])
    def test_non_adversarial_category_ignores_adversarial_answer(self, category):
        q = EvalQuestion(
            question="q", reference_answer="ref", category=category, adversarial_answer="adv"
        )
        assert q.final_answer == "ref"


class TestBaseDataclassDefaults:
    def test_eval_turn_defaults(self):
        turn = EvalTurn(id=0, role="user", content="hi")
        assert turn.timestamp is None
        assert turn.speaker is None
        assert turn.is_compression_point is False
        assert turn.tool_call is None

    def test_eval_question_defaults(self):
        q = EvalQuestion(question="q", reference_answer="a")
        assert q.category is None
        assert q.evidence is None
        assert q.adversarial_answer is None

    def test_eval_sample_defaults(self):
        sample = EvalSample(sample_id="s", turns=[], questions=[])
        assert sample.metadata == {}
        assert sample.initial_goal is None
        assert sample.constraints is None
        assert sample.system_prompt is None
        assert sample.probing_tasks is None
        assert sample.ground_truth is None

    def test_eval_sample_metadata_not_shared_between_instances(self):
        a = EvalSample(sample_id="a", turns=[], questions=[])
        b = EvalSample(sample_id="b", turns=[], questions=[])
        a.metadata["k"] = "v"
        assert b.metadata == {}


class TestBaseGetStatistics:
    def test_statistics_via_template_dataset(self, tmp_path):
        first = _write_json(tmp_path / "a.json", _template("a"))
        second = _write_json(
            tmp_path / "b.json",
            _template("b", turns=[{"turn_id": 1, "role": "user", "content": "x"}], probing_tasks={}),
        )
        ds = TemplateDataset([first, second])
        assert ds.get_statistics() == {
            "name": "TemplateDataset(2 templates)",
            "evaluation_type": "compression",
            "num_samples": 2,
            "total_turns": 3,
            "total_questions": 3,
        }

    def test_statistics_empty_dataset(self, tmp_path):
        ds = TemplateDataset(tmp_path)
        assert ds.get_statistics() == {
            "name": "TemplateDataset(0 templates)",
            "evaluation_type": "compression",
            "num_samples": 0,
            "total_turns": 0,
            "total_questions": 0,
        }


class TestTemplateDatasetLoading:
    def test_load_single_file_str_path(self, tmp_path):
        path = _write_json(tmp_path / "t.json", _template())
        ds = TemplateDataset(str(path))
        assert len(ds) == 1
        assert ds[0].sample_id == "tmpl-1"

    def test_load_single_file_path_object(self, tmp_path):
        path = _write_json(tmp_path / "t.json", _template())
        ds = TemplateDataset(path)
        assert len(ds) == 1

    def test_load_directory_of_json_files(self, tmp_path):
        for tid in ("a", "b", "c"):
            _write_json(tmp_path / f"{tid}.json", _template(tid))
        (tmp_path / "notes.txt").write_text("ignored")
        ds = TemplateDataset(tmp_path)
        assert len(ds) == 3
        assert {s.sample_id for s in ds} == {"a", "b", "c"}

    def test_load_list_of_paths_preserves_order(self, tmp_path):
        paths = [_write_json(tmp_path / f"{tid}.json", _template(tid)) for tid in ("z", "a")]
        ds = TemplateDataset(paths)
        assert [s.sample_id for s in ds] == ["z", "a"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TemplateDataset(tmp_path / "missing.json")


class TestTemplateDatasetConversion:
    @pytest.fixture
    def sample(self, tmp_path):
        return TemplateDataset(_write_json(tmp_path / "t.json", _template()))[0]

    def test_sample_fields(self, sample):
        assert sample.initial_goal == "Build a REST API"
        assert sample.constraints == ["Use FastAPI", "No external DB"]
        assert sample.system_prompt == "You are a helpful assistant."
        assert sample.ground_truth == {"goal_coherence": 1.0}
        assert sample.metadata == {"difficulty": "easy"}
        assert set(sample.probing_tasks) == {"goal_probe", "constraint_probe", "behavioral_test"}

    def test_turn_conversion(self, sample):
        first, second = sample.turns
        assert (first.id, first.role, first.content) == (1, "user", "Start the API")
        assert first.is_compression_point is False
        assert first.tool_call is None
        assert (second.id, second.role) == (2, "assistant")
        assert second.is_compression_point is True
        assert second.tool_call == {"name": "shell", "input": "mkdir api"}

    def test_turn_defaults_when_fields_missing(self, tmp_path):
        path = _write_json(tmp_path / "t.json", _template(turns=[{}, {"content": "x"}]))
        turns = TemplateDataset(path)[0].turns
        assert [t.id for t in turns] == [0, 1]
        assert all(t.role == "user" for t in turns)
        assert [t.content for t in turns] == ["", "x"]
        assert all(t.is_compression_point is False and t.tool_call is None for t in turns)

    def test_probing_tasks_become_questions(self, sample):
        goal_q, constraint_q, behavior_q = sample.questions
        assert goal_q.question == "What is the goal?"
        assert goal_q.reference_answer == "Build a REST API"
        assert constraint_q.question == "What are the constraints?"
        assert constraint_q.reference_answer == "Use FastAPI; No external DB"
        assert behavior_q.question == "Should we add Postgres?"
        assert behavior_q.reference_answer == "Decline: no external DB"
        assert all(q.category is None for q in sample.questions)

    def test_only_present_probes_become_questions(self, tmp_path):
        path = _write_json(
            tmp_path / "t.json", _template(probing_tasks={"constraint_probe": "Constraints?"})
        )
        questions = TemplateDataset(path)[0].questions
        assert len(questions) == 1
        assert questions[0].question == "Constraints?"

    def test_minimal_template_defaults(self, tmp_path):
        sample = TemplateDataset(_write_json(tmp_path / "t.json", {}))[0]
        assert sample.sample_id == "unknown"
        assert sample.turns == []
        assert sample.questions == []
        assert sample.metadata == {}
        assert sample.initial_goal is None
        assert sample.constraints == []
        assert sample.system_prompt is None
        assert sample.probing_tasks == {}
        assert sample.ground_truth is None

    def test_probes_without_setup_use_empty_references(self, tmp_path):
        path = _write_json(
            tmp_path / "t.json",
            {
                "probing_tasks": {
                    "goal_probe": "Goal?",
                    "constraint_probe": "Constraints?",
                    "behavioral_test": {},
                }
            },
        )
        questions = TemplateDataset(path)[0].questions
        assert [(q.question, q.reference_answer) for q in questions] == [
            ("Goal?", ""),
            ("Constraints?", ""),
            ("", ""),
        ]


class TestTemplateDatasetInterface:
    def test_name_single_template(self, tmp_path):
        ds = TemplateDataset(_write_json(tmp_path / "t.json", _template("solo")))
        assert ds.name == "Template(solo)"

    def test_name_multiple_templates(self, tmp_path):
        for tid in ("a", "b"):
            _write_json(tmp_path / f"{tid}.json", _template(tid))
        assert TemplateDataset(tmp_path).name == "TemplateDataset(2 templates)"

    def test_evaluation_type(self, tmp_path):
        ds = TemplateDataset(_write_json(tmp_path / "t.json", _template()))
        assert ds.evaluation_type == "compression"

    def test_len_getitem_and_iteration(self, tmp_path):
        paths = [_write_json(tmp_path / f"{tid}.json", _template(tid)) for tid in ("a", "b")]
        ds = TemplateDataset(paths)
        assert len(ds) == 2
        assert ds[0].sample_id == "a"
        assert ds[-1].sample_id == "b"
        assert [s.sample_id for s in ds] == ["a", "b"]
        assert all(isinstance(s, EvalSample) for s in ds)
        with pytest.raises(IndexError):
            ds[2]


class TestCodingFromDictHelpers:
    def test_specification_full(self):
        spec = CodingSpecification.from_dict(
            {"goal": "g", "requirements": ["r"], "constraints": ["c"], "context": "ctx"}
        )
        assert spec == CodingSpecification(
            goal="g", requirements=["r"], constraints=["c"], context="ctx"
        )

    def test_specification_defaults(self):
        spec = CodingSpecification.from_dict({})
        assert spec == CodingSpecification(goal="", requirements=[], constraints=[], context=None)

    def test_ground_truth_full(self):
        gt = CodingGroundTruth.from_dict(
            {
                "expected_code": "code",
                "expected_files": {"a.py": "x"},
                "test_cases": [
                    {"name": "t", "input": "i", "expected_output": "o", "description": "d"}
                ],
                "acceptance_criteria": ["ok"],
                "file_changes": {"b.py": "y"},
            }
        )
        assert gt.expected_code == "code"
        assert gt.expected_files == {"a.py": "x"}
        assert gt.test_cases == [
            CodingTestCase(name="t", input="i", expected_output="o", description="d")
        ]
        assert gt.acceptance_criteria == ["ok"]
        assert gt.file_changes == {"b.py": "y"}

    def test_ground_truth_defaults(self):
        gt = CodingGroundTruth.from_dict({})
        assert gt == CodingGroundTruth()
        assert gt.test_cases == []
        assert gt.acceptance_criteria == []

    def test_test_case_accepts_expected_alias(self):
        gt = CodingGroundTruth.from_dict({"test_cases": [{"name": "t", "expected": "42"}]})
        assert gt.test_cases[0].expected_output == "42"

    def test_test_case_expected_output_wins_over_expected(self):
        gt = CodingGroundTruth.from_dict(
            {"test_cases": [{"expected_output": "primary", "expected": "alias"}]}
        )
        assert gt.test_cases[0].expected_output == "primary"

    def test_test_case_defaults(self):
        gt = CodingGroundTruth.from_dict({"test_cases": [{}]})
        assert gt.test_cases == [
            CodingTestCase(name="", input="", expected_output="", description=None)
        ]

    def test_tool_call_full_and_defaults(self):
        assert ToolCall.from_dict({"name": "n", "input": "i", "output": "o"}) == ToolCall(
            name="n", input="i", output="o"
        )
        assert ToolCall.from_dict({}) == ToolCall(name="", input="", output=None)


class TestCodingDatasetLoading:
    def test_load_single_task_file(self, tmp_path):
        ds = CodingDataset(_write_json(tmp_path / "task.json", _coding_task()))
        assert len(ds) == 1
        assert ds[0].sample_id == "task-1"

    def test_load_array_of_tasks_file(self, tmp_path):
        path = _write_json(
            tmp_path / "tasks.json",
            [_coding_task("a"), _coding_task("b", "bug_fixing"), _coding_task("c", "refactoring")],
        )
        ds = CodingDataset(str(path))
        assert [s.sample_id for s in ds] == ["a", "b", "c"]

    def test_load_directory_is_sorted_by_filename(self, tmp_path):
        _write_json(tmp_path / "b.json", _coding_task("from-b"))
        _write_json(tmp_path / "a.json", [_coding_task("from-a-1"), _coding_task("from-a-2")])
        ds = CodingDataset(tmp_path)
        assert [s.sample_id for s in ds] == ["from-a-1", "from-a-2", "from-b"]

    def test_missing_path_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Path does not exist"):
            CodingDataset(tmp_path / "nope.json")

    def test_task_types_filter(self, tmp_path):
        path = _write_json(
            tmp_path / "tasks.json",
            [
                _coding_task("gen", "code_generation"),
                _coding_task("bug", "bug_fixing"),
                _coding_task("ref", "refactoring"),
            ],
        )
        ds = CodingDataset(path, task_types=["bug_fixing", "refactoring"])
        assert [s.sample_id for s in ds] == ["bug", "ref"]
        assert [ds.get_task(i).task_id for i in range(len(ds))] == ["bug", "ref"]

    def test_default_filter_drops_unknown_task_types(self, tmp_path):
        path = _write_json(
            tmp_path / "tasks.json",
            [_coding_task("known", "research_synthesis"), _coding_task("odd", "unknown_type")],
        )
        assert [s.sample_id for s in CodingDataset(path)] == ["known"]

    def test_missing_task_type_defaults_to_code_generation(self, tmp_path):
        task = _coding_task()
        del task["task_type"]
        ds = CodingDataset(_write_json(tmp_path / "t.json", task))
        assert ds.get_task(0).task_type == "code_generation"
        assert len(CodingDataset(tmp_path / "t.json", task_types=["bug_fixing"])) == 0

    def test_missing_task_id_uses_position(self, tmp_path):
        tasks = [_coding_task(), _coding_task()]
        for t in tasks:
            del t["task_id"]
        ds = CodingDataset(_write_json(tmp_path / "t.json", tasks))
        assert [s.sample_id for s in ds] == ["task_0", "task_1"]


class TestCodingDatasetConversion:
    @pytest.fixture
    def ds(self, tmp_path):
        return CodingDataset(_write_json(tmp_path / "task.json", _coding_task()))

    def test_task_object(self, ds):
        task = ds.get_task(0)
        assert isinstance(task, CodingTask)
        assert task.task_id == "task-1"
        assert task.task_type == "code_generation"
        assert task.specification.goal == "Implement add()"
        assert task.ground_truth.test_cases[0].expected_output == "3"
        assert task.initial_files == {"README.md": "# utils"}
        assert task.compression_triggers == {"turn_count": 2}
        assert task.metadata == {"difficulty": "easy"}

    def test_turns(self, ds):
        first, second = ds[0].turns
        assert (first.id, first.role, first.content) == (1, "user", "Write add()")
        assert first.timestamp is None
        assert first.is_compression_point is False
        assert first.tool_call is None
        assert second.timestamp == "2026-01-01T00:00:00Z"
        assert second.is_compression_point is True
        assert second.tool_call == {"name": "write_file", "input": "add.py", "output": "ok"}

    def test_multiple_tool_calls_are_wrapped(self, tmp_path):
        task = _coding_task(
            conversation=[
                {
                    "role": "assistant",
                    "content": "two calls",
                    "tool_calls": [{"name": "read", "input": "a"}, {"name": "write", "input": "b"}],
                }
            ]
        )
        turn = CodingDataset(_write_json(tmp_path / "t.json", task))[0].turns[0]
        assert turn.id == 0
        assert turn.tool_call == {
            "tool_calls": [
                {"name": "read", "input": "a", "output": None},
                {"name": "write", "input": "b", "output": None},
            ]
        }

    def test_empty_tool_calls_list_yields_none(self, tmp_path):
        task = _coding_task(conversation=[{"content": "x", "tool_calls": []}])
        assert CodingDataset(_write_json(tmp_path / "t.json", task))[0].turns[0].tool_call is None

    def test_sample_fields(self, ds):
        sample = ds[0]
        assert sample.questions == []
        assert sample.initial_goal == "Implement add()"
        assert sample.constraints == ["No imports"]
        assert sample.metadata == {
            "task_type": "code_generation",
            "initial_files": {"README.md": "# utils"},
            "compression_triggers": {"turn_count": 2},
            "difficulty": "easy",
        }
        assert sample.ground_truth == {
            "expected_code": "def add(a, b):\n    return a + b\n",
            "expected_files": None,
            "test_cases": [
                {"name": "t1", "input": "add(1, 2)", "expected_output": "3", "description": None}
            ],
            "acceptance_criteria": ["Passes tests"],
            "file_changes": None,
        }
        assert set(sample.probing_tasks) == {"goal_probe", "constraint_probe", "behavioral_test"}
        assert "prompt" in sample.probing_tasks["behavioral_test"]


class TestCodingDatasetInterface:
    def test_name_and_evaluation_type(self, tmp_path):
        ds = CodingDataset(_write_json(tmp_path / "my_tasks.json", _coding_task()))
        assert ds.name == "CodingDataset(my_tasks.json)"
        assert ds.evaluation_type == "coding"

    def test_get_task_by_id(self, tmp_path):
        ds = CodingDataset(
            _write_json(tmp_path / "t.json", [_coding_task("a"), _coding_task("b", "bug_fixing")])
        )
        assert ds.get_task_by_id("b").task_type == "bug_fixing"
        assert ds.get_task_by_id("missing") is None

    def test_get_task_index_error(self, tmp_path):
        ds = CodingDataset(_write_json(tmp_path / "t.json", _coding_task()))
        with pytest.raises(IndexError):
            ds.get_task(1)

    def test_get_statistics(self, tmp_path):
        no_tests_no_code = _coding_task(
            "plain",
            "refactoring",
            ground_truth={"acceptance_criteria": ["clean"]},
            conversation=[{"role": "user", "content": "refactor"}],
        )
        files_only = _coding_task(
            "files",
            "bug_fixing",
            ground_truth={"expected_files": {"fix.py": "pass"}},
            conversation=[
                {"role": "user", "content": "a", "is_compression_point": True},
                {"role": "assistant", "content": "b", "is_compression_point": True},
                {"role": "user", "content": "c"},
            ],
        )
        path = _write_json(
            tmp_path / "tasks.json", [_coding_task("gen"), no_tests_no_code, files_only]
        )
        assert CodingDataset(path).get_statistics() == {
            "name": "CodingDataset(tasks.json)",
            "evaluation_type": "coding",
            "num_samples": 3,
            "total_turns": 6,
            "total_questions": 0,
            "task_type_distribution": {"code_generation": 1, "refactoring": 1, "bug_fixing": 1},
            "total_compression_points": 3,
            "tasks_with_compression_points": 2,
            "num_tasks_with_tests": 1,
            "num_tasks_with_expected_code": 2,
        }

    def test_get_statistics_empty_directory(self, tmp_path):
        stats = CodingDataset(tmp_path).get_statistics()
        assert stats["num_samples"] == 0
        assert stats["task_type_distribution"] == {}
        assert stats["total_compression_points"] == 0
        assert stats["tasks_with_compression_points"] == 0
        assert stats["num_tasks_with_tests"] == 0
        assert stats["num_tasks_with_expected_code"] == 0
