"""
Unit tests for evaluation/agents/codex_agent.py.

Covers CodexAgent's goal seeding, turn ingestion and simulated tools (shell,
read_file, write_file, apply_patch), threshold-driven auto-compression, the
heuristic answer_question branches, reset, the copy-returning accessors, and
the create_codex_agent factory. CodexAgent is fully simulated: tokens are
len(content) // 4 and no tool touches the real shell or file system. Strategies
are in-file fakes, and the "codex" factory path runs against a patched
StrategyB_CodexCheckpoint, so no LLM client, API keys, or network are used.
"""

import pytest

import strategies.strategy_b_codex as strategy_b_codex
from evaluation.agents.base import AgentConfig
from evaluation.agents.codex_agent import CodexAgent, create_codex_agent


class FakeStrategy:
    """Records calls and returns a fixed compressed summary."""

    def __init__(self, summary="SUMMARY"):
        self.summary = summary
        self.initialize_calls = []
        self.compress_calls = []
        self.reset_calls = 0

    def name(self):
        return "Fake"

    def initialize(self, goal, constraints):
        self.initialize_calls.append((goal, list(constraints)))

    def compress(self, context, trigger_point):
        self.compress_calls.append((list(context), trigger_point))
        return self.summary

    def reset(self):
        self.reset_calls += 1


class NoResetStrategy:
    def name(self):
        return "NoReset"

    def initialize(self, goal, constraints):
        pass

    def compress(self, context, trigger_point):
        return "x"


def _agent(strategy=None, threshold=80000):
    return CodexAgent(AgentConfig(), strategy=strategy, compaction_threshold=threshold)


def _tool(name, input_data):
    return {"role": "assistant", "content": "", "tool_call": {"name": name, "input": input_data}}


# ---------------------------------------------------------------------------
# Config and naming
# ---------------------------------------------------------------------------


def test_agent_config_defaults():
    config = AgentConfig()
    assert config.model == "claude-sonnet-4-20250514"
    assert config.backend == "anthropic"
    assert config.system_prompt == ""
    assert config.original_goal == ""
    assert config.constraints == []
    assert AgentConfig().constraints is not config.constraints


def test_name_without_strategy():
    assert _agent().name == "CodexAgent(NoCompression)"


def test_name_with_strategy():
    assert _agent(FakeStrategy()).name == "CodexAgent(Fake)"


def test_fresh_agent_is_empty():
    agent = _agent()
    assert agent.get_context_size() == 0
    assert agent.get_file_system() == {}
    assert agent.get_generated_code() == {}


# ---------------------------------------------------------------------------
# initialize_goal
# ---------------------------------------------------------------------------


def test_initialize_goal_seeds_system_turn_and_tokens():
    agent = _agent()
    agent.initialize_goal("Build a FastAPI auth system", ["Use JWT", "Hash passwords"])

    assert len(agent._context) == 1
    system_turn = agent._context[0]
    assert system_turn["role"] == "system"
    assert system_turn["id"] == 0
    content = system_turn["content"]
    assert "GOAL: Build a FastAPI auth system" in content
    assert "CONSTRAINTS:" in content
    assert "  1. Use JWT" in content
    assert "  2. Hash passwords" in content
    assert "- shell: Execute shell commands" in content
    assert agent.get_context_size() == len(content) // 4


def test_initialize_goal_without_constraints_omits_section():
    agent = _agent()
    agent.initialize_goal("Write a CLI", [])
    content = agent._context[0]["content"]
    assert "GOAL: Write a CLI" in content
    assert "CONSTRAINTS:" not in content


def test_initialize_goal_forwards_to_strategy():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    agent.initialize_goal("Goal", ["c1", "c2"])
    assert strategy.initialize_calls == [("Goal", ["c1", "c2"])]


# ---------------------------------------------------------------------------
# ingest_turn and simulated tools
# ---------------------------------------------------------------------------


def test_ingest_turn_appends_context_and_bumps_tokens():
    agent = _agent()
    turn = {"role": "user", "content": "a" * 40}
    agent.ingest_turn(turn)
    agent.ingest_turn({"role": "assistant", "content": "b" * 9})

    assert agent._context[0] is turn
    assert len(agent._context) == 2
    assert agent.get_context_size() == 10 + 2


def test_ingest_turn_non_string_content_counts_zero_tokens():
    agent = _agent()
    agent.ingest_turn({"role": "user", "content": ["not", "a", "string"]})
    agent.ingest_turn({"role": "user"})
    assert len(agent._context) == 2
    assert agent.get_context_size() == 0


def test_write_file_dict_input_updates_fs_and_generated_code():
    agent = _agent()
    agent.ingest_turn(_tool("write_file", {"path": "app.py", "content": "print(1)"}))

    assert agent.get_file_system() == {"app.py": "print(1)"}
    assert agent.get_generated_code() == {"app.py": "print(1)"}
    op = agent._file_operations[-1]
    assert (op.operation, op.path, op.content) == ("write", "app.py", "print(1)")


def test_write_file_string_input_uses_placeholder_content():
    agent = _agent()
    agent.ingest_turn(_tool("write_file", "notes.txt"))
    assert agent.get_file_system() == {"notes.txt": "(file content)"}
    assert agent.get_generated_code() == {"notes.txt": "(file content)"}


def test_read_file_existing_records_operation():
    agent = _agent()
    agent.set_initial_files({"README.md": "hello"})
    agent.ingest_turn(_tool("read_file", "README.md"))

    op = agent._file_operations[-1]
    assert (op.operation, op.path, op.content) == ("read", "README.md", "hello")


def test_read_file_missing_records_nothing():
    agent = _agent()
    agent.ingest_turn(_tool("read_file", "missing.py"))
    assert agent._file_operations == []


def test_read_file_results():
    agent = _agent()
    agent.set_initial_files({"a.py": "x = 1"})
    assert agent._execute_tool({"name": "read_file", "input": "a.py"}) == {"content": "x = 1"}
    assert agent._execute_tool({"name": "read_file", "input": "b.py"}) == {
        "error": "File not found: b.py"
    }


def test_apply_patch_records_operation_without_touching_fs():
    agent = _agent()
    agent.set_initial_files({"a.py": "old"})
    agent.ingest_turn(_tool("apply_patch", "--- a.py\n+++ a.py\n-old\n+new"))

    op = agent._file_operations[-1]
    assert (op.operation, op.path) == ("patch", "(patched)")
    assert op.content.endswith("+new")
    assert agent.get_file_system() == {"a.py": "old"}
    assert agent.get_generated_code() == {}


@pytest.mark.parametrize(
    "command, expected_output",
    [
        ("mkdir src", ""),
        ("pip install fastapi", "Successfully installed packages"),
        ("pytest -q", "===== X passed in Y.YYs ====="),
        ("python main.py", "(execution output)"),
        ("echo hi", "(command executed)"),
    ],
)
def test_shell_simulated_outputs(command, expected_output):
    agent = _agent()
    agent.ingest_turn(_tool("shell", command))

    assert len(agent._shell_history) == 1
    entry = agent._shell_history[0]
    assert entry.command == command
    assert entry.output == expected_output
    assert entry.exit_code == 0


def test_shell_ls_lists_files_or_dot():
    agent = _agent()
    assert agent._execute_tool({"name": "shell", "input": "ls"})["output"] == "."

    agent.set_initial_files({"a.py": "", "b.py": ""})
    assert agent._execute_tool({"name": "shell", "input": "ls -la"}) == {
        "output": "a.py\nb.py",
        "exit_code": 0,
    }


def test_shell_cat_reads_simulated_fs():
    agent = _agent()
    agent.set_initial_files({"a.py": "x = 1"})
    assert agent._execute_tool({"name": "shell", "input": "cat a.py"})["output"] == "x = 1"
    assert (
        agent._execute_tool({"name": "shell", "input": "cat nope.py"})["output"]
        == "cat: nope.py: No such file"
    )


def test_unknown_tool_is_ignored():
    agent = _agent()
    result = agent._execute_tool({"name": "rm_rf", "input": "/"})
    assert result == {"error": "Unknown tool: rm_rf"}

    agent.ingest_turn(_tool("rm_rf", "/"))
    assert agent._shell_history == []
    assert agent._file_operations == []
    assert agent.get_file_system() == {}


def test_tool_calls_list_executes_in_order():
    agent = _agent()
    agent.ingest_turn(
        {
            "role": "assistant",
            "content": "scaffold",
            "tool_calls": [
                {"name": "shell", "input": "mkdir src"},
                {"name": "write_file", "input": {"path": "src/main.py", "content": "app = 1"}},
                {"name": "shell", "input": "cat src/main.py"},
                {"name": "read_file", "input": "src/main.py"},
            ],
        }
    )

    assert [c.command for c in agent._shell_history] == ["mkdir src", "cat src/main.py"]
    assert agent._shell_history[1].output == "app = 1"
    assert [op.operation for op in agent._file_operations] == ["write", "read"]
    assert agent.get_generated_code() == {"src/main.py": "app = 1"}


def test_tool_calls_takes_precedence_over_tool_call():
    agent = _agent()
    agent.ingest_turn(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "shell", "input": "pytest"}],
            "tool_call": {"name": "shell", "input": "echo ignored"},
        }
    )
    assert [c.command for c in agent._shell_history] == ["pytest"]


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_auto_compress_when_tokens_exceed_threshold(capsys):
    strategy = FakeStrategy(summary="S" * 8)
    agent = _agent(strategy, threshold=10)
    agent.ingest_turn({"role": "user", "content": "a" * 40})
    assert strategy.compress_calls == []

    big = {"role": "user", "content": "b" * 40}
    agent.ingest_turn(big)

    assert len(strategy.compress_calls) == 1
    passed_context, trigger_point = strategy.compress_calls[0]
    assert trigger_point == 2
    assert passed_context[-1] is big
    assert agent._context == [
        {"id": 0, "role": "system", "content": "S" * 8, "is_compressed": True}
    ]
    assert agent.get_context_size() == 2
    assert agent._compression_count == 1
    assert "Compressed: 20 -> 2 tokens (compression #1)" in capsys.readouterr().out


def test_no_auto_compress_at_exact_threshold():
    strategy = FakeStrategy()
    agent = _agent(strategy, threshold=10)
    agent.ingest_turn({"role": "user", "content": "a" * 40})
    assert strategy.compress_calls == []
    assert agent._compression_count == 0


def test_compression_count_accumulates_across_triggers():
    strategy = FakeStrategy(summary="")
    agent = _agent(strategy, threshold=5)
    for _ in range(3):
        agent.ingest_turn({"role": "user", "content": "x" * 40})
    assert agent._compression_count == 3
    assert len(strategy.compress_calls) == 3


def test_manual_compress_passes_trigger_point():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    agent.ingest_turn({"role": "user", "content": "one"})
    agent.ingest_turn({"role": "user", "content": "two"})
    agent.compress(trigger_point=1)

    assert strategy.compress_calls[0][1] == 1
    assert agent._context[0]["content"] == "SUMMARY"
    assert agent._compression_count == 1


def test_compress_is_noop_without_strategy():
    agent = _agent(threshold=1)
    agent.ingest_turn({"role": "user", "content": "a" * 40})
    agent.ingest_turn({"role": "user", "content": "b" * 40})
    agent.compress()

    assert len(agent._context) == 2
    assert agent.get_context_size() == 20
    assert agent._compression_count == 0


# ---------------------------------------------------------------------------
# answer_question heuristics
# ---------------------------------------------------------------------------


def test_answer_goal_question():
    agent = _agent()
    assert agent.answer_question("What is the goal?") == "No explicit goal has been set."
    agent.initialize_goal("Ship the API", [])
    assert agent.answer_question("What is the current TASK?") == "The current goal is: Ship the API"


def test_answer_constraints_question():
    agent = _agent()
    assert (
        agent.answer_question("Any constraints?") == "No constraints have been specified."
    )
    agent.initialize_goal("G", ["Use JWT", "No plaintext passwords"])
    assert (
        agent.answer_question("List the requirements")
        == "Constraints: Use JWT; No plaintext passwords"
    )


def test_goal_branch_takes_precedence():
    agent = _agent()
    agent.initialize_goal("Ship it", ["c"])
    assert agent.answer_question("What should the task constraints be?").startswith(
        "The current goal is:"
    )


def test_answer_next_step_empty_context():
    agent = _agent()
    assert (
        agent.answer_question("What should I do?")
        == "Start by understanding the task requirements."
    )


def test_answer_next_step_uses_recent_todo():
    agent = _agent()
    agent.ingest_turn({"role": "assistant", "content": "TODO: add login route " + "z" * 300})
    agent.ingest_turn({"role": "user", "content": "ok"})

    answer = agent.answer_question("What comes next?")
    assert answer.startswith("Based on recent context: TODO: add login route")
    assert answer.endswith("...")
    assert len(answer) == len("Based on recent context: ") + 200 + len("...")


def test_answer_next_step_ignores_turns_outside_last_three():
    agent = _agent()
    agent.ingest_turn({"role": "assistant", "content": "TODO: old item"})
    for i in range(3):
        agent.ingest_turn({"role": "user", "content": f"filler {i}"})
    assert agent.answer_question("What should I do?") == "Continue implementing the current feature."


def test_answer_progress_counts_activity():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    agent.ingest_turn(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "write_file", "input": {"path": "a.py", "content": "a"}},
                {"name": "write_file", "input": {"path": "b.py", "content": "b"}},
                {"name": "shell", "input": "pytest"},
            ],
        }
    )
    agent.compress()
    assert agent.answer_question("How is progress?") == (
        "Progress: 2 files created, 1 commands executed, 1 compressions performed."
    )


def test_answer_code_question():
    agent = _agent()
    assert agent.answer_question("Show me the code") == "No code has been generated yet."
    agent.ingest_turn(_tool("write_file", {"path": "a.py", "content": "1"}))
    agent.ingest_turn(_tool("write_file", {"path": "b.py", "content": "2"}))
    assert agent.answer_question("Which file exists?") == "Generated files: a.py, b.py"


def test_answer_default_summary():
    agent = _agent()
    assert agent.answer_question("Summarize") == "No context available."

    agent.initialize_goal("Ship it", ["c1", "c2"])
    agent.ingest_turn({"role": "user", "content": "hello"})
    expected = (
        f"Goal: Ship it | Constraints: 2 defined | Context: 2 turns | "
        f"Tokens: {agent.get_context_size()}"
    )
    assert agent.answer_question("Summarize") == expected


def test_answer_default_summary_without_goal():
    agent = _agent()
    agent.ingest_turn({"role": "user", "content": "a" * 8})
    assert agent.answer_question("Summarize") == "Context: 1 turns | Tokens: 2"


# ---------------------------------------------------------------------------
# reset, accessors, set_initial_files
# ---------------------------------------------------------------------------


def _busy_agent(strategy):
    agent = _agent(strategy)
    agent.initialize_goal("G", ["c"])
    agent.set_initial_files({"seed.py": "s"})
    agent.ingest_turn(
        {
            "role": "assistant",
            "content": "work",
            "tool_calls": [
                {"name": "write_file", "input": {"path": "a.py", "content": "a"}},
                {"name": "shell", "input": "ls"},
            ],
        }
    )
    agent.compress()
    return agent


def test_reset_clears_state_and_resets_strategy():
    strategy = FakeStrategy()
    agent = _busy_agent(strategy)
    agent.reset()

    assert agent._context == []
    assert agent.get_context_size() == 0
    assert agent._compression_count == 0
    assert agent.get_file_system() == {}
    assert agent._shell_history == []
    assert agent._file_operations == []
    assert agent.get_generated_code() == {}
    assert strategy.reset_calls == 1


def test_reset_tolerates_strategy_without_reset():
    agent = _agent(NoResetStrategy())
    agent.ingest_turn({"role": "user", "content": "hi"})
    agent.reset()
    assert agent._context == []


def test_reset_without_strategy():
    agent = _busy_agent(None)
    agent.reset()
    assert agent._context == []
    assert agent.get_file_system() == {}


def test_accessors_return_copies():
    agent = _agent()
    agent.ingest_turn(_tool("write_file", {"path": "a.py", "content": "a"}))

    code = agent.get_generated_code()
    code["a.py"] = "mutated"
    code["evil.py"] = "x"
    fs = agent.get_file_system()
    fs.clear()

    assert agent.get_generated_code() == {"a.py": "a"}
    assert agent.get_file_system() == {"a.py": "a"}


def test_set_initial_files_merges():
    agent = _agent()
    agent.set_initial_files({"a.py": "1", "b.py": "2"})
    agent.set_initial_files({"b.py": "22", "c.py": "3"})

    assert agent.get_file_system() == {"a.py": "1", "b.py": "22", "c.py": "3"}
    assert agent.get_generated_code() == {}


# ---------------------------------------------------------------------------
# create_codex_agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_name", ["none", "unknown"])
def test_create_codex_agent_without_strategy(strategy_name):
    agent = create_codex_agent(strategy_name=strategy_name, compaction_threshold=123)

    assert isinstance(agent, CodexAgent)
    assert agent.name == "CodexAgent(NoCompression)"
    assert agent._compaction_threshold == 123
    assert agent._config == AgentConfig()


def test_create_codex_agent_codex_uses_patched_strategy_b(monkeypatch):
    created = []

    class FakeStrategyB(FakeStrategy):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

        def name(self):
            return "FakeStrategyB"

    monkeypatch.setattr(strategy_b_codex, "StrategyB_CodexCheckpoint", FakeStrategyB)

    agent = create_codex_agent("codex", compaction_threshold=50, backend="openai")

    assert created == [{"backend": "openai"}]
    assert agent.name == "CodexAgent(FakeStrategyB)"
    assert isinstance(agent._strategy, FakeStrategyB)
    assert agent._compaction_threshold == 50
