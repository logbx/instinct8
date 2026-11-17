"""
Unit tests for evaluation/agents/compression_agent.py.

Covers CompressionAgent's construction, turn ingestion and formatting, token
accounting (len(content) // 4), strategy-driven compression, LLM-backed
answer_question, reset, initialize_goal, and accessors. The strategy is an
in-file fake and the LLM client factory is patched with an in-file fake, so no
API keys, network, or real LLM client are used.
"""

import pytest

import evaluation.agents.compression_agent as compression_agent
from evaluation.agents.base import AgentConfig, BaseAgent
from evaluation.agents.compression_agent import CompressionAgent


class FakeStrategy:
    """Records calls and returns a fixed compressed summary."""

    def __init__(self, summary="SUMMARY"):
        self.summary = summary
        self.initialize_calls = []
        self.compress_calls = []

    def name(self):
        return "Fake"

    def initialize(self, goal, constraints):
        self.initialize_calls.append((goal, list(constraints)))

    def compress(self, context, trigger_point):
        self.compress_calls.append(([dict(t) for t in context], trigger_point))
        return self.summary


class FakeLLMClient:
    """Records prompts and returns a canned reply (or raises)."""

    def __init__(self, reply="ANSWER", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def complete(self, prompt, max_tokens=500):
        self.calls.append((prompt, max_tokens))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def llm(monkeypatch):
    client = FakeLLMClient()
    factory_calls = []

    def fake_factory(backend, model):
        factory_calls.append((backend, model))
        return client

    monkeypatch.setattr(compression_agent, "_create_llm_client", fake_factory)
    client.factory_calls = factory_calls
    return client


def _agent(strategy=None, **config_kwargs):
    return CompressionAgent(AgentConfig(**config_kwargs), strategy or FakeStrategy())


# ---------------------------------------------------------------------------
# Construction and accessors
# ---------------------------------------------------------------------------


def test_constructs_without_llm_client_or_harness():
    agent = _agent()
    assert isinstance(agent, BaseAgent)
    assert agent.context == []
    assert agent.get_context_size() == 0
    assert agent._llm_client is None


def test_construction_initializes_strategy_from_config():
    strategy = FakeStrategy()
    _agent(strategy, original_goal="Ship v1", constraints=["No new deps"])
    assert strategy.initialize_calls == [("Ship v1", ["No new deps"])]


def test_name_uses_strategy_name():
    assert _agent().name == "CompressionAgent(Fake)"


def test_strategy_accessor_returns_strategy():
    strategy = FakeStrategy()
    assert _agent(strategy).strategy is strategy


def test_no_harness_or_mock_agent_reference():
    import inspect

    source = inspect.getsource(compression_agent)
    assert "evaluation.harness" not in source
    assert "MockAgent" not in source


# ---------------------------------------------------------------------------
# ingest_turn
# ---------------------------------------------------------------------------


def test_ingest_turn_preserves_canonical_fields():
    agent = _agent()
    tool_call = {"name": "shell", "input": "ls", "output": "a.py"}
    agent.ingest_turn(
        {
            "id": 7,
            "role": "assistant",
            "content": "x" * 40,
            "is_compression_point": True,
            "tool_call": tool_call,
        }
    )
    assert agent.context == [
        {
            "id": 7,
            "role": "assistant",
            "content": "x" * 40,
            "is_compression_point": True,
            "tool_call": tool_call,
        }
    ]
    assert agent.get_context_size() == 10


def test_ingest_turn_accepts_turn_id_and_text_aliases():
    agent = _agent()
    agent.ingest_turn({"turn_id": 3, "text": "hello world!"})
    assert agent.context[0]["id"] == 3
    assert agent.context[0]["content"] == "hello world!"
    assert agent.get_context_size() == 3


def test_ingest_turn_id_takes_precedence_over_turn_id():
    agent = _agent()
    agent.ingest_turn({"id": 1, "turn_id": 2, "content": "c", "text": "t"})
    assert agent.context[0]["id"] == 1
    assert agent.context[0]["content"] == "c"


def test_ingest_turn_defaults():
    agent = _agent()
    agent.ingest_turn({})
    assert agent.context == [
        {
            "id": 0,
            "role": "user",
            "content": "",
            "is_compression_point": False,
            "tool_call": None,
        }
    ]
    assert agent.get_context_size() == 0


def test_ingest_turn_drops_unknown_keys():
    agent = _agent()
    agent.ingest_turn({"content": "hi", "timestamp": "t", "speaker": "bob"})
    assert set(agent.context[0]) == {"id", "role", "content", "is_compression_point", "tool_call"}


def test_token_count_accumulates():
    agent = _agent()
    agent.ingest_turn({"content": "a" * 8})
    agent.ingest_turn({"content": "b" * 13})
    assert agent.get_context_size() == 2 + 3


def test_ingest_turn_accepts_eval_turn_dict():
    from evaluation.datasets.base import EvalTurn

    agent = _agent()
    turn = EvalTurn(id=5, role="assistant", content="abcd", is_compression_point=True)
    agent.ingest_turn(turn.__dict__)
    assert agent.context[0]["id"] == 5
    assert agent.context[0]["is_compression_point"] is True


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compress_replaces_context_with_strategy_output():
    strategy = FakeStrategy(summary="s" * 20)
    agent = _agent(strategy)
    agent.ingest_turn({"id": 1, "content": "first"})
    agent.ingest_turn({"id": 2, "content": "second"})

    agent.compress(2)

    (passed_context, trigger_point), = strategy.compress_calls
    assert trigger_point == 2
    assert [t["content"] for t in passed_context] == ["first", "second"]
    assert agent.context == [
        {"id": 0, "role": "system", "content": "s" * 20, "is_compression_point": False}
    ]
    assert agent.get_context_size() == 5


def test_compress_without_trigger_point_covers_whole_context():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    for i in range(3):
        agent.ingest_turn({"id": i, "content": "t"})
    agent.compress()
    assert strategy.compress_calls[0][1] == 3


def test_compress_passes_explicit_zero_through():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    agent.ingest_turn({"content": "t"})
    agent.compress(0)
    assert strategy.compress_calls[0][1] == 0


def test_compress_replaces_whole_context_even_with_partial_trigger_point():
    strategy = FakeStrategy(summary="abcdefgh")
    agent = _agent(strategy)
    for i in range(1, 5):
        agent.ingest_turn({"id": i, "content": f"turn {i}"})

    agent.compress(2)

    passed_context, trigger_point = strategy.compress_calls[0]
    assert trigger_point == 2
    assert len(passed_context) == 4
    assert agent.context == [
        {"id": 0, "role": "system", "content": "abcdefgh", "is_compression_point": False}
    ]
    assert agent.get_context_size() == 2


def test_harness_style_compression_at_turn_ids():
    strategy = FakeStrategy(summary="S" * 8)
    agent = _agent(strategy, original_goal="G", constraints=["C"])
    turns = [
        {"id": i, "content": "x" * 4, "is_compression_point": i in (3, 6)}
        for i in range(1, 8)
    ]

    for turn in turns:
        agent.ingest_turn(turn)
        if turn["is_compression_point"]:
            agent.compress(turn["id"])

    assert [call[1] for call in strategy.compress_calls] == [3, 6]
    assert [len(call[0]) for call in strategy.compress_calls] == [3, 4]
    assert [t["content"] for t in agent.context] == ["S" * 8, "x" * 4]
    assert agent.get_context_size() == 2 + 1


def test_turns_after_compression_extend_compressed_context():
    agent = _agent(FakeStrategy(summary="abcd"))
    agent.ingest_turn({"content": "old"})
    agent.compress(1)
    agent.ingest_turn({"id": 9, "content": "new!"})
    assert [t["content"] for t in agent.context] == ["abcd", "new!"]
    assert agent.get_context_size() == 2


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------


def test_answer_question_prompts_llm_with_context(llm):
    agent = _agent()
    agent.ingest_turn({"role": "user", "content": "Build an API"})
    agent.ingest_turn({"role": "assistant", "content": "Sure"})

    assert agent.answer_question("What is the goal?") == "ANSWER"

    (prompt, max_tokens), = llm.calls
    assert max_tokens == 300
    assert prompt == (
        "Context:\n[user]: Build an API\n\n[assistant]: Sure"
        "\n\n---\n\nQuestion: What is the goal?"
    )


def test_answer_question_prepends_system_prompt(llm):
    agent = _agent(system_prompt="You are terse.")
    agent.answer_question("Q?")
    prompt, _ = llm.calls[0]
    assert prompt.startswith("You are terse.\n\nContext:\n")


def test_answer_question_ignores_category_and_reference(llm):
    agent = _agent()
    agent.answer_question("Q?", category=5, reference_answer="secret")
    prompt, _ = llm.calls[0]
    assert "secret" not in prompt


def test_answer_question_uses_compressed_context(llm):
    agent = _agent(FakeStrategy(summary="GOAL: ship it"))
    agent.ingest_turn({"content": "raw detail"})
    agent.compress(1)
    agent.answer_question("Goal?")
    prompt, _ = llm.calls[0]
    assert "[system]: GOAL: ship it" in prompt
    assert "raw detail" not in prompt


def test_llm_client_created_lazily_once_from_config(llm):
    agent = _agent(backend="openai", model="gpt-4o-mini")
    assert llm.factory_calls == []
    agent.answer_question("a")
    agent.answer_question("b")
    assert llm.factory_calls == [("openai", "gpt-4o-mini")]
    assert len(llm.calls) == 2


def test_answer_question_returns_error_string_on_llm_failure(llm, capsys):
    llm.error = RuntimeError("boom")
    assert _agent().answer_question("Q?") == "(error: boom)"
    assert "boom" in capsys.readouterr().out


def test_answer_question_flags_empty_response(llm):
    llm.reply = ""
    assert _agent().answer_question("Q?") == "(error: empty response)"


def test_answer_question_propagates_client_config_errors(monkeypatch):
    def failing_factory(backend, model):
        raise ValueError("No API key found")

    monkeypatch.setattr(compression_agent, "_create_llm_client", failing_factory)
    with pytest.raises(ValueError, match="No API key"):
        _agent().answer_question("Q?")


# ---------------------------------------------------------------------------
# reset and initialize_goal
# ---------------------------------------------------------------------------


def test_reset_clears_context_and_reinitializes_strategy():
    strategy = FakeStrategy()
    agent = _agent(strategy, original_goal="G", constraints=["C"])
    agent.ingest_turn({"content": "abcdefgh"})

    agent.reset()

    assert agent.context == []
    assert agent.get_context_size() == 0
    assert strategy.initialize_calls == [("G", ["C"]), ("G", ["C"])]


def test_reset_after_compression_clears_compressed_context():
    agent = _agent(FakeStrategy(summary="x" * 40))
    agent.ingest_turn({"content": "t"})
    agent.compress(1)
    agent.reset()
    assert agent.context == []
    assert agent.get_context_size() == 0


def test_initialize_goal_updates_config_and_strategy():
    strategy = FakeStrategy()
    agent = _agent(strategy)

    agent.initialize_goal("New goal", ["Rule 1", "Rule 2"])

    assert agent._config.original_goal == "New goal"
    assert agent._config.constraints == ["Rule 1", "Rule 2"]
    assert strategy.initialize_calls[-1] == ("New goal", ["Rule 1", "Rule 2"])


def test_reset_reuses_goal_from_initialize_goal():
    strategy = FakeStrategy()
    agent = _agent(strategy)
    agent.initialize_goal("Goal X", ["Keep it small"])
    agent.reset()
    assert strategy.initialize_calls[-1] == ("Goal X", ["Keep it small"])


def test_initialize_goal_does_not_touch_context():
    agent = _agent()
    agent.ingest_turn({"content": "abcd"})
    agent.initialize_goal("G", [])
    assert len(agent.context) == 1
    assert agent.get_context_size() == 1
