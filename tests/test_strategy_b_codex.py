"""Offline unit tests for Strategy B (Codex-style checkpoint compaction).

No API keys or network are required: the module-level ``_create_llm_client``
factory is monkeypatched to return a recording fake client, and token
estimation is pinned to the deterministic ``ceil(len(text) / 4)`` fallback so
budget-trigger boundaries are exact regardless of tiktoken cache state.
"""

import sys
from types import SimpleNamespace

import pytest

import strategies.strategy_b_codex as codex
from evaluation import token_budget
from evaluation.token_budget import BUDGET_8K, TokenBudget
from strategies.strategy_b_codex import (
    CODEX_SUMMARIZATION_PROMPT_BASELINE,
    CODEX_SUMMARIZATION_PROMPT_ENHANCED,
    CODEX_SUMMARY_PREFIX,
    COMPACT_USER_MESSAGE_MAX_TOKENS,
    StrategyB_CodexCheckpoint,
    create_codex_strategy,
)

ALWAYS_COMPACT = TokenBudget(
    window_tokens=0, trigger_ratio=1.0, output_reserve_tokens=0
)
NEVER_COMPACT = TokenBudget(
    window_tokens=10**9, trigger_ratio=1.0, output_reserve_tokens=0
)


class RecordingLLMClient:
    """Duck-typed stand-in for the strategy's ``LLMClient`` protocol."""

    def __init__(self, response="FAKE SUMMARY", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(token_budget, "TIKTOKEN_AVAILABLE", False)


@pytest.fixture
def fake_llm(monkeypatch):
    client = RecordingLLMClient()
    client.factory_calls = []

    def factory(backend="auto", model=None):
        client.factory_calls.append({"backend": backend, "model": model})
        return client

    monkeypatch.setattr(codex, "_create_llm_client", factory)
    return client


@pytest.fixture
def make_strategy(fake_llm):
    def _make(**kwargs):
        kwargs.setdefault("token_budget", ALWAYS_COMPACT)
        return StrategyB_CodexCheckpoint(**kwargs)

    return _make


def _heuristic_tokens(text: str) -> int:
    return -(-len(text) // 4)


class TestCreateLLMClient:
    @pytest.fixture
    def fake_summarizers(self, monkeypatch):
        created = []

        def fake(kind):
            return lambda model: created.append((kind, model)) or (kind, model)

        monkeypatch.setattr(codex, "OpenAISummarizer", fake("openai"))
        monkeypatch.setattr(codex, "AnthropicSummarizer", fake("anthropic"))
        return created

    def test_auto_without_keys_raises(self):
        with pytest.raises(ValueError, match="No API key found"):
            codex._create_llm_client("auto")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend: gemini"):
            codex._create_llm_client("gemini")

    def test_auto_prefers_openai_when_both_keys_set(
        self, monkeypatch, fake_summarizers
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert codex._create_llm_client("auto") == ("openai", "gpt-4o-mini")

    def test_auto_falls_back_to_anthropic(self, monkeypatch, fake_summarizers):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert codex._create_llm_client("auto") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    @pytest.mark.parametrize(
        "backend,model,expected",
        [
            ("openai", None, ("openai", "gpt-4o-mini")),
            ("openai", "gpt-x", ("openai", "gpt-x")),
            ("anthropic", None, ("anthropic", "claude-sonnet-4-20250514")),
            ("anthropic", "claude-x", ("anthropic", "claude-x")),
        ],
    )
    def test_explicit_backend_and_model_override(
        self, fake_summarizers, backend, model, expected
    ):
        assert codex._create_llm_client(backend, model) == expected
        assert fake_summarizers == [expected]


class TestSummarizerClients:
    def test_openai_complete_sends_request_and_strips(self, monkeypatch):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content="  summary text \n")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        monkeypatch.setitem(
            sys.modules, "openai", SimpleNamespace(OpenAI=lambda: fake_client)
        )

        summarizer = codex.OpenAISummarizer(model="gpt-test")
        assert summarizer.complete("PROMPT", max_tokens=42) == "summary text"
        assert requests == [
            {
                "model": "gpt-test",
                "max_tokens": 42,
                "messages": [{"role": "user", "content": "PROMPT"}],
            }
        ]

    @pytest.mark.parametrize("content", [None, ""])
    def test_openai_empty_content_returns_marker(self, monkeypatch, content):
        message = SimpleNamespace(content=content)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )
        monkeypatch.setitem(
            sys.modules, "openai", SimpleNamespace(OpenAI=lambda: fake_client)
        )

        assert (
            codex.OpenAISummarizer().complete("PROMPT")
            == "(summarization returned empty)"
        )

    def test_anthropic_complete_sends_request_and_strips(self, monkeypatch):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text="\n anthropic summary ")]
            )

        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
        monkeypatch.setitem(
            sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: fake_client)
        )

        summarizer = codex.AnthropicSummarizer(model="claude-test")
        assert summarizer.complete("PROMPT", max_tokens=7) == "anthropic summary"
        assert requests == [
            {
                "model": "claude-test",
                "max_tokens": 7,
                "messages": [{"role": "user", "content": "PROMPT"}],
            }
        ]

    @pytest.mark.parametrize(
        "cls,module_name",
        [(codex.OpenAISummarizer, "openai"), (codex.AnthropicSummarizer, "anthropic")],
    )
    def test_missing_sdk_raises_import_error(self, monkeypatch, cls, module_name):
        monkeypatch.setitem(sys.modules, module_name, None)
        with pytest.raises(ImportError, match=f"pip install {module_name}"):
            cls()


class TestInitialization:
    def test_defaults(self, fake_llm):
        strategy = StrategyB_CodexCheckpoint()

        assert strategy.client is fake_llm
        assert fake_llm.factory_calls == [{"backend": "auto", "model": None}]
        assert strategy.system_prompt == ""
        assert strategy.token_budget is BUDGET_8K
        assert strategy.original_goal is None
        assert strategy.constraints == []
        assert strategy.use_goal_preservation is True

    def test_custom_arguments(self, fake_llm):
        budget = TokenBudget(
            window_tokens=1000, trigger_ratio=0.5, output_reserve_tokens=10
        )
        strategy = StrategyB_CodexCheckpoint(
            system_prompt="SYS",
            model="m",
            backend="anthropic",
            token_budget=budget,
            use_goal_preservation=False,
        )

        assert fake_llm.factory_calls == [{"backend": "anthropic", "model": "m"}]
        assert strategy.system_prompt == "SYS"
        assert strategy.token_budget is budget
        assert strategy.use_goal_preservation is False

    def test_without_keys_or_fake_raises(self):
        with pytest.raises(ValueError, match="No API key found"):
            StrategyB_CodexCheckpoint()

    def test_create_codex_strategy(self, fake_llm):
        strategy = create_codex_strategy(system_prompt="You are a helpful assistant.")

        assert isinstance(strategy, StrategyB_CodexCheckpoint)
        assert strategy.system_prompt == "You are a helpful assistant."
        assert strategy.name() == "Strategy B - Codex-Style Checkpoint"
        assert fake_llm.factory_calls == [{"backend": "auto", "model": None}]

    def test_initialize_stores_goal_and_constraints(self, make_strategy, sample_goal):
        strategy = make_strategy()
        strategy.initialize(sample_goal["objective"], sample_goal["constraints"])

        assert strategy.original_goal == sample_goal["objective"]
        assert strategy.constraints == sample_goal["constraints"]

    def test_reinitialize_replaces_previous_state(self, make_strategy):
        strategy = make_strategy()
        strategy.initialize("Old goal", ["Old constraint"])
        strategy.initialize("New goal", [])

        assert strategy.original_goal == "New goal"
        assert strategy.constraints == []

    def test_update_goal_is_not_tracked(self, make_strategy, fake_llm):
        strategy = make_strategy()
        strategy.initialize("Original goal", ["C1"])
        strategy.update_goal("Pivoted goal", rationale="user asked")

        assert strategy.original_goal == "Original goal"
        assert strategy.constraints == ["C1"]
        assert fake_llm.calls == []


class TestRenderReconstructedPrompt:
    def test_full_render(self, make_strategy, sample_conversation):
        strategy = make_strategy(system_prompt="SYS")
        rendered = strategy.render_reconstructed_prompt(
            sample_conversation, user_message="Next?"
        )

        expected = ["System: SYS"]
        expected += [
            f"{t['role'].capitalize()}: {t['content']}" for t in sample_conversation
        ]
        expected.append("User: Next?")
        assert rendered == "\n\n".join(expected)

    def test_omits_empty_system_prompt_and_user_message(self, make_strategy):
        strategy = make_strategy()
        rendered = strategy.render_reconstructed_prompt(
            [{"role": "assistant", "content": "Hi"}]
        )
        assert rendered == "Assistant: Hi"

    def test_missing_role_and_content_defaults(self, make_strategy):
        strategy = make_strategy()
        assert strategy.render_reconstructed_prompt([{}]) == "Unknown: "

    def test_empty_context(self, make_strategy):
        assert make_strategy().render_reconstructed_prompt([]) == ""
        assert (
            make_strategy(system_prompt="SYS").render_reconstructed_prompt([])
            == "System: SYS"
        )


class TestCompressEmptyInput:
    @pytest.mark.parametrize(
        "context,trigger_point",
        [([], 0), ([], 5), ([{"role": "user", "content": "x"}], 0)],
    )
    def test_nothing_to_compress_without_system_prompt(
        self, make_strategy, fake_llm, context, trigger_point
    ):
        result = make_strategy().compress(context, trigger_point)

        assert result == "(No previous conversation)"
        assert fake_llm.calls == []

    def test_nothing_to_compress_with_system_prompt(self, make_strategy, fake_llm):
        result = make_strategy(system_prompt="SYS").compress([], 0)

        assert result == "System: SYS\n\n(No previous conversation)"
        assert fake_llm.calls == []


class TestTokenBudgetGating:
    def test_below_budget_returns_reconstructed_prompt_without_llm(
        self, make_strategy, fake_llm, sample_conversation
    ):
        strategy = make_strategy(system_prompt="SYS", token_budget=NEVER_COMPACT)
        result = strategy.compress(sample_conversation, trigger_point=2)

        assert result == strategy.render_reconstructed_prompt(sample_conversation[:2])
        assert sample_conversation[2]["content"] not in result
        assert fake_llm.calls == []

    def test_trigger_boundary_is_inclusive(
        self, make_strategy, fake_llm, sample_conversation
    ):
        probe = make_strategy(system_prompt="SYS")
        tokens = _heuristic_tokens(
            probe.render_reconstructed_prompt(sample_conversation)
        )

        at_threshold = TokenBudget(
            window_tokens=tokens, trigger_ratio=1.0, output_reserve_tokens=0
        )
        result = make_strategy(system_prompt="SYS", token_budget=at_threshold).compress(
            sample_conversation, len(sample_conversation)
        )
        assert "--- Conversation Summary ---" in result
        assert len(fake_llm.calls) == 1

        above_threshold = TokenBudget(
            window_tokens=tokens + 1, trigger_ratio=1.0, output_reserve_tokens=0
        )
        result = make_strategy(
            system_prompt="SYS", token_budget=above_threshold
        ).compress(sample_conversation, len(sample_conversation))
        assert result == probe.render_reconstructed_prompt(sample_conversation)
        assert len(fake_llm.calls) == 1

    def test_system_prompt_counts_toward_budget(self, make_strategy, fake_llm):
        context = [{"role": "user", "content": "short"}]
        bare_tokens = _heuristic_tokens(
            make_strategy().render_reconstructed_prompt(context)
        )
        budget = TokenBudget(
            window_tokens=bare_tokens + 1, trigger_ratio=1.0, output_reserve_tokens=0
        )

        make_strategy(token_budget=budget).compress(context, 1)
        assert fake_llm.calls == []

        make_strategy(
            system_prompt="A long system prompt.", token_budget=budget
        ).compress(context, 1)
        assert len(fake_llm.calls) == 1

    def test_only_turns_before_trigger_point_are_considered(
        self, make_strategy, fake_llm
    ):
        context = [
            {"role": "user", "content": "early"},
            {"role": "user", "content": "x" * 10_000},
        ]
        budget = TokenBudget(
            window_tokens=1_000, trigger_ratio=1.0, output_reserve_tokens=0
        )

        result = make_strategy(token_budget=budget).compress(context, trigger_point=1)

        assert result == "User: early"
        assert fake_llm.calls == []


class TestCompactionOutput:
    def test_full_compaction_output_shape(
        self, make_strategy, fake_llm, sample_conversation, sample_goal
    ):
        strategy = make_strategy(system_prompt="SYS")
        strategy.initialize(sample_goal["objective"], sample_goal["constraints"])

        result = strategy.compress(
            sample_conversation, trigger_point=len(sample_conversation)
        )

        user_turns = [t["content"] for t in sample_conversation if t["role"] == "user"]
        expected = [
            "System: SYS",
            "\n--- TASK CONTEXT (AUTHORITATIVE - Never forget) ---",
            f"Original Goal: {sample_goal['objective']}",
            "Hard Constraints:",
            *[f"  - {c}" for c in sample_goal["constraints"]],
            "---",
            "\n--- Previous User Messages ---",
            *[f"User message {i}: {m}" for i, m in enumerate(user_turns, 1)],
            "\n--- Conversation Summary ---",
            f"{CODEX_SUMMARY_PREFIX}\n\nFAKE SUMMARY",
        ]
        assert result == "\n\n".join(expected)
        for turn in sample_conversation:
            if turn["role"] == "assistant":
                assert turn["content"] not in result

    def test_baseline_mode_omits_task_context(
        self, make_strategy, fake_llm, sample_conversation, sample_goal
    ):
        strategy = make_strategy(use_goal_preservation=False)
        strategy.initialize(sample_goal["objective"], sample_goal["constraints"])

        result = strategy.compress(
            sample_conversation, trigger_point=len(sample_conversation)
        )

        assert "TASK CONTEXT" not in result
        assert sample_goal["objective"] not in result
        assert not result.startswith("System:")
        assert result.startswith("\n--- Previous User Messages ---")
        assert result.endswith(f"{CODEX_SUMMARY_PREFIX}\n\nFAKE SUMMARY")

    def test_goal_only_omits_constraints_header(self, make_strategy, fake_llm):
        strategy = make_strategy()
        strategy.initialize("Ship it", [])

        result = strategy.compress([{"role": "user", "content": "go"}], 1)

        assert "Original Goal: Ship it" in result
        assert "Hard Constraints:" not in result

    def test_constraints_only_omits_goal_line(self, make_strategy, fake_llm):
        strategy = make_strategy()
        strategy.initialize("", ["Never delete data"])

        result = strategy.compress([{"role": "user", "content": "go"}], 1)

        assert "--- TASK CONTEXT (AUTHORITATIVE - Never forget) ---" in result
        assert "Original Goal:" not in result
        assert "Hard Constraints:\n\n  - Never delete data" in result

    def test_uninitialized_omits_task_context(self, make_strategy, fake_llm):
        result = make_strategy().compress([{"role": "user", "content": "go"}], 1)
        assert "TASK CONTEXT" not in result

    def test_no_user_messages_omits_user_section(self, make_strategy, fake_llm):
        context = [
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "s"},
        ]

        result = make_strategy().compress(context, len(context))

        assert "Previous User Messages" not in result
        assert (
            result
            == f"\n--- Conversation Summary ---\n\n{CODEX_SUMMARY_PREFIX}\n\nFAKE SUMMARY"
        )

    def test_previous_summary_messages_are_filtered(self, make_strategy, fake_llm):
        prior_summary = f"{CODEX_SUMMARY_PREFIX}\nold summary"
        context = [
            {"role": "user", "content": prior_summary},
            {"role": "user", "content": "real request"},
        ]

        result = make_strategy().compress(context, len(context))

        assert "old summary" not in result
        assert "User message 1: real request" in result
        assert "User message 2" not in result

    def test_summarization_failure_degrades_gracefully(
        self, make_strategy, fake_llm, sample_conversation
    ):
        fake_llm.error = RuntimeError("network down")

        result = make_strategy().compress(sample_conversation, len(sample_conversation))

        assert result.endswith(f"{CODEX_SUMMARY_PREFIX}\n\n(summarization failed)")
        assert "User message 1:" in result
        assert len(fake_llm.calls) == 1


class TestSummarizationPrompt:
    def test_enhanced_prompt_includes_goal_and_constraints(
        self, make_strategy, fake_llm, sample_conversation, sample_goal
    ):
        strategy = make_strategy()
        strategy.initialize(sample_goal["objective"], sample_goal["constraints"])
        strategy.compress(sample_conversation, len(sample_conversation))

        constraint_lines = "".join(f"  - {c}\n" for c in sample_goal["constraints"])
        expected = (
            f"{CODEX_SUMMARIZATION_PROMPT_ENHANCED}"
            "\n\nIMPORTANT CONTEXT TO PRESERVE:\n"
            f"Original Goal: {sample_goal['objective']}\n"
            f"Hard Constraints:\n{constraint_lines}"
            "\nYou MUST preserve the goal and all constraints in your summary.\n"
            f"\n\nConversation to summarize:\n\n{strategy.format_context(sample_conversation)}"
        )
        assert fake_llm.calls == [{"prompt": expected, "max_tokens": 500}]

    def test_enhanced_prompt_without_goal_has_no_preserve_block(
        self, make_strategy, fake_llm, sample_conversation
    ):
        strategy = make_strategy()
        strategy.compress(sample_conversation, len(sample_conversation))

        expected = (
            f"{CODEX_SUMMARIZATION_PROMPT_ENHANCED}"
            f"\n\nConversation to summarize:\n\n{strategy.format_context(sample_conversation)}"
        )
        assert fake_llm.calls[0]["prompt"] == expected

    def test_baseline_prompt_ignores_goal(
        self, make_strategy, fake_llm, sample_conversation, sample_goal
    ):
        strategy = make_strategy(use_goal_preservation=False)
        strategy.initialize(sample_goal["objective"], sample_goal["constraints"])
        strategy.compress(sample_conversation, len(sample_conversation))

        expected = (
            f"{CODEX_SUMMARIZATION_PROMPT_BASELINE}"
            f"\n\nConversation to summarize:\n\n{strategy.format_context(sample_conversation)}"
        )
        assert fake_llm.calls == [{"prompt": expected, "max_tokens": 500}]

    def test_prompt_contains_turn_ids_and_tool_calls(self, make_strategy, fake_llm):
        context = [
            {"id": 7, "role": "user", "content": "run tests"},
            {
                "id": 8,
                "role": "assistant",
                "content": "running",
                "tool_call": {"name": "pytest", "output": "3 passed"},
            },
        ]
        make_strategy().compress(context, len(context))

        prompt = fake_llm.calls[0]["prompt"]
        assert "Turn 7 (user): run tests" in prompt
        assert "Turn 8 (assistant): running\n  Tool: pytest -> 3 passed..." in prompt

    def test_summary_prompt_excludes_turns_after_trigger_point(
        self, make_strategy, fake_llm, sample_conversation
    ):
        make_strategy().compress(sample_conversation, trigger_point=2)

        prompt = fake_llm.calls[0]["prompt"]
        assert sample_conversation[1]["content"] in prompt
        assert sample_conversation[2]["content"] not in prompt


class TestTokenHelpers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", 0),
            ("a", 1),
            ("abcd", 1),
            ("abcde", 2),
            ("é", 1),
            ("€€", 2),
            ("x" * 4000, 1000),
        ],
    )
    def test_approx_token_count_uses_utf8_bytes(self, make_strategy, text, expected):
        assert make_strategy()._approx_token_count(text) == expected

    def test_truncate_text_returns_text_that_fits(self, make_strategy):
        assert make_strategy()._truncate_text("abcdefgh", max_tokens=2) == "abcdefgh"

    def test_truncate_text_keeps_first_half_of_byte_budget(self, make_strategy):
        result = make_strategy()._truncate_text("abcdefghijklmnop", max_tokens=2)
        assert result == "abcd...[truncated]"

    @pytest.mark.xfail(
        strict=True,
        reason="_truncate_text slices characters, not UTF-8 bytes, so multi-byte text overshoots the budget",
    )
    @pytest.mark.parametrize("char", ["é", "€", "😀"])
    def test_truncate_text_respects_budget_for_multibyte_text(
        self, make_strategy, char
    ):
        strategy = make_strategy()
        truncated = strategy._truncate_text(char * 1000, max_tokens=100)
        assert strategy._approx_token_count(truncated) <= 100


class TestSelectUserMessages:
    def test_all_messages_fit_in_chronological_order(self, make_strategy):
        messages = ["first", "second", "third"]
        assert make_strategy()._select_user_messages(messages) == messages

    def test_empty_input(self, make_strategy):
        assert make_strategy()._select_user_messages([]) == []

    def test_budget_keeps_most_recent_and_truncates_boundary_message(
        self, make_strategy
    ):
        oldest = "o" * 400
        middle = "m" * (10_000 * 4)
        newest = "n" * ((COMPACT_USER_MESSAGE_MAX_TOKENS - 5_000) * 4)

        selected = make_strategy()._select_user_messages([oldest, middle, newest])

        remaining_bytes = 5_000 * 4
        assert selected == ["m" * (remaining_bytes // 2) + "...[truncated]", newest]

    def test_exhausted_budget_drops_older_messages(self, make_strategy):
        newest = "n" * (COMPACT_USER_MESSAGE_MAX_TOKENS * 4)

        selected = make_strategy()._select_user_messages(["older", newest])

        assert selected == [newest]

    def test_oversized_compaction_keeps_latest_user_messages(
        self, make_strategy, fake_llm
    ):
        huge = "h" * (COMPACT_USER_MESSAGE_MAX_TOKENS * 4)
        context = [
            {"role": "user", "content": "dropped request"},
            {"role": "user", "content": huge},
        ]

        result = make_strategy().compress(context, len(context))

        assert "dropped request" not in result
        assert f"User message 1: {huge}" in result
        assert "User message 2" not in result
