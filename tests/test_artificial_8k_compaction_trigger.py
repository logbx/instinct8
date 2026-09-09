"""
Tests for artificial 8K context window compaction trigger.

Tests the token budget logic to ensure compaction is triggered correctly
at the 8K artificial window threshold.
"""

import pytest
from evaluation.token_budget import TokenBudget, should_compact, estimate_tokens, BUDGET_8K
from strategies.strategy_b_codex import StrategyB_CodexCheckpoint


def test_budget_8k_trigger_tokens():
    """Test that 8K budget calculates correct trigger tokens."""
    # Expected: floor(8192 * 0.90) - 1024 = floor(7372.8) - 1024 = 7372 - 1024 = 6152
    assert BUDGET_8K.trigger_tokens == 7372 - 1024
    assert BUDGET_8K.trigger_tokens == 6348


def test_estimate_tokens_deterministic():
    """Test that token estimation is deterministic and returns reasonable values."""
    # Test basic properties
    text1 = "hello"
    text2 = "hello world"
    tokens1 = estimate_tokens(text1)
    tokens2 = estimate_tokens(text2)

    # Longer text should have more tokens
    assert tokens2 > tokens1
    assert tokens1 > 0
    assert tokens2 > 0

    # Same text should give same result
    assert estimate_tokens(text1) == tokens1
    assert estimate_tokens(text2) == tokens2


def test_should_compact_below_threshold():
    """Test that should_compact returns False for prompts below trigger threshold."""
    budget = BUDGET_8K

    # Create a prompt with just a few tokens - definitely below threshold
    below_threshold_text = "Hello world"
    tokens = estimate_tokens(below_threshold_text)
    assert tokens < budget.trigger_tokens  # Verify it's actually below
    assert not should_compact(below_threshold_text, budget)


def test_should_compact_above_threshold():
    """Test that should_compact returns True for prompts above trigger threshold."""
    budget = BUDGET_8K

    # Create a very large prompt that will definitely exceed the threshold
    above_threshold_text = "This is a very long message. " * 2000  # Should exceed 8K tokens
    tokens = estimate_tokens(above_threshold_text)
    assert tokens > budget.trigger_tokens  # Verify it's actually above
    assert should_compact(above_threshold_text, budget)


@pytest.mark.integration
def test_codex_strategy_renders_reconstructed_prompt():
    """Test that Codex strategy can render reconstructed prompts."""
    strategy = StrategyB_CodexCheckpoint(system_prompt="You are a helpful assistant.")

    # Create sample context
    context = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    # Render reconstructed prompt
    prompt = strategy.render_reconstructed_prompt(context, "Tell me a joke")

    # Check structure
    assert "System: You are a helpful assistant." in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there!" in prompt
    assert "User: How are you?" in prompt
    assert "User: Tell me a joke" in prompt

    # Check formatting
    lines = prompt.split("\n\n")
    assert len(lines) == 5  # system + 3 turns + user message


@pytest.mark.integration
def test_codex_strategy_skips_compression_when_below_budget():
    """Test that Codex strategy skips compression when below token budget."""
    # Create a budget with very high threshold
    large_budget = TokenBudget(
        window_tokens=100000,  # Very large window
        trigger_ratio=0.90,
        output_reserve_tokens=1024,
    )

    strategy = StrategyB_CodexCheckpoint(
        system_prompt="You are a helpful assistant.",
        token_budget=large_budget,
    )

    # Create context that's small enough to not trigger compression
    context = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # This should not compress and return the reconstructed prompt
    result = strategy.compress(context, trigger_point=2)

    # Should contain the original conversation
    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result
    assert "System: You are a helpful assistant." in result


@pytest.mark.integration
def test_codex_strategy_compresses_when_above_budget():
    """Test that Codex strategy compresses when above token budget."""
    # Create a budget with very low threshold to force compression
    small_budget = TokenBudget(
        window_tokens=100,  # Very small window
        trigger_ratio=0.90,
        output_reserve_tokens=10,
    )

    strategy = StrategyB_CodexCheckpoint(
        system_prompt="You are a helpful assistant.",
        token_budget=small_budget,
    )

    # Create context that's large enough to trigger compression
    # Make it large enough to exceed the small budget
    large_content = "This is a very long message. " * 100
    context = [
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": "I understand."},
    ]

    # This should compress (though we can't test the actual LLM call)
    # We'll mock the _summarize method to avoid calling the LLM
    original_summarize = strategy._summarize
    strategy._summarize = lambda x: "Mock summary of the conversation."

    try:
        result = strategy.compress(context, trigger_point=2)

        # Should be compressed format, not the raw reconstructed prompt
        assert "System: You are a helpful assistant." in result
        assert "--- Conversation Summary ---" in result
        assert "Mock summary of the conversation." in result

    finally:
        # Restore original method
        strategy._summarize = original_summarize


def test_deterministic_growth_to_trigger():
    """Test deterministic growth of prompt until it crosses trigger threshold."""
    budget = BUDGET_8K
    trigger_tokens = budget.trigger_tokens

    # Start with empty text
    text = ""
    current_tokens = 0

    # Grow text deterministically until we cross the threshold
    while current_tokens <= trigger_tokens:
        # Add a fixed amount each time
        text += "word "  # 5 chars
        current_tokens = estimate_tokens(text)

        # Once we cross the threshold, should_compact should return True
        if current_tokens > trigger_tokens:
            assert should_compact(text, budget)
            break

    # The text should now trigger compaction
    assert should_compact(text, budget)

    # If we have text, test that removing some makes it not trigger
    if text:
        # Remove some characters and verify it no longer triggers
        shorter_text = text[:-10]  # Remove last 10 chars
        shorter_tokens = estimate_tokens(shorter_text)
        if shorter_tokens <= trigger_tokens:
            assert not should_compact(shorter_text, budget)
