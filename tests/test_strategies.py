"""Tests for compression strategies."""

import pytest
from strategies.strategy_base import CompressionStrategy, Turn


class TestCompressionStrategyBase:
    """Test the base compression strategy interface."""

    def test_turn_dataclass(self):
        """Test Turn dataclass creation."""
        turn = Turn(
            id=1,
            role="user",
            content="Hello, world!",
            timestamp=None,
        )
        assert turn.id == 1
        assert turn.role == "user"
        assert turn.content == "Hello, world!"

    def test_strategy_is_abstract(self):
        """Test that CompressionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CompressionStrategy()


class TestCodexStrategy:
    """Tests for Codex-style checkpoint strategy."""

    def test_import(self):
        """Test that strategy can be imported."""
        from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
        assert StrategyB_CodexCheckpoint is not None

    @pytest.mark.integration
    def test_create_strategy(self):
        """Test strategy creation."""
        from strategies.strategy_b_codex import create_codex_strategy
        strategy = create_codex_strategy(
            system_prompt="You are a helpful assistant.",
        )
        assert strategy is not None
        assert strategy.name() == "Strategy B - Codex-Style Checkpoint"


class TestNaiveStrategy:
    """Tests for Naive Summarization strategy."""

    def test_import(self):
        """Test that strategy can be imported."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        assert StrategyA_NaiveSummarization is not None

    @pytest.mark.integration
    def test_create_strategy(self):
        """Test strategy creation."""
        from strategies.strategy_a_naive import create_naive_strategy
        strategy = create_naive_strategy(backend="openai")
        assert strategy is not None
        assert strategy.name() == "Strategy A - Naive Summarization"

    @pytest.mark.integration
    def test_initialize(self):
        """Test strategy initialization."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        
        # Mock the LLM client to avoid API calls
        class MockLLMClient:
            def complete(self, prompt: str, max_tokens: int = 500) -> str:
                return "Mock summary"
        
        strategy = StrategyA_NaiveSummarization(backend="openai")
        strategy.client = MockLLMClient()
        
        strategy.initialize(
            original_goal="Test goal",
            constraints=["Constraint 1", "Constraint 2"]
        )
        
        assert strategy.original_goal == "Test goal"
        assert strategy.constraints == ["Constraint 1", "Constraint 2"]

    @pytest.mark.integration
    def test_update_goal_noop(self):
        """Test that update_goal is a no-op (doesn't track goal updates)."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        
        class MockLLMClient:
            def complete(self, prompt: str, max_tokens: int = 500) -> str:
                return "Mock summary"
        
        strategy = StrategyA_NaiveSummarization(backend="openai")
        strategy.client = MockLLMClient()
        strategy.initialize("Original goal", ["Constraint 1"])
        
        # Update goal - should not change anything
        strategy.update_goal("New goal", "Rationale")
        
        # Original goal should still be stored (but not used)
        assert strategy.original_goal == "Original goal"

    @pytest.mark.integration
    def test_compress_summarizes_all(self):
        """Test that compress summarizes all context without protection."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        
        summary_called = False
        prompt_received = None
        
        class MockLLMClient:
            def complete(self, prompt: str, max_tokens: int = 500) -> str:
                nonlocal summary_called, prompt_received
                summary_called = True
                prompt_received = prompt
                return "This is a test summary of the conversation."
        
        strategy = StrategyA_NaiveSummarization(backend="openai")
        strategy.client = MockLLMClient()
        strategy.initialize("Test goal", ["Constraint 1"])
        
        context = [
            {"id": 1, "role": "user", "content": "First message"},
            {"id": 2, "role": "assistant", "content": "First response"},
            {"id": 3, "role": "user", "content": "Second message"},
        ]
        
        result = strategy.compress(context, trigger_point=3)
        
        assert summary_called
        assert "Summarize this conversation in 3-4 sentences" in prompt_received
        assert "Previous conversation summary:" in result
        assert "This is a test summary" in result
        # Should NOT contain goal or constraints in output
        assert "Test goal" not in result
        assert "Constraint 1" not in result

    @pytest.mark.integration
    def test_compress_empty_context(self):
        """Test compress with empty context."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        
        class MockLLMClient:
            def complete(self, prompt: str, max_tokens: int = 500) -> str:
                return "Summary"
        
        strategy = StrategyA_NaiveSummarization(backend="openai")
        strategy.client = MockLLMClient()
        
        result = strategy.compress([], trigger_point=0)
        
        assert "Previous conversation summary:" in result
        assert "(No previous conversation)" in result

    @pytest.mark.integration
    def test_always_compresses(self):
        """Test that strategy always compresses (no token budget checking)."""
        from strategies.strategy_a_naive import StrategyA_NaiveSummarization
        
        compress_called = False
        
        class MockLLMClient:
            def complete(self, prompt: str, max_tokens: int = 500) -> str:
                nonlocal compress_called
                compress_called = True
                return "Summary"
        
        strategy = StrategyA_NaiveSummarization(backend="openai")
        strategy.client = MockLLMClient()
        
        # Even with very small context, should still compress
        context = [{"id": 1, "role": "user", "content": "Hi"}]
        result = strategy.compress(context, trigger_point=1)
        
        assert compress_called
        assert "Previous conversation summary:" in result
