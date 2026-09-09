"""Unit tests for Strategy F Protected Core (no API keys required)."""

import pytest

from strategies.strategy_f_protected_core import StrategyF_ProtectedCore, ProtectedCore


class MockLLMClient:
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        return "MOCK_HALO_SUMMARY"


@pytest.fixture
def strategy():
    s = StrategyF_ProtectedCore(system_prompt="You are a helpful agent.", backend="openai")
    s.client = MockLLMClient()
    s.initialize(
        original_goal="Build a B2B analytics dashboard",
        constraints=["Budget: Maximum $5k monthly cloud infrastructure cost", "Ship in 8 weeks"],
    )
    return s


class TestStrategyFUnit:
    def test_construct_without_api_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = StrategyF_ProtectedCore(system_prompt="sys", backend="openai")
        assert s.protected_core is None
        # Client is lazy — still not created
        assert s._client is None

    def test_initialize_sets_protected_core(self, strategy):
        assert isinstance(strategy.protected_core, ProtectedCore)
        assert strategy.protected_core.original_goal == "Build a B2B analytics dashboard"
        assert strategy.protected_core.current_goal == strategy.protected_core.original_goal
        assert len(strategy.protected_core.hard_constraints) == 2

    def test_skip_compression_preserves_real_history(self, strategy, monkeypatch):
        """Under-budget compress must not return the placeholder halo summary."""
        from evaluation import token_budget as tb

        # Force under-budget path
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: False)

        context = [
            {"id": 1, "role": "user", "content": "First real user message about schema choice"},
            {"id": 2, "role": "assistant", "content": "First real assistant reply"},
            {"id": 3, "role": "user", "content": "Second real user message"},
        ]
        result = strategy.compress(context, trigger_point=3)

        assert "[Compressed conversation summary would go here]" not in result
        assert "First real user message about schema choice" in result
        assert "PROTECTED CORE" in result
        assert "Build a B2B analytics dashboard" in result

    def test_compress_when_over_budget_uses_halo_summary(self, strategy, monkeypatch):
        from evaluation import token_budget as tb

        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)

        context = [
            {"id": 1, "role": "user", "content": "Old turn one"},
            {"id": 2, "role": "assistant", "content": "Old turn two"},
            {"id": 3, "role": "user", "content": "Old turn three"},
            {"id": 4, "role": "assistant", "content": "Recent assistant"},
            {"id": 5, "role": "user", "content": "Recent user"},
        ]
        result = strategy.compress(context, trigger_point=5)

        assert "MOCK_HALO_SUMMARY" in result
        assert "PROTECTED CORE" in result
        assert "Recent user" in result

    def test_update_goal_records_decision(self, strategy):
        strategy.update_goal("Build a B2C analytics app", rationale="Pivot to consumers")
        assert strategy.protected_core.current_goal == "Build a B2C analytics app"
        assert strategy.protected_core.original_goal == "Build a B2B analytics dashboard"
        assert any("Goal updated" in d.decision for d in strategy.protected_core.key_decisions)
