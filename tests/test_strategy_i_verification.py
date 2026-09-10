"""Unit tests for Strategy I (A-MEM + Protected Core Hybrid) verification.

Extends the verification harness to cover Strategy I's core properties:
- Protected core integrity (delegated to Strategy F)
- Token budget edge cases
- Constraint preservation

Tests run WITHOUT API keys (uses mocks).
"""

import pytest
from strategies.strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore
from evaluation.strategy_verification import MockLLMClient


@pytest.fixture
def strategy_i():
    """Strategy I initialized with mock client."""
    s = StrategyI_AMemProtectedCore(system_prompt="Test agent", backend="openai")
    # Mock the LLM client in the underlying protected core strategy
    s._protected_core_strategy._client = MockLLMClient("MOCK_STRATEGY_I_SUMMARY")
    s.initialize(
        original_goal="Build a hybrid analytics system with memory recall",
        constraints=["Budget: $10k", "Ship in 12 weeks", "Support 100+ users"]
    )
    return s


class TestStrategyIVerification:
    """Verification tests for Strategy I hybrid approach."""
    
    def test_protected_core_integrity_via_delegation(self, strategy_i, monkeypatch):
        """Strategy I should preserve protected core via Strategy F delegation."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Old turn 1" * 50},
            {"id": 2, "role": "assistant", "content": "Old turn 2" * 50},
            {"id": 3, "role": "user", "content": "Recent turn"},
        ]
        
        result = strategy_i.compress(context, trigger_point=3)
        
        # Protected core should be present (delegated to Strategy F)
        assert "Build a hybrid analytics system" in result or "PROTECTED CORE" in result
        # At least one constraint should be preserved
        assert any(c in result for c in ["$10k", "12 weeks", "100+ users"])
    
    def test_token_budget_under_threshold(self, strategy_i, monkeypatch):
        """Strategy I should skip compression when under budget."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: False)
        
        context = [{"id": 1, "role": "user", "content": "Short message"}]
        result = strategy_i.compress(context, trigger_point=1)
        
        # Should not have mock summary (compression skipped)
        assert "MOCK_STRATEGY_I_SUMMARY" not in result
        # But should still have the actual content
        assert "Short message" in result
    
    def test_token_budget_over_threshold(self, strategy_i, monkeypatch):
        """Strategy I should compress when over budget."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Old turn 1" * 50},
            {"id": 2, "role": "assistant", "content": "Old turn 2" * 50},
            {"id": 3, "role": "user", "content": "Old turn 3" * 50},
            {"id": 4, "role": "assistant", "content": "Recent turn 4"},
            {"id": 5, "role": "user", "content": "Recent turn 5"},
        ]
        
        result = strategy_i.compress(context, trigger_point=5)
        
        # Compression should have happened (either via Strategy F summary or A-MEM memories)
        # We verify by checking that old turns are not fully present
        old_content = "Old turn 1" * 50
        assert old_content not in result  # Old content should be compressed
    
    def test_constraints_preserved_after_compression(self, strategy_i, monkeypatch):
        """Strategy I should preserve all constraints after compression."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Turn 1" * 50},
            {"id": 2, "role": "assistant", "content": "Turn 2" * 50},
        ]
        
        result = strategy_i.compress(context, trigger_point=2)
        
        # All constraints should be preserved (via Protected Core)
        assert "$10k" in result or "10k" in result
        assert "12 weeks" in result or "weeks" in result
        assert "100+ users" in result or "users" in result
    
    def test_goal_update_synchronization(self, strategy_i):
        """Strategy I should keep Protected Core and A-MEM synchronized on goal updates."""
        original_goal = strategy_i._protected_core_strategy.protected_core.current_goal
        
        new_goal = "Build B2C analytics for freelancers"
        strategy_i.update_goal(new_goal, rationale="Pivot to B2C market")
        
        # Protected Core should be updated
        assert strategy_i._protected_core_strategy.protected_core.current_goal == new_goal
        assert strategy_i._protected_core_strategy.protected_core.current_goal != original_goal
    
    def test_empty_context_handling(self, strategy_i):
        """Strategy I should handle empty context gracefully."""
        context = []
        result = strategy_i.compress(context, trigger_point=0)
        
        # Should not crash and should return something
        assert result is not None
        assert len(result) > 0


class TestStrategyIHarnessCompatibility:
    """Test that Strategy I works with the verification harness."""
    
    def test_harness_can_create_strategy_i(self):
        """Verification harness should be able to instantiate Strategy I."""
        from evaluation.strategy_verification import StrategyVerificationHarness
        
        harness = StrategyVerificationHarness(StrategyI_AMemProtectedCore)
        strategy = harness._create_strategy(system_prompt="Test")
        
        assert strategy is not None
        assert hasattr(strategy, '_protected_core_strategy')
    
    def test_harness_protected_core_check_works(self):
        """Harness should successfully verify Strategy I's protected core integrity."""
        from evaluation.strategy_verification import StrategyVerificationHarness
        
        harness = StrategyVerificationHarness(StrategyI_AMemProtectedCore)
        passed = harness.verify_protected_core_integrity()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "protected_core_integrity"
    
    def test_harness_token_budget_checks_work(self):
        """Harness should successfully verify Strategy I's token budget handling."""
        from evaluation.strategy_verification import StrategyVerificationHarness
        
        harness = StrategyVerificationHarness(StrategyI_AMemProtectedCore)
        
        under_passed = harness.verify_token_budget_under_threshold()
        over_passed = harness.verify_token_budget_over_threshold()
        
        assert under_passed is True
        assert over_passed is True
    
    def test_harness_constraint_preservation_works(self):
        """Harness should successfully verify Strategy I's constraint preservation."""
        from evaluation.strategy_verification import StrategyVerificationHarness
        
        harness = StrategyVerificationHarness(StrategyI_AMemProtectedCore)
        passed = harness.verify_constraint_preservation()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "constraint_preservation"
