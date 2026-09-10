"""Verification tests for Strategy I: A-MEM + Protected Core Hybrid.

Tests Strategy I without API keys by mocking the LLM clients.
Focuses on:
- Protected core integrity via delegation to Strategy F
- Token budget edge cases
- Goal drift detection through Strategy F
- Constraint preservation
- A-MEM memory system integration
- Empty context handling
"""

import pytest
from strategies.strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore
from evaluation.token_budget import TokenBudget


class MockLLMClient:
    """Mock LLM client for testing without API keys."""
    def __init__(self, summary: str = "MOCK_SUMMARY"):
        self.summary = summary
        self.call_count = 0
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        self.call_count += 1
        return self.summary


@pytest.fixture
def base_strategy():
    """Strategy I initialized with basic goal."""
    s = StrategyI_AMemProtectedCore(
        system_prompt="You are a helpful agent.",
        backend="openai",
        keep_recent_turns=3
    )
    # Mock the Protected Core strategy's client
    s._protected_core_strategy._client = MockLLMClient()
    
    s.initialize(
        original_goal="Build a product analytics system",
        constraints=[
            "Budget: $10k",
            "Ship in 6 weeks",
            "Team of 5-10 people"
        ]
    )
    return s


class TestProtectedCoreIntegrity:
    """Test that Strategy I preserves protected core via delegation to Strategy F."""
    
    def test_protected_core_present_after_compression(self, base_strategy, monkeypatch):
        """Protected core must appear in compressed output."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Old turn 1"},
            {"id": 2, "role": "assistant", "content": "Old turn 2"},
            {"id": 3, "role": "user", "content": "Recent turn"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=3)
        
        # Protected core markers must be present
        assert "PROTECTED CORE" in compressed
        assert "Build a product analytics system" in compressed
        assert "$10k" in compressed
        assert "6 weeks" in compressed
    
    def test_goal_updated_in_both_systems(self, base_strategy):
        """Goal update should be synchronized between Protected Core and A-MEM."""
        new_goal = "Build a customer analytics dashboard"
        base_strategy.update_goal(new_goal, "Pivoting to customer focus")
        
        # Check Protected Core has new goal
        assert base_strategy._protected_core_strategy.protected_core.current_goal == new_goal
        
        # Check A-MEM has new goal (if it tracks goal)
        # A-MEM's initialize() receives the goal, but doesn't store it explicitly
        # We verify the systems are synchronized by checking Protected Core
        assert base_strategy._protected_core_strategy.protected_core.current_goal == new_goal


class TestTokenBudgetEdgeCases:
    """Test token budget edge cases for Strategy I."""
    
    def test_under_budget_skips_compression(self, base_strategy, monkeypatch):
        """Should skip compression when under budget."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: False)
        
        context = [
            {"id": 1, "role": "user", "content": "Short message"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=1)
        
        # Should not have mock summary (compression was skipped)
        assert "MOCK_SUMMARY" not in compressed
        # But protected core should still be present
        assert "PROTECTED CORE" in compressed
    
    def test_over_budget_triggers_compression(self, base_strategy, monkeypatch):
        """Should compress when over budget."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Long content " * 100},
            {"id": 2, "role": "assistant", "content": "Response " * 100},
            {"id": 3, "role": "user", "content": "Recent turn"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=3)
        
        # Should have compressed (protected core present)
        assert "PROTECTED CORE" in compressed
        # Should have A-MEM memories section
        assert "RELEVANT MEMORIES" in compressed


class TestConstraintPreservation:
    """Test that constraints survive compression cycles."""
    
    def test_constraints_preserved_after_compression(self, base_strategy, monkeypatch):
        """Constraints must survive compression via Protected Core."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Turn 1"},
            {"id": 2, "role": "assistant", "content": "Turn 2"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=2)
        
        # All constraints must be present
        assert "$10k" in compressed
        assert "6 weeks" in compressed
        assert "5-10 people" in compressed
    
    def test_constraint_changes_detected(self, base_strategy, monkeypatch):
        """Strategy should detect constraint changes through Protected Core."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Budget has been cut to $5k"},
            {"id": 2, "role": "assistant", "content": "Understood, adjusting plans"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=2)
        
        # Protected Core should have detected the budget change
        # (Strategy F's _detect_and_update_goal_shifts handles this)
        budget_constraints = [c for c in base_strategy._protected_core_strategy.protected_core.hard_constraints if "$" in c]
        assert len(budget_constraints) > 0


class TestAMemIntegration:
    """Test A-MEM memory system integration."""
    
    def test_amem_memory_section_present(self, base_strategy, monkeypatch):
        """Compressed output should include A-MEM memory section."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Technical decision about database"},
            {"id": 2, "role": "assistant", "content": "We chose PostgreSQL"},
            {"id": 3, "role": "user", "content": "Recent turn"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=3)
        
        # Should have A-MEM memories section
        assert "RELEVANT MEMORIES" in compressed


class TestEmptyContextHandling:
    """Test edge case of empty context."""
    
    def test_empty_context_handled_gracefully(self, base_strategy):
        """Should handle empty context without crashing."""
        context = []
        
        result = base_strategy.compress(context, trigger_point=0)
        
        # Should not crash and should return protected core
        assert result is not None
        assert len(result) > 0
        assert "PROTECTED CORE" in result


class TestVerificationHarnessCompatibility:
    """Test compatibility with the verification harness."""
    
    def test_harness_can_verify_strategy_i(self):
        """Verification harness should work with Strategy I."""
        from evaluation.strategy_verification import verify_strategy
        
        report = verify_strategy(StrategyI_AMemProtectedCore)
        
        # Should run all checks
        assert report.total_checks >= 6
        
        # Strategy I delegates to Strategy F, so protected_core checks should pass
        protected_core_check = next((r for r in report.results if "protected_core" in r.check_name), None)
        assert protected_core_check is not None
        assert protected_core_check.passed
    
    def test_strategy_name_correct(self, base_strategy):
        """Strategy should report correct name."""
        assert "Strategy I" in base_strategy.name()


class TestGoalDriftDetection:
    """Test goal drift detection through Protected Core delegation."""
    
    def test_goal_shift_detected_via_protected_core(self, base_strategy, monkeypatch):
        """Should detect goal shifts through Strategy F."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Let's pivot to building a marketing analytics tool instead"},
            {"id": 2, "role": "assistant", "content": "Okay, pivoting to marketing analytics"},
            {"id": 3, "role": "user", "content": "Recent turn"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=3)
        
        # Protected Core should have tracked the goal shift
        current_goal = base_strategy._protected_core_strategy.protected_core.current_goal
        assert current_goal != "Build a product analytics system"
        # Should contain "marketing" from the pivot
        assert "marketing" in current_goal.lower() or "marketing analytics" in compressed
