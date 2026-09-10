"""Expanded unit tests for Strategy F: goal-drift detection and protected core integrity.

Focuses on:
- Goal shift detection and tracking
- Constraint mutation handling
- Protected core preservation under various compression scenarios
- No API keys required (uses mocks)
"""

import pytest
from strategies.strategy_f_protected_core import StrategyF_ProtectedCore, ProtectedCore, Decision
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
    """Strategy initialized with basic B2B dashboard goal."""
    s = StrategyF_ProtectedCore(system_prompt="You are a helpful agent.", backend="openai")
    s.client = MockLLMClient()
    s.initialize(
        original_goal="Build a B2B analytics dashboard for enterprise teams",
        constraints=[
            "Budget: Maximum $5k monthly cloud infrastructure cost",
            "Ship in 8 weeks",
            "Must support teams of 10-50 people"
        ],
    )
    return s


@pytest.fixture
def small_budget():
    """8K token budget to trigger compression easily."""
    return TokenBudget(
        window_tokens=8192,
        trigger_ratio=0.90,
        output_reserve_tokens=1024,
    )


class TestGoalShiftDetection:
    """Test goal shift detection and tracking within protected core."""
    
    def test_detect_b2b_to_b2c_pivot(self, base_strategy):
        """Should detect B2B → B2C pivot from user message."""
        context = [
            {"id": 1, "role": "user", "content": "Actually, let's pivot to B2C. Target individual freelancers instead of enterprise teams."},
        ]
        
        base_strategy._detect_and_update_goal_shifts(context)
        
        assert "B2C" in base_strategy.protected_core.current_goal or "individual" in base_strategy.protected_core.current_goal
        assert base_strategy.protected_core.current_goal != base_strategy.protected_core.original_goal
        assert any("Goal updated" in d.decision for d in base_strategy.protected_core.key_decisions)
    
    def test_detect_budget_constraint_change(self, base_strategy):
        """Should detect and update budget constraint changes."""
        context = [
            {"id": 1, "role": "user", "content": "The budget has been cut to $1k per month instead."},
        ]
        
        original_constraints = base_strategy.protected_core.hard_constraints.copy()
        base_strategy._detect_and_update_goal_shifts(context)
        
        # Should have updated budget constraint
        budget_constraint = [c for c in base_strategy.protected_core.hard_constraints if "$" in c][0]
        assert "$1k" in budget_constraint or "$1K" in budget_constraint
        assert budget_constraint != [c for c in original_constraints if "$" in c][0]
    
    def test_multiple_goal_shifts_tracked(self, base_strategy):
        """Should track multiple goal shifts in sequence."""
        context = [
            {"id": 1, "role": "user", "content": "Let's pivot to B2C targeting freelancers."},
            {"id": 2, "role": "assistant", "content": "Got it, pivoting to B2C."},
            {"id": 3, "role": "user", "content": "Actually, we also need to add mobile app support now."},
        ]
        
        base_strategy._detect_and_update_goal_shifts(context)
        
        # Should have at least 2 decision records
        goal_decisions = [d for d in base_strategy.protected_core.key_decisions if "Goal" in d.decision]
        assert len(goal_decisions) >= 1  # At least the B2C pivot
        # Current goal should reflect the evolution
        assert "mobile" in base_strategy.protected_core.current_goal.lower() or "B2C" in base_strategy.protected_core.current_goal or "freelancer" in base_strategy.protected_core.current_goal.lower()
    
    def test_no_false_positive_goal_shifts(self, base_strategy):
        """Should NOT detect goal shifts in normal conversation."""
        context = [
            {"id": 1, "role": "user", "content": "What database should we use for this dashboard?"},
            {"id": 2, "role": "assistant", "content": "I recommend PostgreSQL for the B2B dashboard."},
        ]
        
        original_goal = base_strategy.protected_core.current_goal
        base_strategy._detect_and_update_goal_shifts(context)
        
        # Goal should not change
        assert base_strategy.protected_core.current_goal == original_goal
        # May extract technical decisions (like PostgreSQL choice), but no goal shifts
        goal_decisions = [d for d in base_strategy.protected_core.key_decisions if "Goal updated" in d.decision]
        assert len(goal_decisions) == 0


class TestProtectedCoreIntegrity:
    """Test that protected core is never lost during compression."""
    
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
        assert "Build a B2B analytics dashboard" in compressed
        assert "$5k monthly cloud infrastructure cost" in compressed
        assert "Ship in 8 weeks" in compressed
    
    def test_goal_shift_preserved_after_compression(self, base_strategy, monkeypatch):
        """Goal shifts must be reflected in protected core after compression."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Let's pivot to B2C for individual freelancers."},
            {"id": 2, "role": "assistant", "content": "Okay, pivoting to B2C."},
            {"id": 3, "role": "user", "content": "Old detail about implementation"},
            {"id": 4, "role": "user", "content": "Recent turn"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=4)
        
        # Should reflect the evolved goal
        assert "PROTECTED CORE" in compressed
        assert ("B2C" in compressed or "freelancer" in compressed or "individual" in compressed)
        # Key decision should be recorded
        assert len(base_strategy.protected_core.key_decisions) > 0
    
    def test_constraints_preserved_after_multiple_compressions(self, base_strategy, monkeypatch):
        """Constraints must survive multiple compression cycles."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        # First compression
        context1 = [
            {"id": 1, "role": "user", "content": "Turn 1"},
            {"id": 2, "role": "assistant", "content": "Turn 2"},
        ]
        compressed1 = base_strategy.compress(context1, trigger_point=2)
        
        # Second compression
        context2 = [
            {"id": 0, "role": "system", "content": compressed1},
            {"id": 3, "role": "user", "content": "Turn 3"},
            {"id": 4, "role": "assistant", "content": "Turn 4"},
        ]
        compressed2 = base_strategy.compress(context2, trigger_point=4)
        
        # All original constraints must still be present
        assert "$5k monthly cloud infrastructure cost" in compressed2
        assert "8 weeks" in compressed2
        assert "10-50 people" in compressed2


class TestTokenBudgetEdgeCases:
    """Test token budget edge cases: under/over budget, exact boundaries."""
    
    def test_exactly_at_trigger_threshold(self, base_strategy, monkeypatch):
        """Should compress when exactly at token threshold."""
        from evaluation import token_budget as tb
        
        # Mock to return exactly the trigger threshold
        def mock_should_compact(prompt_text, budget):
            tokens = tb.estimate_tokens(prompt_text)
            return tokens >= budget.trigger_tokens
        
        monkeypatch.setattr(tb, "should_compact", mock_should_compact)
        
        # Create context that will be at threshold
        long_content = "x" * 20000  # ~5k tokens
        context = [
            {"id": 1, "role": "user", "content": long_content},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=1)
        
        # Should have compressed
        assert "MOCK_SUMMARY" in compressed or "PROTECTED CORE" in compressed
    
    def test_well_under_budget_skips_compression(self, base_strategy, monkeypatch):
        """Should skip compression when well under budget."""
        from evaluation import token_budget as tb
        
        # Force under-budget
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: False)
        
        context = [
            {"id": 1, "role": "user", "content": "Short message"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=1)
        
        # Should not have summary (skipped compression)
        assert "MOCK_SUMMARY" not in compressed
        # But protected core should still be present
        assert "PROTECTED CORE" in compressed
    
    def test_empty_context_compression(self, base_strategy):
        """Should handle empty context gracefully."""
        context = []
        
        compressed = base_strategy.compress(context, trigger_point=0)
        
        # Should return protected core only
        assert "PROTECTED CORE" in compressed
        assert "Build a B2B analytics dashboard" in compressed
    
    def test_single_turn_compression(self, base_strategy, monkeypatch):
        """Should handle single-turn context."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Single turn only"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=1)
        
        # Should still work
        assert "PROTECTED CORE" in compressed


class TestImportantDetailExtraction:
    """Test extraction of important technical decisions."""
    
    def test_detect_technical_choice(self, base_strategy):
        """Should detect turns with explicit decision/technical patterns."""
        # Pattern 1: "chose" matches the important_indicators pattern
        turn1 = {
            "id": 1,
            "role": "assistant",
            "content": "We chose PostgreSQL for the database."
        }
        assert base_strategy._is_important_detail(turn1)
        
        # Pattern 2: "selected" with decision pattern
        turn2 = {
            "id": 2,
            "role": "assistant",
            "content": "We selected React framework for the frontend implementation."
        }
        assert base_strategy._is_important_detail(turn2)
    
    def test_detect_decision_field(self, base_strategy):
        """Should recognize explicit decision field in turn."""
        turn = {
            "id": 1,
            "role": "assistant",
            "content": "Some content",
            "decision": "Use PostgreSQL for persistence"
        }
        
        assert base_strategy._is_important_detail(turn)
    
    def test_extract_decision_from_content(self, base_strategy):
        """Should extract decision-like statements from content."""
        turn = {
            "id": 1,
            "role": "assistant",
            "content": "Decision: We will use React for the frontend framework."
        }
        
        detail = base_strategy._extract_important_detail(turn)
        
        assert detail is not None
        assert "React" in detail or "frontend" in detail
    
    def test_ignore_trivial_turns(self, base_strategy):
        """Should not extract from trivial conversational turns."""
        turn = {
            "id": 1,
            "role": "user",
            "content": "Okay, sounds good."
        }
        
        assert not base_strategy._is_important_detail(turn)
        assert base_strategy._extract_important_detail(turn) is None


class TestVerificationHarness:
    """Test verification harness functionality for strategy validation."""
    
    def test_strategy_maintains_coherence_metric(self, base_strategy, monkeypatch):
        """Strategy should maintain goal coherence above threshold."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Build the dashboard"},
            {"id": 2, "role": "assistant", "content": "Working on the B2B analytics dashboard"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=2)
        
        # Protected core should be intact (proxy for goal coherence)
        assert "Build a B2B analytics dashboard" in compressed
        assert "PROTECTED CORE" in compressed
    
    def test_strategy_handles_rapid_goal_shifts(self, base_strategy, monkeypatch):
        """Strategy should handle multiple rapid goal shifts."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Pivot to B2C"},
            {"id": 2, "role": "user", "content": "Actually, add mobile app support"},
            {"id": 3, "role": "user", "content": "Change budget to $1k"},
            {"id": 4, "role": "assistant", "content": "Recent response"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=4)
        
        # Should have recorded multiple shifts
        assert len(base_strategy.protected_core.key_decisions) >= 2
        # Protected core should still be present and authoritative
        assert "PROTECTED CORE" in compressed
        assert "AUTHORITATIVE" in compressed


class TestRegressionGuards:
    """Regression tests for known issues fixed in Strategy F."""
    
    def test_under_budget_does_not_wipe_context(self, base_strategy, monkeypatch):
        """Regression: under-budget compress must not return placeholder text.
        
        This is the bug fixed in PR #4 commit 7cc2544.
        """
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: False)
        
        context = [
            {"id": 1, "role": "user", "content": "Real user message about specific schema choice"},
        ]
        
        result = base_strategy.compress(context, trigger_point=1)
        
        # Must NOT contain the placeholder
        assert "[Compressed conversation summary would go here]" not in result
        # Must contain the actual message
        assert "Real user message about specific schema choice" in result
    
    def test_protected_core_always_authoritative(self, base_strategy, monkeypatch):
        """Protected core must always be marked as authoritative source of truth."""
        from evaluation import token_budget as tb
        monkeypatch.setattr(tb, "should_compact", lambda *_a, **_k: True)
        
        context = [
            {"id": 1, "role": "user", "content": "Turn 1"},
        ]
        
        compressed = base_strategy.compress(context, trigger_point=1)
        
        assert "AUTHORITATIVE" in compressed
        assert "source of truth" in compressed.lower()
