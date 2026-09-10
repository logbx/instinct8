"""
Strategy Verification Harness

Provides automated verification gates for compression strategy changes.
Validates that strategies maintain core properties across scenarios:
- Token budget edge cases
- Protected core integrity
- Goal drift prevention
- Constraint preservation

This harness runs WITHOUT API keys (uses mocks) for fast CI validation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import pytest


@dataclass
class VerificationResult:
    """Result from a single verification check."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class StrategyVerificationReport:
    """Aggregated verification report for a strategy."""
    strategy_name: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[VerificationResult]
    
    @property
    def all_passed(self) -> bool:
        return self.failed_checks == 0
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.all_passed else "❌ FAILED"
        return f"{status}: {self.passed_checks}/{self.total_checks} checks passed for {self.strategy_name}"


class MockLLMClient:
    """Mock LLM client for verification without API keys."""
    def __init__(self, summary: str = "VERIFICATION_MOCK_SUMMARY"):
        self.summary = summary
        self.call_count = 0
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        self.call_count += 1
        return self.summary


class StrategyVerificationHarness:
    """
    Verification harness for compression strategies.
    
    Runs automated checks to verify strategy correctness:
    1. Token budget edge cases (under, at, over threshold)
    2. Protected core integrity (never lost)
    3. Goal drift prevention (shifts tracked)
    4. Constraint preservation (survive compression)
    5. Important detail extraction (technical decisions)
    """
    
    def __init__(self, strategy_class, mock_client: Optional[MockLLMClient] = None):
        """
        Initialize verification harness.
        
        Args:
            strategy_class: The compression strategy class to verify
            mock_client: Optional mock LLM client (creates default if None)
        """
        self.strategy_class = strategy_class
        self.mock_client = mock_client or MockLLMClient()
        self.results: List[VerificationResult] = []
    
    def _create_strategy(self, **kwargs) -> Any:
        """Create strategy instance with mock client."""
        strategy = self.strategy_class(**kwargs)
        # Set _client directly to avoid triggering lazy initialization
        if hasattr(strategy, '_client'):
            strategy._client = self.mock_client
        return strategy
    
    def _record_result(self, check_name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Record a verification result."""
        result = VerificationResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details=details
        )
        self.results.append(result)
    
    def verify_token_budget_under_threshold(self) -> bool:
        """Verify strategy skips compression when under budget."""
        try:
            from evaluation import token_budget as tb
            
            strategy = self._create_strategy(system_prompt="Test")
            strategy.initialize(
                original_goal="Test goal",
                constraints=["Test constraint"]
            )
            
            # Mock to force under-budget
            original_should_compact = tb.should_compact
            tb.should_compact = lambda *_a, **_k: False
            
            context = [{"id": 1, "role": "user", "content": "Short message"}]
            result = strategy.compress(context, trigger_point=1)
            
            # Restore
            tb.should_compact = original_should_compact
            
            # Verify no mock summary (compression was skipped)
            passed = "VERIFICATION_MOCK_SUMMARY" not in result
            msg = "Under-budget compression correctly skipped" if passed else "Under-budget compression incorrectly triggered"
            
            self._record_result("token_budget_under_threshold", passed, msg, {
                "compressed_result_length": len(result)
            })
            return passed
            
        except Exception as e:
            self._record_result("token_budget_under_threshold", False, f"Exception: {e}")
            return False
    
    def verify_token_budget_over_threshold(self) -> bool:
        """Verify strategy compresses when over budget."""
        try:
            from evaluation import token_budget as tb
            
            strategy = self._create_strategy(system_prompt="Test", keep_recent_turns=2)
            strategy.initialize(
                original_goal="Test goal",
                constraints=["Test constraint"]
            )
            
            # Mock to force over-budget
            original_should_compact = tb.should_compact
            tb.should_compact = lambda *_a, **_k: True
            
            # Create enough turns so that some will be compressed (not kept as recent)
            context = [
                {"id": 1, "role": "user", "content": "Old turn 1" * 50},
                {"id": 2, "role": "assistant", "content": "Old turn 2" * 50},
                {"id": 3, "role": "user", "content": "Old turn 3" * 50},
                {"id": 4, "role": "assistant", "content": "Recent turn 4"},
                {"id": 5, "role": "user", "content": "Recent turn 5"},
            ]
            result = strategy.compress(context, trigger_point=5)
            
            # Restore
            tb.should_compact = original_should_compact
            
            # Verify mock summary present (compression happened)
            passed = "VERIFICATION_MOCK_SUMMARY" in result
            msg = "Over-budget compression correctly triggered" if passed else "Over-budget compression failed to trigger"
            
            self._record_result("token_budget_over_threshold", passed, msg, {
                "compressed_result_length": len(result)
            })
            return passed
            
        except Exception as e:
            self._record_result("token_budget_over_threshold", False, f"Exception: {e}")
            return False
    
    def verify_protected_core_integrity(self) -> bool:
        """Verify protected core survives compression."""
        try:
            from evaluation import token_budget as tb
            
            strategy = self._create_strategy(system_prompt="Test")
            original_goal = "Build a production analytics system"
            constraints = ["Budget: $5k", "Ship in 8 weeks"]
            
            strategy.initialize(original_goal=original_goal, constraints=constraints)
            
            # Force compression
            original_should_compact = tb.should_compact
            tb.should_compact = lambda *_a, **_k: True
            
            context = [
                {"id": 1, "role": "user", "content": "Turn 1"},
                {"id": 2, "role": "assistant", "content": "Turn 2"},
            ]
            result = strategy.compress(context, trigger_point=2)
            
            # Restore
            tb.should_compact = original_should_compact
            
            # Verify protected elements present
            has_goal = original_goal in result
            has_constraint_1 = constraints[0] in result
            has_constraint_2 = constraints[1] in result
            passed = has_goal and has_constraint_1 and has_constraint_2
            
            msg = "Protected core intact after compression" if passed else "Protected core lost elements"
            
            self._record_result("protected_core_integrity", passed, msg, {
                "has_goal": has_goal,
                "has_constraint_1": has_constraint_1,
                "has_constraint_2": has_constraint_2
            })
            return passed
            
        except Exception as e:
            self._record_result("protected_core_integrity", False, f"Exception: {e}")
            return False
    
    def verify_goal_drift_detection(self) -> bool:
        """Verify strategy detects and tracks goal shifts."""
        try:
            strategy = self._create_strategy(system_prompt="Test")
            strategy.initialize(
                original_goal="Build B2B dashboard",
                constraints=["Enterprise focus"]
            )
            
            # Simulate goal shift in context
            context = [
                {"id": 1, "role": "user", "content": "Let's pivot to B2C targeting freelancers instead."},
            ]
            
            if hasattr(strategy, '_detect_and_update_goal_shifts'):
                strategy._detect_and_update_goal_shifts(context)
                
                # Check if goal was updated
                current_goal = strategy.protected_core.current_goal if hasattr(strategy, 'protected_core') else None
                original_goal = "Build B2B dashboard"
                
                passed = current_goal and current_goal != original_goal
                msg = "Goal shift correctly detected and tracked" if passed else "Goal shift not detected"
                
                self._record_result("goal_drift_detection", passed, msg, {
                    "original_goal": original_goal,
                    "current_goal": current_goal
                })
                return passed
            else:
                # Strategy doesn't have drift detection - that's okay
                self._record_result("goal_drift_detection", True, "Strategy does not implement drift detection (N/A)")
                return True
                
        except Exception as e:
            self._record_result("goal_drift_detection", False, f"Exception: {e}")
            return False
    
    def verify_constraint_preservation(self) -> bool:
        """Verify constraints survive multiple compression cycles."""
        try:
            from evaluation import token_budget as tb
            
            strategy = self._create_strategy(system_prompt="Test")
            constraints = ["Budget: $5k", "Ship in 8 weeks", "Team of 10-50"]
            strategy.initialize(
                original_goal="Build system",
                constraints=constraints
            )
            
            # Force compression
            original_should_compact = tb.should_compact
            tb.should_compact = lambda *_a, **_k: True
            
            # First compression
            context1 = [{"id": 1, "role": "user", "content": "Turn 1"}]
            result1 = strategy.compress(context1, trigger_point=1)
            
            # Second compression
            context2 = [
                {"id": 0, "role": "system", "content": result1},
                {"id": 2, "role": "user", "content": "Turn 2"}
            ]
            result2 = strategy.compress(context2, trigger_point=2)
            
            # Restore
            tb.should_compact = original_should_compact
            
            # All constraints must survive
            has_c1 = constraints[0] in result2
            has_c2 = constraints[1] in result2
            has_c3 = constraints[2] in result2
            passed = has_c1 and has_c2 and has_c3
            
            msg = "Constraints preserved through multiple compressions" if passed else "Constraints lost during compression"
            
            self._record_result("constraint_preservation", passed, msg, {
                "has_constraint_1": has_c1,
                "has_constraint_2": has_c2,
                "has_constraint_3": has_c3
            })
            return passed
            
        except Exception as e:
            self._record_result("constraint_preservation", False, f"Exception: {e}")
            return False
    
    def verify_empty_context_handling(self) -> bool:
        """Verify strategy handles edge case of empty context."""
        try:
            strategy = self._create_strategy(system_prompt="Test")
            strategy.initialize(
                original_goal="Test goal",
                constraints=["Test constraint"]
            )
            
            context = []
            result = strategy.compress(context, trigger_point=0)
            
            # Should not crash and should return something
            passed = result is not None and len(result) > 0
            msg = "Empty context handled gracefully" if passed else "Empty context caused failure"
            
            self._record_result("empty_context_handling", passed, msg, {
                "result_length": len(result) if result else 0
            })
            return passed
            
        except Exception as e:
            self._record_result("empty_context_handling", False, f"Exception: {e}")
            return False
    
    def run_all_verifications(self) -> StrategyVerificationReport:
        """
        Run all verification checks and return aggregated report.
        
        Returns:
            StrategyVerificationReport with results from all checks
        """
        self.results = []  # Reset results
        
        # Run all verification checks
        self.verify_token_budget_under_threshold()
        self.verify_token_budget_over_threshold()
        self.verify_protected_core_integrity()
        self.verify_goal_drift_detection()
        self.verify_constraint_preservation()
        self.verify_empty_context_handling()
        
        # Compile report
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        report = StrategyVerificationReport(
            strategy_name=self.strategy_class.__name__,
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            results=self.results
        )
        
        return report


def verify_strategy(strategy_class) -> StrategyVerificationReport:
    """
    Convenience function to verify a strategy.
    
    Args:
        strategy_class: The compression strategy class to verify
    
    Returns:
        StrategyVerificationReport with verification results
    """
    harness = StrategyVerificationHarness(strategy_class)
    return harness.run_all_verifications()


def print_verification_report(report: StrategyVerificationReport):
    """Print a human-readable verification report."""
    print("\n" + "=" * 60)
    print(f"Strategy Verification Report: {report.strategy_name}")
    print("=" * 60)
    print(report.summary())
    print("\nDetailed Results:")
    print("-" * 60)
    
    for result in report.results:
        status = "✅" if result.passed else "❌"
        print(f"{status} {result.check_name}: {result.message}")
        if result.details:
            for key, value in result.details.items():
                print(f"   - {key}: {value}")
    
    print("=" * 60 + "\n")
