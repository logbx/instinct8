"""Test the strategy verification harness itself."""

import pytest
from evaluation.strategy_verification import (
    StrategyVerificationHarness,
    verify_strategy,
    StrategyVerificationReport,
)
from strategies.strategy_f_protected_core import StrategyF_ProtectedCore


class TestVerificationHarness:
    """Test the verification harness infrastructure."""
    
    def test_harness_creates_strategy_with_mock(self):
        """Harness should create strategy instance with mock client."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        strategy = harness._create_strategy(system_prompt="Test")
        
        assert strategy is not None
        assert hasattr(strategy, 'client')
        assert strategy.client == harness.mock_client
    
    def test_harness_records_results(self):
        """Harness should record verification results."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        harness._record_result("test_check", True, "Test passed")
        
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "test_check"
        assert harness.results[0].passed is True
    
    def test_run_all_verifications_strategy_f(self):
        """Should run all verifications for Strategy F."""
        report = verify_strategy(StrategyF_ProtectedCore)
        
        assert isinstance(report, StrategyVerificationReport)
        assert report.strategy_name == "StrategyF_ProtectedCore"
        assert report.total_checks >= 6  # At least 6 verification checks
        assert report.passed_checks > 0
        
        # Strategy F should pass all checks
        assert report.all_passed, f"Failed checks: {[r.check_name for r in report.results if not r.passed]}"
    
    def test_verification_report_properties(self):
        """Verification report should have correct properties."""
        report = verify_strategy(StrategyF_ProtectedCore)
        
        assert report.total_checks == report.passed_checks + report.failed_checks
        assert len(report.results) == report.total_checks
        assert isinstance(report.summary(), str)
        assert report.strategy_name in report.summary()


class TestIndividualVerifications:
    """Test individual verification checks."""
    
    def test_verify_token_budget_under_threshold(self):
        """Should verify under-budget behavior."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_token_budget_under_threshold()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "token_budget_under_threshold"
    
    def test_verify_token_budget_over_threshold(self):
        """Should verify over-budget behavior."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_token_budget_over_threshold()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "token_budget_over_threshold"
    
    def test_verify_protected_core_integrity(self):
        """Should verify protected core preservation."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_protected_core_integrity()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "protected_core_integrity"
    
    def test_verify_goal_drift_detection(self):
        """Should verify goal drift detection."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_goal_drift_detection()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "goal_drift_detection"
    
    def test_verify_constraint_preservation(self):
        """Should verify constraint preservation."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_constraint_preservation()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "constraint_preservation"
    
    def test_verify_empty_context_handling(self):
        """Should verify empty context handling."""
        harness = StrategyVerificationHarness(StrategyF_ProtectedCore)
        passed = harness.verify_empty_context_handling()
        
        assert passed is True
        assert len(harness.results) == 1
        assert harness.results[0].check_name == "empty_context_handling"
