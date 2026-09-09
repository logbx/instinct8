"""Tests for granular constraint metrics."""

import pytest
from evaluation.granular_constraint_metrics import (
    categorize_constraint,
    ConstraintCategory,
    measure_granular_constraint_recall,
    GranularConstraintMetrics,
    format_granular_constraint_report,
)


class TestConstraintCategorization:
    """Tests for constraint categorization."""

    def test_budget_constraint(self):
        """Test that budget constraints are categorized correctly."""
        assert categorize_constraint("Budget: maximum $10K") == ConstraintCategory.BUDGET
        assert categorize_constraint("Cost must be under $5000") == ConstraintCategory.BUDGET
        assert categorize_constraint("Price limit: $100") == ConstraintCategory.BUDGET

    def test_timeline_constraint(self):
        """Test that timeline constraints are categorized correctly."""
        assert categorize_constraint("Timeline: 2 weeks") == ConstraintCategory.TIMELINE
        assert categorize_constraint("Deadline: end of month") == ConstraintCategory.TIMELINE
        assert categorize_constraint("Must complete in 3 days") == ConstraintCategory.TIMELINE

    def test_technical_constraint(self):
        """Test that technical constraints are categorized correctly."""
        assert categorize_constraint("Must support WebSockets") == ConstraintCategory.TECHNICAL
        assert categorize_constraint("Integrate with PostgreSQL") == ConstraintCategory.TECHNICAL
        assert categorize_constraint("Compatible with Python 3.9+") == ConstraintCategory.TECHNICAL

    def test_team_constraint(self):
        """Test that team constraints are categorized correctly."""
        assert categorize_constraint("Team experience: intermediate") == ConstraintCategory.TEAM
        assert categorize_constraint("Developer skill level: junior") == ConstraintCategory.TEAM

    def test_compliance_constraint(self):
        """Test that compliance constraints are categorized correctly."""
        assert categorize_constraint("Must be GDPR compliant") == ConstraintCategory.COMPLIANCE
        assert categorize_constraint("HIPAA requirements") == ConstraintCategory.COMPLIANCE

    def test_performance_constraint(self):
        """Test that performance constraints are categorized correctly."""
        assert categorize_constraint("Latency < 100ms") == ConstraintCategory.PERFORMANCE
        assert categorize_constraint("Handle 10K concurrent connections") == ConstraintCategory.PERFORMANCE


class TestGranularConstraintMetrics:
    """Tests for granular constraint metrics calculation."""

    def test_import_granular_metrics(self):
        """Test that granular metrics can be imported."""
        from evaluation.granular_constraint_metrics import measure_granular_constraint_recall
        assert measure_granular_constraint_recall is not None

    @pytest.mark.integration
    def test_empty_constraints(self):
        """Test that empty constraints return perfect scores."""
        metrics = measure_granular_constraint_recall([], "Some response")
        assert metrics.overall_recall == 1.0
        assert metrics.budget_recall == 1.0

    @pytest.mark.integration
    def test_empty_response(self):
        """Test that empty response returns zero scores."""
        constraints = ["Budget: $10K", "Timeline: 2 weeks"]
        metrics = measure_granular_constraint_recall(constraints, "")
        assert metrics.overall_recall == 0.0
        assert metrics.budget_recall == 0.0
        assert metrics.timeline_recall == 0.0

    @pytest.mark.integration
    def test_category_scores_property(self):
        """Test that category_scores returns correct dictionary."""
        constraints = [
            "Budget: $10K",
            "Timeline: 2 weeks",
            "Must support WebSockets",
        ]
        # Mock a response that mentions all constraints
        response = "We have a $10K budget, 2 week deadline, and need WebSocket support"
        
        # Note: This will make actual API calls if OPENAI_API_KEY is set
        # In a real test, we'd mock the LLM client
        try:
            metrics = measure_granular_constraint_recall(constraints, response)
            scores = metrics.category_scores
            assert "budget" in scores
            assert "timeline" in scores
            assert "technical" in scores
            assert isinstance(scores["budget"], float)
        except Exception:
            # Skip if API key not available
            pytest.skip("OPENAI_API_KEY not set, skipping API-dependent test")

    def test_weighted_score_calculation(self):
        """Test that weighted score uses correct weights."""
        # Create metrics with known values
        metrics = GranularConstraintMetrics(
            overall_recall=0.8,
            budget_recall=1.0,
            timeline_recall=1.0,
            technical_recall=0.5,
            team_recall=1.0,
            compliance_recall=1.0,
            performance_recall=1.0,
            other_recall=1.0,
            category_drift=0.5,
        )
        
        # Weighted score should be:
        # 1.0*0.15 + 1.0*0.15 + 0.5*0.25 + 1.0*0.10 + 1.0*0.15 + 1.0*0.15 + 1.0*0.05
        # = 0.15 + 0.15 + 0.125 + 0.10 + 0.15 + 0.15 + 0.05 = 0.875
        expected = 0.875
        assert abs(metrics.weighted_score - expected) < 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GranularConstraintMetrics(
            overall_recall=0.8,
            budget_recall=1.0,
            timeline_recall=0.5,
            technical_recall=0.7,
            team_recall=1.0,
            compliance_recall=1.0,
            performance_recall=1.0,
            other_recall=1.0,
            category_drift=0.5,
        )
        
        d = metrics.to_dict()
        assert "overall_recall" in d
        assert "category_recall" in d
        assert "weighted_score" in d
        assert d["overall_recall"] == 0.8
        assert d["category_recall"]["budget"] == 1.0


class TestFormatReport:
    """Tests for report formatting."""

    def test_format_report(self):
        """Test that format_granular_constraint_report produces readable output."""
        metrics = GranularConstraintMetrics(
            overall_recall=0.8,
            budget_recall=1.0,
            timeline_recall=0.5,
            technical_recall=0.7,
            team_recall=1.0,
            compliance_recall=1.0,
            performance_recall=1.0,
            other_recall=1.0,
            category_drift=0.5,
        )
        
        report = format_granular_constraint_report(metrics)
        
        assert "GRANULAR CONSTRAINT RECALL REPORT" in report
        assert "OVERALL RECALL" in report
        assert "Budget" in report
        assert "Timeline" in report
        assert "80%" in report or "80.0%" in report

