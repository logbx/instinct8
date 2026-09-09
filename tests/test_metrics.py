"""Tests for evaluation metrics."""

import pytest


class TestMetricCalculator:
    """Tests for metric calculation."""

    def test_import_metrics(self):
        """Test that metrics module can be imported."""
        from evaluation.metrics import MetricsCollector
        assert MetricsCollector is not None

    def test_import_qa_metrics(self):
        """Test that QA metrics can be imported."""
        from evaluation.qa_metrics import QAMetricCalculator
        assert QAMetricCalculator is not None

    def test_import_coding_metrics(self):
        """Test that coding metrics can be imported."""
        from evaluation.coding_metrics import CodingMetricCalculator
        assert CodingMetricCalculator is not None


class TestStatistics:
    """Tests for statistical utilities."""

    def test_import_statistics(self):
        """Test that statistics module can be imported."""
        from evaluation.statistics import (
            StatisticalResult,
            paired_t_test,
            calculate_effect_size,
        )
        assert StatisticalResult is not None
        assert paired_t_test is not None
        assert calculate_effect_size is not None

    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation."""
        from evaluation.statistics import calculate_effect_size
        import numpy as np

        # Two identical groups should have effect size of 0
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.0, 2.0, 3.0, 4.0, 5.0]

        effect_size = calculate_effect_size(
            mean_a=np.mean(group_a),
            mean_b=np.mean(group_b),
            std_a=np.std(group_a, ddof=1),
            std_b=np.std(group_b, ddof=1),
            n_a=len(group_a),
            n_b=len(group_b)
        )
        assert abs(effect_size) < 0.01  # Should be ~0
