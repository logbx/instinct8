"""
Comprehensive unit tests for evaluation.statistics module.

Tests all pure statistical helper functions with deterministic inputs,
covering edge cases and error conditions. No live API keys required.
"""

import pytest
import numpy as np
from evaluation.statistics import (
    bootstrap_confidence_interval,
    parametric_confidence_interval,
    paired_t_test,
    independent_t_test,
    mann_whitney_test,
    wilcoxon_signed_rank_test,
    bonferroni_correction,
    benjamini_hochberg_correction,
    calculate_effect_size,
    interpret_effect_size,
    compute_statistical_summary,
    compare_strategies,
    format_comparison_table,
    StatisticalResult,
    ComparisonResult,
)


class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_bootstrap_ci_basic(self):
        """Bootstrap CI with fixed random state should be deterministic."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_confidence_interval(
            data, confidence=0.95, n_bootstrap=1000, random_state=42
        )
        assert lower < np.mean(data) < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_reproducible(self):
        """Same random state should produce same results."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_confidence_interval(data, random_state=42)
        ci2 = bootstrap_confidence_interval(data, random_state=42)
        assert ci1 == ci2

    def test_bootstrap_ci_empty_data(self):
        """Empty data should return (0.0, 0.0)."""
        lower, upper = bootstrap_confidence_interval([])
        assert lower == 0.0
        assert upper == 0.0

    def test_bootstrap_ci_single_value(self):
        """Single value should return (value, value)."""
        lower, upper = bootstrap_confidence_interval([5.5])
        assert lower == 5.5
        assert upper == 5.5

    def test_parametric_ci_basic(self):
        """Parametric CI should produce valid interval."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = parametric_confidence_interval(data, confidence=0.95)
        assert lower < np.mean(data) < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_parametric_ci_empty_data(self):
        """Empty data should return (0.0, 0.0)."""
        lower, upper = parametric_confidence_interval([])
        assert lower == 0.0
        assert upper == 0.0

    def test_parametric_ci_single_value(self):
        """Single value should return (value, value)."""
        lower, upper = parametric_confidence_interval([3.14])
        assert lower == 3.14
        assert upper == 3.14


class TestPairedTests:
    """Tests for paired statistical tests."""

    def test_paired_t_test_basic(self):
        """Paired t-test with different groups should detect difference."""
        scores_a = [5.0, 6.0, 7.0, 8.0, 9.0]
        scores_b = [4.0, 5.0, 6.0, 7.0, 8.0]
        result = paired_t_test(scores_a, scores_b)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "cohens_d" in result
        assert "significant_05" in result
        assert "significant_01" in result
        assert isinstance(result["t_statistic"], float)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_paired_t_test_identical_groups(self):
        """Identical groups should have d=0 (p-value may be NaN from scipy)."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(scores, scores)

        # When all differences are zero, scipy returns NaN for p-value
        # This is expected behavior for degenerate case
        assert result["cohens_d"] == 0.0
        assert not result["significant_05"]
        assert not result["significant_01"]

    def test_paired_t_test_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        scores_a = [1.0, 2.0, 3.0]
        scores_b = [1.0, 2.0]

        with pytest.raises(ValueError, match="same length"):
            paired_t_test(scores_a, scores_b)

    def test_paired_t_test_short_data(self):
        """Single pair should return safe defaults."""
        scores_a = [5.0]
        scores_b = [4.0]
        result = paired_t_test(scores_a, scores_b)

        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["cohens_d"] == 0.0
        assert not result["significant_05"]

    def test_paired_t_test_zero_difference_std(self):
        """Constant differences should handle zero std gracefully."""
        scores_a = [5.0, 5.0, 5.0]
        scores_b = [5.0, 5.0, 5.0]
        result = paired_t_test(scores_a, scores_b)

        assert result["cohens_d"] == 0.0

    def test_wilcoxon_basic(self):
        """Wilcoxon test should detect paired differences."""
        scores_a = [5.0, 6.0, 7.0, 8.0, 9.0]
        scores_b = [4.0, 5.0, 6.0, 7.0, 8.0]
        result = wilcoxon_signed_rank_test(scores_a, scores_b)

        assert "statistic" in result
        assert "p_value" in result
        assert "significant_05" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_wilcoxon_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        scores_a = [1.0, 2.0, 3.0]
        scores_b = [1.0, 2.0]

        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank_test(scores_a, scores_b)

    def test_wilcoxon_all_equal(self):
        """Identical scores should return p=1."""
        scores = [3.0, 3.0, 3.0]
        result = wilcoxon_signed_rank_test(scores, scores)

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["significant_05"]

    def test_wilcoxon_short_data(self):
        """Single pair should return safe defaults."""
        result = wilcoxon_signed_rank_test([5.0], [4.0])

        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["significant_05"]


class TestIndependentTests:
    """Tests for independent sample statistical tests."""

    def test_independent_t_test_basic(self):
        """Independent t-test should detect group differences."""
        scores_a = [8.0, 9.0, 10.0, 11.0, 12.0]
        scores_b = [3.0, 4.0, 5.0, 6.0, 7.0]
        result = independent_t_test(scores_a, scores_b)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "cohens_d" in result
        assert result["p_value"] < 0.05
        assert result["significant_05"]

    def test_independent_t_test_equal_var(self):
        """Should support equal_var parameter."""
        scores_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores_b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = independent_t_test(scores_a, scores_b, equal_var=True)

        assert isinstance(result["t_statistic"], float)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_independent_t_test_short_data(self):
        """Single sample in either group should return safe defaults."""
        scores_a = [5.0]
        scores_b = [3.0, 4.0, 5.0]
        result = independent_t_test(scores_a, scores_b)

        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["cohens_d"] == 0.0
        assert not result["significant_05"]

    def test_mann_whitney_basic(self):
        """Mann-Whitney U test should detect differences."""
        scores_a = [8.0, 9.0, 10.0, 11.0, 12.0]
        scores_b = [3.0, 4.0, 5.0, 6.0, 7.0]
        result = mann_whitney_test(scores_a, scores_b)

        assert "u_statistic" in result
        assert "p_value" in result
        assert "rank_biserial_r" in result
        assert "significant_05" in result
        assert result["p_value"] < 0.05

    def test_mann_whitney_short_data(self):
        """Single sample should return safe defaults."""
        result = mann_whitney_test([5.0], [3.0, 4.0])

        assert result["u_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["significant_05"]


class TestMultipleComparisonCorrection:
    """Tests for multiple comparison corrections."""

    def test_bonferroni_basic(self):
        """Bonferroni should multiply p-values by number of tests."""
        p_values = [0.01, 0.02, 0.03]
        corrected = bonferroni_correction(p_values)

        assert len(corrected) == 3
        assert corrected[0] == 0.03  # 0.01 * 3
        assert corrected[1] == 0.06  # 0.02 * 3
        assert corrected[2] == 0.09  # 0.03 * 3

    def test_bonferroni_caps_at_one(self):
        """Bonferroni should cap corrected p-values at 1.0."""
        p_values = [0.5, 0.6, 0.7]
        corrected = bonferroni_correction(p_values)

        assert all(p <= 1.0 for p in corrected)
        assert corrected[0] == 1.0  # Would be 1.5, capped
        assert corrected[1] == 1.0  # Would be 1.8, capped
        assert corrected[2] == 1.0  # Would be 2.1, capped

    def test_bonferroni_empty(self):
        """Empty list should return empty list."""
        assert bonferroni_correction([]) == []

    def test_benjamini_hochberg_basic(self):
        """BH correction should be less conservative than Bonferroni."""
        p_values = [0.01, 0.02, 0.03]
        corrected = benjamini_hochberg_correction(p_values)

        assert len(corrected) == 3
        # BH corrected values should be <= Bonferroni
        bonf = bonferroni_correction(p_values)
        assert all(c <= b for c, b in zip(corrected, bonf))

    def test_benjamini_hochberg_caps_at_one(self):
        """BH should cap corrected p-values at 1.0."""
        p_values = [0.5, 0.6, 0.7]
        corrected = benjamini_hochberg_correction(p_values)

        assert all(p <= 1.0 for p in corrected)

    def test_benjamini_hochberg_empty(self):
        """Empty list should return empty list."""
        assert benjamini_hochberg_correction([]) == []

    def test_benjamini_hochberg_single(self):
        """Single p-value should be returned unchanged (capped at 1)."""
        corrected = benjamini_hochberg_correction([0.05])
        assert corrected[0] == 0.05

        corrected = benjamini_hochberg_correction([1.5])
        assert corrected[0] == 1.0


class TestEffectSize:
    """Tests for effect size calculations and interpretations."""

    def test_calculate_effect_size_identical_groups(self):
        """Identical groups should have zero effect size."""
        effect = calculate_effect_size(
            mean_a=5.0, mean_b=5.0, std_a=1.0, std_b=1.0, n_a=10, n_b=10
        )
        assert effect == 0.0

    def test_calculate_effect_size_different_groups(self):
        """Different groups should have non-zero effect size."""
        effect = calculate_effect_size(
            mean_a=6.0, mean_b=4.0, std_a=1.0, std_b=1.0, n_a=10, n_b=10
        )
        assert effect > 0.0

    def test_calculate_effect_size_zero_pooled_std(self):
        """Zero pooled std should return 0.0 to avoid division by zero."""
        effect = calculate_effect_size(
            mean_a=5.0, mean_b=3.0, std_a=0.0, std_b=0.0, n_a=10, n_b=10
        )
        assert effect == 0.0

    def test_interpret_effect_size_negligible(self):
        """Effect sizes < 0.2 should be negligible."""
        assert interpret_effect_size(0.0) == "negligible"
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(-0.1) == "negligible"

    def test_interpret_effect_size_small(self):
        """Effect sizes 0.2 <= d < 0.5 should be small."""
        assert interpret_effect_size(0.2) == "small"
        assert interpret_effect_size(0.35) == "small"
        assert interpret_effect_size(-0.4) == "small"

    def test_interpret_effect_size_medium(self):
        """Effect sizes 0.5 <= d < 0.8 should be medium."""
        assert interpret_effect_size(0.5) == "medium"
        assert interpret_effect_size(0.65) == "medium"
        assert interpret_effect_size(-0.7) == "medium"

    def test_interpret_effect_size_large(self):
        """Effect sizes >= 0.8 should be large."""
        assert interpret_effect_size(0.8) == "large"
        assert interpret_effect_size(1.5) == "large"
        assert interpret_effect_size(-2.0) == "large"


class TestStatisticalSummary:
    """Tests for statistical summary computation."""

    def test_compute_summary_basic(self):
        """Should compute mean, std, CI, and n."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_statistical_summary(data)

        assert isinstance(result, StatisticalResult)
        assert result.mean == 3.0
        assert result.std > 0
        assert result.ci_lower < result.mean < result.ci_upper
        assert result.n == 5

    def test_compute_summary_empty(self):
        """Empty data should return zeros."""
        result = compute_statistical_summary([])

        assert result.mean == 0.0
        assert result.std == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0
        assert result.n == 0

    def test_compute_summary_single_value(self):
        """Single value should have zero std."""
        result = compute_statistical_summary([5.5])

        assert result.mean == 5.5
        assert result.std == 0.0
        assert result.ci_lower == 5.5
        assert result.ci_upper == 5.5
        assert result.n == 1

    def test_compute_summary_bootstrap(self):
        """Should use bootstrap CI when requested."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_statistical_summary(data, use_bootstrap=True)

        assert result.mean == 3.0
        assert result.ci_lower < result.mean < result.ci_upper

    def test_compute_summary_parametric(self):
        """Should use parametric CI when requested."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_statistical_summary(data, use_bootstrap=False)

        assert result.mean == 3.0
        assert result.ci_lower < result.mean < result.ci_upper


class TestStrategyComparison:
    """Tests for multi-metric strategy comparison."""

    def test_compare_strategies_basic(self):
        """Should compare strategies across multiple metrics."""
        scores_a = {
            "accuracy": [0.8, 0.85, 0.9],
            "f1": [0.75, 0.8, 0.85],
        }
        scores_b = {
            "accuracy": [0.7, 0.75, 0.8],
            "f1": [0.65, 0.7, 0.75],
        }

        results = compare_strategies("Strategy A", "Strategy B", scores_a, scores_b)

        assert len(results) == 2
        assert all(isinstance(r, ComparisonResult) for r in results)
        assert results[0].strategy_a == "Strategy A"
        assert results[0].strategy_b == "Strategy B"
        assert results[0].metric in ["accuracy", "f1"]

    def test_compare_strategies_paired(self):
        """Should use paired test when specified."""
        scores_a = {"metric1": [1.0, 2.0, 3.0]}
        scores_b = {"metric1": [1.5, 2.5, 3.5]}

        results = compare_strategies("A", "B", scores_a, scores_b, paired=True)

        assert len(results) == 1
        assert results[0].metric == "metric1"

    def test_compare_strategies_independent(self):
        """Should use independent test when specified."""
        scores_a = {"metric1": [1.0, 2.0, 3.0, 4.0]}
        scores_b = {"metric1": [1.5, 2.5, 3.5]}

        results = compare_strategies("A", "B", scores_a, scores_b, paired=False)

        assert len(results) == 1
        assert results[0].metric == "metric1"

    def test_compare_strategies_correction(self):
        """Should apply Bonferroni correction when requested."""
        scores_a = {
            "m1": [1.0, 2.0, 3.0],
            "m2": [2.0, 3.0, 4.0],
            "m3": [3.0, 4.0, 5.0],
        }
        scores_b = {
            "m1": [1.1, 2.1, 3.1],
            "m2": [2.1, 3.1, 4.1],
            "m3": [3.1, 4.1, 5.1],
        }

        results = compare_strategies("A", "B", scores_a, scores_b, apply_correction=True)

        assert len(results) == 3
        assert all(r.p_value_corrected is not None for r in results)
        # Corrected p-values should be >= uncorrected
        assert all(r.p_value_corrected >= r.p_value for r in results)

    def test_compare_strategies_no_correction(self):
        """Should not apply correction when not requested."""
        scores_a = {"m1": [1.0, 2.0, 3.0]}
        scores_b = {"m1": [2.0, 3.0, 4.0]}

        results = compare_strategies("A", "B", scores_a, scores_b, apply_correction=False)

        assert len(results) == 1
        # p_value_corrected is set to None initially but not updated
        assert results[0].p_value_corrected is None

    def test_compare_strategies_disjoint_metrics(self):
        """Should only compare metrics present in both strategies."""
        scores_a = {"m1": [1.0, 2.0], "m2": [3.0, 4.0]}
        scores_b = {"m2": [3.5, 4.5], "m3": [5.0, 6.0]}

        results = compare_strategies("A", "B", scores_a, scores_b)

        assert len(results) == 1
        assert results[0].metric == "m2"

    def test_compare_strategies_empty(self):
        """No common metrics should return empty list."""
        scores_a = {"m1": [1.0, 2.0]}
        scores_b = {"m2": [3.0, 4.0]}

        results = compare_strategies("A", "B", scores_a, scores_b)

        assert results == []


class TestFormatting:
    """Tests for comparison result formatting."""

    def test_format_comparison_table_basic(self):
        """Should format results as readable table."""
        results = [
            ComparisonResult(
                strategy_a="A",
                strategy_b="B",
                metric="accuracy",
                mean_diff=0.1,
                t_statistic=2.5,
                p_value=0.03,
                p_value_corrected=0.06,
                cohens_d=0.6,
                effect_interpretation="medium",
                significant_05=True,
                significant_01=False,
            )
        ]

        table = format_comparison_table(results)

        assert "Comparison: A vs B" in table
        assert "accuracy" in table
        assert "0.1000" in table
        assert "*" in table  # Significance marker

    def test_format_comparison_table_corrected(self):
        """Should show corrected p-values when requested."""
        results = [
            ComparisonResult(
                strategy_a="A",
                strategy_b="B",
                metric="m1",
                mean_diff=0.1,
                t_statistic=2.0,
                p_value=0.02,
                p_value_corrected=0.04,
                cohens_d=0.5,
                effect_interpretation="medium",
                significant_05=True,
                significant_01=False,
            )
        ]

        table = format_comparison_table(results, show_corrected=True)

        assert "p (corrected)" in table
        assert "0.0400" in table

    def test_format_comparison_table_uncorrected(self):
        """Should show uncorrected p-values when requested."""
        results = [
            ComparisonResult(
                strategy_a="A",
                strategy_b="B",
                metric="m1",
                mean_diff=0.1,
                t_statistic=2.0,
                p_value=0.02,
                p_value_corrected=0.04,
                cohens_d=0.5,
                effect_interpretation="medium",
                significant_05=True,
                significant_01=False,
            )
        ]

        table = format_comparison_table(results, show_corrected=False)

        assert "p-value" in table
        assert "0.0200" in table

    def test_format_comparison_table_empty(self):
        """Empty results should return message."""
        table = format_comparison_table([])

        assert table == "No comparisons to display"

    def test_format_comparison_table_significance_markers(self):
        """Should show correct significance markers."""
        results = [
            ComparisonResult(
                strategy_a="A",
                strategy_b="B",
                metric="m1",
                mean_diff=0.1,
                t_statistic=3.0,
                p_value=0.001,
                p_value_corrected=None,
                cohens_d=1.0,
                effect_interpretation="large",
                significant_05=True,
                significant_01=True,
            ),
            ComparisonResult(
                strategy_a="A",
                strategy_b="B",
                metric="m2",
                mean_diff=0.05,
                t_statistic=2.0,
                p_value=0.03,
                p_value_corrected=None,
                cohens_d=0.4,
                effect_interpretation="small",
                significant_05=True,
                significant_01=False,
            ),
        ]

        table = format_comparison_table(results, show_corrected=False)

        assert "**" in table  # p < 0.01
        assert table.count("*") >= 2  # At least one * for p < 0.05
