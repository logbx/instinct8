"""
Unit tests for evaluation/unified_aggregator.py.

Covers AggregateStats, UnifiedMetricAggregator collection/aggregation, and the
compression/QA summary helpers. All inputs are hand-built MetricResult objects,
so no API keys, LLM calls, or heavy metric backends are exercised.
"""

import statistics

import pytest

from evaluation.metric_interfaces import MetricResult, MetricType
from evaluation.unified_aggregator import AggregateStats, UnifiedMetricAggregator


def _mr(name, value, category=None, metric_type=MetricType.SCORE_0_1, metadata=None):
    return MetricResult(
        name=name,
        value=value,
        metric_type=metric_type,
        category=category,
        metadata=metadata if metadata is not None else {},
    )


@pytest.fixture
def aggregator():
    return UnifiedMetricAggregator()


class TestAggregateStats:
    def test_to_dict_keys_and_values(self):
        stats = AggregateStats(mean=0.5, std=0.1, median=0.4, min=0.2, max=0.9, count=3)
        assert stats.to_dict() == {
            "mean": 0.5,
            "std": 0.1,
            "median": 0.4,
            "min": 0.2,
            "max": 0.9,
            "count": 3,
        }


class TestCollection:
    def test_new_aggregator_is_empty(self, aggregator):
        assert len(aggregator) == 0
        assert aggregator.get_results_list() == []

    def test_add_single(self, aggregator):
        aggregator.add(_mr("f1", 0.5))
        assert len(aggregator) == 1

    def test_add_batch(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.5), _mr("f1", 0.7), _mr("exact_match", 1)])
        assert len(aggregator) == 3

    def test_add_batch_empty_list(self, aggregator):
        aggregator.add_batch([])
        assert len(aggregator) == 0

    def test_add_and_add_batch_accumulate(self, aggregator):
        aggregator.add(_mr("f1", 0.5))
        aggregator.add_batch([_mr("f1", 0.7), _mr("f1", 0.9)])
        assert len(aggregator) == 3

    def test_reset_clears_results(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.5), _mr("goal_drift", 0.3)])
        aggregator.reset()
        assert len(aggregator) == 0
        assert aggregator.aggregate() == {"overall": {}}

    def test_get_results_list_uses_metric_result_to_dict(self, aggregator):
        results = [
            _mr("f1", 0.5, category="1", metadata={"q": "a"}),
            _mr("compression_ratio", 3, metric_type=MetricType.COUNT),
        ]
        aggregator.add_batch(results)
        assert aggregator.get_results_list() == [r.to_dict() for r in results]
        assert aggregator.get_results_list()[1] == {
            "name": "compression_ratio",
            "value": 3,
            "metric_type": "count",
            "category": None,
            "metadata": {},
        }


class TestComputeStats:
    def test_empty_values_returns_zeros(self, aggregator):
        stats = aggregator._compute_stats([])
        assert stats.to_dict() == {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }

    def test_single_value_has_zero_std(self, aggregator):
        stats = aggregator._compute_stats([0.42])
        assert stats.mean == pytest.approx(0.42)
        assert stats.std == 0.0
        assert stats.median == pytest.approx(0.42)
        assert stats.min == pytest.approx(0.42)
        assert stats.max == pytest.approx(0.42)
        assert stats.count == 1

    def test_multi_value_stats(self, aggregator):
        values = [0.1, 0.4, 0.2, 0.9]
        stats = aggregator._compute_stats(values)
        assert stats.mean == pytest.approx(0.4)
        assert stats.median == pytest.approx(0.3)
        assert stats.min == pytest.approx(0.1)
        assert stats.max == pytest.approx(0.9)
        assert stats.count == 4
        # Sample (n-1) standard deviation, matching statistics.stdev.
        assert stats.std == pytest.approx(statistics.stdev(values))
        assert stats.std == pytest.approx((0.38 / 3) ** 0.5)

    def test_odd_count_median(self, aggregator):
        stats = aggregator._compute_stats([3.0, 1.0, 2.0])
        assert stats.median == 2.0

    def test_identical_values_zero_std(self, aggregator):
        stats = aggregator._compute_stats([0.5, 0.5, 0.5])
        assert stats.std == 0.0
        assert stats.mean == 0.5


class TestAggregateOverall:
    def test_empty_aggregator_returns_empty_overall(self, aggregator):
        assert aggregator.aggregate() == {"overall": {}}
        assert aggregator.aggregate(group_by_category=False) == {"overall": {}}

    def test_groups_by_metric_name(self, aggregator):
        aggregator.add_batch(
            [
                _mr("f1", 0.2),
                _mr("f1", 0.6),
                _mr("custom_metric", 1.0),
            ]
        )
        overall = aggregator.aggregate()["overall"]
        assert set(overall) == {"f1", "custom_metric"}
        assert overall["f1"]["count"] == 2
        assert overall["f1"]["mean"] == pytest.approx(0.4)
        assert overall["f1"]["min"] == pytest.approx(0.2)
        assert overall["f1"]["max"] == pytest.approx(0.6)
        assert overall["custom_metric"]["count"] == 1
        assert overall["custom_metric"]["std"] == 0.0

    def test_overall_includes_uncategorized_and_categorized(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.2, category="1"), _mr("f1", 0.8)])
        overall = aggregator.aggregate()["overall"]
        assert overall["f1"]["count"] == 2
        assert overall["f1"]["mean"] == pytest.approx(0.5)

    def test_int_values_are_aggregated_as_floats(self, aggregator):
        aggregator.add_batch(
            [_mr("exact_match", 1), _mr("exact_match", 0)]
        )
        overall = aggregator.aggregate()["overall"]
        assert overall["exact_match"]["mean"] == pytest.approx(0.5)
        assert isinstance(overall["exact_match"]["max"], float)

    def test_unknown_metrics_produce_no_summaries(self, aggregator):
        aggregator.add(_mr("custom_metric", 0.5))
        output = aggregator.aggregate()
        assert set(output) == {"overall"}


class TestAggregateByCategory:
    def test_group_by_category_true_builds_category_keys(self, aggregator):
        aggregator.add_batch(
            [
                _mr("f1", 0.2, category="2"),
                _mr("f1", 0.4, category="2"),
                _mr("f1", 0.9, category="1"),
                _mr("exact_match", 1, category="1"),
            ]
        )
        by_category = aggregator.aggregate(group_by_category=True)["by_category"]
        assert list(by_category) == ["category_1", "category_2"]
        assert set(by_category["category_1"]) == {"f1", "exact_match"}
        assert by_category["category_1"]["f1"]["count"] == 1
        assert by_category["category_1"]["f1"]["mean"] == pytest.approx(0.9)
        assert set(by_category["category_2"]) == {"f1"}
        assert by_category["category_2"]["f1"]["count"] == 2
        assert by_category["category_2"]["f1"]["mean"] == pytest.approx(0.3)

    def test_uncategorized_results_excluded_from_by_category(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.2, category="1"), _mr("f1", 0.8)])
        by_category = aggregator.aggregate()["by_category"]
        assert list(by_category) == ["category_1"]
        assert by_category["category_1"]["f1"]["count"] == 1

    def test_group_by_category_false_omits_by_category(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.2, category="1"), _mr("f1", 0.4, category="2")])
        output = aggregator.aggregate(group_by_category=False)
        assert "by_category" not in output
        assert output["overall"]["f1"]["count"] == 2

    def test_no_categories_omits_by_category(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.2), _mr("f1", 0.4)])
        assert "by_category" not in aggregator.aggregate(group_by_category=True)

    def test_default_is_group_by_category(self, aggregator):
        aggregator.add(_mr("f1", 0.2, category="3"))
        assert "category_3" in aggregator.aggregate()["by_category"]


class TestCompressionSummary:
    def test_full_compression_summary(self, aggregator):
        aggregator.add_batch(
            [
                _mr("goal_drift", 0.05, metric_type=MetricType.DELTA),
                _mr("goal_drift", 0.3, metric_type=MetricType.DELTA),
                _mr("goal_drift", 0.2, metric_type=MetricType.DELTA),
                _mr("goal_coherence_after", 0.8),
                _mr("goal_coherence_after", 0.6),
                _mr("constraint_loss", 0.25),
                _mr("constraint_recall_after", 0.75),
                _mr("behavioral_alignment_after", 4, metric_type=MetricType.SCORE_1_5),
                _mr("compression_ratio", 4.0),
                _mr("compression_ratio", 2.0),
            ]
        )
        summary = aggregator.aggregate()["compression_summary"]
        assert set(summary) == {
            "avg_goal_drift",
            "max_goal_drift",
            "final_goal_coherence",
            "avg_constraint_loss",
            "final_constraint_recall",
            "avg_behavioral_alignment",
            "avg_compression_ratio",
            "drift_events_detected",
        }
        assert summary["avg_goal_drift"] == pytest.approx(0.55 / 3)
        assert summary["max_goal_drift"] == pytest.approx(0.3)
        assert summary["final_goal_coherence"] == pytest.approx(0.7)
        assert summary["avg_constraint_loss"] == pytest.approx(0.25)
        assert summary["final_constraint_recall"] == pytest.approx(0.75)
        assert summary["avg_behavioral_alignment"] == pytest.approx(4.0)
        assert summary["avg_compression_ratio"] == pytest.approx(3.0)
        assert summary["drift_events_detected"] == 2

    def test_drift_threshold_is_strictly_greater_than_point_one(self, aggregator):
        aggregator.add_batch(
            [
                _mr("goal_drift", 0.1, metric_type=MetricType.DELTA),
                _mr("goal_drift", 0.1000001, metric_type=MetricType.DELTA),
                _mr("goal_drift", -0.5, metric_type=MetricType.DELTA),
                _mr("goal_drift", 0.0, metric_type=MetricType.DELTA),
            ]
        )
        summary = aggregator.aggregate()["compression_summary"]
        assert summary["drift_events_detected"] == 1

    def test_drift_events_zero_when_no_drift_exceeds_threshold(self, aggregator):
        aggregator.add(_mr("goal_drift", 0.05, metric_type=MetricType.DELTA))
        summary = aggregator.aggregate()["compression_summary"]
        assert summary["drift_events_detected"] == 0
        assert summary["avg_goal_drift"] == pytest.approx(0.05)
        assert summary["max_goal_drift"] == pytest.approx(0.05)

    def test_drift_events_counted_across_categories(self, aggregator):
        aggregator.add_batch(
            [
                _mr("goal_drift", 0.5, category="1", metric_type=MetricType.DELTA),
                _mr("goal_drift", 0.5, category="2", metric_type=MetricType.DELTA),
            ]
        )
        summary = aggregator.aggregate()["compression_summary"]
        assert summary["drift_events_detected"] == 2

    def test_constraint_only_summary(self, aggregator):
        aggregator.add_batch([_mr("constraint_loss", 0.1), _mr("constraint_recall_after", 0.9)])
        summary = aggregator.aggregate()["compression_summary"]
        assert summary == {
            "avg_constraint_loss": pytest.approx(0.1),
            "final_constraint_recall": pytest.approx(0.9),
        }

    def test_compression_ratio_only_summary(self, aggregator):
        aggregator.add(_mr("compression_ratio", 5.0))
        summary = aggregator.aggregate()["compression_summary"]
        assert summary == {"avg_compression_ratio": pytest.approx(5.0)}

    def test_before_metrics_alone_produce_no_summary(self, aggregator):
        aggregator.add_batch(
            [
                _mr("goal_coherence_before", 0.9),
                _mr("constraint_recall_before", 0.9),
                _mr("behavioral_alignment_before", 5, metric_type=MetricType.SCORE_1_5),
            ]
        )
        assert "compression_summary" not in aggregator.aggregate()

    def test_helper_returns_none_without_compression_metrics(self, aggregator):
        assert aggregator._compression_summary({}) is None


class TestQASummary:
    def test_full_qa_summary(self, aggregator):
        aggregator.add_batch(
            [
                _mr("exact_match", 1),
                _mr("exact_match", 0),
                _mr("f1", 0.6),
                _mr("bert_f1", 0.85),
                _mr("rouge1_f", 0.6),
                _mr("rouge2_f", 0.3),
                _mr("rougeL_f", 0.45),
                _mr("bleu1", 0.4),
                _mr("bleu2", 0.3),
                _mr("bleu3", 0.2),
                _mr("bleu4", 0.1),
                _mr("meteor", 0.55),
                _mr("sbert_similarity", 0.9),
            ]
        )
        summary = aggregator.aggregate()["qa_summary"]
        assert set(summary) == {
            "accuracy",
            "f1_score",
            "semantic_similarity",
            "avg_rouge",
            "avg_bleu",
            "meteor",
            "sbert_similarity",
        }
        assert summary["accuracy"] == pytest.approx(0.5)
        assert summary["f1_score"] == pytest.approx(0.6)
        assert summary["semantic_similarity"] == pytest.approx(0.85)
        assert summary["avg_rouge"] == pytest.approx(0.45)
        assert summary["avg_bleu"] == pytest.approx(0.25)
        assert summary["meteor"] == pytest.approx(0.55)
        assert summary["sbert_similarity"] == pytest.approx(0.9)

    def test_avg_rouge_uses_only_present_keys(self, aggregator):
        aggregator.add_batch([_mr("rouge1_f", 0.8), _mr("rougeL_f", 0.4)])
        summary = aggregator.aggregate()["qa_summary"]
        assert summary == {"avg_rouge": pytest.approx(0.6)}

    def test_avg_rouge_averages_per_metric_means(self, aggregator):
        # rouge1_f mean is 0.5 from two samples; rouge2_f mean is 0.2 from one.
        aggregator.add_batch(
            [_mr("rouge1_f", 0.4), _mr("rouge1_f", 0.6), _mr("rouge2_f", 0.2)]
        )
        summary = aggregator.aggregate()["qa_summary"]
        assert summary["avg_rouge"] == pytest.approx(0.35)

    def test_avg_bleu_uses_only_present_keys(self, aggregator):
        aggregator.add(_mr("bleu4", 0.12))
        summary = aggregator.aggregate()["qa_summary"]
        assert summary == {"avg_bleu": pytest.approx(0.12)}

    def test_bert_precision_recall_alone_produce_no_summary(self, aggregator):
        aggregator.add_batch([_mr("bert_precision", 0.8), _mr("bert_recall", 0.7)])
        assert "qa_summary" not in aggregator.aggregate()

    def test_helper_returns_none_without_qa_metrics(self, aggregator):
        assert aggregator._qa_summary({}) is None


class TestMixedMetrics:
    def test_compression_and_qa_summaries_coexist(self, aggregator):
        aggregator.add_batch(
            [
                _mr("goal_drift", 0.2, category="1", metric_type=MetricType.DELTA),
                _mr("f1", 0.7, category="1"),
                _mr("exact_match", 1, category="2"),
            ]
        )
        output = aggregator.aggregate()
        assert set(output) == {"overall", "by_category", "compression_summary", "qa_summary"}
        assert output["compression_summary"]["drift_events_detected"] == 1
        assert output["qa_summary"]["f1_score"] == pytest.approx(0.7)
        assert output["qa_summary"]["accuracy"] == pytest.approx(1.0)
        assert set(output["by_category"]) == {"category_1", "category_2"}

    def test_aggregate_is_repeatable(self, aggregator):
        aggregator.add_batch([_mr("f1", 0.3), _mr("goal_drift", 0.4)])
        assert aggregator.aggregate() == aggregator.aggregate()
        assert len(aggregator) == 2
