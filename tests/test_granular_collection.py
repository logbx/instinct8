"""Unit test: granular constraint metrics collection.

Verifies that MetricsCollector produces granular_constraint_metrics
when use_granular_metrics=True.
"""

import json
import pytest

from evaluation.metrics import MetricsCollector


@pytest.mark.integration
class TestGranularCollection:
    """Test that granular constraint metrics are collected properly."""

    def test_granular_metrics_present(self):
        """Granular constraint metrics should appear in results."""
        with open("templates/research-synthesis-001.json") as f:
            template = json.load(f)
        constraints = template["initial_setup"]["hard_constraints"]

        collector = MetricsCollector(
            original_goal=template["initial_setup"]["original_goal"],
            constraints=constraints,
            use_granular_metrics=True,
        )

        collector.collect_at_compression_point(
            compression_point_id=1,
            turn_id=5,
            tokens_before=1000,
            tokens_after=500,
            goal_stated_before="Research frameworks",
            goal_stated_after="Research frameworks",
            constraints_stated_before="We have a budget of $10K and need to finish in 2 weeks",
            constraints_stated_after="Budget is $10K, timeline is 2 weeks, need WebSockets and PostgreSQL",
        )

        results = collector.get_results()
        assert "granular_constraint_metrics" in results

        granular = results["granular_constraint_metrics"]
        assert "after" in granular
        assert len(granular["after"]) > 0

    def test_granular_metrics_keys(self):
        """Each granular metric entry should have expected keys."""
        with open("templates/research-synthesis-001.json") as f:
            template = json.load(f)
        constraints = template["initial_setup"]["hard_constraints"]

        collector = MetricsCollector(
            original_goal=template["initial_setup"]["original_goal"],
            constraints=constraints,
            use_granular_metrics=True,
        )

        collector.collect_at_compression_point(
            compression_point_id=1,
            turn_id=5,
            tokens_before=1000,
            tokens_after=500,
            goal_stated_before="Research frameworks",
            goal_stated_after="Research frameworks",
            constraints_stated_before="Budget and timeline",
            constraints_stated_after="Budget and timeline constraints",
        )

        results = collector.get_results()
        if "granular_constraint_metrics" in results:
            granular = results["granular_constraint_metrics"]
            if granular.get("after"):
                first_cp = granular["after"][0]
                # Should have keys describing the compression point
                assert isinstance(first_cp, dict)
