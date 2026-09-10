#!/usr/bin/env python3
"""Trace the granular metrics flow to find the issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import MetricsCollector

# Step 1: Check initialization
print("=" * 70)
print("STEP 1: Initialization")
print("=" * 70)
collector = MetricsCollector(
    original_goal='test goal',
    constraints=['Budget: $10K', 'Timeline: 2 weeks'],
    use_granular_metrics=True
)

print(f"use_granular_metrics: {collector.use_granular_metrics}")
print(f"hasattr granular_metrics_before: {hasattr(collector, 'granular_metrics_before')}")
print(f"hasattr granular_metrics_after: {hasattr(collector, 'granular_metrics_after')}")
if hasattr(collector, 'granular_metrics_before'):
    print(f"granular_metrics_before: {collector.granular_metrics_before}")
    print(f"granular_metrics_after: {collector.granular_metrics_after}")

# Step 2: Simulate collect_at_compression_point
print("\n" + "=" * 70)
print("STEP 2: collect_at_compression_point")
print("=" * 70)

constraints_before = 'We have a budget of $10K'
constraints_after = 'Budget is $10K, timeline is 2 weeks'

try:
    metrics = collector.collect_at_compression_point(
        compression_point_id=1,
        turn_id=5,
        tokens_before=1000,
        tokens_after=500,
        goal_stated_before='test goal',
        goal_stated_after='test goal',
        constraints_stated_before=constraints_before,
        constraints_stated_after=constraints_after,
    )
    print(f"✓ collect_at_compression_point succeeded")
    print(f"granular_metrics_before length: {len(collector.granular_metrics_before)}")
    print(f"granular_metrics_after length: {len(collector.granular_metrics_after)}")
    if collector.granular_metrics_before:
        print(f"First granular_before type: {type(collector.granular_metrics_before[0])}")
        print(f"First granular_before has to_dict: {hasattr(collector.granular_metrics_before[0], 'to_dict')}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Check get_results
print("\n" + "=" * 70)
print("STEP 3: get_results()")
print("=" * 70)
try:
    results = collector.get_results()
    print(f"results has 'granular_constraint_metrics': {'granular_constraint_metrics' in results}")
    if 'granular_constraint_metrics' in results:
        granular = results['granular_constraint_metrics']
        print(f"  Keys: {list(granular.keys())}")
        print(f"  before length: {len(granular.get('before', []))}")
        print(f"  after length: {len(granular.get('after', []))}")
    else:
        print(f"  Available keys: {list(results.keys())}")
        print(f"  use_granular_metrics: {collector.use_granular_metrics}")
        print(f"  granular_metrics_before length: {len(collector.granular_metrics_before)}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Check TrialResult
print("\n" + "=" * 70)
print("STEP 4: TrialResult.to_dict()")
print("=" * 70)
from evaluation.template_utils import TrialResult

granular_metrics = results.get("granular_constraint_metrics") if 'results' in locals() else None
trial_result = TrialResult(
    trial_id=1,
    strategy_name="test",
    template_id="test",
    compression_points=[],
    summary={},
    granular_constraint_metrics=granular_metrics,
)

trial_dict = trial_result.to_dict()
print(f"trial_dict has 'granular_constraint_metrics': {'granular_constraint_metrics' in trial_dict}")
if 'granular_constraint_metrics' in trial_dict:
    print(f"  ✓ Granular metrics in TrialResult!")
else:
    print(f"  ✗ Granular metrics NOT in TrialResult")
    print(f"  trial_dict keys: {list(trial_dict.keys())}")
    print(f"  trial_result.granular_constraint_metrics: {trial_result.granular_constraint_metrics}")

