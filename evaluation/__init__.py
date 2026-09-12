# Evaluation Framework
# This package contains metrics collection and probing functions
# for measuring goal coherence under compression and QA evaluation.

# Core metrics (existing)
from .metrics import (
    measure_goal_coherence,
    measure_constraint_recall,
    measure_behavioral_alignment,
    MetricsCollector,
    CompressionPointMetrics,
)

# Unified metric interfaces (new)
from .metric_interfaces import (
    MetricType,
    MetricResult,
    MetricCalculator,
    COMPRESSION_METRICS,
    QA_METRICS,
)

# QA metrics adapter (new)
from .qa_metrics import QAMetricCalculator

# Coding metrics (new)
from .coding_metrics import CodingMetricCalculator, CODING_METRICS

# Unified aggregator (new)
from .unified_aggregator import UnifiedMetricAggregator, AggregateStats

# Agent abstractions (new)
from .agents import (
    BaseAgent,
    AgentConfig,
    CompressionAgent,
    AMemAgent,
    CodexAgent,
    create_codex_agent,
)

# Dataset abstractions (new)
from .datasets import (
    BaseDataset,
    EvalTurn,
    EvalQuestion,
    EvalSample,
    TemplateDataset,
    LoCoMoDataset,
    CodingDataset,
    CodingGroundTruth,
    CodingTask,
)

# Cache manager (new)
from .cache_manager import CacheManager

# Unified harness (new)
from .unified_harness import (
    UnifiedHarness,
    QAResult,
    SampleResult,
    CodingResult,
    EvaluationResults,
)

# Strategy comparison - removed (use unified_harness directly)

# Statistical utilities (new)
from .statistics import (
    StatisticalResult,
    ComparisonResult,
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
    compare_strategies as compare_strategies_statistical,
    format_comparison_table,
)

# Baseline strategies (new)
from .baseline_strategies import (
    NoCompressionBaseline,
    RandomTruncationBaseline,
    RecencyOnlyBaseline,
    FirstLastBaseline,
    SlidingWindowBaseline,
    get_all_baselines,
    get_baseline_by_name,
)

# Multi-run evaluation - integrated into unified_harness

# Ablation studies - removed (unused)

# Codex CLI wrapper - removed (external tool not in-repo)

# Hierarchical metrics (new)
from .hierarchical_metrics import (
    HierarchicalMetrics,
    HierarchicalMetricsCalculator,
    ProbeResult,
    BehavioralTestResult,
    load_hierarchical_template,
    format_hierarchical_report,
    measure_element_recall,
    measure_depth_precision,
    measure_reasoning_fidelity,
)

# CLI benchmark client - removed (external app-server tool)

__all__ = [
    # Core metrics
    "measure_goal_coherence",
    "measure_constraint_recall",
    "measure_behavioral_alignment",
    "MetricsCollector",
    "CompressionPointMetrics",
    # Metric interfaces
    "MetricType",
    "MetricResult",
    "MetricCalculator",
    "COMPRESSION_METRICS",
    "QA_METRICS",
    # QA metrics
    "QAMetricCalculator",
    # Coding metrics
    "CodingMetricCalculator",
    "CODING_METRICS",
    # Aggregator
    "UnifiedMetricAggregator",
    "AggregateStats",
    # Agents
    "BaseAgent",
    "AgentConfig",
    "CompressionAgent",
    "AMemAgent",
    "CodexAgent",
    "create_codex_agent",
    # Datasets
    "BaseDataset",
    "EvalTurn",
    "EvalQuestion",
    "EvalSample",
    "TemplateDataset",
    "LoCoMoDataset",
    "CodingDataset",
    "CodingGroundTruth",
    "CodingTask",
    # Cache
    "CacheManager",
    # Unified harness
    "UnifiedHarness",
    "QAResult",
    "SampleResult",
    "CodingResult",
    "EvaluationResults",
    # Statistical utilities
    "StatisticalResult",
    "ComparisonResult",
    "bootstrap_confidence_interval",
    "parametric_confidence_interval",
    "paired_t_test",
    "independent_t_test",
    "mann_whitney_test",
    "wilcoxon_signed_rank_test",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    "calculate_effect_size",
    "interpret_effect_size",
    "compute_statistical_summary",
    "compare_strategies_statistical",
    "format_comparison_table",
    # Baseline strategies
    "NoCompressionBaseline",
    "RandomTruncationBaseline",
    "RecencyOnlyBaseline",
    "FirstLastBaseline",
    "SlidingWindowBaseline",
    "get_all_baselines",
    "get_baseline_by_name",
    # Hierarchical metrics
    "HierarchicalMetrics",
    "HierarchicalMetricsCalculator",
    "ProbeResult",
    "BehavioralTestResult",
    "load_hierarchical_template",
    "format_hierarchical_report",
    "measure_element_recall",
    "measure_depth_precision",
    "measure_reasoning_fidelity",
]

