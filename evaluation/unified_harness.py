"""
Unified Evaluation Harness

This module provides a unified harness for evaluating both compression-based
and memory-based agents on various datasets.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .agents.base import BaseAgent
from .cache_manager import CacheManager
from .coding_metrics import CodingMetricCalculator
from .datasets.base import BaseDataset, EvalSample
from .metric_interfaces import MetricResult, MetricType
from .metrics import MetricsCollector
from .qa_metrics import QAMetricCalculator
from .unified_aggregator import UnifiedMetricAggregator
from .statistics import compute_statistical_summary, StatisticalResult


@dataclass
class QAResult:
    """Result for a single QA evaluation."""

    question: str
    prediction: str
    reference: str
    category: Optional[int]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "prediction": self.prediction,
            "reference": self.reference,
            "category": self.category,
            "metrics": self.metrics,
        }


@dataclass
class CodingResult:
    """Result for coding task evaluation."""

    task_type: str
    generated_code: Dict[str, str]  # file -> content
    metrics: Dict[str, float]
    compression_points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "generated_code": self.generated_code,
            "metrics": self.metrics,
            "compression_points": self.compression_points,
        }


@dataclass
class SampleResult:
    """Result for evaluating a single sample."""

    sample_id: str
    agent_name: str
    evaluation_type: str  # "compression", "qa", or "coding"
    # QA metrics (for LoComo)
    qa_results: List[QAResult] = field(default_factory=list)
    aggregate_qa_metrics: Dict[str, Any] = field(default_factory=dict)
    # Compression metrics (for templates)
    compression_points: List[Dict[str, Any]] = field(default_factory=list)
    compression_summary: Dict[str, Any] = field(default_factory=dict)
    # Coding metrics (for coding tasks)
    coding_result: Optional[CodingResult] = None
    coding_metrics: Dict[str, float] = field(default_factory=dict)
    # Common
    context_sizes: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "sample_id": self.sample_id,
            "agent_name": self.agent_name,
            "evaluation_type": self.evaluation_type,
            "qa_results": [r.to_dict() for r in self.qa_results],
            "aggregate_qa_metrics": self.aggregate_qa_metrics,
            "compression_points": self.compression_points,
            "compression_summary": self.compression_summary,
            "coding_metrics": self.coding_metrics,
            "context_sizes": self.context_sizes,
            "timestamp": self.timestamp,
        }
        if self.coding_result:
            result["coding_result"] = self.coding_result.to_dict()
        return result


@dataclass
class EvaluationResults:
    """Aggregated results across all samples."""

    agent_name: str
    dataset_name: str
    evaluation_type: str
    num_samples: int
    sample_results: List[SampleResult]
    aggregate_metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Multi-run evaluation fields
    n_runs: int = 1
    confidence: float = 0.95
    statistical_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "agent_name": self.agent_name,
            "dataset_name": self.dataset_name,
            "evaluation_type": self.evaluation_type,
            "num_samples": self.num_samples,
            "sample_results": [r.to_dict() for r in self.sample_results],
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp,
        }
        # Include multi-run info if applicable
        if self.n_runs > 1:
            result["n_runs"] = self.n_runs
            result["confidence"] = self.confidence
            result["statistical_summary"] = self.statistical_summary
        return result

    def save(self, filepath: str) -> None:
        """Save results to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class UnifiedHarness:
    """
    Unified evaluation harness supporting both compression and QA evaluation.

    This harness automatically routes to the appropriate evaluation method
    based on the dataset's evaluation_type property.

    Usage:
        # QA Evaluation (LoComo)
        dataset = LoCoMoDataset("path/to/locomo.json", ratio=0.1)
        agent = AMemAgent(config)
        harness = UnifiedHarness(agent, dataset, cache_dir="./cache")
        results = harness.run_evaluation()

        # Compression Evaluation (Templates)
        dataset = TemplateDataset("path/to/template.json")
        agent = CompressionAgent(config, strategy)
        harness = UnifiedHarness(agent, dataset)
        results = harness.run_evaluation()
    """

    def __init__(
        self,
        agent: BaseAgent,
        dataset: BaseDataset,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the unified harness.

        Args:
            agent: The agent to evaluate
            dataset: The dataset to evaluate on
            cache_dir: Optional directory for caching agent states
        """
        self.agent = agent
        self.dataset = dataset
        self.cache_dir = cache_dir

        # Initialize cache manager if caching is enabled
        self._cache: Optional[CacheManager] = None
        if cache_dir:
            self._cache = CacheManager(
                cache_dir=cache_dir,
                agent_name=agent.name,
                backend=getattr(agent, "_config", {}).backend
                if hasattr(agent, "_config")
                else "",
            )

        # Initialize metric calculators
        self._qa_calc = QAMetricCalculator()
        self._coding_calc = CodingMetricCalculator()

    def run_evaluation(
        self,
        num_samples: Optional[int] = None,
        n_runs: int = 1,
        confidence: float = 0.95,
        verbose: bool = True,
    ) -> EvaluationResults:
        """
        Run full evaluation on the dataset.

        Args:
            num_samples: Limit number of samples (None = all)
            n_runs: Number of runs per sample for variance capture (default 1)
            confidence: Confidence level for CI when n_runs > 1 (default 0.95)
            verbose: Print progress information

        Returns:
            EvaluationResults with all sample results and aggregated metrics
        """
        samples = list(self.dataset)
        if num_samples:
            samples = samples[:num_samples]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"UNIFIED EVALUATION")
            print(f"Agent: {self.agent.name}")
            print(f"Dataset: {self.dataset.name}")
            print(f"Type: {self.dataset.evaluation_type}")
            print(f"Samples: {len(samples)}")
            if n_runs > 1:
                print(f"Runs per sample: {n_runs}")
                print(f"Confidence level: {confidence:.0%}")
            print(f"{'=' * 60}\n")

        # Use multi-run evaluation if n_runs > 1
        if n_runs > 1:
            return self._run_multi_run_evaluation(
                samples, n_runs, confidence, verbose
            )

        # Standard single-run evaluation
        sample_results = []
        aggregator = UnifiedMetricAggregator()

        for idx, sample in enumerate(samples):
            if verbose:
                print(f"Processing sample {idx + 1}/{len(samples)}: {sample.sample_id}")

            if self.dataset.evaluation_type == "qa":
                result = self._evaluate_qa_sample(sample, aggregator, verbose)
            elif self.dataset.evaluation_type == "coding":
                result = self._evaluate_coding_sample(sample, aggregator, verbose)
            else:
                result = self._evaluate_compression_sample(sample, aggregator, verbose)

            sample_results.append(result)
            self.agent.reset()

        # Aggregate metrics
        aggregate = aggregator.aggregate(group_by_category=True)

        if verbose:
            print(f"\n{'=' * 60}")
            print("EVALUATION COMPLETE")
            self._print_summary(aggregate)
            print(f"{'=' * 60}\n")

        return EvaluationResults(
            agent_name=self.agent.name,
            dataset_name=self.dataset.name,
            evaluation_type=self.dataset.evaluation_type,
            num_samples=len(sample_results),
            sample_results=sample_results,
            aggregate_metrics=aggregate,
        )

    def _run_multi_run_evaluation(
        self,
        samples: List[EvalSample],
        n_runs: int,
        confidence: float,
        verbose: bool,
    ) -> EvaluationResults:
        """
        Run multi-run evaluation to capture LLM variance.

        This runs each sample n_runs times, collecting variance statistics
        to provide confidence intervals and capture stochastic variation.
        """
        all_sample_results = []
        all_metric_values: Dict[str, List[float]] = {}

        for idx, sample in enumerate(samples):
            if verbose:
                print(f"Processing sample {idx + 1}/{len(samples)}: {sample.sample_id} ({n_runs} runs)")

            run_results = []

            for run_id in range(n_runs):
                if verbose and n_runs > 3:
                    print(f"  Run {run_id + 1}/{n_runs}")

                # Reset agent for each run
                self.agent.reset()

                # Create fresh aggregator for this run
                run_aggregator = UnifiedMetricAggregator()

                # Evaluate based on type
                if self.dataset.evaluation_type == "qa":
                    result = self._evaluate_qa_sample(sample, run_aggregator, False)
                elif self.dataset.evaluation_type == "coding":
                    result = self._evaluate_coding_sample(sample, run_aggregator, False)
                else:
                    result = self._evaluate_compression_sample(sample, run_aggregator, False)

                run_results.append(result)

            # Aggregate across runs for this sample
            # Use the first run's result as the base, but add variance info
            base_result = run_results[0]

            # Collect metric values across runs for this sample
            sample_metric_values: Dict[str, List[float]] = {}

            for result in run_results:
                # Extract metrics based on evaluation type
                if self.dataset.evaluation_type == "qa":
                    for qa_result in result.qa_results:
                        for metric_name, value in qa_result.metrics.items():
                            if metric_name not in sample_metric_values:
                                sample_metric_values[metric_name] = []
                            sample_metric_values[metric_name].append(value)
                elif self.dataset.evaluation_type == "coding":
                    for metric_name, value in result.coding_metrics.items():
                        if metric_name not in sample_metric_values:
                            sample_metric_values[metric_name] = []
                        sample_metric_values[metric_name].append(value)
                else:
                    # Compression - use compression_summary
                    for metric_name, value in result.compression_summary.items():
                        if isinstance(value, (int, float)):
                            if metric_name not in sample_metric_values:
                                sample_metric_values[metric_name] = []
                            sample_metric_values[metric_name].append(float(value))

            # Add sample metrics to overall collection
            for metric_name, values in sample_metric_values.items():
                if metric_name not in all_metric_values:
                    all_metric_values[metric_name] = []
                # Use mean of this sample's runs
                all_metric_values[metric_name].append(sum(values) / len(values))

            all_sample_results.append(base_result)

        # Compute statistical summary across all samples
        statistical_summary = {}
        for metric_name, values in all_metric_values.items():
            stat_result = compute_statistical_summary(
                values, confidence=confidence, use_bootstrap=True
            )
            statistical_summary[metric_name] = {
                "mean": stat_result.mean,
                "std": stat_result.std,
                "ci_lower": stat_result.ci_lower,
                "ci_upper": stat_result.ci_upper,
                "n": stat_result.n,
            }

        # Build aggregate metrics with CI info
        aggregate_metrics = {
            "overall": statistical_summary,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"MULTI-RUN EVALUATION COMPLETE ({n_runs} runs/sample)")
            print(f"Confidence level: {confidence:.0%}")
            self._print_statistical_summary(statistical_summary)
            print(f"{'=' * 60}\n")

        return EvaluationResults(
            agent_name=self.agent.name,
            dataset_name=self.dataset.name,
            evaluation_type=self.dataset.evaluation_type,
            num_samples=len(all_sample_results),
            sample_results=all_sample_results,
            aggregate_metrics=aggregate_metrics,
            n_runs=n_runs,
            confidence=confidence,
            statistical_summary=statistical_summary,
        )

    def _print_statistical_summary(self, summary: Dict[str, Any]) -> None:
        """Print statistical summary with confidence intervals."""
        print("\nStatistical Summary (with CI):")
        for metric_name, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                ci_width = (stats.get("ci_upper", 0) - stats.get("ci_lower", 0)) / 2
                print(f"  {metric_name}: {stats['mean']:.4f} ± {ci_width:.4f} "
                      f"(95% CI: [{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}])")

    def _evaluate_qa_sample(
        self,
        sample: EvalSample,
        aggregator: UnifiedMetricAggregator,
        verbose: bool,
    ) -> SampleResult:
        """Evaluate a QA-style sample (LoComo)."""
        # Check cache
        cached = False
        if self._cache and self._cache.has_amem_cache(sample.sample_id):
            if hasattr(self.agent, "memory_system"):
                cached = self._cache.load_amem_state(
                    sample.sample_id, self.agent.memory_system
                )
                if cached and verbose:
                    print(f"  Loaded cached memories")

        if not cached:
            # Ingest all turns
            for turn in sample.turns:
                self.agent.ingest_turn(turn.__dict__)

            # Cache state
            if self._cache and hasattr(self.agent, "memory_system"):
                self._cache.save_amem_state(sample.sample_id, self.agent.memory_system)
                if verbose:
                    print(f"  Cached {len(sample.turns)} turns")

        # Answer questions
        qa_results = []
        all_metrics: List[Dict[str, float]] = []
        all_categories: List[int] = []

        for q in sample.questions:
            prediction = self.agent.answer_question(
                question=q.question,
                category=q.category,
                reference_answer=q.adversarial_answer or q.reference_answer,
            )

            # Calculate metrics
            metric_results = self._qa_calc.calculate(
                prediction=prediction,
                reference=q.final_answer,
                category=q.category,
            )

            # Add to aggregator
            aggregator.add_batch(metric_results)

            # Convert to dict for storage
            metrics_dict = {r.name: r.value for r in metric_results}

            qa_results.append(
                QAResult(
                    question=q.question,
                    prediction=prediction,
                    reference=q.final_answer,
                    category=q.category,
                    metrics=metrics_dict,
                )
            )

            all_metrics.append(metrics_dict)
            if q.category:
                all_categories.append(q.category)

        # Aggregate QA metrics for this sample
        sample_aggregator = UnifiedMetricAggregator()
        for q_result in qa_results:
            for name, value in q_result.metrics.items():
                sample_aggregator.add(
                    MetricResult(
                        name=name,
                        value=value,
                        metric_type=MetricType.SCORE_0_1,
                        category=str(q_result.category) if q_result.category else None,
                    )
                )

        aggregate_qa = sample_aggregator.aggregate(group_by_category=True)

        return SampleResult(
            sample_id=sample.sample_id,
            agent_name=self.agent.name,
            evaluation_type="qa",
            qa_results=qa_results,
            aggregate_qa_metrics=aggregate_qa,
            context_sizes=[self.agent.get_context_size()],
        )

    def _evaluate_compression_sample(
        self,
        sample: EvalSample,
        aggregator: UnifiedMetricAggregator,
        verbose: bool,
    ) -> SampleResult:
        """Evaluate a compression-style sample (templates)."""
        # Initialize goal/constraints if available
        if sample.initial_goal:
            self.agent.initialize_goal(
                sample.initial_goal, sample.constraints or []
            )

        # Create metrics collector
        collector = MetricsCollector(
            original_goal=sample.initial_goal or "",
            constraints=sample.constraints or [],
        )

        compression_points_data = []
        context_sizes = []
        compression_point_counter = 0

        probing_tasks = sample.probing_tasks or {}

        for turn in sample.turns:
            self.agent.ingest_turn(turn.__dict__)
            context_sizes.append(self.agent.get_context_size())

            if turn.is_compression_point:
                compression_point_counter += 1

                if verbose:
                    print(f"  Compression point {compression_point_counter}")

                # Probe before compression
                goal_before = self.agent.answer_question(
                    probing_tasks.get("goal_probe", "What is your current goal?")
                )
                constraints_before = self.agent.answer_question(
                    probing_tasks.get("constraint_probe", "What constraints apply?")
                )

                tokens_before = self.agent.get_context_size()

                # Compress
                self.agent.compress(turn.id)

                tokens_after = self.agent.get_context_size()

                # Probe after compression
                goal_after = self.agent.answer_question(
                    probing_tasks.get("goal_probe", "What is your current goal?")
                )
                constraints_after = self.agent.answer_question(
                    probing_tasks.get("constraint_probe", "What constraints apply?")
                )

                # Behavioral test
                behavioral_test = probing_tasks.get("behavioral_test", {})
                behavioral_response = self.agent.answer_question(
                    behavioral_test.get("prompt", "What should we do next?")
                )

                # Collect metrics
                metrics = collector.collect_at_compression_point(
                    compression_point_id=compression_point_counter,
                    turn_id=turn.id,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    goal_stated_before=goal_before,
                    goal_stated_after=goal_after,
                    constraints_stated_before=constraints_before,
                    constraints_stated_after=constraints_after,
                    behavioral_response_after=behavioral_response,
                    behavioral_test_context=behavioral_test.get("prompt", ""),
                )

                compression_points_data.append(metrics.to_dict())

                # Add to unified aggregator
                aggregator.add_batch(
                    [
                        MetricResult(
                            "goal_coherence_before",
                            metrics.goal_coherence_before,
                            MetricType.SCORE_0_1,
                        ),
                        MetricResult(
                            "goal_coherence_after",
                            metrics.goal_coherence_after,
                            MetricType.SCORE_0_1,
                        ),
                        MetricResult(
                            "goal_drift", metrics.goal_drift, MetricType.DELTA
                        ),
                        MetricResult(
                            "constraint_recall_before",
                            metrics.constraint_recall_before,
                            MetricType.PERCENTAGE,
                        ),
                        MetricResult(
                            "constraint_recall_after",
                            metrics.constraint_recall_after,
                            MetricType.PERCENTAGE,
                        ),
                        MetricResult(
                            "constraint_loss", metrics.constraint_loss, MetricType.DELTA
                        ),
                        MetricResult(
                            "behavioral_alignment_after",
                            metrics.behavioral_alignment_after,
                            MetricType.SCORE_1_5,
                        ),
                        MetricResult(
                            "compression_ratio",
                            metrics.compression_ratio,
                            MetricType.PERCENTAGE,
                        ),
                    ]
                )

        return SampleResult(
            sample_id=sample.sample_id,
            agent_name=self.agent.name,
            evaluation_type="compression",
            compression_points=compression_points_data,
            compression_summary=collector.get_summary(),
            context_sizes=context_sizes,
        )

    def _evaluate_coding_sample(
        self,
        sample: EvalSample,
        aggregator: UnifiedMetricAggregator,
        verbose: bool,
    ) -> SampleResult:
        """Evaluate a coding task sample."""
        # Initialize goal/constraints if available
        if sample.initial_goal:
            self.agent.initialize_goal(
                sample.initial_goal, sample.constraints or []
            )

        # Set initial files if the agent supports it and sample has them
        initial_files = sample.metadata.get("initial_files", {})
        if initial_files and hasattr(self.agent, "set_initial_files"):
            self.agent.set_initial_files(initial_files)

        # Get ground truth
        ground_truth = sample.ground_truth or {}
        task_type = sample.metadata.get("task_type", "code_generation")

        # Create metrics collector for compression points
        collector = MetricsCollector(
            original_goal=sample.initial_goal or "",
            constraints=sample.constraints or [],
        )

        compression_points_data = []
        context_sizes = []
        compression_point_counter = 0
        probing_tasks = sample.probing_tasks or {}

        # Process turns
        for turn in sample.turns:
            self.agent.ingest_turn(turn.__dict__)
            context_sizes.append(self.agent.get_context_size())

            if turn.is_compression_point:
                compression_point_counter += 1

                if verbose:
                    print(f"  Compression point {compression_point_counter}")

                # Probe before compression
                goal_before = self.agent.answer_question(
                    probing_tasks.get("goal_probe", "What is the current coding task you're working on?")
                )
                constraints_before = self.agent.answer_question(
                    probing_tasks.get("constraint_probe", "What constraints or requirements must the solution meet?")
                )

                tokens_before = self.agent.get_context_size()

                # Compress
                self.agent.compress(turn.id)

                tokens_after = self.agent.get_context_size()

                # Probe after compression
                goal_after = self.agent.answer_question(
                    probing_tasks.get("goal_probe", "What is the current coding task you're working on?")
                )
                constraints_after = self.agent.answer_question(
                    probing_tasks.get("constraint_probe", "What constraints or requirements must the solution meet?")
                )

                # Behavioral test
                behavioral_test = probing_tasks.get("behavioral_test", {})
                behavioral_response = self.agent.answer_question(
                    behavioral_test.get("prompt", "What would be your next step to complete this task?")
                )

                # Collect metrics
                metrics = collector.collect_at_compression_point(
                    compression_point_id=compression_point_counter,
                    turn_id=turn.id,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    goal_stated_before=goal_before,
                    goal_stated_after=goal_after,
                    constraints_stated_before=constraints_before,
                    constraints_stated_after=constraints_after,
                    behavioral_response_after=behavioral_response,
                    behavioral_test_context=behavioral_test.get("prompt", ""),
                )

                compression_points_data.append(metrics.to_dict())

                # Add compression metrics to aggregator
                aggregator.add_batch(
                    [
                        MetricResult(
                            "goal_coherence_after",
                            metrics.goal_coherence_after,
                            MetricType.SCORE_0_1,
                        ),
                        MetricResult(
                            "goal_drift", metrics.goal_drift, MetricType.DELTA
                        ),
                        MetricResult(
                            "constraint_recall_after",
                            metrics.constraint_recall_after,
                            MetricType.PERCENTAGE,
                        ),
                        MetricResult(
                            "compression_ratio",
                            metrics.compression_ratio,
                            MetricType.PERCENTAGE,
                        ),
                    ]
                )

        # Get generated code from agent (if supported)
        generated_code = {}
        if hasattr(self.agent, "get_generated_code"):
            generated_code = self.agent.get_generated_code()

        # Calculate final coding metrics
        coding_metrics_results = self._coding_calc.calculate(
            generated_code="\n".join(generated_code.values()) if generated_code else "",
            expected_code=ground_truth.get("expected_code"),
            expected_files=ground_truth.get("expected_files"),
            test_cases=ground_truth.get("test_cases", []),
            acceptance_criteria=ground_truth.get("acceptance_criteria", []),
            language="python",
            task_type=task_type,
            tokens_before_compression=context_sizes[0] if context_sizes else 0,
            tokens_after_compression=context_sizes[-1] if context_sizes else 0,
        )

        # Add coding metrics to aggregator
        aggregator.add_batch(coding_metrics_results)

        # Convert to dict for storage
        coding_metrics_dict = {r.name: r.value for r in coding_metrics_results}

        # Create coding result
        coding_result = CodingResult(
            task_type=task_type,
            generated_code=generated_code,
            metrics=coding_metrics_dict,
            compression_points=compression_points_data,
        )

        if verbose:
            print(f"  Task type: {task_type}")
            print(f"  Files generated: {len(generated_code)}")
            print(f"  Compression points: {compression_point_counter}")
            if "syntax_validity" in coding_metrics_dict:
                print(f"  Syntax valid: {coding_metrics_dict['syntax_validity']:.2f}")

        return SampleResult(
            sample_id=sample.sample_id,
            agent_name=self.agent.name,
            evaluation_type="coding",
            coding_result=coding_result,
            coding_metrics=coding_metrics_dict,
            compression_points=compression_points_data,
            compression_summary=collector.get_summary() if compression_points_data else {},
            context_sizes=context_sizes,
        )

    def _print_summary(self, aggregate: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results."""
        if "qa_summary" in aggregate and aggregate["qa_summary"]:
            print("\nQA Summary:")
            for key, value in aggregate["qa_summary"].items():
                print(f"  {key}: {value:.4f}")

        if "compression_summary" in aggregate and aggregate["compression_summary"]:
            print("\nCompression Summary:")
            for key, value in aggregate["compression_summary"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        if "coding_summary" in aggregate and aggregate["coding_summary"]:
            print("\nCoding Summary:")
            for key, value in aggregate["coding_summary"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
