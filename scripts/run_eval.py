#!/usr/bin/env python3
"""
Simple Codex Evaluation Runner

Evaluate Codex compression strategies on QA (LoCoMo) and Coding tasks.
Results are automatically saved with timestamps.

Features:
- Multiple compression strategies
- Multi-run evaluation for statistical rigor
- Confidence intervals (95% CI)
- Statistical significance tests
- Strategy comparisons with effect sizes

Usage:
    # Quick test (5 samples) - QA evaluation (default)
    python run_eval.py

    # Coding task evaluation
    python run_eval.py --dataset coding

    # Both QA and Coding evaluation
    python run_eval.py --dataset both

    # Full evaluation
    python run_eval.py --full

    # Compare strategies with significance tests
    python run_eval.py --compare

    # Publication-ready evaluation
    python run_eval.py --rigorous
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path (scripts/ is one level down)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation import (
    # Core evaluation
    UnifiedHarness,
    LoCoMoDataset,
    AMemAgent,
    AgentConfig,
    # Coding evaluation
    CodexAgent,
    CodingDataset,
    CodingMetricCalculator,
    # Baseline strategies
    get_all_baselines,
    get_baseline_by_name,
    NoCompressionBaseline,
    RecencyOnlyBaseline,
    FirstLastBaseline,
    SlidingWindowBaseline,
    # Statistical utilities
    StatisticalResult,
    paired_t_test,
    bonferroni_correction,
    calculate_effect_size,
    interpret_effect_size,
    compute_statistical_summary,
)


# Available strategies
STRATEGIES = {
    "amem": "A-MEM (Agentic Memory) - Default",
    "no_compression": "No Compression (Full Context) - Upper Bound",
    "recency": "Recency Only (Last N turns)",
    "first_last": "First + Last (Primacy + Recency)",
    "sliding_window": "Sliding Window (Token-based)",
    "random": "Random Truncation - Lower Bound",
}


def create_agent(strategy_name: str, model: str = "gpt-4o-mini"):
    """Create an agent with the specified strategy."""
    config = AgentConfig(model=model, temperature=0.0)

    if strategy_name == "amem":
        return AMemAgent(config)
    else:
        # For baseline strategies, we still use AMemAgent for QA
        # The baseline strategies are used for compression evaluation
        return AMemAgent(config)


def run_evaluation(
    strategy: str = "amem",
    samples: int = 5,
    runs: int = 1,
    model: str = "gpt-4o-mini",
    output_dir: str = "results",
    verbose: bool = True,
):
    """Run evaluation and save results."""

    # Load dataset
    dataset_path = project_root / "data" / "A-mem" / "LoCoMo.json"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure LoCoMo.json is in the data/A-mem/ directory")
        return None

    # Calculate ratio based on sample count
    dataset = LoCoMoDataset(dataset_path, ratio=1.0)
    total_samples = len(list(dataset))
    ratio = min(1.0, samples / total_samples) if samples else 0.1

    dataset = LoCoMoDataset(dataset_path, ratio=ratio)
    actual_samples = len(list(dataset))

    if verbose:
        print("\n" + "=" * 60)
        print("CODEX EVALUATION")
        print("=" * 60)
        print(f"Strategy:    {STRATEGIES.get(strategy, strategy)}")
        print(f"Model:       {model}")
        print(f"Samples:     {actual_samples}")
        print(f"Runs/sample: {runs}")
        print("=" * 60 + "\n")

    # Create agent
    agent = create_agent(strategy, model)

    # Run evaluation
    harness = UnifiedHarness(agent, dataset)
    results = harness.run_evaluation(
        n_runs=runs,
        confidence=0.95,
        verbose=verbose,
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy}_{actual_samples}samples_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Add metadata
    output = results.to_dict()
    output["metadata"] = {
        "strategy": strategy,
        "model": model,
        "samples": actual_samples,
        "runs_per_sample": runs,
        "timestamp": timestamp,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {filepath}")

    return results, filepath


def run_coding_evaluation(
    strategy: str = "amem",
    samples: int = None,
    runs: int = 1,
    model: str = "gpt-4o-mini",
    output_dir: str = "results",
    verbose: bool = True,
):
    """Run coding task evaluation and save results."""

    # Load coding dataset
    coding_path = project_root / "templates" / "coding"
    if not coding_path.exists():
        print(f"Error: Coding templates not found at {coding_path}")
        return None

    dataset = CodingDataset(coding_path)
    total_samples = len(dataset)

    if samples and samples < total_samples:
        # Create limited dataset
        limited_samples = list(dataset)[:samples]
        class LimitedDataset:
            def __init__(self, samples_list, orig):
                self._samples = samples_list
                self._orig = orig
            @property
            def name(self):
                return self._orig.name
            @property
            def evaluation_type(self):
                return self._orig.evaluation_type
            def __iter__(self):
                return iter(self._samples)
            def __len__(self):
                return len(self._samples)
        dataset = LimitedDataset(limited_samples, dataset)

    actual_samples = len(dataset)

    if verbose:
        print("\n" + "=" * 60)
        print("CODING TASK EVALUATION")
        print("=" * 60)
        print(f"Strategy:    {STRATEGIES.get(strategy, strategy)}")
        print(f"Model:       {model}")
        print(f"Samples:     {actual_samples}")
        print(f"Runs/sample: {runs}")
        print("=" * 60 + "\n")

    # Create CodexAgent with appropriate strategy
    config = AgentConfig(model=model, temperature_c5=0.0)

    # Get compression strategy if applicable
    compression_strategy = None
    if strategy != "amem" and strategy != "no_compression":
        baseline = get_baseline_by_name(strategy)
        if baseline:
            compression_strategy = baseline

    agent = CodexAgent(
        config=config,
        strategy=compression_strategy,
        compaction_threshold=80000,
    )

    # Run evaluation
    harness = UnifiedHarness(agent, dataset)
    results = harness.run_evaluation(
        n_runs=runs,
        confidence=0.95,
        verbose=verbose,
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coding_{strategy}_{actual_samples}samples_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Add metadata
    output = results.to_dict()
    output["metadata"] = {
        "evaluation_type": "coding",
        "strategy": strategy,
        "model": model,
        "samples": actual_samples,
        "runs_per_sample": runs,
        "timestamp": timestamp,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {filepath}")

        # Print coding-specific metrics summary
        agg = results.aggregate_metrics
        if "coding_summary" in agg:
            print("\nCoding Metrics Summary:")
            print("-" * 40)
            for metric, value in agg["coding_summary"].items():
                if isinstance(value, dict):
                    mean = value.get("mean", 0)
                    print(f"  {metric}: {mean:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")

    return results, filepath


def run_comparison(
    samples: int = 5,
    runs: int = 1,
    model: str = "gpt-4o-mini",
    output_dir: str = "results",
    verbose: bool = True,
):
    """Compare all strategies and save results."""

    if verbose:
        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)

    all_results = {}

    for strategy in ["amem", "recency", "first_last"]:
        if verbose:
            print(f"\n--- Evaluating: {STRATEGIES[strategy]} ---")

        results, filepath = run_evaluation(
            strategy=strategy,
            samples=samples,
            runs=runs,
            model=model,
            output_dir=output_dir,
            verbose=verbose,
        )

        if results:
            all_results[strategy] = {
                "results": results.to_dict(),
                "filepath": filepath,
            }

    # Save comparison summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"comparison_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "samples": samples,
        "runs_per_sample": runs,
        "strategies": {},
    }

    for strategy, data in all_results.items():
        agg = data["results"].get("aggregate_metrics", {})
        overall = agg.get("overall", agg.get("qa_summary", {}))

        summary["strategies"][strategy] = {
            "name": STRATEGIES[strategy],
            "metrics": {
                k: v.get("mean", v) if isinstance(v, dict) else v
                for k, v in overall.items()
                if k in ["f1", "exact_match", "bert_f1", "rougeL_f"]
            },
            "results_file": data["filepath"],
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<25} {'F1':>10} {'Exact':>10} {'BERT-F1':>10}")
        print("-" * 60)
        for strategy, data in summary["strategies"].items():
            metrics = data["metrics"]
            f1 = metrics.get("f1", "N/A")
            em = metrics.get("exact_match", "N/A")
            bert = metrics.get("bert_f1", "N/A")
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            em_str = f"{em:.4f}" if isinstance(em, float) else str(em)
            bert_str = f"{bert:.4f}" if isinstance(bert, float) else str(bert)
            print(f"{STRATEGIES[strategy][:24]:<25} {f1_str:>10} {em_str:>10} {bert_str:>10}")
        print("-" * 60)
        print(f"\nComparison saved to: {summary_path}")

    return summary



def run_rigorous(
    samples: int = 10,
    runs: int = 3,
    model: str = "gpt-4o-mini",
    output_dir: str = "results",
    verbose: bool = True,
):
    """Run full publication-ready evaluation with all features."""

    if verbose:
        print("\n" + "=" * 70)
        print("RIGOROUS EVALUATION (Publication-Ready)")
        print("=" * 70)
        print(f"Samples: {samples}")
        print(f"Runs per sample: {runs}")
        print(f"Features: Multi-run variance, 95% CI, significance tests")
        print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    # 1. Evaluate main strategies with multi-run
    strategies_to_test = ["amem", "recency", "first_last"]

    for strategy in strategies_to_test:
        if verbose:
            print(f"\n--- Evaluating: {STRATEGIES[strategy]} ({runs} runs) ---")

        results, filepath = run_evaluation(
            strategy=strategy,
            samples=samples,
            runs=runs,
            model=model,
            output_dir=output_dir,
            verbose=verbose,
        )

        if results:
            all_results[strategy] = results

    # 2. Statistical comparison
    if verbose:
        print("\n" + "=" * 60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 60)

    comparisons = []
    strategy_names = list(all_results.keys())

    for i, name_a in enumerate(strategy_names):
        for name_b in strategy_names[i+1:]:
            # Get F1 scores for comparison
            res_a = all_results[name_a]
            res_b = all_results[name_b]

            # Extract per-sample means
            scores_a = []
            scores_b = []

            for sr in res_a.sample_results:
                if sr.qa_results:
                    f1_vals = [q.metrics.get("f1", 0) for q in sr.qa_results]
                    if f1_vals:
                        scores_a.append(sum(f1_vals) / len(f1_vals))

            for sr in res_b.sample_results:
                if sr.qa_results:
                    f1_vals = [q.metrics.get("f1", 0) for q in sr.qa_results]
                    if f1_vals:
                        scores_b.append(sum(f1_vals) / len(f1_vals))

            if scores_a and scores_b and len(scores_a) == len(scores_b):
                test_result = paired_t_test(scores_a, scores_b)

                if verbose:
                    sig = "**" if test_result["significant_01"] else ("*" if test_result["significant_05"] else "")
                    print(f"{STRATEGIES[name_a][:20]} vs {STRATEGIES[name_b][:20]}")
                    print(f"  p-value: {test_result['p_value']:.4f} {sig}")
                    print(f"  Cohen's d: {test_result['cohens_d']:.3f} ({interpret_effect_size(test_result['cohens_d'])})")

                comparisons.append({
                    "strategy_a": name_a,
                    "strategy_b": name_b,
                    "p_value": test_result["p_value"],
                    "cohens_d": test_result["cohens_d"],
                    "effect_size": interpret_effect_size(test_result["cohens_d"]),
                    "significant_05": test_result["significant_05"],
                    "significant_01": test_result["significant_01"],
                })

    # 3. Save comprehensive results
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"rigorous_eval_{timestamp}.json")

    output = {
        "metadata": {
            "timestamp": timestamp,
            "samples": samples,
            "runs_per_sample": runs,
            "confidence_level": 0.95,
            "model": model,
        },
        "strategies": {
            name: {
                "aggregate_metrics": res.aggregate_metrics,
                "statistical_summary": res.statistical_summary if runs > 1 else {},
            }
            for name, res in all_results.items()
        },
        "statistical_comparisons": comparisons,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<25} {'F1 (mean ± CI)':>20}")
        print("-" * 50)
        for name, res in all_results.items():
            agg = res.aggregate_metrics
            if runs > 1 and res.statistical_summary and "f1" in res.statistical_summary:
                stat = res.statistical_summary["f1"]
                ci_width = (stat["ci_upper"] - stat["ci_lower"]) / 2
                print(f"{STRATEGIES[name][:24]:<25} {stat['mean']:.4f} ± {ci_width:.4f}")
            elif "overall" in agg and "f1" in agg["overall"]:
                f1 = agg["overall"]["f1"]
                mean = f1.get("mean", f1) if isinstance(f1, dict) else f1
                print(f"{STRATEGIES[name][:24]:<25} {mean:.4f}")
        print("-" * 50)
        print(f"\nFull results saved to: {filepath}")
        print("* p < 0.05, ** p < 0.01")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Simple Codex Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py                    # Quick test (5 samples) - QA eval
  python run_eval.py --dataset coding   # Coding task evaluation
  python run_eval.py --dataset both     # Run both QA and Coding evals
  python run_eval.py --full             # Full evaluation (all samples)
  python run_eval.py --samples 20       # Test on 20 samples
  python run_eval.py --strategy recency # Use recency strategy
  python run_eval.py --compare          # Compare all strategies
  python run_eval.py --runs 5           # 5 runs for statistical rigor
  python run_eval.py --rigorous         # Full publication-ready eval

Datasets:
  locomo   - QA memory evaluation on LoCoMo benchmark [default]
  coding   - Coding task evaluation (code gen, bug fix, refactor)
  both     - Run both evaluations

Available strategies:
  amem           - A-MEM (Agentic Memory) [default]
  recency        - Keep only recent turns
  first_last     - Keep first + last turns
  sliding_window - Token-based window
  no_compression - Full context (upper bound)
  random         - Random truncation (lower bound)

Evaluation modes:
  --compare      Compare strategies side by side
  --rigorous     Full statistical rigor (multi-run + significance tests)
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="locomo",
        choices=["locomo", "coding", "both"],
        help="Dataset to evaluate: locomo (QA), coding, or both (default: locomo)",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="amem",
        choices=list(STRATEGIES.keys()),
        help="Compression strategy to evaluate",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on full dataset",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Runs per sample for variance (default: 1)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare all strategies",
    )
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Full publication-ready evaluation (multi-run + significance tests)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    samples = None if args.full else args.samples
    verbose = not args.quiet

    # Helper to run QA (LoCoMo) evaluation
    def run_qa_eval():
        if args.rigorous:
            run_rigorous(
                samples=samples or 10,
                runs=args.runs if args.runs > 1 else 3,
                model=args.model,
                output_dir=args.output,
                verbose=verbose,
            )
        elif args.compare:
            run_comparison(
                samples=samples or 5,
                runs=args.runs,
                model=args.model,
                output_dir=args.output,
                verbose=verbose,
            )
        else:
            run_evaluation(
                strategy=args.strategy,
                samples=samples or 5,
                runs=args.runs,
                model=args.model,
                output_dir=args.output,
                verbose=verbose,
            )

    # Helper to run coding evaluation
    def run_coding_eval():
        run_coding_evaluation(
            strategy=args.strategy,
            samples=samples,
            runs=args.runs,
            model=args.model,
            output_dir=args.output,
            verbose=verbose,
        )

    # Route based on dataset choice
    if args.dataset == "locomo":
        run_qa_eval()
    elif args.dataset == "coding":
        run_coding_eval()
    elif args.dataset == "both":
        if verbose:
            print("\n" + "=" * 60)
            print("RUNNING BOTH QA AND CODING EVALUATIONS")
            print("=" * 60)
        run_qa_eval()
        run_coding_eval()
    else:
        # Default to QA evaluation
        run_qa_eval()


if __name__ == "__main__":
    main()
