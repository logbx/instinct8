# Codex Compaction Evaluation Guide

Benchmark and compare your custom Codex compaction implementations against the baseline.

## Overview

This framework evaluates Codex CLI's context compaction performance by measuring:
- **Goal Retention**: How well Codex remembers task objectives after compression
- **Task Completion**: Success rate on coding tasks
- **QA Accuracy**: Performance on long-context question answering (LoCoMo dataset)

## Repository Structure

```
instinct8/main/                        # BASELINE + EVALUATION FRAMEWORK
├── codex/                             # BASELINE Codex (do not modify)
│   └── codex-rs/                      # Rust source
│       └── target/release/codex       # Baseline binary (after build)
│
├── codex-experimental/                # YOUR MODIFIED CODEX (default location)
│   └── codex-rs/                      # Your modified Rust source
│       └── target/release/codex       # Your modified binary
│
├── evaluation/                        # Python evaluation framework
├── strategies/                        # Compression strategy implementations
├── templates/                         # Test templates
│   └── coding/                        # Coding task definitions
├── data/                              # External datasets
│   └── A-mem/                         # LoCoMo QA dataset
├── docs/                              # Documentation
├── results/                           # Evaluation results
│
├── scripts/                           # Evaluation entry points
│   ├── run_comparison_eval.py         # Compare baseline vs modified
│   ├── run_codex_cli_eval.py          # Evaluate single Codex version
│   ├── run_eval.py                    # General evaluation runner
│   └── run_locomo_eval.py             # LoCoMo QA evaluation
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview
```

**Custom naming:** You can name your modified Codex directory anything:
- `codex-graphrag/` - Use `--name graphrag`
- `codex-my-version/` - Use `--name my-version`
- `codex-experimental/` - Default (no `--name` needed)

---

## Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/LoganLiangMay/instinct8.git
cd instinct8/main
```

### Step 2: Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Step 4: Install Rust (if not installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

---

## Building Codex Binaries

### Build Baseline Codex

```bash
cd codex/codex-rs
cargo build --release
cd ../..
```

Verify:
```bash
./codex/codex-rs/target/release/codex --version
```

### Build Your Modified Codex

```bash
# If your modified version is in ../graphrag-rs/
cd ../graphrag-rs/codex/codex-rs
cargo build --release
cd ../../../main
```

Verify:
```bash
../graphrag-rs/codex/codex-rs/target/release/codex --version
```

### Build Both at Once

```bash
python scripts/run_comparison_eval.py --build-all
```

---

## Running Evaluations

### 1. Quick Smoke Test (No Codex Required)

Test the framework works with simulation mode:

```bash
python scripts/run_eval.py --dataset coding --samples 1
```

Expected output:
```
============================================================
CODING TASK EVALUATION
============================================================
Strategy:    A-MEM (Agentic Memory) - Default
Model:       gpt-4o-mini
...
Results saved to: results/coding_amem_1samples_YYYYMMDD.json
```

### 2. LoCoMo QA Evaluation

Test long-context question answering:

```bash
# Quick test (10% of dataset)
python scripts/run_locomo_eval.py --ratio 0.1

# Full evaluation
python scripts/run_locomo_eval.py --ratio 1.0
```

### 3. Coding Task Evaluation

```bash
# Single sample
python scripts/run_eval.py --dataset coding --samples 1

# All coding tasks
python scripts/run_eval.py --dataset coding
```

### 4. Compare Baseline vs Your Modified Codex

This is the main evaluation for testing your compaction changes:

```bash
# Compare (both binaries must be built)
python scripts/run_comparison_eval.py --verbose

# Build and compare in one command
python scripts/run_comparison_eval.py --build-all --verbose

# Save results to file
python scripts/run_comparison_eval.py --verbose --output results/my_comparison.json
```

Expected output:
```
======================================================================
CODEX COMPARISON EVALUATION
======================================================================

Baseline:  codex 1.0.0
Modified:  codex 1.0.0-custom
Tasks:     3

----------------------------------------------------------------------
RESULTS COMPARISON
----------------------------------------------------------------------

Metric               Baseline     Modified         Diff     Change
----------------------------------------------------------------------
Goal Retention          0.650        0.920       +0.270     BETTER
Task Completion         0.700        0.850       +0.150     BETTER
Has Code                0.800        1.000       +0.200     BETTER
No Errors               0.900        0.900       +0.000       SAME
----------------------------------------------------------------------

======================================================================
SUMMARY
======================================================================

Improvements: Goal Retention: +41.5%, Task Completion: +21.4%
```

### 5. Rigorous Statistical Evaluation

For publication-quality results with confidence intervals:

```bash
# Multiple runs per sample
python scripts/run_eval.py --rigorous --samples 10
```

---

## Full Test Suite

Run all evaluations in sequence:

```bash
#!/bin/bash
# full_test.sh

set -e  # Exit on error
cd /path/to/instinct8/main
source .venv/bin/activate
export OPENAI_API_KEY="your-key"

echo "=== 1. Testing Imports ==="
python -c "from evaluation import CodexCLIWrapper, UnifiedHarness; print('OK')"

echo "=== 2. Coding Evaluation (1 sample) ==="
python scripts/run_eval.py --dataset coding --samples 1

echo "=== 3. LoCoMo Evaluation (10%) ==="
python scripts/run_locomo_eval.py --ratio 0.1

echo "=== 4. Building Codex Binaries ==="
python scripts/run_comparison_eval.py --build-all

echo "=== 5. Comparison Evaluation ==="
python scripts/run_comparison_eval.py --verbose --output results/full_comparison.json

echo "=== ALL TESTS PASSED ==="
```

---

## Creating Your Modified Codex

### Option A: Use Default Location (Recommended)

```bash
cd instinct8/main

# Copy baseline to experimental directory
cp -r codex codex-experimental

# Make your modifications
cd codex-experimental/codex-rs
vim core/src/compaction/mod.rs  # Edit compaction logic
cargo build --release

# Run comparison (no --name needed for default)
cd ..
python scripts/run_comparison_eval.py --verbose
```

### Option B: Custom Named Version

```bash
cd instinct8/main

# Create a named version
cp -r codex codex-graphrag
cd codex-graphrag/codex-rs

# Make your modifications
vim core/src/compaction/mod.rs
cargo build --release

# Run comparison with --name flag
cd ..
python scripts/run_comparison_eval.py --name graphrag --verbose
```

### Option C: Multiple Versions

You can have multiple versions side-by-side:

```
instinct8/main/
├── codex/                    # Baseline (never modify)
├── codex-experimental/       # Your primary experiment
├── codex-graphrag/           # GraphRAG experiment
└── codex-semantic/           # Semantic chunking experiment
```

Compare each version:
```bash
python scripts/run_comparison_eval.py --name experimental --verbose
python scripts/run_comparison_eval.py --name graphrag --verbose
python scripts/run_comparison_eval.py --name semantic --verbose
```

---

## Where to Modify Compaction Logic

Key files in `codex/codex-rs/`:

```
codex-rs/
├── core/
│   └── src/
│       ├── compaction/           # <-- MAIN COMPACTION LOGIC
│       │   ├── mod.rs
│       │   └── strategies/
│       ├── context/              # Context window management
│       │   ├── mod.rs
│       │   └── window.rs
│       └── memory/               # Memory/state handling
├── exec/
│   └── src/
│       └── agent/                # Agent execution flow
└── Cargo.toml
```

### Common Modifications

1. **Change compression threshold**: Edit when compaction triggers
2. **Modify summarization**: Change how context is summarized
3. **Adjust retention policy**: What information to keep vs discard
4. **Add semantic chunking**: Implement smarter context splitting

---

## Understanding the Metrics

| Metric | Description | Range | Target |
|--------|-------------|-------|--------|
| **Goal Retention** | How well Codex remembers original task after compression | 0-1 | Higher is better |
| **Task Completion** | Weighted score: output (30%) + code (40%) + no errors (30%) | 0-1 | Higher is better |
| **Has Code** | Whether response contains code blocks | 0 or 1 | 1.0 |
| **No Errors** | Absence of error messages in output | 0 or 1 | 1.0 |

### Interpreting Results

| Goal Retention | Interpretation |
|----------------|----------------|
| > 0.90 | Excellent - goals well preserved through compression |
| 0.70 - 0.90 | Good - minor information loss |
| 0.50 - 0.70 | Fair - significant drift from original goal |
| < 0.50 | Poor - compaction losing critical information |

---

## Adding Custom Test Tasks

Create JSON files in `templates/coding/`:

```json
{
  "task_id": "my-custom-task",
  "specification": {
    "goal": "Implement a rate limiter using the token bucket algorithm",
    "language": "python",
    "constraints": [
      "Must be thread-safe",
      "Support configurable rate and burst size",
      "Include async support"
    ],
    "starting_code": "class RateLimiter:\n    def __init__(self, rate: float, burst: int):\n        pass\n"
  },
  "ground_truth": {
    "key_components": ["token bucket", "thread-safe", "async"],
    "expected_functions": ["__init__", "acquire", "try_acquire"]
  }
}
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python scripts/run_comparison_eval.py` | Compare baseline vs codex-experimental/ |
| `python scripts/run_comparison_eval.py --name graphrag` | Compare baseline vs codex-graphrag/ |
| `python scripts/run_comparison_eval.py --build-all` | Build both, then compare |
| `python scripts/run_comparison_eval.py --build-all --name graphrag` | Build and compare named version |
| `python scripts/run_comparison_eval.py --modified /path/to/codex` | Use explicit binary path |
| `python scripts/run_codex_cli_eval.py` | Evaluate single Codex binary |
| `python scripts/run_codex_cli_eval.py --codex-path /path/to/codex` | Evaluate specific binary |
| `python scripts/run_eval.py --dataset coding` | Run coding task evaluation |
| `python scripts/run_eval.py --dataset locomo` | Run LoCoMo QA evaluation |
| `python scripts/run_locomo_eval.py --ratio 0.1` | Quick LoCoMo test (10%) |
| `python scripts/run_eval.py --rigorous` | Statistical evaluation |

---

## Troubleshooting

### "Baseline Codex not found"

```bash
# Build baseline
cd codex/codex-rs
cargo build --release
```

### "Modified Codex not found"

```bash
# For default codex-experimental/:
ls -la codex-experimental/codex-rs/target/release/codex

# If missing, copy baseline and build:
cp -r codex codex-experimental
cd codex-experimental/codex-rs
cargo build --release

# For custom named version (e.g., graphrag):
ls -la codex-graphrag/codex-rs/target/release/codex
```

### "cargo: command not found"

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Python import errors

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### OpenAI API errors

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  1. SETUP                                                    │
│     git clone → python venv → pip install → export API key  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. BUILD BASELINE                                           │
│     cd codex/codex-rs → cargo build --release               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. CREATE MODIFIED CODEX                                    │
│     cp -r codex codex-experimental                          │
│     cd codex-experimental/codex-rs                          │
│     Edit compaction logic → cargo build --release           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. EVALUATE                                                 │
│     python scripts/run_comparison_eval.py --verbose                 │
│     (or: --name <version> for custom named versions)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. ITERATE                                                  │
│     Review results → Improve compaction → Re-evaluate       │
└─────────────────────────────────────────────────────────────┘
```

---

## Support

- **Issues**: https://github.com/LoganLiangMay/instinct8/issues
- **Documentation**: See `docs/` directory
