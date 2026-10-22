# Testing and Evaluation Guide

This document explains how to run tests and evaluations for the instinct8 compression evaluation framework.

## Quick Start

```bash
# See all available commands
make help

# Run unit tests
make test

# Quick sanity check (5 samples, ~2 minutes)
make eval-quick

# Compare compression strategies
make eval-compare
```

---

## Test Matrix

Choose the right evaluation for your use case:

| Use Case | Command | Time | Description |
|----------|---------|------|-------------|
| Development testing | `make test` | ~10s | All tests |
| Unit tests only | `make test-quick` | ~5s | Skips integration (no API keys needed) |
| Integration tests | `make test-integration` | ~5m | API-dependent tests only |
| Quick sanity check | `make eval-quick` | ~2m | 5 samples, basic metrics |
| Hierarchical testing | `make eval-hierarchical` | ~5m | Domain/category/episode recall |
| Full benchmark | `make eval-full` | ~30m | All LoCoMo samples |
| Strategy comparison | `make eval-compare` | ~15m | Compare all strategies |
| Publication-ready | `make eval-rigorous` | ~1hr | Multi-run, statistical tests |
| Real binary test | `make eval-binary` | ~10m | Test actual Codex CLI |
| App-server compaction | `make eval-appserver` | ~5m | Test compaction via JSON-RPC |

---

## Evaluation Scripts Reference

### Essential Scripts (Primary Use)

#### 1. `run_eval.py` - Master Evaluation Script
The main entry point for most evaluations.

```bash
# Basic evaluation
python scripts/run_eval.py --dataset locomo

# Compare strategies
python scripts/run_eval.py --compare

# Publication-ready with statistical rigor
python scripts/run_eval.py --rigorous --runs 3

# Quick test (5 samples)
python scripts/run_eval.py --dataset locomo --samples 5
```

**Strategies:** `amem`, `no_compression`, `recency`, `first_last`, `sliding_window`

#### 2. `run_hierarchical_eval.py` - Hierarchical Depth Evaluation
Tests information retention at different hierarchy levels.

```bash
# Single strategy
python scripts/run_hierarchical_eval.py --strategy amem

# Compare strategies
python scripts/run_hierarchical_eval.py --compare

# Available strategies: amem, codex, no_compression, recency, first_last
```

**Metrics:**
- Domain Recall (high-level summary retention)
- Category Recall (mid-level pattern retention)
- Episode Recall (specific detail retention)
- Hierarchy Drift (degradation at deeper levels)

#### 3. `run_codex_cli_eval.py` - Real Codex Binary Test
Tests against an installed Codex CLI binary.

```bash
# Auto-detect Codex installation
python scripts/run_codex_cli_eval.py

# Specify path
python scripts/run_codex_cli_eval.py --codex-path /path/to/codex
```

#### 4. `run_comparison_eval.py` - A/B Binary Comparison
Compare two Codex binary variants side-by-side.

```bash
# Compare baseline vs variant
python scripts/run_comparison_eval.py --name my-variant

# Auto-build from source
python scripts/run_comparison_eval.py --name my-variant --build-all
```

#### 5. `run_appserver_eval.py` - App-Server Compaction Test
Tests Codex compaction via JSON-RPC app-server protocol.

```bash
# Basic evaluation
python scripts/run_appserver_eval.py

# Custom settings
python scripts/run_appserver_eval.py --token-limit 5000 --turns 15

# Specify binary path
python scripts/run_appserver_eval.py --codex-path /path/to/codex
```

**Metrics:**
- Goal Retention (keyword presence after compaction)
- Constraint Retention (constraint preservation)
- Decision Retention (architectural decisions preserved)
- Compaction Events (number of compaction triggers)

### Deprecated Scripts (Use Alternatives)

| Script | Use Instead |
|--------|-------------|
| `run_rigorous_eval.py` | `run_eval.py --rigorous` |
| `run_locomo_eval.py` | `run_eval.py --dataset locomo` |
| `scripts/archive/run_codex_eval.py` | Kept as baseline reference only |

---

## Understanding Results

### Output Location
All results are saved to `results/` directory:
```
results/
├── evaluation_*.json           # run_eval.py output
├── hierarchical_*.json         # run_hierarchical_eval.py output
├── comparison_*.json           # run_comparison_eval.py output
└── codex_cli_eval_*.json       # run_codex_cli_eval.py output
```

### Key Metrics

#### QA Metrics (LoCoMo Dataset)
| Metric | Description | Good Score |
|--------|-------------|------------|
| F1 | Token overlap | > 0.6 |
| Exact Match | Exact string match | > 0.3 |
| ROUGE-L | Longest common subsequence | > 0.5 |
| BERT-F1 | Semantic similarity | > 0.7 |

#### Compression Metrics
| Metric | Description | Good Score |
|--------|-------------|------------|
| Goal Coherence | Goal retention after compression | > 0.8 |
| Constraint Recall | Constraint preservation | > 0.7 |
| Behavioral Alignment | Decision consistency | > 0.75 |

#### Hierarchical Metrics
| Metric | Description | Good Score |
|--------|-------------|------------|
| Domain Recall | High-level summary retention | > 0.8 |
| Category Recall | Mid-level pattern retention | > 0.7 |
| Episode Recall | Specific detail retention | > 0.6 |
| Hierarchy Drift | Domain - Episode (lower is better) | < 0.15 |

#### Retention Metrics (App-Server Eval)
| Metric | Description | Good Score |
|--------|-------------|------------|
| Goal Retention | Goal keywords preserved after compaction | > 0.7 |
| Constraint Retention | Project constraints preserved | > 0.6 |
| Decision Retention | Architectural decisions preserved | > 0.5 |
| Overall Retention | Average of all retention metrics | > 0.6 |

---

## Unit Tests

### Running Tests
```bash
# All tests (unit + integration)
make test

# Unit tests only — no API keys required
make test-quick

# Integration tests only — requires OPENAI_API_KEY or ANTHROPIC_API_KEY
make test-integration

# Specific test file
pytest tests/test_hierarchical_eval.py -v
```

### Test Files
| Test File | Type | Description |
|-----------|------|-------------|
| `test_metrics.py` | Unit | Metric calculator imports |
| `test_strategies.py` | Unit | Strategy interfaces (A, B) |
| `test_granular_constraint_metrics.py` | Unit | Constraint categorization |
| `test_artificial_8k_compaction_trigger.py` | Unit | Token budget triggers |
| `test_strategy_h.py` | Unit | Strategy H (mocked) |
| `test_hierarchical_eval.py` | Unit | Hierarchical template validation |
| `test_granular_collection.py` | Unit | Granular metrics collection |
| `test_selective_salience_package.py` | Integration | Package API |
| `test_strategy_h_integration.py` | Integration | Strategy H full flow |
| `test_budget_judge.py` | Integration | LLM-as-judge accuracy |
| `test_strategy_f_product_pivot.py` | Integration | Strategy F on multi-shift template |
| `test_strategy_i_product_design.py` | Integration | Strategy I on healthcare template |
| `test_with_actual_compression.py` | Integration | Compression trigger + constraint retention |

Integration tests are marked with `@pytest.mark.integration` and skipped automatically by `make test-quick`.

---

## Environment Setup

### Requirements
```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variable for API calls
export OPENAI_API_KEY="your-key-here"
```

### Optional: Codex Binary
For `run_codex_cli_eval.py` and `run_comparison_eval.py`:
```bash
# Build from source
cd codex/codex-rs && cargo build --release

# Or install globally
npm install -g @openai/codex
```

---

## Troubleshooting

### Common Issues

**"No module named 'openai'"**
```bash
pip install openai
```

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**"Codex CLI not found"**
```bash
# Check if installed
which codex

# Or specify path
python scripts/run_codex_cli_eval.py --codex-path /path/to/codex
```

**Tests timing out**
```bash
# Reduce sample count
python scripts/run_eval.py --samples 3
```
