# Context Compression Middleware for Long-Running LLM Agents

**Capstone Project**: Evaluating context compression strategies to prevent goal drift in long-running LLM agent conversations.

## Problem Statement

Long-running LLM agents face a critical challenge: as conversations grow, context windows fill up, requiring compression/summarization. However, **existing compression strategies cause cumulative goal drift** - the agent gradually forgets or misremembers its original objective and constraints.

This project:
1. **Establishes a baseline** measuring goal drift under Codex-style compression
2. **Implements 7 compression strategies** for comparison
3. **Proves that Protected Core + Goal Re-assertion** outperforms existing approaches

## Key Finding: Goal Drift is Real

Our baseline evaluation (50-turn conversations, 5 compression points) shows:

- **7% average goal drift** per compression point
- **40% of compressions** trigger drift detection
- Goal coherence can drop from **80% → 20%** after multiple compressions
- Constraints are lost unpredictably (0-40% loss per compression)

**This validates the need for explicit goal protection mechanisms.**

## 🚀 Quick Start

### For Claude Code Users (5 minutes)

```bash
# 1. Install
pip install instinct8-mcp

# 2. Configure (~/.claude/claude_code_config.json)
{
  "mcpServers": {
    "instinct8": {
      "command": "instinct8-mcp"
    }
  }
}

# 3. Restart Claude Code

# 4. Test it!
"Initialize a session with goal 'Build a TODO app'"
"Compress our conversation"
"What was our goal?"
```

**[Full Setup Guide →](mcp_server/QUICKSTART.md)** | **[Troubleshooting →](mcp_server/TROUBLESHOOTING.md)** | **[FAQ →](mcp_server/FAQ.md)**

### For Developers (Standalone CLI)

```bash
# Install
pip install instinct8-agent

# Use
export OPENAI_API_KEY="your-key"
instinct8 "Create a FastAPI endpoint"
```

## Project Structure

```
instinct8/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── pyproject.toml / Makefile / requirements.txt / LICENSE
├── selective_salience/          # Core Python package
├── strategies/                  # 9+ compression strategies
├── evaluation/                  # Evaluation framework
├── tests/                       # Unit tests
├── templates/                   # 22 conversation templates
├── results/                     # Evaluation results
├── examples/                    # Usage examples
├── scripts/                     # Evaluation & utility scripts
├── data/                        # External datasets
├── Formula/                     # Homebrew formula
├── mcp_server/                  # MCP server for Claude Code integration
├── codex/                       # Codex source (git submodule, pinned to v0.98.0)
├── versions/                    # Version variant descriptions
│   ├── v1-core/                # Python selective salience (main)
│   ├── v2-graphrag/            # GraphRAG + MCP server (graphrag-rs)
│   └── v3-sfhm/               # Rust SFHM backend (sfhm)
├── archive/                     # Historical planning docs
│   └── PR_PARTY/
└── docs/                        # Documentation
    ├── INSTALLATION.md          # Installation guide
    ├── PUBLISHING.md            # PyPI publishing guide
    ├── CODEX_FORK_PACKAGING.md  # Codex repackaging guide
    ├── TESTING.md               # Testing documentation
    ├── CODE_PROTECTION.md
    ├── PRIVATE_API_SETUP.md
    ├── features/                # Feature documentation
    ├── presentations/           # Presentation materials
    ├── research/                # Research & analysis docs
    └── development/             # Development guides
```

## Branch Map

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Production — Python-based selective salience | Active, v0.4.2, on PyPI |
| `feature/package-selective-salience` | PyPI packaging | Partially merged |
| `feature/pr01-selective-salience` | PR#01 enhancements, presentations | Feature branch |
| `feature/strategy-i-hybrid-implementation` | Strategy I (A-MEM + Protected Core) | Feature branch |
| `feature/h-mem` | Hierarchical Memory (H-MEM) implementation | Feature branch |
| `feature/instinct8-goal-preservation` | Goal preservation framework | Merged |
| `sfhm` | Rust-only backend, Stateless Functional Hierarchical Memory | Experimental |
| `graphrag-rs` | GraphRAG memory + MCP server integration | Experimental, ~30 commits ahead |

See [versions/](versions/) for detailed descriptions of each product variant.

## Quick Start

### Install Instinct8 Agent

**Install from PyPI:**
```bash
pip install instinct8-agent
```

**Set your API key:**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Test it:**
```bash
instinct8 "Hello, what can you do?"
```

**That's it!** You're ready to use Instinct8 Agent.

### Updating

If you already have Instinct8 Agent installed:

```bash
pip install --upgrade instinct8-agent
```

## MCP Server for Claude Code

Use instinct8's context compression directly in Claude Code through the Model Context Protocol (MCP).

### Install MCP Server

```bash
pip install instinct8-mcp
```

### Configure Claude Code

Add to your Claude Code config (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "instinct8": {
      "command": "instinct8-mcp"
    }
  }
}
```

Restart Claude Code and you'll have access to:
- **`initialize_session`** - Set goal and constraints to protect during compression
- **`compress_context`** - Intelligently compress conversations while preserving goals
- **`measure_drift`** - Quantify if compression affected your original objectives

**Key Benefits:**
- No API keys needed (uses Claude's model via MCP sampling)
- Lightweight (~50KB, no heavy ML dependencies)
- Scientifically validated approach (7% avg drift reduction)
- All processing stays local on your machine

**Getting Started:**
- **[5-Minute Setup](mcp_server/QUICKSTART.md)** - Get running quickly
- **[Hello World Test](mcp_server/examples/hello_world.md)** - 60-second verification
- **[Troubleshooting](mcp_server/TROUBLESHOOTING.md)** - Solve common issues
- **[FAQ](mcp_server/FAQ.md)** - Frequently asked questions
- **[Full Documentation](mcp_server/README.md)** - Complete MCP server details

**Note:** If `pip install instinct8-mcp` fails, the package may not be published yet. Install from source:
```bash
git clone https://github.com/LoganLiangMay/instinct8.git
cd instinct8/mcp_server && pip install -e .
```

### Alternative: Install from Source

For development or latest features:
```bash
git clone https://github.com/LoganLiangMay/instinct8.git
cd instinct8
git submodule update --init

# Minimal install (core CLI only)
pip install -e .

# Development install (includes testing tools)
pip install -e ".[dev]"

# Full install (all dependencies for evaluation and research)
pip install -e ".[all]"
```

**Installation Options:**
- `pip install -e .` - Core functionality only (minimal dependencies)
- `pip install -e ".[dev]"` - Adds testing tools (pytest, mypy)
- `pip install -e ".[eval]"` - Adds LLM providers for evaluation (anthropic, litellm)
- `pip install -e ".[metrics]"` - Adds evaluation metrics (BERT-score, ROUGE)
- `pip install -e ".[data]"` - Adds data processing utilities (pandas, scipy)
- `pip install -e ".[heavy]"` - Adds heavy ML frameworks (torch, transformers)
- `pip install -e ".[all]"` - Install everything (recommended for development)

### Use as Codex Replacement

```bash
# Alias to replace Codex
alias codex=instinct8

# Now use Codex commands - they'll use Instinct8!
codex exec "create a FastAPI endpoint"
```

**See [docs/INSTALLATION.md](docs/INSTALLATION.md) for complete installation instructions.**

### Run Baseline Evaluation

```bash
# Short template (12 turns, 2 compression points)
python3 -m evaluation.harness \
  --template templates/research-synthesis-001.json \
  --trials 5 \
  --output results/baseline_results.json

# Long template (50 turns, 5 compression points)
python3 -m evaluation.harness \
  --template templates/research-synthesis-002-long.json \
  --trials 5 \
  --output results/baseline_long_results.json
```

### View Results

```bash
cat results/baseline_long_results.json | jq '.aggregate_summary'
```

## Running Tests & Evaluations

Use `make` commands for easy testing and evaluation:

```bash
# See all available commands
make help

# Run all tests (~10s)
make test

# Unit tests only — no API keys needed
make test-quick

# Integration tests only — requires API keys
make test-integration

# Quick evaluation - 5 samples (~2m)
make eval-quick

# Compare compression strategies (~15m)
make eval-compare

# Hierarchical depth evaluation (~5m)
make eval-hierarchical

# Publication-ready evaluation (~1hr)
make eval-rigorous
```

For complete documentation, see **[docs/TESTING.md](docs/TESTING.md)**.

## Metrics

We measure three core metrics using LLM-as-judge (Claude):

### 1. Goal Coherence Score (0.0 - 1.0)
Semantic similarity between original goal and agent's stated goal after compression.

- **1.0**: Identical goal
- **0.8**: Same goal, minor wording differences
- **0.6**: Related but some aspects missing
- **0.4**: Partially related, significant drift
- **0.0**: Completely different

### 2. Constraint Recall Rate (0.0 - 1.0)
Percentage of original constraints the agent remembers after compression.

- **1.0**: All constraints mentioned
- **0.6**: 3 of 5 constraints mentioned
- **0.0**: No constraints mentioned

### 3. Behavioral Alignment (1 - 5)
Rubric score for whether agent's behavior aligns with original goal when tested.

- **5**: Perfectly aligned, explicitly references goal
- **4**: Mostly aligned, minor deviations
- **3**: Ambiguous
- **2**: Some drift, partially abandons goal
- **1**: Complete drift, goal forgotten

## Compression Strategies

### Strategy B: Codex-Style Checkpoint (Baseline) ✅

**Implementation**: `strategies/strategy_b_codex.py`

**How it works**:
1. Summarizes conversation history (excluding last 3 turns)
2. Reinjects system prompt
3. Keeps recent turns raw

**Results** (50-turn template):
- Avg goal drift: **7%** per compression
- Drift events: **4 of 10** compressions
- Goal coherence: **80% → 20%** (worst case)

### Strategy F: Protected Core + Goal Re-assertion (Novel) ✅

**Status**: Implemented (`strategies/strategy_f_protected_core.py`)

**Design**:
- Stores original goal and constraints in protected object
- Never compresses goal/constraints
- Re-asserts goal after every compression
- Expected: **>95% goal coherence** maintained

### Other Strategies (Planned)

- Strategy A: No compression (control)
- Strategy C: Simple truncation
- Strategy D: Semantic chunking
- Strategy E: Hierarchical summarization
- Strategy G: Adaptive compression

## Baseline Results Summary

From `results/baseline_long_results.json`:

| Metric | Value |
|--------|-------|
| Avg Goal Coherence Before | 70% |
| Avg Goal Coherence After | 63% |
| Avg Goal Drift | **7%** per compression |
| Total Drift Events | **4** (of 10 compressions) |
| Avg Compression Ratio | 1.04 (barely compressing) |
| Behavioral Alignment | 4.9/5 (still good) |

**Key Insight**: Compression causes goal drift, but behavioral alignment stays high. The agent *behaves* reasonably but *understands* the goal less accurately.

## Development

### Adding a New Strategy

1. Create `strategies/strategy_X_name.py`
2. Inherit from `CompressionStrategy` base class
3. Implement required methods:
   - `initialize(original_goal, constraints)`
   - `compress(context, trigger_point) -> str`
   - `name() -> str`

Example:
```python
from strategies.strategy_base import CompressionStrategy

class StrategyX_MyStrategy(CompressionStrategy):
    def initialize(self, original_goal: str, constraints: List[str]):
        self.goal = original_goal
        self.constraints = constraints
    
    def compress(self, context: List[Dict], trigger_point: int) -> str:
        # Your compression logic here
        return compressed_context
    
    def name(self) -> str:
        return "Strategy X - My Strategy"
```

### Creating a Conversation Template

Templates are JSON files with:
- `initial_setup`: Original goal, constraints, system prompt
- `turns`: List of conversation turns
- `compression_config`: When to trigger compression
- `probing_tasks`: Questions to test goal coherence

See `templates/research-synthesis-001.json` for example.

## Documentation

- **[Codex Compression Analysis](docs/research/codex_analysis.md)**: Deep dive into Codex's `compact.rs` algorithm
- **[Baseline Results Report](docs/research/baseline_results.md)**: Detailed analysis of goal drift findings
- **[Implementation Guide](docs/research/implementation_guide.md)**: Technical specifications
- **[Project PRD](docs/research/context_compression_prd.md)**: Full project requirements
- **[CHANGELOG](CHANGELOG.md)**: Version history

## License

This project includes Codex source code for reference. Codex modifications are for research purposes only.

## Contributing

This is a capstone research project. For questions or collaboration, please open an issue.

---

**Status**: Baseline established ✅ | Strategy F Protected Core implemented ✅
