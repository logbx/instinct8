# Instinct8 Agent - Installation Guide

Complete guide to install and use Instinct8 Agent.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Git (for cloning the repository)

## Quick Install (30 seconds)

```bash
pip install instinct8-agent
export OPENAI_API_KEY="your-api-key"
instinct8 "Hello!"
```

## Installation Methods

### Method 1: pip (Recommended)

```bash
pip install instinct8-agent
```

On macOS/Linux, you might need:
```bash
pip3 install instinct8-agent
```

Verify installation:
```bash
instinct8 --help
instinct8-agent --help
```

### Method 2: pipx (Recommended for CLI Tools)

```bash
# Install pipx if you don't have it
brew install pipx  # macOS
# or: pip install pipx  # Linux/Windows

# Install Instinct8 Agent
pipx install instinct8-agent

# Verify
instinct8 --help
```

### Method 3: From Source (Development)

```bash
git clone https://github.com/logbx/instinct8.git
cd instinct8
git submodule update --init

# Choose your installation level:

# Minimal install (core CLI only - ~50MB)
pip install -e .

# Development install (includes testing tools)
pip install -e ".[dev]"

# Full install (all dependencies for evaluation and research - ~2GB)
pip install -e ".[all]"
```

**Installation Options:**

| Command | Use Case | Dependencies |
|---------|----------|--------------|
| `pip install -e .` | **End users** - Core CLI functionality only | Minimal (openai, numpy, sentence-transformers, scikit-learn, tiktoken) |
| `pip install -e ".[dev]"` | **Contributors** - Adds testing tools | Core + pytest, mypy, python-dotenv |
| `pip install -e ".[eval]"` | **Researchers** - LLM providers for evaluation | Core + anthropic, litellm |
| `pip install -e ".[metrics]"` | **Evaluators** - Metrics for benchmarking | Core + BERT-score, ROUGE, nltk |
| `pip install -e ".[data]"` | **Data scientists** - Data processing utilities | Core + pandas, scipy, tqdm |
| `pip install -e ".[heavy]"` | **ML engineers** - Heavy ML frameworks | Core + torch, transformers (~1.5GB) |
| `pip install -e ".[all]"` | **Full development** - Everything | All of the above (~2GB) |

**Recommended Installations:**
- **Quick start / testing**: `pip install -e .` (minimal, ~50MB, installs in 30 seconds)
- **Development work**: `pip install -e ".[dev]"` (adds testing tools)
- **Evaluation / research**: `pip install -e ".[all]"` (complete environment)

This installs Instinct8 Agent and makes the `instinct8` and `instinct8-agent` commands available.

## Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Get your API key from: https://platform.openai.com/api-keys

**To make this permanent**, add it to your shell configuration file:

```bash
# For bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Verify Installation

```bash
# Check that commands are available
instinct8 --help
instinct8-agent --help

# Test basic functionality
instinct8 "Hello, what can you do?"
```

If you see help text and a response, installation is successful!

## Quick Start

### Use as Codex Replacement

```bash
# Alias Instinct8 to replace Codex
alias codex=instinct8

# Now use Codex commands - they'll use Instinct8!
codex exec "create a FastAPI endpoint"
```

### Use Instinct8 Directly

```bash
# Basic usage
instinct8 "create a FastAPI endpoint"

# With exec command (Codex-compatible)
instinct8 exec "explain this codebase"
```

### File Operations (Default)

```bash
instinct8
```

File operations work immediately:
- `review my files` - Read files
- `create hello.py` - Create files
- `edit main.py` - Edit files

### Command Execution (Optional)

```bash
instinct8 --allow-execution
```

Enables shell command execution:
- `run npm install` - Run shell commands
- `execute python test.py` - Execute scripts

### Interactive Agent Mode

```bash
instinct8-agent interactive \
  --goal "Build a FastAPI authentication system" \
  --constraints "Use JWT" "Hash passwords"
```

## Configuration (Optional)

Create a config file to set default goal and constraints:

```bash
mkdir -p ~/.instinct8
```

Create `~/.instinct8/config.json`:

```json
{
  "goal": "Build a FastAPI authentication system",
  "constraints": [
    "Use JWT tokens",
    "Hash passwords with bcrypt",
    "Must be production-ready"
  ],
  "model": "gpt-4o"
}
```

## Usage Examples

### Basic Task Execution

```bash
# Simple prompt
instinct8 "create a login endpoint"

# With exec (Codex-compatible)
instinct8 exec "refactor this code to use async/await"

# JSON output
instinct8 exec --json "explain this codebase"
```

### With Goal and Constraints

```bash
instinct8 exec \
  --goal "Build authentication system" \
  --constraints "Use JWT" "Hash passwords" "Support refresh tokens" \
  "create a login endpoint"
```

### Interactive Mode

```bash
instinct8-agent interactive \
  --goal "Research async Python frameworks" \
  --constraints "Budget $10K" "Timeline 2 weeks"
```

Commands in interactive mode:
- `ask <question>` - Ask the agent a question
- `say <message>` - Add a user message
- `compress` - Manually trigger compression
- `salience` - Show preserved salience set
- `stats` - Show agent statistics
- `quit` - Exit

### Test Mode

```bash
instinct8-agent test \
  --goal "Build FastAPI app" \
  --constraints "Use JWT" \
  --questions "What is the goal?" "What constraints exist?"
```

## Python API Usage

```python
from selective_salience import Instinct8Agent

# Create agent
agent = Instinct8Agent()
agent.initialize(
    goal="Build a FastAPI auth system",
    constraints=["Use JWT", "Hash passwords"]
)

# Use it
agent.ingest_turn({"role": "user", "content": "Create login endpoint"})
response = agent.answer_question("What are we building?")
print(response)
```

## Virtual Environment Setup

On macOS with Homebrew Python, you may need a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install Instinct8
pip install instinct8-agent
```

## Troubleshooting

### "Command not found: instinct8"

```bash
# Make sure pip install completed successfully
pip install instinct8-agent

# Or try pip3
pip3 install instinct8-agent

# Or use pipx
pipx install instinct8-agent

# Verify installation
which instinct8
```

### "OPENAI_API_KEY environment variable not set"

```bash
export OPENAI_API_KEY="your-api-key-here"

# Verify it's set
echo $OPENAI_API_KEY
```

### "ModuleNotFoundError: No module named 'selective_salience'"

```bash
# If installed from source, make sure you're in the project root
cd instinct8

# Install with pip (dependencies are managed by pyproject.toml)
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

**Note**: `requirements.txt` now points to `pyproject.toml`, so `pip install -e .` is the recommended approach.

### "Permission denied"

```bash
# Use virtual environment
python3 -m venv venv && source venv/bin/activate
pip install instinct8-agent

# Or use pipx
pipx install instinct8-agent
```

### Import Errors (v0.1.0)

```bash
# Upgrade to latest version
pip install --upgrade instinct8-agent
```

## Updating

```bash
# Upgrade to latest version
pip install --upgrade instinct8-agent

# Or with pipx
pipx upgrade instinct8-agent

# Verify version
pip show instinct8-agent
```

## Uninstallation

```bash
# Remove the package
pip uninstall instinct8-agent

# Remove config (optional)
rm -rf ~/.instinct8
```

## What Makes Instinct8 Different?

**Codex's compression** uses simple summarization - it may lose goal-critical information.

**Instinct8's compression** preserves goal-critical information **verbatim** by:
1. Using GPT-4o to identify what's important
2. Keeping important info verbatim (not summarized)
3. Only compressing background context

**Result**: Better goal coherence and constraint retention in long conversations!

## Next Steps

- See [features/CODEX_REPLACEMENT.md](features/CODEX_REPLACEMENT.md) for migration details
- Check [examples/](../examples/) for more usage patterns
- Read the [Full Documentation](../selective_salience/README.md)

## Getting Help

- Check the [README](../README.md) for project overview
- See [Troubleshooting](#troubleshooting) section above
- Open an issue on GitHub: https://github.com/logbx/instinct8/issues
