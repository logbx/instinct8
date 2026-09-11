# Selective Salience Compression

**Preserve goal-critical information in long-running LLM agent conversations**

Selective Salience Compression is a compression strategy that uses model-judged salience extraction to preserve goal-critical information verbatim while compressing background context. Perfect for long-running LLM agents that need to maintain goal coherence across many conversation turns.

## Features

- 🎯 **Goal-Preserving**: Extracts and preserves goal-critical information verbatim
- 🤖 **Model-Judged**: Uses GPT-4o to identify what's important (no fixed schema)
- 🔄 **Cumulative**: Maintains a growing salience set across multiple compressions
- 🧠 **Semantic Deduplication**: Prevents duplicate information using sentence embeddings
- ⚡ **Efficient**: Compresses background context with GPT-4o-mini

## Installation

```bash
pip install selective-salience-compression
```

Or install from source:

```bash
git clone https://github.com/LoganLiangMay/instinct8.git
cd instinct8
pip install -e .
```

## Replace Codex

**Instinct8 can be used as a drop-in replacement for Codex!**

```bash
# Alias Codex to Instinct8
alias codex=instinct8

# Now use Codex commands - they'll use Instinct8!
codex exec "create a FastAPI endpoint"
```

See [Codex Replacement Guide](../docs/CODEX_REPLACEMENT.md) for details.

## Quick Start

### Python API

```python
from selective_salience import SelectiveSalienceCompressor

# Initialize compressor
compressor = SelectiveSalienceCompressor()

# Set your task's goal and constraints
compressor.initialize(
    original_goal="Research async Python frameworks and recommend one",
    constraints=["Budget max $10K", "Timeline 2 weeks", "Must support WebSockets"]
)

# Your conversation context (list of turns)
context = [
    {"id": 1, "role": "user", "content": "What async frameworks exist?"},
    {"id": 2, "role": "assistant", "content": "FastAPI, Django Async, Quart..."},
    {"id": 3, "role": "user", "content": "Which ones support WebSockets?"},
    {"id": 4, "role": "assistant", "content": "FastAPI and Quart both support WebSockets natively"},
]

# Compress when context gets too long
compressed = compressor.compress(context, trigger_point=3)

# Use compressed as your agent's context
print(compressed)
```

### Command Line

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Compress a conversation context
selective-salience compress \
  --context conversation.json \
  --trigger 10 \
  --goal "Research async frameworks" \
  --constraints "Budget $10K" "Timeline 2 weeks"
```

## How It Works

1. **Salience Extraction**: GPT-4o identifies goal-critical information from the conversation
2. **Semantic Deduplication**: New salient items are compared against existing ones using embeddings
3. **Background Compression**: Everything else is compressed into a lightweight summary
4. **Context Rebuild**: Final context = System Prompt + Salient Items + Background Summary + Recent Turns

## Configuration

```python
compressor = SelectiveSalienceCompressor(
    extraction_model="gpt-4o",        # Model for salience extraction
    compression_model="gpt-4o-mini",  # Model for background compression
    similarity_threshold=0.85,        # Cosine similarity for deduplication
)
```

## Instinct8 Agent

**Instinct8 Agent** - A coding agent with Selective Salience Compression built-in!

```python
from selective_salience import Instinct8Agent

# Create Instinct8 agent
agent = Instinct8Agent()
agent.initialize(
    goal="Build a FastAPI auth system",
    constraints=["Use JWT", "Hash passwords"]
)

# Use like a normal coding agent
agent.ingest_turn({"role": "user", "content": "Create login endpoint"})
response = agent.answer_question("What are we building?")

# Compression happens automatically when context exceeds threshold
# Goal-critical information is preserved verbatim!
```

### CLI Usage

Test Instinct8 Agent interactively:

```bash
# Interactive mode
instinct8-agent interactive \
  --goal "Build a FastAPI auth system" \
  --constraints "Use JWT" "Hash passwords"

# Test mode
instinct8-agent test \
  --goal "Research frameworks" \
  --constraints "Budget $10K" \
  --questions "What is the goal?" "What constraints exist?"
```

**Interactive Commands:**
- `ask <question>` - Ask the agent a question
- `say <message>` - Add a user message
- `compress` - Manually trigger compression
- `salience` - Show preserved salience set
- `stats` - Show agent statistics
- `reset` - Reset agent state
- `quit` - Exit

See `examples/instinct8_agent_example.py` for a complete example.

## Example: Long-Running Agent

```python
from selective_salience import SelectiveSalienceCompressor

compressor = SelectiveSalienceCompressor()
compressor.initialize(
    original_goal="Help user build a web app",
    constraints=["Use Python", "Budget $5K"]
)

# Simulate a long conversation
context = []
for turn_id in range(1, 51):
    # ... add turns to context ...
    
    # Compress every 10 turns
    if turn_id % 10 == 0:
        compressed = compressor.compress(context, trigger_point=turn_id)
        # Reset context to compressed version
        context = [{"id": 0, "role": "system", "content": compressed}]
```

## Why Instinct8 Agent?

**Traditional compression** (like Codex's default) uses simple summarization - it may lose goal-critical information.

**Instinct8 Agent with Selective Salience Compression** preserves goal-critical information verbatim by:
1. Extracting salient information using GPT-4o
2. Keeping it verbatim (not summarized)
3. Only compressing background context

**Result**: Better goal coherence and constraint retention in long conversations.

## API Reference

### `SelectiveSalienceCompressor`

#### `__init__(extraction_model="gpt-4o", compression_model="gpt-4o-mini", similarity_threshold=0.85)`

Initialize the compressor.

#### `initialize(original_goal: str, constraints: List[str])`

Set the task's goal and constraints. Call once at the start.

#### `compress(context: List[Dict], trigger_point: int) -> str`

Compress conversation context up to `trigger_point`.

**Context format**: Each dict should have:
- `id`: Turn ID (int)
- `role`: "user", "assistant", or "system" (str)
- `content`: Turn content (str)
- Optional: `tool_call`, `decision`, etc.

#### `update_goal(new_goal: str, rationale: str = "")`

Update the goal if it evolves during the task.

#### `salience_set: List[str]`

Property returning the current salience set (goal-critical information preserved verbatim).

#### `reset()`

Reset the compressor state (clears salience set).

### `Instinct8Agent`

Coding agent with Selective Salience Compression built-in.

#### `__init__(model="gpt-4o", compaction_threshold=80000, extraction_model="gpt-4o", compression_model="gpt-4o-mini")`

Initialize Instinct8 agent with Selective Salience Compression.

#### `initialize(goal: str, constraints: List[str])`

Set the task's goal and constraints.

#### `ingest_turn(turn: Dict)`

Add a conversation turn.

#### `answer_question(question: str) -> str`

Answer a question using current context.

#### `compress(trigger_point: Optional[int])`

Manually trigger compression.

#### `salience_set: List[str]`

Property returning preserved goal-critical information.

#### `context_length: int`

Property returning current context length in tokens.

#### `reset()`

Reset agent state.

## Requirements

- Python >= 3.9
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Research

This implementation is based on research evaluating context compression strategies for long-running LLM agents. For more details, see:

- [Selective Salience Compression Paper](Selective%20Salience%20Compression.md)

## License

Apache 2.0

## Contributing

This is part of the Instinct8 research project. For questions or collaboration, please open an issue.
