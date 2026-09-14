# ❓ Frequently Asked Questions

Quick answers to common questions about instinct8 MCP server.

---

## General Questions

### What is instinct8?

instinct8 is a context compression tool for Claude Code that prevents "goal drift" - the problem where AI assistants gradually forget or misremember their original objectives during long conversations. It uses intelligent compression that preserves your goals while reducing context size.

### What is goal drift?

Goal drift occurs when an AI assistant loses track of the original task during a long conversation. For example, you start debugging a login issue but end up refactoring unrelated code. Research shows standard compression causes ~7% drift per compression. instinct8 reduces this dramatically.

### Why do I need this?

If you use Claude Code for:
- Long debugging sessions (2+ hours)
- Multi-day projects
- Complex features with many requirements
- Code reviews with architectural decisions

You'll benefit from instinct8's goal preservation during context compression.

### How is this different from regular context compression?

**Regular compression**: Summarizes everything equally, often losing critical details
**instinct8**: Identifies and preserves goal-critical information verbatim while compressing only background context

---

## Installation Questions

### How do I install instinct8?

```bash
pip install instinct8-mcp
```

If the package isn't on PyPI yet, install from source:
```bash
git clone https://github.com/logbx/instinct8.git
cd instinct8/mcp_server
pip install -e .
```

### What are the requirements?

- Python 3.10 or higher
- Claude Code desktop app
- pip (Python package manager)
- No API keys needed!

### Do I need an OpenAI API key?

No! instinct8 uses MCP sampling, which means it uses Claude's model directly. All processing happens through Claude Code - no external API calls.

### Can I use this with other MCP servers?

Yes! instinct8 works alongside other MCP servers like GitHub, Perplexity, etc. Just add it to your existing config.

---

## Configuration Questions

### Where is the Claude Code config file?

- **macOS/Linux**: `~/.claude/claude_code_config.json`
- **Windows**: `%USERPROFILE%\.claude\claude_code_config.json`

### What if the config file doesn't exist?

Create it! Make the directory and file:
```bash
# macOS/Linux
mkdir -p ~/.claude
echo '{"mcpServers": {"instinct8": {"command": "instinct8-mcp"}}}' > ~/.claude/claude_code_config.json
```

### How do I add instinct8 to existing MCP servers?

```json
{
  "mcpServers": {
    "github": {
      "command": "github-mcp"
    },
    "instinct8": {
      "command": "instinct8-mcp"
    }
  }
}
```

### Do I need to restart Claude Code?

Yes! After changing the config, you must fully quit Claude Code (Cmd+Q on Mac, Alt+F4 on Windows) and reopen it.

---

## Usage Questions

### When should I use compression?

Compress when:
- Context feels large (every 1-2 hours)
- Before starting a new component
- After completing a major section
- When Claude seems to forget earlier context

### What tools does instinct8 provide?

1. **`initialize_session`** - Set your goal and constraints to protect
2. **`compress_context`** - Compress while preserving critical information
3. **`measure_drift`** - Check if compression maintained your goals

### How do I know it's working?

After compression, you'll see:
- Token count reduced
- Compression statistics
- When you ask about original goals, Claude remembers them perfectly

### Can I compress multiple times?

Yes! instinct8 maintains a "salience set" that grows with each compression, preserving all critical information cumulatively.

### What should I put in constraints?

Include:
- What NOT to do ("Don't modify database schema")
- Hard requirements ("Must be backwards compatible")
- Performance targets ("Under 2 second load time")
- Security requirements ("No hardcoded credentials")

---

## Technical Questions

### How does MCP sampling work?

MCP sampling allows the server to construct prompts that Claude Code's model executes. This means:
- No API keys needed
- All processing uses Claude's model
- Data stays local on your machine
- Works with whatever model Claude Code uses

### What compression strategy does it use?

instinct8 uses "Selective Salience with Protected Core":
1. Extract goal-critical information (using Claude)
2. Preserve it verbatim (never summarized)
3. Compress only background context
4. Deduplicate semantically similar content

### How much compression can I expect?

Typically 3:1 to 5:1 compression ratio while maintaining 90%+ goal coherence. Exact ratio depends on conversation content.

### Are my conversations sent anywhere?

No! All processing happens locally through Claude Code. No data is sent to external servers.

### Do sessions persist across restarts?

No, sessions are in-memory only. When Claude Code restarts, you'll need to re-initialize. This is a known limitation - session persistence is planned for future versions.

---

## Troubleshooting Questions

### Why don't I see instinct8 tools?

Most common causes:
1. Didn't restart Claude Code after config change
2. Config file in wrong location
3. JSON syntax error in config
4. Package not installed correctly

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

### "Command not found: instinct8-mcp"

The command isn't in your PATH. Either:
- Add Python scripts to PATH: `export PATH="$PATH:$(python3 -m site --user-base)/bin"`
- Use full path in config: `"command": "/full/path/to/instinct8-mcp"`

### "Session not initialized" error

You need to use `initialize_session` first before using other tools. Set your goal and constraints, then you can compress.

### Can I use this in VS Code?

Currently, instinct8 is designed for Claude Code's MCP implementation. VS Code support would require a different integration method.

---

## Advanced Questions

### Can I customize compression strategies?

Currently uses the default Selective Salience strategy. Custom strategies are planned for future versions.

### How do I measure compression effectiveness?

Use the `measure_drift` tool to get:
- Goal coherence score (0.0-1.0)
- Constraint recall score
- Behavior alignment score

### Can I export/import sessions?

Not yet, but this is a planned feature. Currently, sessions are in-memory only.

### Is there a token limit?

Compression triggers are based on your Claude Code context limits. instinct8 works within whatever limits your Claude Code instance has.

### Can I see what was preserved vs compressed?

Check the session resource (`session://current`) to see the current salience set - these are the preserved critical elements.

---

## Project Questions

### Is this open source?

Yes! instinct8 is open source under Apache-2.0 license (main project) and BSL-1.1 for the MCP server (converts to Apache-2.0 after 3 years).

### How can I contribute?

- Report bugs: https://github.com/logbx/instinct8/issues
- Submit PRs for fixes or features
- Share usage examples
- Help with documentation

### What's the roadmap?

Planned features:
- Session persistence across restarts
- Custom compression strategies
- Multiple session management
- Export/import capabilities
- Performance analytics dashboard

### Who maintains this?

instinct8 is maintained by LoganLiangMay and the open source community.

---

## Comparison Questions

### How does this compare to other compression tools?

| Feature | instinct8 | Regular Compression |
|---------|-----------|-------------------|
| Preserves goals | ✅ Yes, verbatim | ❌ Often lost |
| Cumulative preservation | ✅ Grows over time | ❌ Resets each time |
| Semantic deduplication | ✅ Yes | ❌ No |
| MCP sampling | ✅ No API keys | ❌ Requires API |
| Goal drift measurement | ✅ Built-in | ❌ No |

### Is this better than manual copy/paste?

Yes! Manual copy/paste:
- Loses context structure
- No intelligent selection
- No drift measurement
- Time-consuming
- Error-prone

### Why not just start a new conversation?

Starting fresh means losing:
- All context and decisions
- Code understanding
- Problem analysis
- Discussion history

instinct8 preserves what matters while reducing size.

---

**Don't see your question?** Open an issue: https://github.com/logbx/instinct8/issues