# 🚀 Quick Start: instinct8 MCP Server

Get instinct8 working with Claude Code in **5 minutes or less**.

## What You'll Get

✨ **Intelligent context compression** that preserves your goals
🎯 **Goal drift prevention** in long Claude Code conversations
💾 **Token savings** without losing important information
🔒 **No API keys needed** - uses Claude's own model

---

## Step 1: Install (30 seconds)

```bash
pip install instinct8-mcp
```

**Verify installation:**
```bash
# Check it installed
which instinct8-mcp

# Should output something like:
# /usr/local/bin/instinct8-mcp
```

⚠️ **If "command not found"**, add Python scripts to PATH:
```bash
# macOS/Linux
export PATH="$PATH:$(python3 -m site --user-base)/bin"

# Windows
set PATH=%PATH%;%APPDATA%\Python\Scripts
```

---

## Step 2: Configure Claude Code (1 minute)

Find your config file:
- **macOS/Linux**: `~/.claude/claude_code_config.json`
- **Windows**: `%USERPROFILE%\.claude\claude_code_config.json`

Add instinct8 to your config:

```json
{
  "mcpServers": {
    "instinct8": {
      "command": "instinct8-mcp"
    }
  }
}
```

**If you already have other MCP servers:**
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

---

## Step 3: Restart Claude Code (10 seconds)

1. **Fully quit Claude Code** (Cmd+Q on Mac, Alt+F4 on Windows)
2. **Reopen Claude Code**

---

## Step 4: Verify It Works (30 seconds)

In Claude Code, type:
```
What MCP tools do you have from instinct8?
```

You should see:
- `initialize_session` - Set goals to protect
- `compress_context` - Compress while preserving goals
- `measure_drift` - Check if goals are maintained

✅ **Success!** instinct8 is ready to use.

---

## Step 5: Your First Test (60 seconds)

Try this hello world example:

```
1. Initialize a session with goal "Build a TODO app" and
   constraints "Use React" and "Include local storage"

2. Let's discuss some features for the TODO app...
   [Have a short conversation about features]

3. Now compress our context to save tokens while keeping the goal

4. Check the drift to see if we're still focused on the TODO app

5. Ask: What was our original goal?
```

Claude should:
1. Set up the protected core ✓
2. Compress the conversation ✓
3. Show drift measurement (~0.9 coherence) ✓
4. Remember "Build a TODO app" perfectly ✓

---

## 💡 When to Use instinct8

Use these tools during:
- **Long debugging sessions** - Compress investigation history
- **Multi-day projects** - Maintain continuity across sessions
- **Complex features** - Keep requirements intact
- **Code reviews** - Preserve architectural decisions

**Best Practice:** Compress every 1-2 hours or when context feels large

---

## 🔧 Troubleshooting

### "instinct8 tools not appearing"

1. Check installation: `pip show instinct8-mcp`
2. Verify config JSON is valid: Use jsonlint.com
3. Ensure Claude Code was fully restarted
4. Check the exact config file path for your OS

### "Command not found: instinct8-mcp"

```bash
# Find where it installed
pip show instinct8-mcp | grep Location

# Add that location + /bin to your PATH
```

### "Config file doesn't exist"

```bash
# Create the directory and file
mkdir -p ~/.claude
echo '{"mcpServers": {"instinct8": {"command": "instinct8-mcp"}}}' > ~/.claude/claude_code_config.json
```

---

## 📚 Next Steps

- **[Examples](examples/)** - See real-world usage patterns
- **[Troubleshooting](TROUBLESHOOTING.md)** - Detailed problem solving
- **[FAQ](FAQ.md)** - Common questions answered
- **[Full Guide](README.md)** - Complete MCP server details

---

## 🆘 Need Help?

- **Issues**: https://github.com/LoganLiangMay/instinct8/issues
- **Discussions**: Coming soon!
- **Quick Test**: Run the hello world example above

---

**Remember:** instinct8 prevents goal drift. Use it whenever your Claude Code conversations get long!