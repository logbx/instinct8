"""Instinct8 MCP server — context compression quality assurance for LLM agents.

Entry point for the FastMCP server. Wires up tools, prompts, and resources.

Usage:
    instinct8-mcp                     # stdio transport (default)
    python -m instinct8_mcp.server    # same thing
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from .session_manager import SessionManager
from .tools import register_tools
from .prompts import register_prompts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server + shared state
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "instinct8",
    instructions=(
        "Instinct8 provides context compression with goal drift prevention for LLM agents. "
        "Start by calling initialize_session with your goal and constraints, "
        "then use compress_context when you need to compress your conversation history. "
        "Use measure_drift to verify goal preservation after compression."
    ),
)

session_manager = SessionManager()

# ---------------------------------------------------------------------------
# Register tools and prompts
# ---------------------------------------------------------------------------

register_tools(mcp, session_manager)
register_prompts(mcp)

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("session://current")
def get_session_state() -> str:
    """Current session state including salience set, compression count, and tokens saved."""
    if not session_manager.has_session:
        return json.dumps({"status": "no_active_session", "message": "Call initialize_session first."})
    return json.dumps(session_manager.session.to_dict(), indent=2)


@mcp.resource("strategies://list")
def get_strategies() -> str:
    """Available compression strategies and their characteristics."""
    return json.dumps({
        "strategies": [
            {
                "id": "selective_salience",
                "name": "Strategy H — Selective Salience Compression",
                "description": (
                    "Model-judged salience extraction with semantic deduplication. "
                    "Extracts goal-critical quotes verbatim, compresses everything else."
                ),
                "characteristics": {
                    "salience_extraction": "LLM-judged (via MCP sampling)",
                    "deduplication": "String similarity (lightweight, no torch)",
                    "goal_protection": "Verbatim quote preservation",
                    "compression_target": "Background context only",
                },
                "best_for": "General-purpose compression with high goal preservation",
                "sampling_calls_per_compression": 2,
            },
            {
                "id": "protected_core",
                "name": "Strategy F — Protected Core + Goal Re-assertion",
                "description": (
                    "Explicit goal protection via a first-class Protected Core object. "
                    "Goal/constraints are NEVER compressed, only re-asserted."
                ),
                "characteristics": {
                    "goal_protection": "Structural (Protected Core dataclass)",
                    "compression_target": "Conversation halo only",
                },
                "best_for": "Tasks requiring strict constraint adherence",
                "sampling_calls_per_compression": 1,
            },
        ],
        "note": (
            "The MCP server currently uses a combined approach: "
            "Protected Core (from Strategy F) for structural goal protection, "
            "plus Selective Salience (from Strategy H) for intelligent compression. "
            "All LLM calls use MCP sampling — no API keys needed."
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the instinct8 MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
