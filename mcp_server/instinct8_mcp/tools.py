"""MCP tool definitions for instinct8.

Three tools:
  1. initialize_session — set up protected core (pure state, no sampling)
  2. compress_context   — compress conversation preserving goals (2 sampling calls)
  3. measure_drift      — quantify goal coherence after compression (1–3 sampling calls)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import Context

from .session_manager import SessionManager
from . import strategy_adapter as sa

logger = logging.getLogger(__name__)


def register_tools(mcp_server: Any, session_manager: SessionManager) -> None:
    """Register all instinct8 tools on the given FastMCP server."""

    @mcp_server.tool()
    async def initialize_session(
        goal: str,
        constraints: list[str],
    ) -> str:
        """Set up a new compression session with a protected core.

        Call this at the start of a task to establish the goal and constraints
        that must be preserved across all future context compressions.

        Args:
            goal: The task's primary goal statement.
            constraints: Hard constraints the agent must follow (e.g. "Budget max $10K").

        Returns:
            Confirmation with the initialized session state.
        """
        session = session_manager.initialize(goal, constraints)
        return json.dumps({
            "status": "initialized",
            "protected_core": {
                "goal": session.protected_core.goal,
                "constraints": session.protected_core.hard_constraints,
            },
            "message": (
                f"Session initialized. Goal and {len(constraints)} constraint(s) "
                "are now protected and will be preserved across compressions."
            ),
        }, indent=2)

    @mcp_server.tool()
    async def compress_context(
        conversation: list[dict[str, Any]],
        ctx: Context,
        keep_recent_turns: int = 3,
    ) -> str:
        """Compress a conversation while preserving goals and critical information.

        Uses Selective Salience extraction: the client's own LLM identifies
        goal-critical quotes, then compresses everything else into a summary.
        The Protected Core (goal + constraints) is NEVER compressed.

        Requires an active session (call initialize_session first).

        Args:
            conversation: List of conversation turns, each with "role" and "content"
                         (optionally "id" and "decision").
            keep_recent_turns: Number of recent turns to preserve raw (default 3).

        Returns:
            The compressed context string ready to use as a replacement.
        """
        session_manager.require_session()

        # Get the MCP server session for sampling
        mcp_session = ctx.session

        # Determine what to compress vs keep raw
        trigger_point = max(0, len(conversation) - keep_recent_turns)
        to_compress = conversation[:trigger_point]
        recent_turns = conversation[trigger_point:]

        if not to_compress:
            compressed = sa.build_compressed_context(
                session_manager, "", recent_turns
            )
            return json.dumps({
                "compressed_context": compressed,
                "stats": {
                    "original_turns": len(conversation),
                    "salience_items": len(session_manager.session.salience_set),
                    "sampling_calls": 0,
                },
            }, indent=2)

        # Step 1: Extract salient information (1 sampling call)
        salience = await sa.extract_salience(mcp_session, session_manager, to_compress)

        # Step 2: Compress background (1 sampling call)
        background = await sa.compress_background(mcp_session, session_manager, to_compress)

        # Step 3: Assemble final output
        compressed = sa.build_compressed_context(
            session_manager, background, recent_turns
        )

        # Estimate tokens saved (rough: 4 chars ≈ 1 token)
        original_chars = sum(len(str(t.get("content", ""))) for t in conversation)
        compressed_chars = len(compressed)
        tokens_saved = max(0, (original_chars - compressed_chars) // 4)
        session_manager.record_compression(tokens_saved)

        return json.dumps({
            "compressed_context": compressed,
            "stats": {
                "original_turns": len(conversation),
                "compressed_length": compressed_chars,
                "salience_items": len(salience),
                "estimated_tokens_saved": tokens_saved,
                "compression_count": session_manager.session.compression_count,
                "sampling_calls": 2,
            },
        }, indent=2)

    @mcp_server.tool()
    async def measure_drift(
        stated_goal: str,
        ctx: Context,
        stated_constraints: str = "",
        agent_response: str = "",
        test_context: str = "",
    ) -> str:
        """Quantify goal coherence and constraint recall after compression.

        This is instinct8's unique differentiator — nobody else measures
        compression quality. Uses LLM-as-judge evaluation via the client's model.

        Requires an active session (call initialize_session first).

        Args:
            stated_goal: What the agent currently says its goal is.
            stated_constraints: What the agent says its constraints are (optional).
            agent_response: An agent response to evaluate for alignment (optional).
            test_context: Context for the behavioral alignment test (optional).

        Returns:
            Drift metrics including goal coherence, constraint recall,
            and optionally behavioral alignment scores.
        """
        state = session_manager.require_session()
        mcp_session = ctx.session

        original_goal = state.protected_core.goal
        constraints = state.protected_core.hard_constraints

        # Goal coherence (always measured)
        goal_coherence = await sa.measure_goal_coherence(
            mcp_session, original_goal, stated_goal
        )

        result: dict[str, Any] = {
            "goal_coherence": goal_coherence,
            "goal_coherence_interpretation": _interpret_coherence(goal_coherence),
            "sampling_calls": 1,
        }

        # Constraint recall (if stated_constraints provided)
        if stated_constraints and constraints:
            constraint_recall = await sa.measure_constraint_recall(
                mcp_session, constraints, stated_constraints
            )
            result["constraint_recall"] = constraint_recall
            result["constraints_remembered"] = f"{int(constraint_recall * len(constraints))}/{len(constraints)}"
            result["sampling_calls"] += len(constraints)

        # Behavioral alignment (if agent_response provided)
        if agent_response:
            alignment = await sa.measure_behavioral_alignment(
                mcp_session, original_goal, constraints, agent_response, test_context
            )
            result["behavioral_alignment"] = alignment
            result["behavioral_interpretation"] = _interpret_alignment(alignment)
            result["sampling_calls"] += 1

        # Overall assessment
        result["drift_detected"] = goal_coherence < 0.7
        result["session_stats"] = {
            "compression_count": state.compression_count,
            "total_tokens_saved": state.total_tokens_saved,
            "salience_items": len(state.salience_set),
        }

        return json.dumps(result, indent=2)


def _interpret_coherence(score: float) -> str:
    if score >= 0.9:
        return "Excellent — goal is well preserved"
    if score >= 0.7:
        return "Good — minor drift but core goal intact"
    if score >= 0.5:
        return "Warning — significant drift detected"
    if score >= 0.3:
        return "Critical — major goal drift"
    return "Severe — goal appears lost"


def _interpret_alignment(score: int) -> str:
    return {
        5: "Perfectly aligned",
        4: "Mostly aligned, minor deviations",
        3: "Ambiguous alignment",
        2: "Some drift detected",
        1: "Complete drift — goal forgotten",
    }.get(score, "Unknown")
