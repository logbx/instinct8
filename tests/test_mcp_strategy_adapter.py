"""Unit tests for MCP strategy_adapter pure helpers — no API keys required.

Tests pure assembly functions that don't require live LLM calls:
- build_compressed_context: assembles protected core + salience + background + recent turns
- _deduplicate: removes near-duplicate strings
- _prioritize: sorts salience by type (constraints → decisions → facts)
- _format_turns: formats conversation turns to text
"""

import pytest

from mcp_server.instinct8_mcp.session_manager import (
    Decision,
    ProtectedCore,
    SessionState,
    SessionManager,
)
from mcp_server.instinct8_mcp.strategy_adapter import (
    build_compressed_context,
    _deduplicate,
    _prioritize,
    _format_turns,
)


class TestFormatTurns:
    """Test conversation turn formatting."""

    def test_format_empty_turns(self):
        result = _format_turns([])
        assert result == ""

    def test_format_single_turn(self):
        turns = [{"id": 1, "role": "user", "content": "Hello"}]
        result = _format_turns(turns)
        assert result == "Turn 1 (user): Hello"

    def test_format_multiple_turns(self):
        turns = [
            {"id": 1, "role": "user", "content": "Build an API"},
            {"id": 2, "role": "assistant", "content": "I'll help with that"},
            {"id": 3, "role": "user", "content": "Add auth"},
        ]
        result = _format_turns(turns)

        lines = result.split("\n")
        assert len(lines) == 3
        assert "Turn 1 (user): Build an API" in lines[0]
        assert "Turn 2 (assistant): I'll help with that" in lines[1]
        assert "Turn 3 (user): Add auth" in lines[2]

    def test_format_turn_missing_fields(self):
        turns = [{"content": "No id or role"}]
        result = _format_turns(turns)
        assert "Turn ? (unknown): No id or role" in result


class TestDeduplicate:
    """Test near-duplicate string removal."""

    def test_deduplicate_empty(self):
        assert _deduplicate([]) == []

    def test_deduplicate_single_item(self):
        assert _deduplicate(["Item A"]) == ["Item A"]

    def test_deduplicate_no_duplicates(self):
        items = ["Item A", "Item B", "Item C"]
        result = _deduplicate(items)
        assert len(result) == 3
        assert set(result) == set(items)

    def test_deduplicate_exact_duplicates(self):
        items = ["Same item", "Same item", "Different"]
        result = _deduplicate(items)
        assert len(result) == 2
        assert "Same item" in result
        assert "Different" in result

    def test_deduplicate_near_duplicates_keeps_shorter(self):
        """Test that highly similar strings (ratio >= 0.85) are deduplicated."""
        items = [
            "Must implement JWT authentication with secure token storage",
            "Must implement JWT authentication with token storage",
            "Different constraint entirely",
        ]
        result = _deduplicate(items)

        # These two are similar enough (ratio >= 0.85) that one should be removed
        assert len(result) == 2
        assert "Different constraint entirely" in result
        # One of the JWT items should remain (prefer shorter)
        jwt_items = [item for item in result if "JWT" in item]
        assert len(jwt_items) == 1

    def test_deduplicate_case_insensitive(self):
        items = ["Use PostgreSQL", "use postgresql", "Other"]
        result = _deduplicate(items)
        assert len(result) == 2

    def test_deduplicate_custom_threshold(self):
        items = ["The quick brown fox", "The fast brown fox", "The slow green turtle"]
        result = _deduplicate(items, threshold=0.6)
        # First two should be considered duplicates at 0.6 threshold
        assert len(result) <= 2


class TestPrioritize:
    """Test salience item prioritization."""

    def test_prioritize_empty(self):
        assert _prioritize([]) == []

    def test_prioritize_only_constraints(self):
        items = ["Must use FastAPI", "Cannot use eval", "Required to have auth"]
        result = _prioritize(items)
        assert result == items

    def test_prioritize_only_decisions(self):
        items = ["Chose PostgreSQL", "Decided to use JWT", "Selected REST architecture"]
        result = _prioritize(items)
        assert result == items

    def test_prioritize_only_facts(self):
        items = ["API endpoint defined", "Schema created", "Tests written"]
        result = _prioritize(items)
        assert result == items

    def test_prioritize_mixed_items(self):
        items = [
            "Schema created",  # fact
            "Must use FastAPI",  # constraint
            "Decided to use JWT",  # decision
            "Cannot use eval",  # constraint
            "Chose PostgreSQL",  # decision
            "Tests written",  # fact
        ]
        result = _prioritize(items)

        # Constraints should come first
        assert "Must use FastAPI" in result[:2]
        assert "Cannot use eval" in result[:2]

        # Decisions next
        assert "Decided to use JWT" in result[2:4]
        assert "Chose PostgreSQL" in result[2:4]

        # Facts last
        assert "Schema created" in result[4:]
        assert "Tests written" in result[4:]

    def test_prioritize_constraint_keywords(self):
        """Test various constraint keyword forms."""
        items = [
            "Fact A",
            "Must not use globals",
            "mustn't hardcode secrets",
            "Should not expose internals",
            "Forbidden to modify core",
            "Fact B",
        ]
        result = _prioritize(items)

        # All constraints should come before facts
        facts_start = result.index("Fact A")
        assert all(
            "must" in item.lower()
            or "forbidden" in item.lower()
            or "should not" in item.lower()
            for item in result[:facts_start]
        )

    def test_prioritize_decision_keywords(self):
        """Test various decision keyword forms."""
        items = [
            "Fact X",
            "opted for SQLite",
            "going with REST",
            "will use async",
            "picked FastAPI",
            "Fact Y",
        ]
        result = _prioritize(items)

        # Decisions should come before facts
        facts_start = result.index("Fact X")
        assert all(
            any(kw in item.lower() for kw in ["opted", "going", "will use", "picked"])
            for item in result[:facts_start]
        )


class TestBuildCompressedContext:
    """Test full context assembly without requiring live LLM."""

    def test_build_compressed_context_minimal(self):
        """Test with minimal session: just goal and constraints."""
        mgr = SessionManager()
        mgr.initialize(goal="Build a calculator", constraints=["Must handle negatives"])

        result = build_compressed_context(mgr, background_summary="", recent_turns=[])

        # Must include protected core
        assert "PROTECTED CORE" in result
        assert "Build a calculator" in result
        assert "Must handle negatives" in result

        # Should not have salience or background sections
        assert "SALIENT INFORMATION" not in result
        assert "BACKGROUND SUMMARY" not in result
        assert "RECENT TURNS" not in result

    def test_build_compressed_context_with_salience(self):
        """Test context includes prioritized salience."""
        mgr = SessionManager()
        mgr.initialize(goal="Build API", constraints=["Use FastAPI"])
        mgr.update_salience(
            [
                "Endpoint POST /users defined",
                "Must have JWT auth",
                "Decided to use PostgreSQL",
            ]
        )

        result = build_compressed_context(mgr, background_summary="", recent_turns=[])

        assert "PROTECTED CORE" in result
        assert "SALIENT INFORMATION" in result
        assert "Must have JWT auth" in result
        assert "Decided to use PostgreSQL" in result
        assert "Endpoint POST /users defined" in result

    def test_build_compressed_context_salience_prioritized(self):
        """Test salience items are prioritized correctly."""
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])
        mgr.update_salience(
            [
                "Fact A",
                "Must not use eval",
                "Decided to use async",
                "Fact B",
            ]
        )

        result = build_compressed_context(mgr, background_summary="", recent_turns=[])

        # Find salience section
        salience_start = result.index("SALIENT INFORMATION")
        salience_section = result[salience_start : salience_start + 500]

        # Constraint should appear before decision and facts
        constraint_pos = salience_section.index("Must not use eval")
        decision_pos = salience_section.index("Decided to use async")
        fact_pos = salience_section.index("Fact A")

        assert constraint_pos < decision_pos < fact_pos

    def test_build_compressed_context_with_background(self):
        """Test context includes background summary."""
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        background = "User discussed API design and chose REST architecture."
        result = build_compressed_context(mgr, background_summary=background, recent_turns=[])

        assert "BACKGROUND SUMMARY" in result
        assert background in result

    def test_build_compressed_context_with_recent_turns(self):
        """Test context includes recent conversation turns."""
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        recent = [
            {"id": 10, "role": "user", "content": "Add logging"},
            {"id": 11, "role": "assistant", "content": "I'll add structured logging"},
        ]
        result = build_compressed_context(mgr, background_summary="", recent_turns=recent)

        assert "RECENT TURNS" in result
        assert "Turn 10 (user): Add logging" in result
        assert "Turn 11 (assistant): I'll add structured logging" in result

    def test_build_compressed_context_full(self):
        """Integration: test full context with all sections."""
        mgr = SessionManager()
        mgr.initialize(
            goal="Build REST API",
            constraints=["Use FastAPI", "JWT auth required"],
        )
        mgr.add_decision("Use PostgreSQL", rationale="Relational model")
        mgr.update_salience(
            [
                "Endpoint: POST /users",
                "Must validate input",
                "Chose async handlers",
            ]
        )

        background = "User set up FastAPI project with poetry. Database schema created."
        recent = [
            {"id": 20, "role": "user", "content": "Add error handling"},
            {"id": 21, "role": "assistant", "content": "Adding HTTP exception handlers"},
        ]

        result = build_compressed_context(mgr, background_summary=background, recent_turns=recent)

        # All sections present
        assert "PROTECTED CORE" in result
        assert "SALIENT INFORMATION" in result
        assert "BACKGROUND SUMMARY" in result
        assert "RECENT TURNS" in result

        # Protected core content
        assert "Build REST API" in result
        assert "Use FastAPI" in result
        assert "JWT auth required" in result

        # Salience (prioritized)
        assert "Must validate input" in result
        assert "Chose async handlers" in result
        assert "Endpoint: POST /users" in result

        # Background
        assert background in result

        # Recent turns
        assert "Turn 20 (user): Add error handling" in result
        assert "Turn 21 (assistant): Adding HTTP exception handlers" in result

    def test_build_compressed_context_preserves_structure(self):
        """Test section ordering is consistent."""
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=["C1"])
        mgr.update_salience(["S1"])

        result = build_compressed_context(
            mgr,
            background_summary="B1",
            recent_turns=[{"id": 1, "role": "user", "content": "R1"}],
        )

        # Order: CORE → SALIENCE → BACKGROUND → RECENT
        core_idx = result.index("PROTECTED CORE")
        salience_idx = result.index("SALIENT INFORMATION")
        background_idx = result.index("BACKGROUND SUMMARY")
        recent_idx = result.index("RECENT TURNS")

        assert core_idx < salience_idx < background_idx < recent_idx
