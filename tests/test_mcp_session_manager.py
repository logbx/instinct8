"""Unit tests for MCP SessionManager — no API keys required.

Covers pure state management: initialization, goal/decision tracking,
salience updates, compression recording, serialization.
"""

import pytest
from datetime import datetime

from mcp_server.instinct8_mcp.session_manager import (
    Decision,
    ProtectedCore,
    SessionState,
    SessionManager,
)


class TestDecision:
    """Test Decision dataclass."""

    def test_decision_with_defaults(self):
        d = Decision(decision="Use FastAPI", rationale="Modern async framework")
        assert d.decision == "Use FastAPI"
        assert d.rationale == "Modern async framework"
        assert d.timestamp
        assert isinstance(d.timestamp, str)

    def test_decision_with_custom_timestamp(self):
        ts = "2026-09-15T10:00:00"
        d = Decision(
            decision="Use pytest", rationale="Standard test framework", timestamp=ts
        )
        assert d.timestamp == ts


class TestProtectedCore:
    """Test ProtectedCore dataclass and rendering."""

    def test_protected_core_creation(self):
        core = ProtectedCore(
            original_goal="Build REST API",
            current_goal="Build REST API",
            hard_constraints=["Must use FastAPI", "Must have auth"],
        )
        assert core.original_goal == "Build REST API"
        assert core.current_goal == "Build REST API"
        assert len(core.hard_constraints) == 2
        assert core.key_decisions == []

    def test_protected_core_render_no_decisions(self):
        core = ProtectedCore(
            original_goal="Test goal",
            current_goal="Test goal",
            hard_constraints=["Constraint A", "Constraint B"],
        )
        rendered = core.render()

        assert "PROTECTED CORE" in rendered
        assert "Original Goal: Test goal" in rendered
        assert "Current Goal: Test goal" in rendered
        assert "Constraint A" in rendered
        assert "Constraint B" in rendered
        assert "(none yet)" in rendered

    def test_protected_core_render_with_decisions(self):
        core = ProtectedCore(
            original_goal="Build API",
            current_goal="Build API with auth",
            hard_constraints=["Use FastAPI"],
            key_decisions=[
                Decision(
                    decision="Add JWT auth",
                    rationale="Industry standard",
                    timestamp="2026-09-15T10:00:00",
                )
            ],
        )
        rendered = core.render()

        assert "Add JWT auth" in rendered
        assert "Rationale: Industry standard" in rendered
        assert "CURRENT GOAL" in rendered


class TestSessionState:
    """Test SessionState dataclass and serialization."""

    def test_session_state_creation(self):
        core = ProtectedCore(
            original_goal="Goal A",
            current_goal="Goal A",
            hard_constraints=["Constraint X"],
        )
        state = SessionState(protected_core=core)

        assert state.protected_core == core
        assert state.salience_set == []
        assert state.compression_count == 0
        assert state.total_tokens_saved == 0
        assert state.created_at

    def test_session_state_to_dict(self):
        core = ProtectedCore(
            original_goal="Original",
            current_goal="Current",
            hard_constraints=["C1", "C2"],
            key_decisions=[
                Decision(
                    decision="D1", rationale="R1", timestamp="2026-09-15T10:00:00"
                )
            ],
        )
        state = SessionState(
            protected_core=core,
            salience_set=["Item 1", "Item 2"],
            compression_count=3,
            total_tokens_saved=1500,
        )

        data = state.to_dict()

        assert data["protected_core"]["original_goal"] == "Original"
        assert data["protected_core"]["current_goal"] == "Current"
        assert data["protected_core"]["hard_constraints"] == ["C1", "C2"]
        assert len(data["protected_core"]["key_decisions"]) == 1
        assert data["protected_core"]["key_decisions"][0]["decision"] == "D1"
        assert data["salience_set"] == ["Item 1", "Item 2"]
        assert data["compression_count"] == 3
        assert data["total_tokens_saved"] == 1500
        assert "created_at" in data


class TestSessionManager:
    """Test SessionManager lifecycle and state operations."""

    def test_manager_starts_empty(self):
        mgr = SessionManager()
        assert mgr.session is None
        assert not mgr.has_session

    def test_initialize_creates_session(self):
        mgr = SessionManager()
        session = mgr.initialize(
            goal="Build a calculator", constraints=["Must handle negatives", "No eval"]
        )

        assert mgr.has_session
        assert mgr.session is session
        assert session.protected_core.original_goal == "Build a calculator"
        assert session.protected_core.current_goal == "Build a calculator"
        assert len(session.protected_core.hard_constraints) == 2
        assert "Must handle negatives" in session.protected_core.hard_constraints

    def test_require_session_raises_when_empty(self):
        mgr = SessionManager()
        with pytest.raises(ValueError, match="No active session"):
            mgr.require_session()

    def test_require_session_returns_session(self):
        mgr = SessionManager()
        session = mgr.initialize(goal="Test", constraints=[])
        retrieved = mgr.require_session()
        assert retrieved is session

    def test_update_goal(self):
        mgr = SessionManager()
        mgr.initialize(goal="Initial goal", constraints=["C1"])

        mgr.update_goal("Updated goal", rationale="User changed requirements")

        session = mgr.require_session()
        assert session.protected_core.current_goal == "Updated goal"
        assert len(session.protected_core.key_decisions) == 1
        decision = session.protected_core.key_decisions[0]
        assert "Updated goal" in decision.decision
        assert decision.rationale == "User changed requirements"

    def test_update_goal_default_rationale(self):
        mgr = SessionManager()
        mgr.initialize(goal="Goal A", constraints=[])

        mgr.update_goal("Goal B")

        session = mgr.require_session()
        assert session.protected_core.current_goal == "Goal B"
        assert len(session.protected_core.key_decisions) == 1
        assert "Goal evolution" in session.protected_core.key_decisions[0].rationale

    def test_add_decision(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        mgr.add_decision("Use SQLite", rationale="Lightweight and portable")

        session = mgr.require_session()
        assert len(session.protected_core.key_decisions) == 1
        decision = session.protected_core.key_decisions[0]
        assert decision.decision == "Use SQLite"
        assert decision.rationale == "Lightweight and portable"

    def test_add_multiple_decisions(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        mgr.add_decision("D1", rationale="R1")
        mgr.add_decision("D2", rationale="R2")

        session = mgr.require_session()
        assert len(session.protected_core.key_decisions) == 2
        assert session.protected_core.key_decisions[0].decision == "D1"
        assert session.protected_core.key_decisions[1].decision == "D2"

    def test_update_salience(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        mgr.update_salience(["Fact A", "Fact B", "Fact C"])

        session = mgr.require_session()
        assert session.salience_set == ["Fact A", "Fact B", "Fact C"]

    def test_update_salience_replaces(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])
        mgr.update_salience(["Old A", "Old B"])

        mgr.update_salience(["New X", "New Y"])

        session = mgr.require_session()
        assert session.salience_set == ["New X", "New Y"]
        assert "Old A" not in session.salience_set

    def test_record_compression(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        mgr.record_compression(tokens_saved=500)

        session = mgr.require_session()
        assert session.compression_count == 1
        assert session.total_tokens_saved == 500

    def test_record_multiple_compressions(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=[])

        mgr.record_compression(tokens_saved=300)
        mgr.record_compression(tokens_saved=450)
        mgr.record_compression(tokens_saved=250)

        session = mgr.require_session()
        assert session.compression_count == 3
        assert session.total_tokens_saved == 1000

    def test_reset(self):
        mgr = SessionManager()
        mgr.initialize(goal="Test", constraints=["C1"])
        assert mgr.has_session

        mgr.reset()

        assert not mgr.has_session
        assert mgr.session is None

    def test_full_lifecycle(self):
        """Integration: initialize, update, compress, serialize, reset."""
        mgr = SessionManager()

        # Initialize
        mgr.initialize(
            goal="Build a REST API",
            constraints=["Use FastAPI", "JWT auth required"],
        )
        assert mgr.has_session

        # Track decisions
        mgr.add_decision("Use PostgreSQL", rationale="Relational data model")
        mgr.update_goal("Build a REST API with admin panel", rationale="User request")

        # Update salience
        mgr.update_salience(
            ["API endpoint: POST /users", "Auth uses JWT tokens", "PostgreSQL schema"]
        )

        # Record compressions
        mgr.record_compression(tokens_saved=400)
        mgr.record_compression(tokens_saved=350)

        # Serialize
        session = mgr.require_session()
        data = session.to_dict()

        assert data["protected_core"]["current_goal"] == "Build a REST API with admin panel"
        assert len(data["protected_core"]["key_decisions"]) == 2
        assert data["salience_set"] == [
            "API endpoint: POST /users",
            "Auth uses JWT tokens",
            "PostgreSQL schema",
        ]
        assert data["compression_count"] == 2
        assert data["total_tokens_saved"] == 750

        # Reset
        mgr.reset()
        assert not mgr.has_session
