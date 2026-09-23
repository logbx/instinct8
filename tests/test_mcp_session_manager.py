"""Unit tests for MCP SessionManager — no API keys required.

Covers pure state management: initialization, salience updates,
compression recording, serialization.
"""

import pytest

from mcp_server.instinct8_mcp.session_manager import (
    ProtectedCore,
    SessionState,
    SessionManager,
)


class TestProtectedCore:
    """Test ProtectedCore dataclass and rendering."""

    def test_protected_core_creation(self):
        core = ProtectedCore(
            goal="Build REST API",
            hard_constraints=["Must use FastAPI", "Must have auth"],
        )
        assert core.goal == "Build REST API"
        assert len(core.hard_constraints) == 2
        assert core.timestamp_updated

    def test_protected_core_render(self):
        core = ProtectedCore(
            goal="Test goal",
            hard_constraints=["Constraint A", "Constraint B"],
        )
        rendered = core.render()

        assert "PROTECTED CORE" in rendered
        assert "Goal: Test goal" in rendered
        assert "Constraint A" in rendered
        assert "Constraint B" in rendered
        assert "AUTHORITATIVE" in rendered


class TestSessionState:
    """Test SessionState dataclass and serialization."""

    def test_session_state_creation(self):
        core = ProtectedCore(
            goal="Goal A",
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
            goal="Current Goal",
            hard_constraints=["C1", "C2"],
        )
        state = SessionState(
            protected_core=core,
            salience_set=["Item 1", "Item 2"],
            compression_count=3,
            total_tokens_saved=1500,
        )

        data = state.to_dict()

        assert data["protected_core"]["goal"] == "Current Goal"
        assert data["protected_core"]["hard_constraints"] == ["C1", "C2"]
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
        assert session.protected_core.goal == "Build a calculator"
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

    def test_full_lifecycle(self):
        """Integration: initialize, update salience, compress, serialize."""
        mgr = SessionManager()

        # Initialize
        mgr.initialize(
            goal="Build a REST API",
            constraints=["Use FastAPI", "JWT auth required"],
        )
        assert mgr.has_session

        # Update salience (this is where we track decisions now)
        mgr.update_salience(
            [
                "Decision: Use PostgreSQL (relational data model)",
                "API endpoint: POST /users",
                "Auth uses JWT tokens",
                "PostgreSQL schema",
            ]
        )

        # Record compressions
        mgr.record_compression(tokens_saved=400)
        mgr.record_compression(tokens_saved=350)

        # Serialize
        session = mgr.require_session()
        data = session.to_dict()

        assert data["protected_core"]["goal"] == "Build a REST API"
        assert "Use FastAPI" in data["protected_core"]["hard_constraints"]
        assert "Decision: Use PostgreSQL (relational data model)" in data["salience_set"]
        assert data["compression_count"] == 2
        assert data["total_tokens_saved"] == 750
