"""Unit tests for the MCP tool layer (mcp_server/instinct8_mcp/tools.py) — no API keys required.

The tool handlers are captured with a fake server whose ``tool()`` decorator
records them, and each call gets a fake ``ctx`` whose ``.session`` is a stub.
Most tests monkeypatch the async ``strategy_adapter`` functions with canned
values; ``TestSamplingCallsEndToEnd`` instead routes canned replies through a
stub MCP session so the reported ``sampling_calls`` can be checked against the
number of ``create_message`` calls actually made.

The ``mcp`` package is not part of the root ``[dev]`` extra, so when it is not
installed a minimal stand-in is used for the imports tools.py and sampling.py
need.
"""

import importlib
import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_server.instinct8_mcp import strategy_adapter as sa
from mcp_server.instinct8_mcp.session_manager import SessionManager


def _import_tools_module():
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        stub_names = ("mcp", "mcp.server", "mcp.server.fastmcp")
        stubs = {name: types.ModuleType(name) for name in stub_names}
        stubs["mcp.server.fastmcp"].Context = object
        added = [name for name in stub_names if name not in sys.modules]
        sys.modules.update({name: stubs[name] for name in added})
        try:
            return importlib.import_module("mcp_server.instinct8_mcp.tools")
        finally:
            for name in added:
                sys.modules.pop(name, None)
    return importlib.import_module("mcp_server.instinct8_mcp.tools")


tools_module = _import_tools_module()

NO_SESSION_MESSAGE = "No active session. Call initialize_session first."


class _FakeServer:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def sm():
    return SessionManager()


@pytest.fixture
def registered(sm):
    server = _FakeServer()
    tools_module.register_tools(server, sm)
    return server.tools


@pytest.fixture
def ctx():
    return SimpleNamespace(session=object())


def _turns(n, content="turn content"):
    return [
        {"id": i, "role": "user" if i % 2 else "assistant", "content": f"{content} {i}"}
        for i in range(1, n + 1)
    ]


def _forbid(*names):
    async def _fail(*args, **kwargs):
        raise AssertionError("sampling helper should not be called")

    return {name: _fail for name in names}


def test_register_tools_registers_exactly_three_tools(registered):
    assert set(registered) == {"initialize_session", "compress_context", "measure_drift"}


class TestInitializeSession:
    async def test_returns_status_core_and_message(self, registered, sm):
        result = json.loads(
            await registered["initialize_session"](
                goal="Build a REST API",
                constraints=["Use FastAPI", "Budget max $10K"],
            )
        )

        assert result == {
            "status": "initialized",
            "protected_core": {
                "goal": "Build a REST API",
                "constraints": ["Use FastAPI", "Budget max $10K"],
            },
            "message": (
                "Session initialized. Goal and 2 constraint(s) are now protected "
                "and will be preserved across compressions."
            ),
        }
        assert sm.session.protected_core.goal == "Build a REST API"
        assert sm.session.protected_core.hard_constraints == ["Use FastAPI", "Budget max $10K"]

    async def test_zero_constraints(self, registered):
        result = json.loads(await registered["initialize_session"](goal="G", constraints=[]))

        assert result["protected_core"] == {"goal": "G", "constraints": []}
        assert result["message"].startswith("Session initialized. Goal and 0 constraint(s)")

    async def test_constraints_are_copied_from_input(self, registered, sm):
        constraints = ["C1"]
        await registered["initialize_session"](goal="G", constraints=constraints)
        constraints.append("C2")

        assert sm.session.protected_core.hard_constraints == ["C1"]

    async def test_reinitialize_replaces_previous_session(self, registered, sm):
        await registered["initialize_session"](goal="Old", constraints=["Old C"])
        sm.update_salience(["stale item"])
        sm.record_compression(100)

        await registered["initialize_session"](goal="New", constraints=["New C"])

        assert sm.session.protected_core.goal == "New"
        assert sm.session.salience_set == []
        assert sm.session.compression_count == 0
        assert sm.session.total_tokens_saved == 0


class TestRequireSession:
    async def test_compress_context_without_session_raises(self, registered, ctx, monkeypatch):
        for name, fn in _forbid("extract_salience", "compress_background").items():
            monkeypatch.setattr(tools_module.sa, name, fn)

        with pytest.raises(ValueError, match=NO_SESSION_MESSAGE):
            await registered["compress_context"](conversation=_turns(5), ctx=ctx)

    async def test_measure_drift_without_session_raises(self, registered, ctx, monkeypatch):
        names = (
            "measure_goal_coherence",
            "measure_constraint_recall",
            "measure_behavioral_alignment",
        )
        for name, fn in _forbid(*names).items():
            monkeypatch.setattr(tools_module.sa, name, fn)

        with pytest.raises(ValueError, match=NO_SESSION_MESSAGE):
            await registered["measure_drift"](
                stated_goal="G",
                ctx=ctx,
                stated_constraints="C",
                agent_response="R",
            )


class TestCompressContextNoCompress:
    @pytest.fixture(autouse=True)
    def _no_sampling(self, monkeypatch):
        for name, fn in _forbid("extract_salience", "compress_background").items():
            monkeypatch.setattr(tools_module.sa, name, fn)

    @pytest.mark.parametrize(
        "n_turns,keep_recent_turns",
        [(3, 3), (2, 3), (0, 3), (1, 1)],
    )
    async def test_all_turns_recent_returns_raw_context(
        self, registered, sm, ctx, n_turns, keep_recent_turns
    ):
        sm.initialize("Build API", ["Use FastAPI"])
        conversation = _turns(n_turns)

        result = json.loads(
            await registered["compress_context"](
                conversation=conversation,
                ctx=ctx,
                keep_recent_turns=keep_recent_turns,
            )
        )

        assert result == {
            "compressed_context": sa.build_compressed_context(sm, "", conversation),
            "stats": {
                "original_turns": n_turns,
                "salience_items": 0,
                "sampling_calls": 0,
            },
        }
        assert sm.session.compression_count == 0
        assert sm.session.total_tokens_saved == 0

    async def test_default_keep_recent_turns_is_three(self, registered, sm, ctx):
        sm.initialize("G", [])

        result = json.loads(await registered["compress_context"](conversation=_turns(3), ctx=ctx))

        assert result["stats"]["sampling_calls"] == 0

    async def test_reports_existing_salience_items(self, registered, sm, ctx):
        sm.initialize("G", [])
        sm.update_salience(["Must use FastAPI", "Chose PostgreSQL"])

        result = json.loads(await registered["compress_context"](conversation=_turns(2), ctx=ctx))

        assert result["stats"]["salience_items"] == 2
        assert "=== SALIENT INFORMATION ===" in result["compressed_context"]
        assert "=== BACKGROUND SUMMARY ===" not in result["compressed_context"]


class TestCompressContextFullPath:
    @pytest.fixture
    def calls(self, monkeypatch, sm):
        recorded = {"extract_salience": [], "compress_background": []}

        async def fake_extract_salience(session, manager, context):
            recorded["extract_salience"].append((session, manager, context))
            items = ["Must use FastAPI", "Decided to use PostgreSQL"]
            manager.update_salience(items)
            return items

        async def fake_compress_background(session, manager, context):
            recorded["compress_background"].append((session, manager, context))
            return "BACKGROUND"

        monkeypatch.setattr(tools_module.sa, "extract_salience", fake_extract_salience)
        monkeypatch.setattr(tools_module.sa, "compress_background", fake_compress_background)
        return recorded

    async def test_splits_conversation_and_reports_exact_stats(self, registered, sm, ctx, calls):
        sm.initialize("Build API", ["Use FastAPI"])
        conversation = _turns(5)

        result = json.loads(
            await registered["compress_context"](
                conversation=conversation,
                ctx=ctx,
                keep_recent_turns=2,
            )
        )

        assert calls["extract_salience"] == [(ctx.session, sm, conversation[:3])]
        assert calls["compress_background"] == [(ctx.session, sm, conversation[:3])]

        expected_context = sa.build_compressed_context(sm, "BACKGROUND", conversation[3:])
        original_chars = sum(len(t["content"]) for t in conversation)
        expected_saved = max(0, (original_chars - len(expected_context)) // 4)
        assert result == {
            "compressed_context": expected_context,
            "stats": {
                "original_turns": 5,
                "compressed_length": len(expected_context),
                "salience_items": 2,
                "estimated_tokens_saved": expected_saved,
                "compression_count": 1,
                "sampling_calls": 2,
            },
        }
        assert "Turn 4 (assistant): turn content 4" in expected_context
        assert "Turn 3 (user)" not in expected_context

    async def test_short_conversation_saves_zero_tokens_but_counts_compression(
        self, registered, sm, ctx, calls
    ):
        sm.initialize("G", [])

        result = json.loads(
            await registered["compress_context"](conversation=_turns(4), ctx=ctx, keep_recent_turns=1)
        )

        assert result["stats"]["estimated_tokens_saved"] == 0
        assert result["stats"]["compression_count"] == 1
        assert sm.session.compression_count == 1
        assert sm.session.total_tokens_saved == 0

    async def test_long_conversation_accumulates_tokens_saved(self, registered, sm, ctx, calls):
        sm.initialize("G", [])
        conversation = _turns(6, content="x" * 2000)

        first = json.loads(
            await registered["compress_context"](conversation=conversation, ctx=ctx, keep_recent_turns=1)
        )
        second = json.loads(
            await registered["compress_context"](conversation=conversation, ctx=ctx, keep_recent_turns=1)
        )

        saved = first["stats"]["estimated_tokens_saved"]
        assert saved > 0
        assert second["stats"]["estimated_tokens_saved"] == saved
        assert second["stats"]["compression_count"] == 2
        assert sm.session.total_tokens_saved == 2 * saved

    async def test_keep_zero_recent_turns_compresses_everything(self, registered, sm, ctx, calls):
        sm.initialize("G", [])
        conversation = _turns(3)

        result = json.loads(
            await registered["compress_context"](conversation=conversation, ctx=ctx, keep_recent_turns=0)
        )

        assert calls["extract_salience"][0][2] == conversation
        assert "=== RECENT TURNS ===" not in result["compressed_context"]
        assert result["stats"]["sampling_calls"] == 2


class TestMeasureDrift:
    @pytest.fixture
    def fakes(self, monkeypatch):
        state = {
            "coherence": 0.85,
            "recall": 1.0,
            "alignment": 5,
            "calls": {"coherence": [], "recall": [], "alignment": []},
        }

        async def fake_goal_coherence(session, original_goal, stated_goal):
            state["calls"]["coherence"].append((session, original_goal, stated_goal))
            return state["coherence"]

        async def fake_constraint_recall(session, known_constraints, stated_constraints):
            state["calls"]["recall"].append((session, known_constraints, stated_constraints))
            return state["recall"]

        async def fake_behavioral_alignment(session, goal, constraints, response, test_context):
            state["calls"]["alignment"].append((session, goal, constraints, response, test_context))
            return state["alignment"]

        monkeypatch.setattr(tools_module.sa, "measure_goal_coherence", fake_goal_coherence)
        monkeypatch.setattr(tools_module.sa, "measure_constraint_recall", fake_constraint_recall)
        monkeypatch.setattr(tools_module.sa, "measure_behavioral_alignment", fake_behavioral_alignment)
        return state

    async def test_goal_coherence_only(self, registered, sm, ctx, fakes):
        sm.initialize("Build API", ["Use FastAPI"])

        result = json.loads(await registered["measure_drift"](stated_goal="Build an API", ctx=ctx))

        assert result == {
            "goal_coherence": 0.85,
            "goal_coherence_interpretation": "Good — minor drift but core goal intact",
            "sampling_calls": 1,
            "drift_detected": False,
            "session_stats": {
                "compression_count": 0,
                "total_tokens_saved": 0,
                "salience_items": 0,
            },
        }
        assert fakes["calls"]["coherence"] == [(ctx.session, "Build API", "Build an API")]
        assert fakes["calls"]["recall"] == []
        assert fakes["calls"]["alignment"] == []

    @pytest.mark.parametrize(
        "coherence,drift_detected",
        [(1.0, False), (0.7, False), (0.6999, True), (0.0, True)],
    )
    async def test_drift_detected_threshold(self, registered, sm, ctx, fakes, coherence, drift_detected):
        sm.initialize("G", [])
        fakes["coherence"] = coherence

        result = json.loads(await registered["measure_drift"](stated_goal="S", ctx=ctx))

        assert result["drift_detected"] is drift_detected

    async def test_drift_detected_ignores_recall_and_alignment(self, registered, sm, ctx, fakes):
        sm.initialize("G", ["C1"])
        fakes.update(coherence=0.95, recall=0.0, alignment=1)

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="S", ctx=ctx, stated_constraints="none", agent_response="R"
            )
        )

        assert result["drift_detected"] is False

    async def test_constraint_recall_branch(self, registered, sm, ctx, fakes):
        constraints = ["Use FastAPI", "Budget max $10K", "JWT auth"]
        sm.initialize("G", constraints)
        fakes["recall"] = 2 / 3

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="S", ctx=ctx, stated_constraints="FastAPI and JWT"
            )
        )

        assert result["constraint_recall"] == pytest.approx(2 / 3)
        assert result["constraints_remembered"] == "2/3"
        assert result["sampling_calls"] == 1 + len(constraints)
        assert "behavioral_alignment" not in result
        assert fakes["calls"]["recall"] == [(ctx.session, constraints, "FastAPI and JWT")]

    @pytest.mark.parametrize(
        "recall,remembered",
        [(0.0, "0/4"), (0.25, "1/4"), (0.5, "2/4"), (0.75, "3/4"), (1.0, "4/4")],
    )
    async def test_constraints_remembered_format(self, registered, sm, ctx, fakes, recall, remembered):
        sm.initialize("G", ["C1", "C2", "C3", "C4"])
        fakes["recall"] = recall

        result = json.loads(
            await registered["measure_drift"](stated_goal="S", ctx=ctx, stated_constraints="x")
        )

        assert result["constraints_remembered"] == remembered

    @pytest.mark.xfail(
        strict=True,
        reason="int(recall * n) truncates float error: (15/22)*22 == 14.999..., reported as 14/22",
    )
    async def test_constraints_remembered_exact_for_float_ratio(self, registered, sm, ctx, fakes):
        sm.initialize("G", [f"C{i}" for i in range(22)])
        fakes["recall"] = 15 / 22

        result = json.loads(
            await registered["measure_drift"](stated_goal="S", ctx=ctx, stated_constraints="x")
        )

        assert result["constraints_remembered"] == "15/22"

    async def test_constraint_recall_skipped_without_stated_constraints(self, registered, sm, ctx, fakes):
        sm.initialize("G", ["C1"])

        result = json.loads(await registered["measure_drift"](stated_goal="S", ctx=ctx))

        assert "constraint_recall" not in result
        assert "constraints_remembered" not in result
        assert result["sampling_calls"] == 1
        assert fakes["calls"]["recall"] == []

    async def test_constraint_recall_skipped_when_session_has_no_constraints(
        self, registered, sm, ctx, fakes
    ):
        sm.initialize("G", [])

        result = json.loads(
            await registered["measure_drift"](stated_goal="S", ctx=ctx, stated_constraints="Use FastAPI")
        )

        assert "constraint_recall" not in result
        assert result["sampling_calls"] == 1
        assert fakes["calls"]["recall"] == []

    async def test_behavioral_alignment_branch(self, registered, sm, ctx, fakes):
        sm.initialize("Build API", ["Use FastAPI"])
        fakes["alignment"] = 4

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="S",
                ctx=ctx,
                agent_response="Let's switch to Django",
                test_context="User asks to change framework",
            )
        )

        assert result["behavioral_alignment"] == 4
        assert result["behavioral_interpretation"] == "Mostly aligned, minor deviations"
        assert result["sampling_calls"] == 2
        assert "constraint_recall" not in result
        assert fakes["calls"]["alignment"] == [
            (
                ctx.session,
                "Build API",
                ["Use FastAPI"],
                "Let's switch to Django",
                "User asks to change framework",
            )
        ]

    async def test_all_branches_and_session_stats(self, registered, sm, ctx, fakes):
        sm.initialize("G", ["C1", "C2"])
        sm.update_salience(["s1", "s2", "s3"])
        sm.record_compression(40)
        sm.record_compression(2)
        fakes.update(coherence=0.5, recall=0.5, alignment=2)

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="S", ctx=ctx, stated_constraints="C1", agent_response="R"
            )
        )

        assert result == {
            "goal_coherence": 0.5,
            "goal_coherence_interpretation": "Warning — significant drift detected",
            "sampling_calls": 1 + 2 + 1,
            "constraint_recall": 0.5,
            "constraints_remembered": "1/2",
            "behavioral_alignment": 2,
            "behavioral_interpretation": "Some drift detected",
            "drift_detected": True,
            "session_stats": {
                "compression_count": 2,
                "total_tokens_saved": 42,
                "salience_items": 3,
            },
        }


class TestInterpretCoherence:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (1.0, "Excellent — goal is well preserved"),
            (0.9, "Excellent — goal is well preserved"),
            (0.8999, "Good — minor drift but core goal intact"),
            (0.7, "Good — minor drift but core goal intact"),
            (0.6999, "Warning — significant drift detected"),
            (0.5, "Warning — significant drift detected"),
            (0.4999, "Critical — major goal drift"),
            (0.3, "Critical — major goal drift"),
            (0.2999, "Severe — goal appears lost"),
            (0.0, "Severe — goal appears lost"),
        ],
    )
    def test_bands(self, score, expected):
        assert tools_module._interpret_coherence(score) == expected


class TestInterpretAlignment:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (5, "Perfectly aligned"),
            (4, "Mostly aligned, minor deviations"),
            (3, "Ambiguous alignment"),
            (2, "Some drift detected"),
            (1, "Complete drift — goal forgotten"),
        ],
    )
    def test_known_scores(self, score, expected):
        assert tools_module._interpret_alignment(score) == expected

    @pytest.mark.parametrize("score", [0, 6, -1])
    def test_unknown_fallback(self, score):
        assert tools_module._interpret_alignment(score) == "Unknown"


@dataclass
class _StubTextContent:
    type: str
    text: str


@dataclass
class _StubSamplingMessage:
    role: str
    content: Any


@pytest.fixture
def mcp_types(monkeypatch):
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        module = types.ModuleType("mcp.types")
        module.SamplingMessage = _StubSamplingMessage
        module.TextContent = _StubTextContent
        if "mcp" not in sys.modules:
            monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
        monkeypatch.setitem(sys.modules, "mcp.types", module)


class _RoutingSession:
    """Stub MCP session that answers ``create_message`` by matching prompt text."""

    def __init__(self, routes):
        self.routes = routes
        self.prompts = []

    async def create_message(self, messages, *, max_tokens, system_prompt=None):
        prompt = messages[0].content.text
        self.prompts.append(prompt)
        for marker, reply in self.routes:
            if marker in prompt:
                text = reply(prompt) if callable(reply) else reply
                return SimpleNamespace(content=SimpleNamespace(text=text))
        raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")


class TestSamplingCallsEndToEnd:
    @pytest.fixture(autouse=True)
    def _types(self, mcp_types):
        pass

    async def test_compress_context_makes_two_sampling_calls(self, registered, sm):
        session = _RoutingSession(
            [
                ("selective salience extraction", '{"salient_items": ["Must use FastAPI"]}'),
                ("compressing conversation background", "User set up the project."),
            ]
        )
        sm.initialize("Build API", ["Use FastAPI"])

        result = json.loads(
            await registered["compress_context"](
                conversation=_turns(5),
                ctx=SimpleNamespace(session=session),
                keep_recent_turns=2,
            )
        )

        assert len(session.prompts) == result["stats"]["sampling_calls"] == 2
        assert sm.session.salience_set == ["Must use FastAPI"]
        assert "1. Must use FastAPI" in result["compressed_context"]
        assert "User set up the project." in result["compressed_context"]

    async def test_measure_drift_sampling_calls_match_reality(self, registered, sm):
        session = _RoutingSession(
            [
                ("evaluating goal coherence", "0.85"),
                ("Does this statement mention", lambda p: "yes" if "Use FastAPI" in p else "no"),
                ("aligns with its original goal", "4"),
            ]
        )
        sm.initialize("Build API", ["Use FastAPI", "Budget max $10K"])

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="Build a web service",
                ctx=SimpleNamespace(session=session),
                stated_constraints="I am using FastAPI",
                agent_response="Continuing with FastAPI",
            )
        )

        assert len(session.prompts) == result["sampling_calls"] == 4
        assert result["goal_coherence"] == 0.85
        assert result["constraint_recall"] == 0.5
        assert result["constraints_remembered"] == "1/2"
        assert result["behavioral_alignment"] == 4

    async def test_out_of_range_scores_are_clamped(self, registered, sm):
        session = _RoutingSession(
            [
                ("evaluating goal coherence", "1.5"),
                ("aligns with its original goal", "7"),
            ]
        )
        sm.initialize("Build API", [])

        result = json.loads(
            await registered["measure_drift"](
                stated_goal="Build a web service",
                ctx=SimpleNamespace(session=session),
                agent_response="R",
            )
        )

        assert result["goal_coherence"] == 1.0
        assert result["behavioral_alignment"] == 5
        assert result["behavioral_interpretation"] == "Perfectly aligned"
