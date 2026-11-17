"""Unit tests for MCP sampling helpers and prompt templates — no API keys required.

- sampling.py: response-text extraction, JSON parsing (plain, fenced, bare
  braces, failure), numeric score parsing, and ``_extract_json_block``
- prompts.py: the two registered prompt builders, captured with a fake server

``sample_text`` imports ``mcp.types`` lazily; when the ``mcp`` package is not
installed a minimal stand-in is provided for that import.
"""

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_server.instinct8_mcp import sampling
from mcp_server.instinct8_mcp.prompts import register_prompts


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


class _RecordingSession:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def create_message(self, messages, *, max_tokens, system_prompt=None):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "system_prompt": system_prompt}
        )
        return SimpleNamespace(content=self.content)


@pytest.fixture
def canned_text(monkeypatch):
    """Replace ``sample_text`` with a canned reply and record its arguments."""
    state = {"reply": "", "calls": []}

    async def fake_sample_text(session, prompt, *, max_tokens=2000, system_hint=None):
        state["calls"].append(
            {"session": session, "prompt": prompt, "max_tokens": max_tokens, "system_hint": system_hint}
        )
        return state["reply"]

    monkeypatch.setattr(sampling, "sample_text", fake_sample_text)
    return state


class TestSampleText:
    @pytest.fixture(autouse=True)
    def _types(self, mcp_types):
        pass

    async def test_sends_single_user_message_with_defaults(self):
        session = _RecordingSession(SimpleNamespace(text="hello"))

        assert await sampling.sample_text(session, "the prompt") == "hello"

        (call,) = session.calls
        assert call["max_tokens"] == 2000
        assert call["system_prompt"] is None
        (message,) = call["messages"]
        assert message.role == "user"
        assert message.content.type == "text"
        assert message.content.text == "the prompt"

    async def test_forwards_max_tokens_and_system_hint(self):
        session = _RecordingSession(SimpleNamespace(text="ok"))

        await sampling.sample_text(session, "p", max_tokens=5, system_hint="be brief")

        assert session.calls[0]["max_tokens"] == 5
        assert session.calls[0]["system_prompt"] == "be brief"

    async def test_joins_text_blocks_from_list_content(self):
        content = [SimpleNamespace(text="first"), SimpleNamespace(data="img"), SimpleNamespace(text="second")]
        session = _RecordingSession(content)

        assert await sampling.sample_text(session, "p") == "first\nsecond"

    async def test_falls_back_to_str_for_other_content(self):
        session = _RecordingSession(42)

        assert await sampling.sample_text(session, "p") == "42"


class TestSampleJson:
    async def test_parses_plain_json(self, canned_text):
        canned_text["reply"] = '{"salient_items": ["a", "b"]}'

        assert await sampling.sample_json(object(), "p") == {"salient_items": ["a", "b"]}

    async def test_parses_json_in_fenced_block(self, canned_text):
        canned_text["reply"] = 'Here you go:\n```json\n{"salient_items": ["a"]}\n```\nDone.'

        assert await sampling.sample_json(object(), "p") == {"salient_items": ["a"]}

    async def test_parses_json_in_unlabeled_fence(self, canned_text):
        canned_text["reply"] = '```\n{"k": 1}\n```'

        assert await sampling.sample_json(object(), "p") == {"k": 1}

    async def test_parses_bare_braces_in_prose(self, canned_text):
        canned_text["reply"] = 'Sure! {"k": [1, 2]} hope that helps'

        assert await sampling.sample_json(object(), "p") == {"k": [1, 2]}

    async def test_non_json_raises_value_error_with_truncated_text(self, canned_text):
        canned_text["reply"] = "x" * 300

        with pytest.raises(ValueError) as excinfo:
            await sampling.sample_json(object(), "p")

        assert str(excinfo.value) == "Client returned non-JSON response: " + "x" * 200

    async def test_malformed_json_inside_fence_raises_value_error(self, canned_text):
        canned_text["reply"] = "```json\n{not valid}\n```"

        with pytest.raises(ValueError):
            await sampling.sample_json(object(), "p")

    async def test_forwards_arguments_to_sample_text(self, canned_text):
        canned_text["reply"] = "{}"
        session = object()

        await sampling.sample_json(session, "the prompt", max_tokens=123, system_hint="json only")

        assert canned_text["calls"] == [
            {"session": session, "prompt": "the prompt", "max_tokens": 123, "system_hint": "json only"}
        ]


class TestSampleScore:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("0.85", 0.85),
            ("  0.85\n", 0.85),
            ("4", 4.0),
            ("Score: 0.7", 0.7),
            ("I'd rate this 3 out of 5", 3.0),
        ],
    )
    async def test_parses_numeric_reply(self, canned_text, reply, expected):
        canned_text["reply"] = reply

        assert await sampling.sample_score(object(), "p") == expected

    async def test_does_not_clamp(self, canned_text):
        canned_text["reply"] = "7.5"

        assert await sampling.sample_score(object(), "p") == 7.5

    async def test_non_numeric_raises_value_error(self, canned_text):
        canned_text["reply"] = " no idea "

        with pytest.raises(ValueError, match=r"Client returned non-numeric response: 'no idea'"):
            await sampling.sample_score(object(), "p")

    async def test_default_and_custom_max_tokens(self, canned_text):
        canned_text["reply"] = "1"

        await sampling.sample_score(object(), "p")
        await sampling.sample_score(object(), "p", max_tokens=5)

        assert [c["max_tokens"] for c in canned_text["calls"]] == [10, 5]
        assert all(c["system_hint"] is None for c in canned_text["calls"])


class TestExtractJsonBlock:
    def test_fenced_json(self):
        assert sampling._extract_json_block('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fenced_without_language(self):
        assert sampling._extract_json_block('```\n[1, 2]\n```') == "[1, 2]"

    def test_prefers_fence_over_outer_braces(self):
        text = 'prefix {"outer": 0} ```json\n{"inner": 1}\n```'

        assert sampling._extract_json_block(text) == '{"inner": 1}'

    def test_bare_braces_span_first_to_last(self):
        assert sampling._extract_json_block('a {"x": {"y": 1}} b') == '{"x": {"y": 1}}'

    def test_no_json_returns_none(self):
        assert sampling._extract_json_block("plain text, no json") is None


class _FakePromptServer:
    def __init__(self):
        self.prompts = {}

    def prompt(self):
        def decorator(fn):
            self.prompts[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def prompts():
    server = _FakePromptServer()
    register_prompts(server)
    return server.prompts


def test_register_prompts_registers_exactly_two(prompts):
    assert set(prompts) == {"selective_salience_extract", "protected_core_compress"}


class TestSelectiveSalienceExtract:
    def test_interpolates_inputs(self, prompts):
        text = prompts["selective_salience_extract"](
            goal="Build API",
            constraints="Use FastAPI, JWT auth",
            conversation="user: hi\nassistant: hello",
        )

        assert text.startswith("You are performing selective salience extraction for context compression.\n")
        assert "Original Goal: Build API\nConstraints: Use FastAPI, JWT auth\n" in text
        assert "Conversation to analyze:\nuser: hi\nassistant: hello\n" in text
        assert text.endswith(
            'Output format (JSON):\n{\n  "salient_items": [\n'
            '    "exact quote 1",\n    "exact quote 2"\n  ]\n}'
        )


class TestProtectedCoreCompress:
    def test_renders_constraints_and_decisions_as_bullets(self, prompts):
        text = prompts["protected_core_compress"](
            goal="Build API",
            constraints=" Use FastAPI ,JWT auth, ,Budget max $10K",
            decisions="Chose PostgreSQL\n\n  Use async handlers  \n",
            conversation="user: hi",
        )

        assert "Goal: Build API\n" in text
        assert (
            "Hard Constraints (MUST FOLLOW):\n"
            "  - Use FastAPI\n"
            "  - JWT auth\n"
            "  - Budget max $10K\n"
            "\n"
            "Key Decisions Made:\n"
            "  - Chose PostgreSQL\n"
            "  - Use async handlers\n"
            "================================================\n"
        ) in text
        assert "Conversation to compress:\nuser: hi\n" in text
        assert text.endswith(
            "Output the PROTECTED CORE block above verbatim, followed by your summary "
            "under a '--- Previous Conversation Summary ---' header."
        )

    @pytest.mark.parametrize("decisions", ["", "   ", "\n\n"])
    def test_blank_decisions_render_placeholder(self, prompts, decisions):
        text = prompts["protected_core_compress"](
            goal="G", constraints="C1", decisions=decisions, conversation="c"
        )

        assert "Key Decisions Made:\n  (none yet)\n" in text

    def test_blank_constraints_render_no_bullets(self, prompts):
        text = prompts["protected_core_compress"](
            goal="G", constraints=" , ", decisions="D1", conversation="c"
        )

        assert "Hard Constraints (MUST FOLLOW):\n\n\nKey Decisions Made:\n  - D1\n" in text
