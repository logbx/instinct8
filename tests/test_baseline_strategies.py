"""
Unit tests for evaluation/baseline_strategies.py.

Covers the five comparison baselines plus get_all_baselines() and
get_baseline_by_name(). Baselines are pure, local transformations of a turn
list, so no API keys, network, embeddings, or LLM clients are involved.
Expected output is built with the inherited CompressionStrategy.format_context
so assertions pin turn selection and order rather than re-deriving formatting.
"""

import pytest

from evaluation.baseline_strategies import (
    FirstLastBaseline,
    NoCompressionBaseline,
    RandomTruncationBaseline,
    RecencyOnlyBaseline,
    SlidingWindowBaseline,
    get_all_baselines,
    get_baseline_by_name,
)


def _turns(n, start=1):
    return [
        {"id": i, "role": "user" if i % 2 else "assistant", "content": f"message {i}"}
        for i in range(start, start + n)
    ]


ALL_CLASSES = [
    NoCompressionBaseline,
    RandomTruncationBaseline,
    RecencyOnlyBaseline,
    FirstLastBaseline,
    SlidingWindowBaseline,
]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_initialize_and_update_goal(cls):
    strategy = cls()
    assert strategy.original_goal is None
    assert strategy.constraints == []

    strategy.initialize("Build an API", ["Use FastAPI", "No ORMs"])
    assert strategy.original_goal == "Build an API"
    assert strategy.constraints == ["Use FastAPI", "No ORMs"]

    strategy.update_goal("Build a CLI", rationale="pivot")
    assert strategy.original_goal == "Build a CLI"
    assert strategy.constraints == ["Use FastAPI", "No ORMs"]


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (NoCompressionBaseline(), "NoCompression (Full Context)"),
        (RandomTruncationBaseline(), "RandomTruncation (30%)"),
        (RandomTruncationBaseline(keep_ratio=0.5), "RandomTruncation (50%)"),
        (RecencyOnlyBaseline(), "RecencyOnly (last 10)"),
        (RecencyOnlyBaseline(n_recent=3), "RecencyOnly (last 3)"),
        (FirstLastBaseline(), "FirstLast (5+5)"),
        (FirstLastBaseline(n_first=2, n_last=4), "FirstLast (2+4)"),
        (SlidingWindowBaseline(), "SlidingWindow (20k tokens)"),
        (SlidingWindowBaseline(max_tokens=8000), "SlidingWindow (8k tokens)"),
    ],
)
def test_name(strategy, expected):
    assert strategy.name() == expected


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_turns_after_trigger_point_are_ignored(cls):
    context = _turns(20)
    output = cls().compress(context, trigger_point=8)
    for turn in context[8:]:
        assert f"Turn {turn['id']} " not in output


class TestNoCompression:
    def test_empty_context(self):
        assert NoCompressionBaseline().compress([], trigger_point=0) == ""

    def test_keeps_everything_up_to_trigger_point(self):
        strategy = NoCompressionBaseline()
        context = _turns(6)
        assert strategy.compress(context, trigger_point=4) == strategy.format_context(context[:4])

    def test_output_shape(self):
        context = [
            {"id": 1, "role": "user", "content": "hi"},
            {"id": 2, "role": "assistant", "content": "hello"},
        ]
        output = NoCompressionBaseline().compress(context, trigger_point=2)
        assert output == "Turn 1 (user): hi\nTurn 2 (assistant): hello"


class TestRandomTruncation:
    def test_empty_context(self):
        assert RandomTruncationBaseline(seed=0).compress([], trigger_point=0) == "(No context)"

    @pytest.mark.parametrize(
        "n, keep_ratio, expected_count",
        [(10, 0.3, 3), (10, 0.5, 5), (7, 0.3, 2), (3, 0.1, 1), (4, 1.0, 4)],
    )
    def test_keep_ratio_determines_count(self, n, keep_ratio, expected_count):
        context = _turns(n)
        output = RandomTruncationBaseline(keep_ratio=keep_ratio, seed=42).compress(
            context, trigger_point=n
        )
        lines = output.split("\n")
        assert len(lines) == expected_count == max(1, int(n * keep_ratio))
        all_lines = set(NoCompressionBaseline().format_context(context).split("\n"))
        assert set(lines) <= all_lines

    def test_kept_turns_are_chronological_by_id(self):
        output = RandomTruncationBaseline(keep_ratio=0.5, seed=7).compress(
            _turns(20), trigger_point=20
        )
        ids = [int(line.split()[1]) for line in output.split("\n")]
        assert ids == sorted(ids)

    def test_turn_id_takes_precedence_over_id_for_ordering(self):
        context = [
            {"id": i, "turn_id": 10 - i, "role": "user", "content": f"m{i}"}
            for i in range(1, 6)
        ]
        strategy = RandomTruncationBaseline(keep_ratio=1.0, seed=0)
        output = strategy.compress(context, trigger_point=5)
        assert output == strategy.format_context(list(reversed(context)))

    def test_seed_is_reproducible(self):
        context = _turns(30)
        first = RandomTruncationBaseline(keep_ratio=0.3, seed=123).compress(context, 30)
        second = RandomTruncationBaseline(keep_ratio=0.3, seed=123).compress(context, 30)
        assert first == second


class TestRecencyOnly:
    def test_empty_context(self):
        assert RecencyOnlyBaseline().compress([], trigger_point=0) == "(No context)"

    @pytest.mark.parametrize("n", [1, 3])
    def test_keeps_all_when_within_limit(self, n):
        strategy = RecencyOnlyBaseline(n_recent=3)
        context = _turns(n)
        assert strategy.compress(context, trigger_point=n) == strategy.format_context(context)

    def test_keeps_last_n_in_order(self):
        strategy = RecencyOnlyBaseline(n_recent=3)
        context = _turns(8)
        assert strategy.compress(context, trigger_point=8) == strategy.format_context(context[-3:])


class TestFirstLast:
    def test_empty_context(self):
        assert FirstLastBaseline().compress([], trigger_point=0) == "(No context)"

    @pytest.mark.parametrize("n", [1, 5])
    def test_keeps_all_without_separator_when_within_limit(self, n):
        strategy = FirstLastBaseline(n_first=2, n_last=3)
        context = _turns(n)
        output = strategy.compress(context, trigger_point=n)
        assert output == strategy.format_context(context)
        assert "omitted" not in output

    def test_keeps_first_and_last_with_separator(self):
        strategy = FirstLastBaseline(n_first=2, n_last=3)
        context = _turns(12)
        output = strategy.compress(context, trigger_point=12)

        omitted = len(context) - 2 - 3
        separator = {"role": "system", "content": f"[... {omitted} turns omitted ...]"}
        assert output == strategy.format_context(context[:2] + [separator] + context[-3:])
        assert "Turn ? (system): [... 7 turns omitted ...]" in output


class TestSlidingWindow:
    @staticmethod
    def _turn(turn_id, tokens):
        return {"id": turn_id, "role": "user", "content": "x" * (tokens * 4)}

    def test_empty_context(self):
        assert SlidingWindowBaseline().compress([], trigger_point=0) == "(No context)"

    @pytest.mark.parametrize("text, expected", [("", 0), ("abc", 0), ("abcd", 1), ("x" * 41, 10)])
    def test_estimate_tokens(self, text, expected):
        assert SlidingWindowBaseline()._estimate_tokens(text) == expected

    def test_keeps_most_recent_turns_within_budget(self):
        strategy = SlidingWindowBaseline(max_tokens=25)
        context = [self._turn(i, 10) for i in range(1, 6)]
        assert strategy.compress(context, trigger_point=5) == strategy.format_context(context[-2:])

    def test_budget_is_inclusive(self):
        strategy = SlidingWindowBaseline(max_tokens=30)
        context = [self._turn(i, 10) for i in range(1, 6)]
        assert strategy.compress(context, trigger_point=5) == strategy.format_context(context[-3:])

    def test_stops_at_first_turn_that_overflows(self):
        strategy = SlidingWindowBaseline(max_tokens=50)
        context = [self._turn(1, 5), self._turn(2, 100), self._turn(3, 10), self._turn(4, 10)]
        assert strategy.compress(context, trigger_point=4) == strategy.format_context(context[-2:])

    def test_keeps_last_turn_even_if_over_budget(self):
        strategy = SlidingWindowBaseline(max_tokens=10)
        context = [self._turn(1, 1), self._turn(2, 100)]
        assert strategy.compress(context, trigger_point=2) == strategy.format_context(context[-1:])


def test_get_all_baselines():
    baselines = get_all_baselines()
    assert [type(b) for b in baselines] == ALL_CLASSES
    assert [b.name() for b in baselines] == [
        "NoCompression (Full Context)",
        "RandomTruncation (30%)",
        "RecencyOnly (last 10)",
        "FirstLast (5+5)",
        "SlidingWindow (20k tokens)",
    ]


@pytest.mark.parametrize(
    "key, cls",
    [
        ("no_compression", NoCompressionBaseline),
        ("random", RandomTruncationBaseline),
        ("recency", RecencyOnlyBaseline),
        ("first_last", FirstLastBaseline),
        ("sliding_window", SlidingWindowBaseline),
    ],
)
@pytest.mark.parametrize("transform", [str, str.upper, str.title])
def test_get_baseline_by_name(key, cls, transform):
    assert type(get_baseline_by_name(transform(key))) is cls


@pytest.mark.parametrize("key", ["unknown", "", "first-last", "no compression"])
def test_get_baseline_by_name_unknown(key):
    assert get_baseline_by_name(key) is None
