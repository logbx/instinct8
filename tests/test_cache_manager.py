"""
Unit tests for evaluation/cache_manager.py.

Covers path layout/sanitization, cache presence checks, compression and A-mem
save/load round-trips, and cleanup/listing helpers. All filesystem work happens
under pytest's tmp_path, and the A-mem memory system is a minimal in-file stub,
so no API keys, network, embeddings, or real retrievers are exercised.
"""

import pickle

import pytest

from evaluation.cache_manager import CacheManager


class FakeRetriever:
    def __init__(self, label="original", fail_save=False):
        self.label = label
        self.fail_save = fail_save
        self.saved_to = None

    def save(self, retriever_path, embeddings_path):
        if self.fail_save:
            raise RuntimeError("boom")
        self.saved_to = (retriever_path, embeddings_path)
        with open(retriever_path, "wb") as f:
            pickle.dump(self.label, f)

    def load(self, retriever_path, embeddings_path):
        restored = FakeRetriever(label="loaded")
        restored.loaded_from = (retriever_path, embeddings_path)
        return restored

    def load_from_local_memory(self, memories, model_name):
        rebuilt = FakeRetriever(label="rebuilt")
        rebuilt.rebuilt_from = (memories, model_name)
        return rebuilt


class FakeMemorySystem:
    def __init__(self, memories=None, retriever=None):
        self.memories = memories if memories is not None else {}
        self.retriever = retriever if retriever is not None else FakeRetriever()


@pytest.fixture
def cache(tmp_path):
    return CacheManager(str(tmp_path), "agent", "backend")


def _touch_all(cache, sample_id):
    paths = [
        cache.get_memory_cache_path(sample_id),
        cache.get_retriever_cache_path(sample_id),
        cache.get_embeddings_cache_path(sample_id),
        cache.get_context_cache_path(sample_id),
    ]
    for path in paths:
        path.write_bytes(b"x")
    return paths


class TestInit:
    def test_creates_subdir_from_agent_and_backend(self, tmp_path):
        cm = CacheManager(str(tmp_path), "agent", "backend")
        assert cm.cache_dir == tmp_path / "agent_backend"
        assert cm.cache_dir.is_dir()

    def test_sanitizes_slashes_and_backslashes(self, tmp_path):
        cm = CacheManager(str(tmp_path), "org/model", "a\\b/c")
        assert cm.cache_dir == tmp_path / "org_model_a_b_c"
        assert cm.cache_dir.parent == tmp_path

    def test_empty_backend_drops_trailing_underscore(self, tmp_path):
        cm = CacheManager(str(tmp_path), "agent")
        assert cm.cache_dir == tmp_path / "agent"

    def test_creates_nested_missing_parents(self, tmp_path):
        cm = CacheManager(str(tmp_path / "deep" / "nested"), "agent", "b")
        assert cm.cache_dir == tmp_path / "deep" / "nested" / "agent_b"
        assert cm.cache_dir.is_dir()

    def test_reinit_on_existing_dir_preserves_files(self, tmp_path):
        cm = CacheManager(str(tmp_path), "agent", "backend")
        cm.save_compression_state("s1", [], 0)
        cm2 = CacheManager(str(tmp_path), "agent", "backend")
        assert cm2.has_compression_cache("s1")


class TestPathContracts:
    @pytest.mark.parametrize(
        "method, filename",
        [
            ("get_memory_cache_path", "memory_cache_sample_42.pkl"),
            ("get_retriever_cache_path", "retriever_cache_sample_42.pkl"),
            ("get_embeddings_cache_path", "retriever_embeddings_sample_42.npy"),
            ("get_context_cache_path", "context_cache_sample_42.pkl"),
        ],
    )
    def test_filename_contract(self, cache, method, filename):
        path = getattr(cache, method)("42")
        assert path == cache.cache_dir / filename

    def test_paths_do_not_create_files(self, cache):
        cache.get_memory_cache_path("s1")
        cache.get_context_cache_path("s1")
        assert list(cache.cache_dir.iterdir()) == []


class TestHasCache:
    def test_all_false_when_empty(self, cache):
        assert cache.has_cache("s1") is False
        assert cache.has_amem_cache("s1") is False
        assert cache.has_compression_cache("s1") is False

    def test_memory_file_only(self, cache):
        cache.get_memory_cache_path("s1").write_bytes(b"x")
        assert cache.has_cache("s1") is True
        assert cache.has_amem_cache("s1") is True
        assert cache.has_compression_cache("s1") is False

    def test_context_file_only(self, cache):
        cache.get_context_cache_path("s1").write_bytes(b"x")
        assert cache.has_cache("s1") is True
        assert cache.has_amem_cache("s1") is False
        assert cache.has_compression_cache("s1") is True

    def test_retriever_or_embeddings_alone_do_not_count(self, cache):
        cache.get_retriever_cache_path("s1").write_bytes(b"x")
        cache.get_embeddings_cache_path("s1").write_bytes(b"x")
        assert cache.has_cache("s1") is False
        assert cache.has_amem_cache("s1") is False

    def test_scoped_to_sample_id(self, cache):
        cache.get_context_cache_path("s1").write_bytes(b"x")
        assert cache.has_cache("s2") is False


class TestCompressionState:
    def test_round_trip(self, cache):
        context = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "meta": {"n": 1}},
        ]
        cache.save_compression_state("s1", context, 123)

        assert cache.has_compression_cache("s1")
        assert cache.load_compression_state("s1") == {"context": context, "tokens": 123}

    def test_empty_context_round_trip(self, cache):
        cache.save_compression_state("s1", [], 0)
        assert cache.load_compression_state("s1") == {"context": [], "tokens": 0}

    def test_save_overwrites_previous(self, cache):
        cache.save_compression_state("s1", [{"role": "user", "content": "a"}], 1)
        cache.save_compression_state("s1", [{"role": "user", "content": "b"}], 2)
        state = cache.load_compression_state("s1")
        assert state["tokens"] == 2
        assert state["context"][0]["content"] == "b"

    def test_missing_returns_none(self, cache):
        assert cache.load_compression_state("missing") is None

    def test_corrupted_pickle_returns_none(self, cache, capsys):
        cache.get_context_cache_path("s1").write_bytes(b"not a pickle")
        assert cache.load_compression_state("s1") is None
        assert "Error loading compression state" in capsys.readouterr().out

    def test_empty_file_returns_none(self, cache):
        cache.get_context_cache_path("s1").write_bytes(b"")
        assert cache.load_compression_state("s1") is None


class TestAMemState:
    def test_save_writes_memories_and_delegates_retriever(self, cache):
        system = FakeMemorySystem(memories={"m1": "note one", "m2": "note two"})
        cache.save_amem_state("s1", system)

        with open(cache.get_memory_cache_path("s1"), "rb") as f:
            assert pickle.load(f) == {"m1": "note one", "m2": "note two"}
        assert system.retriever.saved_to == (
            str(cache.get_retriever_cache_path("s1")),
            str(cache.get_embeddings_cache_path("s1")),
        )
        assert cache.has_amem_cache("s1")

    def test_save_survives_retriever_failure(self, cache, capsys):
        system = FakeMemorySystem(
            memories={"m1": "x"}, retriever=FakeRetriever(fail_save=True)
        )
        cache.save_amem_state("s1", system)

        assert cache.has_amem_cache("s1")
        assert not cache.get_retriever_cache_path("s1").exists()
        assert "Could not save retriever state" in capsys.readouterr().out

    def test_load_missing_returns_false_and_leaves_system(self, cache):
        original = FakeRetriever()
        system = FakeMemorySystem(memories={"keep": "me"}, retriever=original)

        assert cache.load_amem_state("missing", system) is False
        assert system.memories == {"keep": "me"}
        assert system.retriever is original

    def test_load_with_retriever_cache_uses_load(self, cache):
        cache.save_amem_state("s1", FakeMemorySystem(memories={"m1": "note"}))

        target = FakeMemorySystem()
        assert cache.load_amem_state("s1", target) is True
        assert target.memories == {"m1": "note"}
        assert target.retriever.label == "loaded"
        assert target.retriever.loaded_from == (
            str(cache.get_retriever_cache_path("s1")),
            str(cache.get_embeddings_cache_path("s1")),
        )

    def test_load_without_retriever_cache_rebuilds_from_memories(self, cache):
        cache.save_amem_state(
            "s1",
            FakeMemorySystem(memories={"m1": "note"}, retriever=FakeRetriever(fail_save=True)),
        )

        target = FakeMemorySystem()
        assert cache.load_amem_state("s1", target) is True
        assert target.retriever.label == "rebuilt"
        assert target.retriever.rebuilt_from == ({"m1": "note"}, "all-MiniLM-L6-v2")

    def test_load_corrupted_memory_returns_false(self, cache, capsys):
        cache.get_memory_cache_path("s1").write_bytes(b"not a pickle")
        original = FakeRetriever()
        system = FakeMemorySystem(memories={"keep": "me"}, retriever=original)

        assert cache.load_amem_state("s1", system) is False
        assert system.memories == {"keep": "me"}
        assert system.retriever is original
        assert "Error loading A-mem state" in capsys.readouterr().out


class TestClearSample:
    def test_removes_all_four_paths(self, cache):
        paths = _touch_all(cache, "s1")
        cache.clear_sample("s1")
        assert not any(p.exists() for p in paths)
        assert cache.has_cache("s1") is False

    def test_leaves_other_samples(self, cache):
        _touch_all(cache, "s1")
        other = _touch_all(cache, "s2")
        cache.clear_sample("s1")
        assert all(p.exists() for p in other)

    def test_partial_and_missing_are_noops(self, cache):
        cache.get_context_cache_path("s1").write_bytes(b"x")
        cache.clear_sample("s1")
        cache.clear_sample("never-cached")
        assert list(cache.cache_dir.iterdir()) == []


class TestClearAll:
    def test_empties_and_recreates_dir(self, cache):
        _touch_all(cache, "s1")
        _touch_all(cache, "s2")
        (cache.cache_dir / "unrelated.txt").write_text("x")

        cache.clear_all()

        assert cache.cache_dir.is_dir()
        assert list(cache.cache_dir.iterdir()) == []

    def test_cache_usable_after_clear(self, cache):
        cache.save_compression_state("s1", [], 1)
        cache.clear_all()
        cache.save_compression_state("s2", [], 2)
        assert cache.list_cached_samples() == ["s2"]

    def test_does_not_touch_sibling_dirs(self, tmp_path):
        a = CacheManager(str(tmp_path), "a")
        b = CacheManager(str(tmp_path), "b")
        b.save_compression_state("s1", [], 0)
        a.clear_all()
        assert b.has_compression_cache("s1")


class TestListCachedSamples:
    def test_empty(self, cache):
        assert cache.list_cached_samples() == []

    def test_dedupes_across_cache_types_and_sorts(self, cache):
        _touch_all(cache, "b")
        cache.get_context_cache_path("a").write_bytes(b"x")
        cache.get_memory_cache_path("c").write_bytes(b"x")
        assert cache.list_cached_samples() == ["a", "b", "c"]

    def test_sort_is_lexicographic(self, cache):
        for sample_id in ["2", "10", "1"]:
            cache.get_context_cache_path(sample_id).write_bytes(b"x")
        assert cache.list_cached_samples() == ["1", "10", "2"]

    def test_ignores_embeddings_and_unrelated_files(self, cache):
        cache.get_embeddings_cache_path("emb").write_bytes(b"x")
        (cache.cache_dir / "notes.pkl").write_bytes(b"x")
        (cache.cache_dir / "context_cache_sample_txt.json").write_bytes(b"x")
        assert cache.list_cached_samples() == []
