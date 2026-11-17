"""Offline tests for data/A-mem/nltk_resources.py (nltk is faked; no network or nltk_data needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

MODULE_PATH = Path(__file__).parent.parent / "data" / "A-mem" / "nltk_resources.py"


def _make_fake_nltk(installed, download_result=True, modern=True, download_installs=True):
    """Fake ``nltk`` whose data.find succeeds only for paths in ``installed``.

    A successful download installs every path registered under that package name.
    """
    packages = {
        "punkt_tab": ["tokenizers/punkt_tab/english/"],
        "punkt": ["tokenizers/punkt/english.pickle"],
        "wordnet": ["corpora/wordnet.zip/wordnet/"],
    }
    installed = set(installed)

    def find(path):
        if path not in installed:
            raise LookupError(f"Resource {path} not found.")
        return path

    def download(name, quiet=False):
        if isinstance(download_result, BaseException):
            raise download_result
        if download_result and download_installs:
            installed.update(packages[name])
        return download_result

    punkt = SimpleNamespace(PunktTokenizer=object) if modern else SimpleNamespace()
    return SimpleNamespace(
        data=SimpleNamespace(find=MagicMock(side_effect=find)),
        download=MagicMock(side_effect=download),
        tokenize=SimpleNamespace(punkt=punkt),
    )


@pytest.fixture
def load(monkeypatch):
    def _load(fake_nltk):
        monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
        spec = importlib.util.spec_from_file_location("amem_nltk_resources", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


def test_import_does_not_touch_nltk(load):
    fake = _make_fake_nltk(installed=[])
    load(fake)
    fake.download.assert_not_called()
    fake.data.find.assert_not_called()


def test_present_resources_are_not_downloaded(load):
    fake = _make_fake_nltk(installed=["tokenizers/punkt_tab/english/", "corpora/wordnet.zip/wordnet/"])
    module = load(fake)

    module.ensure_nltk_data("punkt", "wordnet")

    fake.download.assert_not_called()


def test_missing_punkt_downloads_punkt_tab_on_modern_nltk(load):
    fake = _make_fake_nltk(installed=[])
    module = load(fake)

    module.ensure_nltk_data("punkt")

    fake.download.assert_called_once_with("punkt_tab", quiet=True)


def test_missing_punkt_downloads_punkt_on_old_nltk(load):
    fake = _make_fake_nltk(installed=[], modern=False)
    module = load(fake)

    module.ensure_nltk_data("punkt")

    fake.download.assert_called_once_with("punkt", quiet=True)


def test_missing_wordnet_is_downloaded(load):
    fake = _make_fake_nltk(installed=[])
    module = load(fake)

    module.ensure_nltk_data("wordnet")

    fake.download.assert_called_once_with("wordnet", quiet=True)


def test_unzipped_wordnet_counts_as_installed(load):
    fake = _make_fake_nltk(installed=["corpora/wordnet"])
    module = load(fake)

    module.ensure_nltk_data("wordnet")

    fake.download.assert_not_called()


def test_ensured_resources_are_cached(load):
    fake = _make_fake_nltk(installed=[])
    module = load(fake)

    module.ensure_nltk_data("punkt")
    module.ensure_nltk_data("punkt")
    find_calls = fake.data.find.call_count
    module.ensure_nltk_data("punkt")

    fake.download.assert_called_once()
    assert fake.data.find.call_count == find_calls


def test_failed_download_raises_clear_error(load):
    fake = _make_fake_nltk(installed=[], download_result=False)
    module = load(fake)

    with pytest.raises(LookupError, match=r"punkt_tab.*python -m nltk\.downloader punkt_tab"):
        module.ensure_nltk_data("punkt")


def test_download_exception_raises_clear_error(load):
    error = OSError("network unreachable")
    fake = _make_fake_nltk(installed=[], download_result=error)
    module = load(fake)

    with pytest.raises(LookupError, match="wordnet") as excinfo:
        module.ensure_nltk_data("wordnet")
    assert excinfo.value.__cause__ is error


def test_download_reporting_success_without_installing_raises(load):
    fake = _make_fake_nltk(installed=[], download_installs=False)
    module = load(fake)

    with pytest.raises(LookupError, match="punkt_tab"):
        module.ensure_nltk_data("punkt")


def test_failed_resource_is_retried_on_next_call(load):
    fake = _make_fake_nltk(installed=[], download_result=False)
    module = load(fake)

    for _ in range(2):
        with pytest.raises(LookupError):
            module.ensure_nltk_data("punkt")

    assert fake.download.call_count == 2


def test_unknown_package_is_rejected(load):
    module = load(_make_fake_nltk(installed=[]))

    with pytest.raises(ValueError, match="bogus"):
        module.ensure_nltk_data("bogus")
