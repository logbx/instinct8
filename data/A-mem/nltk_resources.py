"""On-demand provisioning of the NLTK data packages used by A-mem."""

from typing import Set, Tuple

_WORDNET_PATHS = ("corpora/wordnet", "corpora/wordnet.zip/wordnet/")

_ensured: Set[str] = set()


def _resolve(nltk, package: str) -> Tuple[str, Tuple[str, ...]]:
    if package == "punkt":
        # NLTK >= 3.8.2 tokenizes with the pickle-free ``punkt_tab`` tables;
        # older releases still load the ``punkt`` pickles.
        if hasattr(nltk.tokenize.punkt, "PunktTokenizer"):
            return "punkt_tab", ("tokenizers/punkt_tab/english/",)
        return "punkt", ("tokenizers/punkt/english.pickle",)
    if package == "wordnet":
        return "wordnet", _WORDNET_PATHS
    raise ValueError(f"Unknown NLTK data package: {package!r}")


def _is_installed(nltk, paths: Tuple[str, ...]) -> bool:
    for path in paths:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data(*packages: str) -> None:
    """Make sure the given NLTK data packages are available, downloading them on first use.

    Supported packages are ``"punkt"`` (resolved to whichever Punkt data the
    installed NLTK's ``word_tokenize``/``sent_tokenize`` load) and ``"wordnet"``.
    Raises ``LookupError`` if a package is missing and cannot be downloaded.
    """
    import nltk

    for package in packages:
        if package in _ensured:
            continue
        name, paths = _resolve(nltk, package)
        if not _is_installed(nltk, paths):
            try:
                downloaded = nltk.download(name, quiet=True)
            except Exception as e:
                raise LookupError(_missing_message(name)) from e
            if not downloaded or not _is_installed(nltk, paths):
                raise LookupError(_missing_message(name))
        _ensured.add(package)


def _missing_message(name: str) -> str:
    return (
        f"NLTK data package '{name}' is not installed and could not be downloaded. "
        f"Run `python -m nltk.downloader {name}` with network access, or install it "
        f"into a directory listed in the NLTK_DATA environment variable."
    )
