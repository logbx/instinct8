"""
Check that every documented `run_eval.py` invocation only uses flags the
script's argparse accepts. The script is parsed with `ast` rather than
imported so this stays offline and free of heavy evaluation dependencies.
"""

import ast
import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
RUN_EVAL = REPO_ROOT / "scripts" / "run_eval.py"
CALLER_FILES = [
    REPO_ROOT / "Makefile",
    REPO_ROOT / "docs" / "TESTING.md",
    REPO_ROOT / "docs" / "research" / "CODEX_EVALUATION_SETUP.md",
]


def accepted_flags():
    tree = ast.parse(RUN_EVAL.read_text())
    flags = {"-h", "--help"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    flags.add(arg.value)
    return flags


def invocations():
    for path in CALLER_FILES:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r"run_eval\.py([^`|#\n]*)", line)
            if match:
                yield pytest.param(
                    shlex.split(match.group(1)),
                    id=f"{path.relative_to(REPO_ROOT)}:{lineno}",
                )


def test_argparse_flags_found():
    assert {"--samples", "--runs", "--rigorous"} <= accepted_flags()


@pytest.mark.parametrize("args", list(invocations()))
def test_invocation_uses_accepted_flags(args):
    used = {arg.split("=", 1)[0] for arg in args if arg.startswith("-")}
    assert used <= accepted_flags(), f"unknown run_eval.py flags: {used - accepted_flags()}"
