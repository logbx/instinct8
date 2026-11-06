"""
Smoke tests - basic sanity checks that don't require API keys.
These tests verify package installation and basic functionality.
"""

import ast
import pytest
import sys
from pathlib import Path


def test_python_version():
    """Verify Python version meets requirements."""
    assert sys.version_info >= (3, 9), "Python 3.9+ is required"


def test_package_imports():
    """Test that core packages can be imported."""
    # Core package
    import selective_salience
    assert selective_salience is not None
    
    # Strategies
    import strategies
    assert strategies is not None
    
    # Evaluation
    import evaluation
    assert evaluation is not None


def test_compressor_class_available():
    """Test that SelectiveSalienceCompressor class is available and has expected methods."""
    from selective_salience import SelectiveSalienceCompressor
    
    assert SelectiveSalienceCompressor is not None
    assert hasattr(SelectiveSalienceCompressor, 'compress')
    assert hasattr(SelectiveSalienceCompressor, 'initialize')
    assert hasattr(SelectiveSalienceCompressor, 'reset')


def test_strategy_base_available():
    """Test that strategy base class is available."""
    from strategies.strategy_base import CompressionStrategy
    assert CompressionStrategy is not None


def test_cli_entry_points():
    """Test that CLI entry points are defined."""
    import selective_salience.instinct8_cli
    import selective_salience.agent_cli
    
    assert hasattr(selective_salience.instinct8_cli, 'main')
    assert hasattr(selective_salience.agent_cli, 'main')


def test_dependencies_present():
    """Test that critical dependencies are installed."""
    import openai
    import numpy
    import sentence_transformers
    import sklearn
    import tiktoken
    
    # Verify versions meet minimum requirements
    import pkg_resources
    
    # Check sentence-transformers is patched for CVE-2026-68770
    st_version = pkg_resources.get_distribution("sentence-transformers").version
    major, minor = map(int, st_version.split('.')[:2])
    assert (major > 5) or (major == 5 and minor >= 6), \
        f"sentence-transformers {st_version} is vulnerable to CVE-2026-68770. Upgrade to 5.6.0+"


def test_amem_does_not_import_removed_ast_nodes():
    """A-mem must stay importable on Python 3.14, where these ast aliases were removed."""
    removed = {"Str", "Num", "Bytes", "NameConstant", "Ellipsis"}
    amem_dir = Path(__file__).resolve().parent.parent / "data" / "A-mem"
    offenders = []
    for path in sorted(amem_dir.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.module == "ast":
                offenders += [f"{path.name}:{node.lineno} {a.name}" for a in node.names if a.name in removed]
    assert not offenders, f"Removed ast names imported: {offenders}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
