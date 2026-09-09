"""
Smoke tests - basic sanity checks that don't require API keys.
These tests verify package installation and basic functionality.
"""

import pytest
import sys


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
    """Test that SelectiveSalienceCompressor can be imported and instantiated."""
    from selective_salience import SelectiveSalienceCompressor
    
    compressor = SelectiveSalienceCompressor()
    assert compressor is not None
    assert hasattr(compressor, 'compress')
    assert hasattr(compressor, 'initialize')


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
