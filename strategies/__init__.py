# Context Compression Strategies
# This package contains implementations of different compression strategies
# for long-running LLM agents.

# Base imports (no circular dependencies)
from .strategy_base import CompressionStrategy, Turn, ToolCall, ProbeResults

# Strategy imports - lazy to avoid circular dependencies with evaluation module
# Import these directly from their modules when needed to avoid circular imports.
# 
# Example usage:
#   from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
#   from strategies.strategy_h_selective_salience import SelectiveSalienceStrategy
#
# This lazy import pattern prevents circular dependencies between:
# - strategies/__init__.py → strategy modules → evaluation module → strategies (cycle!)
from .strategy_a_naive import StrategyA_NaiveSummarization, create_naive_strategy
from .strategy_b_codex import StrategyB_CodexCheckpoint, create_codex_strategy
from .strategy_d_amem import StrategyD_AMemStyle, create_amem_strategy
from .strategy_f_protected_core import StrategyF_ProtectedCore, create_protected_core_strategy
from .strategy_h_selective_salience import SelectiveSalienceStrategy
from .strategy_h_keyframe import StrategyH_Keyframe, create_keyframe_strategy
from .strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore, create_amem_protected_strategy

__all__ = [
    # Base classes (always available, no circular dependencies)
    "CompressionStrategy",
    "Turn",
    "ToolCall",
    "ProbeResults",
    # Strategy classes (import directly from their modules to avoid circular imports)
    # Strategy A - Naive Summarization
    # "StrategyA_NaiveSummarization",  # from strategies.strategy_a_naive import StrategyA_NaiveSummarization
    # Strategy B - Codex Checkpoint
    # "StrategyB_CodexCheckpoint",  # from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
    # "create_codex_strategy",  # from strategies.strategy_b_codex import create_codex_strategy
    # Strategy D - A-MEM Style
    # "StrategyD_AMemStyle",  # from strategies.strategy_d_amem import StrategyD_AMemStyle
    # "create_amem_strategy",  # from strategies.strategy_d_amem import create_amem_strategy
    # Strategy F - Protected Core + Goal Re-assertion
    # "StrategyF_ProtectedCore",  # from strategies.strategy_f_protected_core import StrategyF_ProtectedCore
    # "create_protected_core_strategy",  # from strategies.strategy_f_protected_core import create_protected_core_strategy
    # Strategy A - Naive Summarization
    "StrategyA_NaiveSummarization",
    "create_naive_strategy",
    # Strategy B - Codex
    "StrategyB_CodexCheckpoint",
    "create_codex_strategy",
    # Strategy D - A-MEM
    "StrategyD_AMemStyle",
    "create_amem_strategy",
    # Strategy F - Protected Core + Goal Re-assertion (Novel)
    "StrategyF_ProtectedCore",
    "create_protected_core_strategy",
    # Strategy H - Selective Salience Compression (Agent-as-Judge)
    # "SelectiveSalienceStrategy",  # from strategies.strategy_h_selective_salience import SelectiveSalienceStrategy
    # Strategy H - Keyframe Compression (alternative implementation)
    # "StrategyH_Keyframe",  # from strategies.strategy_h_keyframe import StrategyH_Keyframe
    # "create_keyframe_strategy",  # from strategies.strategy_h_keyframe import create_keyframe_strategy
    # Strategy I - A-MEM + Protected Core Hybrid
    # "StrategyI_AMemProtectedCore",  # from strategies.strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore
    # "create_amem_protected_strategy",  # from strategies.strategy_i_hybrid_amem_protected import create_amem_protected_strategy
]

