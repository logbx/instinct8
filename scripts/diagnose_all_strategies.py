#!/usr/bin/env python3
"""
Diagnostic script for all compression strategies.

Tests each strategy to verify:
- Can be imported
- Can be instantiated
- Implements all required methods
- Can initialize with goal/constraints
- Can compress a simple context
- Handles goal updates
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies import (
    StrategyB_CodexCheckpoint,
    StrategyD_AMemStyle,
    StrategyF_ProtectedCore,
    StrategyH_Keyframe,
    StrategyI_AMemProtectedCore,
)
from strategies.strategy_base import CompressionStrategy


def test_strategy_import(strategy_class, name: str) -> tuple[bool, str]:
    """Test if strategy can be imported."""
    try:
        return True, f"✓ {name} imported successfully"
    except Exception as e:
        return False, f"✗ {name} import failed: {e}"


def test_strategy_instantiation(strategy_class, name: str, **kwargs) -> tuple[bool, str, object, bool]:
    """Test if strategy can be instantiated.
    
    Returns: (success, message, strategy_instance, requires_api_key)
    """
    try:
        strategy = strategy_class(**kwargs)
        return True, f"✓ {name} instantiated successfully", strategy, False
    except Exception as e:
        error_msg = str(e).lower()
        requires_api = "api key" in error_msg or "openai" in error_msg or "anthropic" in error_msg
        if requires_api:
            return False, f"⚠ {name} requires API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)", None, True
        else:
            return False, f"✗ {name} instantiation failed: {e}", None, False


def test_required_methods(strategy: CompressionStrategy, name: str) -> list[tuple[bool, str]]:
    """Test if strategy implements all required methods."""
    results = []
    required_methods = ['initialize', 'update_goal', 'compress', 'name']
    
    for method_name in required_methods:
        if hasattr(strategy, method_name):
            if callable(getattr(strategy, method_name)):
                results.append((True, f"  ✓ {method_name}() implemented"))
            else:
                results.append((False, f"  ✗ {method_name} exists but is not callable"))
        else:
            results.append((False, f"  ✗ {method_name}() missing"))
    
    return results


def test_initialize(strategy: CompressionStrategy, name: str) -> tuple[bool, str]:
    """Test strategy initialization."""
    try:
        strategy.initialize(
            original_goal="Test goal: Build a simple web app",
            constraints=["Use Python", "Must be fast"]
        )
        return True, f"✓ {name} initialize() works"
    except Exception as e:
        return False, f"✗ {name} initialize() failed: {e}"


def test_compress(strategy: CompressionStrategy, name: str) -> tuple[bool, str]:
    """Test strategy compression with simple context."""
    try:
        context = [
            {"id": 1, "role": "user", "content": "Hello, let's build a web app"},
            {"id": 2, "role": "assistant", "content": "Sure! What kind of web app?"},
            {"id": 3, "role": "user", "content": "A todo list app"},
        ]
        result = strategy.compress(context, trigger_point=3)
        
        if isinstance(result, str) and len(result) > 0:
            return True, f"✓ {name} compress() works (output: {len(result)} chars)"
        else:
            return False, f"✗ {name} compress() returned invalid result: {type(result)}"
    except Exception as e:
        return False, f"✗ {name} compress() failed: {e}"


def test_update_goal(strategy: CompressionStrategy, name: str) -> tuple[bool, str]:
    """Test strategy goal update."""
    try:
        strategy.update_goal("Updated goal: Build a mobile app", rationale="User changed requirements")
        return True, f"✓ {name} update_goal() works"
    except Exception as e:
        return False, f"✗ {name} update_goal() failed: {e}"


def test_name(strategy: CompressionStrategy, name: str) -> tuple[bool, str]:
    """Test strategy name method."""
    try:
        strategy_name = strategy.name()
        if isinstance(strategy_name, str) and len(strategy_name) > 0:
            return True, f"✓ {name} name() returns: '{strategy_name}'"
        else:
            return False, f"✗ {name} name() returned invalid: {strategy_name}"
    except Exception as e:
        return False, f"✗ {name} name() failed: {e}"


def diagnose_strategy(strategy_class, name: str, **init_kwargs) -> dict:
    """Run full diagnostic on a strategy."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {name}")
    print(f"{'='*80}")
    
    results = {
        'name': name,
        'imported': False,
        'instantiated': False,
        'methods_ok': False,
        'initialize_ok': False,
        'compress_ok': False,
        'update_goal_ok': False,
        'name_ok': False,
        'errors': [],
        'warnings': [],
    }
    
    # Test import (already done, but check)
    results['imported'] = True
    print(f"✓ {name} imported")
    
    # Test instantiation
    success, msg, strategy, requires_api = test_strategy_instantiation(strategy_class, name, **init_kwargs)
    print(msg)
    if not success:
        if requires_api:
            results['warnings'].append(msg)
            results['requires_api_key'] = True
            print(f"  (Note: This strategy requires an API key to run)")
        else:
            results['errors'].append(msg)
        return results
    results['instantiated'] = True
    results['requires_api_key'] = False
    
    # Test required methods
    method_results = test_required_methods(strategy, name)
    all_methods_ok = all(r[0] for r in method_results)
    for success, msg in method_results:
        print(msg)
        if not success:
            results['errors'].append(msg)
    results['methods_ok'] = all_methods_ok
    
    # Test initialize
    success, msg = test_initialize(strategy, name)
    print(msg)
    if success:
        results['initialize_ok'] = True
    else:
        results['errors'].append(msg)
    
    # Test name
    success, msg = test_name(strategy, name)
    print(msg)
    if success:
        results['name_ok'] = True
    else:
        results['errors'].append(msg)
    
    # Test update_goal
    success, msg = test_update_goal(strategy, name)
    print(msg)
    if success:
        results['update_goal_ok'] = True
    else:
        results['warnings'].append(msg)  # update_goal is optional for some strategies
    
    # Test compress
    success, msg = test_compress(strategy, name)
    print(msg)
    if success:
        results['compress_ok'] = True
    else:
        results['errors'].append(msg)
    
    return results


def main():
    """Run diagnostics on all strategies."""
    print("="*80)
    print("COMPRESSION STRATEGIES DIAGNOSTIC")
    print("="*80)
    
    # Define all strategies with their initialization parameters
    strategies_to_test = [
        (StrategyB_CodexCheckpoint, "Strategy B - Codex Checkpoint", {}),
        (StrategyD_AMemStyle, "Strategy D - A-MEM Style", {}),
        (StrategyF_ProtectedCore, "Strategy F - Protected Core", {}),
        (StrategyH_Keyframe, "Strategy H - Keyframe Compression", {}),
        (StrategyI_AMemProtectedCore, "Strategy I - A-MEM + Protected Core", {}),
    ]
    
    all_results = []
    
    for strategy_class, name, init_kwargs in strategies_to_test:
        try:
            results = diagnose_strategy(strategy_class, name, **init_kwargs)
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ FATAL ERROR testing {name}: {e}")
            all_results.append({
                'name': name,
                'imported': False,
                'errors': [f"Fatal error: {e}"],
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for results in all_results:
        name = results['name']
        errors = results.get('errors', [])
        warnings = results.get('warnings', [])
        
        if not results.get('imported', False):
            status = "✗ NOT IMPORTED"
        elif errors:
            status = f"✗ HAS ERRORS ({len(errors)} error(s))"
        elif warnings:
            status = f"⚠ HAS WARNINGS ({len(warnings)} warning(s))"
        else:
            status = "✓ OK"
        
        print(f"{status:20} {name}")
        if errors:
            for error in errors[:3]:  # Show first 3 errors
                print(f"  {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more error(s)")
    
    # Count runnable strategies
    runnable = sum(1 for r in all_results if r.get('compress_ok', False) and not r.get('errors', []))
    requires_api = sum(1 for r in all_results if r.get('requires_api_key', False))
    total = len(all_results)
    
    print(f"\n{'='*80}")
    print(f"RUNNABLE: {runnable}/{total} strategies")
    if requires_api > 0:
        print(f"REQUIRES API KEY: {requires_api} strategies (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print(f"{'='*80}")
    
    # Detailed status
    print("\nDETAILED STATUS:")
    for results in all_results:
        name = results['name']
        status_parts = []
        
        if results.get('requires_api_key', False):
            status_parts.append("🔑 Requires API key")
        if results.get('compress_ok', False):
            status_parts.append("✅ Compress works")
        if results.get('initialize_ok', False):
            status_parts.append("✅ Initialize works")
        if results.get('update_goal_ok', False):
            status_parts.append("✅ Update goal works")
        if results.get('errors', []):
            status_parts.append(f"❌ {len(results['errors'])} error(s)")
        
        status_str = " | ".join(status_parts) if status_parts else "⚠ Status unknown"
        print(f"  {name:40} {status_str}")


if __name__ == "__main__":
    main()

