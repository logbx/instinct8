#!/usr/bin/env python3
"""
Diagnostic script to investigate constraint loss at Compression Point 1.

This script will:
1. Run a single compression point with detailed logging
2. Capture the actual agent responses before/after compression
3. Show how the LLM-as-judge evaluates each constraint
4. Display the compressed context to see what was preserved
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
from evaluation.template_utils import MockAgent
from evaluation.metrics import measure_constraint_recall, _constraint_mentioned, _get_client
from evaluation.token_budget import BUDGET_8K


def diagnose_cp1(template_path: str):
    """Run detailed diagnosis of Compression Point 1."""
    
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    original_goal = template["initial_setup"]["original_goal"]
    constraints = template["initial_setup"]["hard_constraints"]
    system_prompt = template["initial_setup"]["system_prompt"]
    
    print("=" * 80)
    print("CONSTRAINT LOSS DIAGNOSIS - Compression Point 1")
    print("=" * 80)
    print(f"\nOriginal Goal: {original_goal}")
    print(f"\nConstraints to test ({len(constraints)}):")
    for i, c in enumerate(constraints, 1):
        print(f"  {i}. {c}")
    
    # Find first compression point
    turns = template["turns"]
    cp1_turn = None
    turns_before_cp1 = []
    
    for turn in turns:
        if turn.get("is_compression_point", False) and cp1_turn is None:
            cp1_turn = turn
            break
        turns_before_cp1.append(turn)
    
    if not cp1_turn:
        print("ERROR: No compression point found in template")
        return
    
    print(f"\n{'='*80}")
    print(f"Compression Point 1 at Turn {cp1_turn['turn_id']}")
    print(f"{'='*80}\n")
    
    # Test both strategies
    for strategy_name, use_goal_preservation in [("Baseline", False), ("Instinct8", True)]:
        print(f"\n{'='*80}")
        print(f"TESTING: {strategy_name} (use_goal_preservation={use_goal_preservation})")
        print(f"{'='*80}\n")
        
        # Create strategy
        strategy = StrategyB_CodexCheckpoint(
            system_prompt=system_prompt,
            backend="openai",
            token_budget=BUDGET_8K,
            use_goal_preservation=use_goal_preservation
        )
        strategy.initialize(original_goal, constraints)
        
        # Create agent
        agent = MockAgent(
            strategy=strategy,
            system_prompt=system_prompt,
            original_goal=original_goal,
            constraints=constraints,
            model="gpt-4o"
        )
        
        # Add turns up to CP1
        print("Building context up to compression point...")
        for turn in turns_before_cp1:
            agent.add_turn({
                "id": turn["turn_id"],
                "role": turn["role"],
                "content": turn["content"],
            })
        
        # Get context before compression
        context_before = agent.strategy.format_context(agent.context)
        tokens_before = agent.get_token_count()
        
        print(f"\nContext size before compression: {tokens_before} tokens")
        print(f"Context length: {len(context_before)} characters\n")
        
        # Probe BEFORE compression
        print("=" * 80)
        print("PROBING BEFORE COMPRESSION")
        print("=" * 80)
        constraint_probe = "What constraints are you operating under for this project?"
        constraints_before_response = agent.call(constraint_probe)
        
        print(f"\nProbe question: {constraint_probe}")
        print(f"\nAgent response:\n{constraints_before_response}\n")
        
        # Evaluate constraint recall before
        recall_before = measure_constraint_recall(constraints, constraints_before_response)
        print(f"\nConstraint Recall BEFORE: {recall_before:.2%}")
        
        # Check each constraint individually
        print("\nIndividual constraint checks BEFORE:")
        client = _get_client()
        for i, constraint in enumerate(constraints, 1):
            mentioned = _constraint_mentioned(constraint, constraints_before_response, client, "gpt-4o")
            status = "✓" if mentioned else "✗"
            print(f"  {status} Constraint {i}: {constraint}")
        
        # COMPRESS
        print(f"\n{'='*80}")
        print("COMPRESSING...")
        print(f"{'='*80}\n")
        agent.compress(cp1_turn["turn_id"])
        
        # Get context after compression
        # After compression, the context is stored in the strategy's compressed form
        # We need to reconstruct what the agent sees
        context_after = agent.strategy.format_context(agent.context) if agent.context else "(compressed)"
        tokens_after = agent.get_token_count()
        
        print(f"Context size after compression: {tokens_after} tokens")
        print(f"Context length: {len(context_after)} characters")
        print(f"Compression ratio: {tokens_after / tokens_before:.2%}\n")
        
        # Show compressed context (first 2000 chars)
        print("=" * 80)
        print("COMPRESSED CONTEXT (first 2000 chars):")
        print("=" * 80)
        print(context_after[:2000])
        if len(context_after) > 2000:
            print(f"\n... ({len(context_after) - 2000} more characters)")
        print()
        
        # Probe AFTER compression
        print("=" * 80)
        print("PROBING AFTER COMPRESSION")
        print("=" * 80)
        constraints_after_response = agent.call(constraint_probe)
        
        print(f"\nProbe question: {constraint_probe}")
        print(f"\nAgent response:\n{constraints_after_response}\n")
        
        # Evaluate constraint recall after
        recall_after = measure_constraint_recall(constraints, constraints_after_response)
        print(f"\nConstraint Recall AFTER: {recall_after:.2%}")
        
        # Check each constraint individually
        print("\nIndividual constraint checks AFTER:")
        for i, constraint in enumerate(constraints, 1):
            mentioned = _constraint_mentioned(constraint, constraints_after_response, client, "gpt-4o")
            status = "✓" if mentioned else "✗"
            print(f"  {status} Constraint {i}: {constraint}")
        
        # Calculate loss
        constraint_loss = recall_before - recall_after
        print(f"\n{'='*80}")
        print(f"SUMMARY: {strategy_name}")
        print(f"{'='*80}")
        print(f"Constraint Recall BEFORE: {recall_before:.2%}")
        print(f"Constraint Recall AFTER:  {recall_after:.2%}")
        print(f"Constraint LOSS:          {constraint_loss:+.2%}")
        print()
        
        # Compare responses
        print("=" * 80)
        print("RESPONSE COMPARISON")
        print("=" * 80)
        print("\nBEFORE compression response:")
        print(f"  {constraints_before_response[:200]}...")
        print("\nAFTER compression response:")
        print(f"  {constraints_after_response[:200]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_constraint_loss.py <template_path>")
        print("Example: python diagnose_constraint_loss.py templates/research-synthesis-008-8k-4compactions-realistic.json")
        sys.exit(1)
    
    template_path = sys.argv[1]
    if not os.path.exists(template_path):
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    diagnose_cp1(template_path)

