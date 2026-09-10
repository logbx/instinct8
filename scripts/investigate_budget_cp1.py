#!/usr/bin/env python3
"""Investigate why Instinct8 loses budget constraint at CP1."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
from evaluation.template_utils import MockAgent, load_template
from evaluation.metrics import measure_constraint_recall, _constraint_mentioned, _get_client

def investigate_cp1():
    """Run CP1 and capture actual agent responses."""
    
    template_path = "templates/research-synthesis-008-8k-4compactions-realistic.json"
    template = load_template(template_path)
    
    original_goal = template["initial_setup"]["original_goal"]
    constraints = template["initial_setup"]["hard_constraints"]
    system_prompt = template["initial_setup"]["system_prompt"]
    
    budget_constraint = "Budget: maximum $10K implementation cost"
    
    print("=" * 80)
    print("INVESTIGATING BUDGET CONSTRAINT LOSS AT CP1")
    print("=" * 80)
    
    # Find CP1
    turns = template["turns"]
    cp1_turn = None
    turns_before_cp1 = []
    
    for turn in turns:
        if turn.get("is_compression_point", False) and cp1_turn is None:
            cp1_turn = turn
            break
        turns_before_cp1.append(turn)
    
    if not cp1_turn:
        print("ERROR: No compression point found")
        return
    
    print(f"\nCompression Point 1 at Turn {cp1_turn['turn_id']}\n")
    
    # Test Instinct8
    print("=" * 80)
    print("TESTING INSTINCT8")
    print("=" * 80)
    
    strategy = StrategyB_CodexCheckpoint(
        system_prompt=system_prompt,
        backend="openai",
        use_goal_preservation=True
    )
    strategy.initialize(original_goal, constraints)
    
    agent = MockAgent(
        strategy=strategy,
        system_prompt=system_prompt,
        original_goal=original_goal,
        constraints=constraints,
        model="gpt-4o"
    )
    
    # Build context
    for turn in turns_before_cp1:
        agent.add_turn({
            "id": turn["turn_id"],
            "role": turn["role"],
            "content": turn["content"],
        })
    
    # Probe BEFORE
    constraint_probe = "What constraints are you operating under for this project?"
    constraints_before = agent.call(constraint_probe)
    
    print("\nBEFORE COMPRESSION:")
    print(f"Agent response:\n{constraints_before}\n")
    
    # Check budget constraint
    client = _get_client()
    budget_mentioned_before = _constraint_mentioned(budget_constraint, constraints_before, client, "gpt-4o")
    print(f"Budget constraint mentioned: {budget_mentioned_before}")
    
    # COMPRESS
    print("\n" + "=" * 80)
    print("COMPRESSING...")
    print("=" * 80)
    agent.compress(cp1_turn["turn_id"])
    
    # Show what's in the compressed context
    compressed_context = agent.context[0]["content"] if agent.context else ""
    print("\nCompressed context (showing constraint section):")
    if "--- TASK CONTEXT" in compressed_context:
        start = compressed_context.find("--- TASK CONTEXT")
        end = compressed_context.find("---", start + 20)
        if end > start:
            print(compressed_context[start:end+3])
        else:
            print(compressed_context[start:start+500])
    else:
        print("⚠ TASK CONTEXT section not found in compressed context!")
        print(f"First 500 chars: {compressed_context[:500]}")
    
    # Probe AFTER
    print("\n" + "=" * 80)
    print("AFTER COMPRESSION")
    print("=" * 80)
    constraints_after = agent.call(constraint_probe)
    
    print(f"\nAgent response:\n{constraints_after}\n")
    
    # Check budget constraint
    budget_mentioned_after = _constraint_mentioned(budget_constraint, constraints_after, client, "gpt-4o")
    print(f"Budget constraint mentioned: {budget_mentioned_after}")
    
    # Analyze why it might not be mentioned
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if not budget_mentioned_after:
        print("\n⚠ Budget constraint NOT mentioned in agent response")
        print("\nPossible reasons:")
        print("1. Agent sees explicit constraints in context and doesn't repeat them")
        print("2. Agent response format doesn't match what LLM-as-judge expects")
        print("3. The explicit format might make agent think it's 'already stated'")
        
        # Check if budget-related words appear
        budget_keywords = ["budget", "$10", "10K", "cost", "10,000"]
        found_keywords = [kw for kw in budget_keywords if kw.lower() in constraints_after.lower()]
        if found_keywords:
            print(f"\n⚠ But budget-related keywords found: {found_keywords}")
            print("   This suggests the LLM-as-judge might be too strict")
        else:
            print("\n⚠ No budget-related keywords found in response")
            print("   Agent truly didn't mention budget")
    else:
        print("\n✓ Budget constraint IS mentioned")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    investigate_cp1()

