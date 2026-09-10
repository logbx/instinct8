#!/usr/bin/env python3
"""Capture actual agent responses at CP1 for both strategies."""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
from evaluation.template_utils import MockAgent, load_template, run_single_trial
from evaluation.metrics import _constraint_mentioned, _get_client

def capture_cp1_responses():
    """Run CP1 and capture exact agent responses."""
    
    template_path = "templates/research-synthesis-008-8k-4compactions-realistic.json"
    template = load_template(template_path)
    
    original_goal = template["initial_setup"]["original_goal"]
    constraints = template["initial_setup"]["hard_constraints"]
    system_prompt = template["initial_setup"]["system_prompt"]
    budget_constraint = "Budget: maximum $10K implementation cost"
    
    print("=" * 80)
    print("CAPTURING CP1 RESPONSES - Both Strategies")
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
    
    # Test both strategies
    for strategy_name, use_goal_preservation in [("Baseline", False), ("Instinct8", True)]:
        print(f"\n{'='*80}")
        print(f"TESTING: {strategy_name}")
        print(f"{'='*80}\n")
        
        strategy = StrategyB_CodexCheckpoint(
            system_prompt=system_prompt,
            backend="openai",
            use_goal_preservation=use_goal_preservation
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
        
        tokens_before = agent.get_token_count()
        print(f"Tokens before: {tokens_before}")
        
        # Probe BEFORE
        constraint_probe = "What constraints are you operating under for this project?"
        constraints_before = agent.call(constraint_probe)
        
        client = _get_client()
        budget_before = _constraint_mentioned(budget_constraint, constraints_before, client, "gpt-4o")
        
        print(f"\nBEFORE compression:")
        print(f"  Budget mentioned: {budget_before}")
        print(f"  Response preview: {constraints_before[:150]}...")
        
        # COMPRESS
        print(f"\nCompressing...")
        agent.compress(cp1_turn["turn_id"])
        
        tokens_after = agent.get_token_count()
        print(f"Tokens after: {tokens_after} (ratio: {tokens_after/tokens_before:.2%})")
        
        # Check if TASK CONTEXT is in compressed context
        compressed = agent.context[0]["content"] if agent.context else ""
        has_task_context = "--- TASK CONTEXT" in compressed
        has_budget_in_context = "Budget: maximum $10K" in compressed or "$10K" in compressed
        
        print(f"  Has TASK CONTEXT section: {has_task_context}")
        print(f"  Has budget in context: {has_budget_in_context}")
        
        if has_task_context:
            start = compressed.find("--- TASK CONTEXT")
            end = compressed.find("---", start + 20)
            if end > start:
                task_section = compressed[start:min(start+300, end)]
                print(f"  TASK CONTEXT preview: {task_section}...")
        
        # Probe AFTER
        constraints_after = agent.call(constraint_probe)
        budget_after = _constraint_mentioned(budget_constraint, constraints_after, client, "gpt-4o")
        
        print(f"\nAFTER compression:")
        print(f"  Budget mentioned: {budget_after}")
        print(f"  Full response:\n{constraints_after}\n")
        
        # Check for budget keywords
        budget_keywords = ["budget", "$10", "10K", "10,000", "ten thousand", "implementation cost"]
        found = [kw for kw in budget_keywords if kw.lower() in constraints_after.lower()]
        print(f"  Budget keywords found: {found}")
        
        if not budget_after and found:
            print(f"\n  ⚠ ISSUE: Agent mentioned budget keywords but judge says NOT mentioned")
            print(f"     This suggests the agent didn't mention the SPECIFIC constraint format")
            print(f"     (e.g., said 'budget' but not '$10K implementation cost')")
        
        if has_task_context and not budget_after:
            print(f"\n  ⚠ ISSUE: Budget IS in TASK CONTEXT but agent didn't mention it")
            print(f"     Hypothesis: Agent sees explicit format and doesn't repeat it")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    capture_cp1_responses()

