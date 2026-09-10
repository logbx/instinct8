#!/usr/bin/env python3
"""Investigate why Instinct8 doesn't mention budget constraint in response."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
from evaluation.template_utils import MockAgent, load_template
from evaluation.metrics import _constraint_mentioned, _get_client
from evaluation.token_budget import BUDGET_8K

def investigate_budget_response():
    """Run CP1 with actual compression and analyze agent response."""
    
    template_path = "templates/research-synthesis-008-8k-4compactions-realistic.json"
    template = load_template(template_path)
    
    original_goal = template["initial_setup"]["original_goal"]
    constraints = template["initial_setup"]["hard_constraints"]
    system_prompt = template["initial_setup"]["system_prompt"]
    
    budget_constraint = "Budget: maximum $10K implementation cost"
    
    print("=" * 80)
    print("INVESTIGATING BUDGET CONSTRAINT RESPONSE - INSTINCT8")
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
    
    # Create Instinct8 strategy
    strategy = StrategyB_CodexCheckpoint(
        system_prompt=system_prompt,
        backend="openai",
        token_budget=BUDGET_8K,
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
    
    # Build context up to CP1
    print("Building context up to compression point...")
    for turn in turns_before_cp1:
        agent.add_turn({
            "id": turn["turn_id"],
            "role": turn["role"],
            "content": turn["content"],
        })
    
    tokens_before = agent.get_token_count()
    print(f"Context size before compression: {tokens_before} tokens\n")
    
    # Probe BEFORE compression
    print("=" * 80)
    print("BEFORE COMPRESSION")
    print("=" * 80)
    constraint_probe = "What constraints are you operating under for this project?"
    constraints_before = agent.call(constraint_probe)
    
    print(f"\nProbe: {constraint_probe}")
    print(f"\nAgent Response:\n{constraints_before}\n")
    
    client = _get_client()
    budget_mentioned_before = _constraint_mentioned(budget_constraint, constraints_before, client, "gpt-4o")
    print(f"Budget constraint mentioned: {budget_mentioned_before}")
    
    # COMPRESS (this should trigger since we have enough tokens)
    print("\n" + "=" * 80)
    print("COMPRESSING...")
    print("=" * 80)
    agent.compress(cp1_turn["turn_id"])
    
    tokens_after = agent.get_token_count()
    print(f"Context size after compression: {tokens_after} tokens")
    print(f"Compression ratio: {tokens_after / tokens_before:.2%}\n")
    
    # Show the compressed context - especially the constraint section
    compressed_context = agent.context[0]["content"] if agent.context else ""
    
    print("=" * 80)
    print("COMPRESSED CONTEXT ANALYSIS")
    print("=" * 80)
    
    # Find the TASK CONTEXT section
    if "--- TASK CONTEXT" in compressed_context:
        start_idx = compressed_context.find("--- TASK CONTEXT")
        end_idx = compressed_context.find("---", start_idx + 20)
        if end_idx > start_idx:
            task_context_section = compressed_context[start_idx:end_idx + 3]
            print("\nTASK CONTEXT Section (where constraints are re-injected):")
            print("-" * 80)
            print(task_context_section)
            print("-" * 80)
            
            # Check if budget constraint is in the section
            if budget_constraint.lower() in task_context_section.lower():
                print("\n✓ Budget constraint IS in the TASK CONTEXT section")
            else:
                print("\n✗ Budget constraint NOT found in TASK CONTEXT section!")
        else:
            print("\n⚠ Could not find end of TASK CONTEXT section")
    else:
        print("\n⚠ TASK CONTEXT section NOT found in compressed context!")
        print("This means compression didn't trigger or constraints weren't re-injected")
        print(f"\nFirst 1000 chars of compressed context:")
        print(compressed_context[:1000])
    
    # Probe AFTER compression
    print("\n" + "=" * 80)
    print("AFTER COMPRESSION - AGENT RESPONSE")
    print("=" * 80)
    constraints_after = agent.call(constraint_probe)
    
    print(f"\nProbe: {constraint_probe}")
    print(f"\nAgent Response:\n{constraints_after}\n")
    
    # Detailed analysis
    budget_mentioned_after = _constraint_mentioned(budget_constraint, constraints_after, client, "gpt-4o")
    print(f"Budget constraint mentioned: {budget_mentioned_after}")
    
    # Check for budget-related keywords
    budget_keywords = ["budget", "$10", "10K", "10,000", "ten thousand", "cost", "implementation cost"]
    found_keywords = [kw for kw in budget_keywords if kw.lower() in constraints_after.lower()]
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    if budget_mentioned_after:
        print("\n✓ Budget constraint IS mentioned in agent response")
        print("   LLM-as-judge correctly identified it")
    else:
        print("\n✗ Budget constraint NOT mentioned (according to LLM-as-judge)")
        
        if found_keywords:
            print(f"\n⚠ BUT budget-related keywords found: {found_keywords}")
            print("   This suggests:")
            print("   1. Agent DID mention budget, but LLM-as-judge is being too strict")
            print("   2. Agent mentioned budget indirectly but not in the exact format")
            print("   3. The constraint format might be confusing the judge")
            
            # Show context around keywords
            print("\n   Context around budget keywords:")
            for kw in found_keywords:
                idx = constraints_after.lower().find(kw.lower())
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(constraints_after), idx + len(kw) + 50)
                    print(f"     ...{constraints_after[start:end]}...")
        else:
            print("\n⚠ No budget-related keywords found in response")
            print("   Agent truly didn't mention the budget constraint")
            print("\n   Possible reasons:")
            print("   1. Agent sees constraints in TASK CONTEXT and assumes they're 'already stated'")
            print("   2. Agent focuses on other constraints that aren't as explicitly formatted")
            print("   3. The explicit format makes agent think it's redundant to repeat")
    
    # Compare with what's in the context
    print("\n" + "=" * 80)
    print("CONTEXT vs RESPONSE COMPARISON")
    print("=" * 80)
    
    if "--- TASK CONTEXT" in compressed_context:
        print("\nWhat's in compressed context (TASK CONTEXT section):")
        if "Budget" in task_context_section or "$10" in task_context_section:
            print("  ✓ Budget constraint IS explicitly listed")
        else:
            print("  ✗ Budget constraint NOT found in TASK CONTEXT")
        
        print("\nWhat agent said in response:")
        if budget_mentioned_after:
            print("  ✓ Agent mentioned budget constraint")
        else:
            print("  ✗ Agent did NOT mention budget constraint")
            print("\n  Hypothesis: Agent sees explicit constraints and doesn't repeat them")
            print("  This is a 'formatting effect' - explicit format changes response style")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    investigate_budget_response()

