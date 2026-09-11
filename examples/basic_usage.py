#!/usr/bin/env python3
"""
Basic usage example for Selective Salience Compression

This example shows how to use Selective Salience Compression in a simple
LLM agent conversation.

Run after installing the package: pip install -e .
"""

import os
import sys

from selective_salience import SelectiveSalienceCompressor


def main():
    """Example: Compress a conversation while preserving goal-critical info."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    # Initialize compressor
    print("Initializing Selective Salience Compressor...")
    compressor = SelectiveSalienceCompressor()
    
    # Set task goal and constraints
    compressor.initialize(
        original_goal="Research async Python web frameworks and recommend the best one",
        constraints=[
            "Budget maximum $10,000",
            "Timeline: 2 weeks",
            "Must support WebSockets",
            "Must be production-ready"
        ]
    )
    
    # Simulate a conversation
    print("\n=== Simulating Conversation ===")
    context = []
    
    # Turn 1: User asks about frameworks
    context.append({
        "id": 1,
        "role": "user",
        "content": "What async Python web frameworks are available?"
    })
    print(f"Turn 1: {context[-1]['content']}")
    
    # Turn 2: Assistant responds
    context.append({
        "id": 2,
        "role": "assistant",
        "content": "The main async Python web frameworks are FastAPI, Django Async, Quart, and Tornado. FastAPI is built on Starlette and Pydantic, Django Async adds async support to Django, Quart is Flask-like but async, and Tornado is older but battle-tested."
    })
    print(f"Turn 2: {context[-1]['content'][:80]}...")
    
    # Turn 3: User asks about WebSocket support
    context.append({
        "id": 3,
        "role": "user",
        "content": "Which of these support WebSockets? That's a hard requirement."
    })
    print(f"Turn 3: {context[-1]['content']}")
    
    # Turn 4: Assistant responds with WebSocket info
    context.append({
        "id": 4,
        "role": "assistant",
        "content": "FastAPI has excellent WebSocket support built-in. Quart also supports WebSockets natively. Django Async can handle WebSockets but requires additional setup. Tornado has WebSocket support but it's less modern."
    })
    print(f"Turn 4: {context[-1]['content'][:80]}...")
    
    # Turn 5: User asks about budget
    context.append({
        "id": 5,
        "role": "user",
        "content": "What about cost? We have a tight budget of $10K."
    })
    print(f"Turn 5: {context[-1]['content']}")
    
    # Turn 6: Assistant responds
    context.append({
        "id": 6,
        "role": "assistant",
        "content": "All of these frameworks are open-source and free. The main costs would be hosting and infrastructure, not the framework itself. FastAPI and Quart are particularly lightweight, which can reduce hosting costs."
    })
    print(f"Turn 6: {context[-1]['content'][:80]}...")
    
    # Now compress at turn 6
    print("\n=== Compressing Context ===")
    print(f"Context has {len(context)} turns")
    print(f"Compressing up to turn 6...")
    
    compressed = compressor.compress(context, trigger_point=6)
    
    print("\n=== Compressed Context ===")
    print(compressed[:500] + "..." if len(compressed) > 500 else compressed)
    
    print("\n=== Salience Set ===")
    salience = compressor.salience_set
    print(f"Preserved {len(salience)} salient items:")
    for i, item in enumerate(salience, 1):
        print(f"  {i}. {item[:100]}..." if len(item) > 100 else f"  {i}. {item}")
    
    print("\n✅ Compression complete!")


if __name__ == "__main__":
    main()
