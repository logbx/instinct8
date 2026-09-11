#!/usr/bin/env python3
"""
Codex Integration Example

Shows how to use Codex with Selective Salience Compression instead of
Codex's default compression.

Run after installing the package: pip install -e .
"""

import os
import sys

from selective_salience import Instinct8Agent, create_instinct8_agent


def main():
    """Example: Use Codex with Selective Salience Compression."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    print("=== Instinct8 Agent - Selective Salience Compression ===\n")
    
    # Create Instinct8 agent with Selective Salience Compression
    print("Creating Instinct8 agent with Selective Salience Compression...")
    agent = create_instinct8_agent(
        goal="Build a FastAPI authentication system",
        constraints=[
            "Use JWT tokens",
            "Hash passwords with bcrypt",
            "Support refresh tokens",
            "Must be production-ready"
        ],
        model="gpt-4o",
        compaction_threshold=80000,  # Compress when context exceeds 80K tokens
    )
    
    print("✅ Agent initialized\n")
    
    # Simulate a coding conversation
    print("=== Simulating Coding Conversation ===\n")
    
    # Turn 1: User asks to create login endpoint
    agent.ingest_turn({
        "role": "user",
        "content": "Create a login endpoint that accepts email and password"
    })
    print("Turn 1: User asks for login endpoint")
    print(f"Context length: {agent.context_length} tokens\n")
    
    # Turn 2: Assistant responds with code
    agent.ingest_turn({
        "role": "assistant",
        "content": """Here's a FastAPI login endpoint:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt
import jwt
from datetime import datetime, timedelta

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    # Hash password and verify
    # Generate JWT token
    # Return token
    pass
```"""
    })
    print("Turn 2: Assistant provides login endpoint code")
    print(f"Context length: {agent.context_length} tokens\n")
    
    # Turn 3: User asks about password hashing
    agent.ingest_turn({
        "role": "user",
        "content": "How should I hash passwords? Show me the bcrypt implementation."
    })
    print("Turn 3: User asks about password hashing")
    print(f"Context length: {agent.context_length} tokens\n")
    
    # Turn 4: Assistant provides hashing code
    agent.ingest_turn({
        "role": "assistant",
        "content": """Here's how to hash passwords with bcrypt:

```python
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed.encode('utf-8')
    )
```"""
    })
    print("Turn 4: Assistant provides password hashing code")
    print(f"Context length: {agent.context_length} tokens\n")
    
    # Probe: What is the current goal?
    print("=== Probing Goal Retention ===\n")
    goal_response = agent.answer_question("What is the current task?")
    print(f"Goal probe: {goal_response}\n")
    
    # Probe: What constraints are we following?
    constraints_response = agent.answer_question("What constraints must we follow?")
    print(f"Constraints probe: {constraints_response}\n")
    
    # Show salience set
    print("=== Salience Set (Goal-Critical Information) ===\n")
    salience = agent.salience_set
    if salience:
        for i, item in enumerate(salience, 1):
            print(f"{i}. {item[:150]}..." if len(item) > 150 else f"{i}. {item}")
    else:
        print("(No compression triggered yet - salience set is empty)")
    
    print("\n✅ Example complete!")
    print("\nNote: Compression will trigger automatically when context exceeds 80K tokens.")
    print("The Selective Salience Compression will preserve goal-critical information verbatim.")


if __name__ == "__main__":
    main()
