"""
A-mem Agent Adapter

This module wraps the A-mem advancedMemAgent to conform to the BaseAgent interface.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentConfig, BaseAgent

_amem_path = Path(__file__).resolve().parent.parent.parent / "data" / "A-mem"


class AMemAgent(BaseAgent):
    """
    Adapter wrapping A-mem's advancedMemAgent for the unified interface.

    This agent uses A-mem's memory system for storing and retrieving
    conversation context, with hybrid retrieval (BM25 + semantic search).
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the A-mem agent.

        Args:
            config: Agent configuration including model, backend, and retrieval settings
        """
        self._config = config
        self._turn_count = 0
        self._agent = self._create_agent()

    def _create_agent(self):
        """Create a new advancedMemAgent instance."""
        # Lazy import to avoid loading heavy dependencies until needed.
        # A-mem modules import each other by bare name, and cached memories are
        # pickled as memory_layer.*, so the directory goes on sys.path (appended
        # so it cannot shadow same-named modules elsewhere).
        if str(_amem_path) not in sys.path:
            sys.path.append(str(_amem_path))
        from test_advanced import advancedMemAgent

        return advancedMemAgent(
            model=self._config.model,
            backend=self._config.backend,
            retrieve_k=self._config.retrieve_k,
            temperature_c5=self._config.temperature_c5,
            sglang_host=self._config.sglang_host,
            sglang_port=self._config.sglang_port,
        )

    def ingest_turn(self, turn: Dict[str, Any]) -> None:
        """
        Add a turn to the agent's memory system.

        Args:
            turn: Dictionary with turn content and metadata
        """
        # Extract content and speaker
        speaker = turn.get("speaker", turn.get("role", "unknown"))
        content = turn.get("content", turn.get("text", ""))
        timestamp = turn.get("timestamp", turn.get("date_time"))

        # Format for A-mem memory system
        memory_content = f"Speaker {speaker} says: {content}"
        self._agent.add_memory(memory_content, time=timestamp)
        self._turn_count += 1

    def answer_question(
        self,
        question: str,
        category: Optional[int] = None,
        reference_answer: Optional[str] = None,
    ) -> str:
        """
        Answer a question based on stored memories.

        Args:
            question: The question to answer
            category: Question category (1-5) for category-aware prompting
            reference_answer: Reference answer for category-5 adversarial questions

        Returns:
            The agent's answer string
        """
        # Default to category 1 if not specified
        cat = category if category is not None else 1
        answer = reference_answer or ""

        # Call A-mem's answer_question method
        response, user_prompt, raw_context = self._agent.answer_question(
            question, cat, answer
        )

        # Parse JSON response
        try:
            return json.loads(response)["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Return raw response if JSON parsing fails
            return str(response).strip()

    def get_context_size(self) -> int:
        """Get the number of memory notes stored."""
        return len(self._agent.memory_system.memories)

    def compress(self, trigger_point: Optional[int] = None) -> None:
        """
        Trigger memory consolidation.

        A-mem continuously evolves memories, but this allows explicit consolidation.
        """
        self._agent.memory_system.consolidate_memories()

    def reset(self) -> None:
        """Reset the agent for a new evaluation sample."""
        self._agent = self._create_agent()
        self._turn_count = 0

    @property
    def name(self) -> str:
        """Return human-readable agent name."""
        return f"AMemAgent({self._config.model}, {self._config.backend})"

    @property
    def memory_system(self):
        """Access the underlying memory system."""
        return self._agent.memory_system

    @property
    def memories(self) -> Dict:
        """Access stored memories."""
        return self._agent.memory_system.memories

    @property
    def retriever(self):
        """Access the retriever component."""
        return self._agent.memory_system.retriever

    def get_raw_context(self, query: str, k: Optional[int] = None) -> str:
        """
        Get raw retrieved context for a query.

        Args:
            query: The query to retrieve context for
            k: Number of memories to retrieve (defaults to config value)

        Returns:
            Raw context string from memory retrieval
        """
        retrieve_k = k if k is not None else self._config.retrieve_k
        return self._agent.retrieve_memory(query, k=retrieve_k)
