"""
Compression Agent Adapter

This module adapts a CompressionStrategy to the BaseAgent interface.
"""

from typing import Any, Dict, List, Optional

from strategies import CompressionStrategy
from strategies.strategy_b_codex import LLMClient, _create_llm_client

from .base import AgentConfig, BaseAgent


class CompressionAgent(BaseAgent):
    """
    Agent whose context is managed by a CompressionStrategy.

    Turns accumulate in context until compress() replaces them with the
    strategy's compressed output. Questions are answered by an LLM over the
    current (possibly compressed) context.
    """

    def __init__(self, config: AgentConfig, strategy: CompressionStrategy):
        """
        Initialize the compression agent.

        Args:
            config: Agent configuration
            strategy: The compression strategy to use
        """
        self._config = config
        self._strategy = strategy
        self._llm_client: Optional[LLMClient] = None
        self._context: List[Dict[str, Any]] = []
        self._total_tokens = 0
        self._strategy.initialize(config.original_goal, config.constraints)

    def ingest_turn(self, turn: Dict[str, Any]) -> None:
        """Add a turn to the agent's context."""
        formatted_turn = {
            "id": turn.get("id", turn.get("turn_id", 0)),
            "role": turn.get("role", "user"),
            "content": turn.get("content", turn.get("text", "")),
            "is_compression_point": turn.get("is_compression_point", False),
            "tool_call": turn.get("tool_call"),
        }
        self._context.append(formatted_turn)
        # Approximate: ~4 chars per token
        self._total_tokens += len(formatted_turn["content"]) // 4

    def answer_question(
        self,
        question: str,
        category: Optional[int] = None,
        reference_answer: Optional[str] = None,
    ) -> str:
        """
        Answer a question based on current context.

        Args:
            question: The question to answer
            category: Ignored for compression agent
            reference_answer: Ignored for compression agent

        Returns:
            The agent's response
        """
        if self._llm_client is None:
            self._llm_client = _create_llm_client(self._config.backend, self._config.model)

        context_text = "\n\n".join(
            f"[{turn.get('role', 'unknown')}]: {turn.get('content', '')}"
            for turn in self._context
        )
        prompt = f"Context:\n{context_text}\n\n---\n\nQuestion: {question}"
        if self._config.system_prompt:
            prompt = f"{self._config.system_prompt}\n\n{prompt}"

        try:
            return self._llm_client.complete(prompt, max_tokens=300) or "(error: empty response)"
        except Exception as e:
            print(f"[{self.name}] Error calling LLM: {e}")
            return f"(error: {e})"

    def get_context_size(self) -> int:
        """Get approximate token count of current context."""
        return self._total_tokens

    def compress(self, trigger_point: Optional[int] = None) -> None:
        """Compress the context using the configured strategy."""
        if trigger_point is None:
            trigger_point = len(self._context)
        compressed = self._strategy.compress(self._context, trigger_point)
        self._context = [
            {
                "id": 0,
                "role": "system",
                "content": compressed,
                "is_compression_point": False,
            }
        ]
        self._total_tokens = len(compressed) // 4

    def reset(self) -> None:
        """Reset the agent for a new evaluation sample."""
        self._context = []
        self._total_tokens = 0
        self._strategy.initialize(
            self._config.original_goal,
            self._config.constraints,
        )

    def initialize_goal(self, goal: str, constraints: List[str]) -> None:
        """Initialize goal and constraints for the strategy."""
        self._config.original_goal = goal
        self._config.constraints = constraints
        self._strategy.initialize(goal, constraints)

    @property
    def name(self) -> str:
        """Return human-readable agent name."""
        return f"CompressionAgent({self._strategy.name()})"

    @property
    def strategy(self) -> CompressionStrategy:
        """Access the underlying compression strategy."""
        return self._strategy

    @property
    def context(self) -> List[Dict[str, Any]]:
        """Access the current context."""
        return self._context
