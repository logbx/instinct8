"""
Strategy D: A-MEM Style Agentic Memory

This strategy wraps the A-mem AgenticMemorySystem for compression evaluation.

Key Features:
- Converts conversation turns to memory notes
- Uses hybrid retrieval (BM25 + semantic embeddings)
- Memory consolidation and evolution
- Explicit goal/constraint protection through tagged memories

Algorithm:
1. Initialize with goal/constraints as protected memories
2. For each turn, add as memory note with LLM-extracted metadata
3. On compression trigger:
   - Consolidate memories
   - Retrieve most relevant memories for current context
   - Format as compressed context string
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .strategy_base import CompressionStrategy

_amem_path = Path(__file__).resolve().parent.parent / "data" / "A-mem"


class StrategyD_AMemStyle(CompressionStrategy):
    """
    A-MEM style compression using agentic memory with hybrid retrieval.

    This implementation wraps the AgenticMemorySystem from A-mem,
    adapting it to the CompressionStrategy interface.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the A-MEM style strategy.

        Args:
            model_name: SentenceTransformer model for embeddings
            retrieve_k: Number of memories to retrieve during compression
            llm_backend: Backend for memory metadata extraction ("openai", "ollama", "sglang")
            llm_model: Model to use for metadata extraction
            evo_threshold: How often to consolidate memories
            api_key: Optional API key for the LLM backend
        """
        self._model_name = model_name
        self._retrieve_k = retrieve_k
        self._llm_backend = llm_backend
        self._llm_model = llm_model
        self._evo_threshold = evo_threshold
        self._api_key = api_key

        self._memory_system = None
        self._original_goal: Optional[str] = None
        self._constraints: List[str] = []
        self._goal_memory_id: Optional[str] = None
        self._constraint_memory_ids: List[str] = []

        self._initialize_memory_system()

    def _initialize_memory_system(self) -> None:
        """Initialize or reinitialize the memory system."""
        # Appended so data/A-mem cannot shadow same-named modules elsewhere.
        if str(_amem_path) not in sys.path:
            sys.path.append(str(_amem_path))
        try:
            from memory_layer import AgenticMemorySystem

            self._memory_system = AgenticMemorySystem(
                model_name=self._model_name,
                llm_backend=self._llm_backend,
                llm_model=self._llm_model,
                evo_threshold=self._evo_threshold,
                api_key=self._api_key,
            )
        except ImportError as e:
            self.log(f"Warning: Could not import AgenticMemorySystem: {e}")
            self._memory_system = None

    def initialize(self, original_goal: str, constraints: List[str]) -> None:
        """
        Store initial goal and constraints as protected memories.

        Unlike Codex-style compression, A-MEM explicitly stores and protects
        these as high-importance memory notes with special tags.

        Args:
            original_goal: The task's original goal statement
            constraints: List of hard constraints the agent must follow
        """
        self._original_goal = original_goal
        self._constraints = constraints

        if self._memory_system is None:
            self._initialize_memory_system()

        if self._memory_system:
            # Add goal as protected memory with high importance
            self._goal_memory_id = self._memory_system.add_note(
                content=f"GOAL: {original_goal}",
                tags=["protected", "goal", "critical"],
                keywords=["goal", "objective", "task"],
                context="Primary task objective that must be preserved",
                importance_score=10.0,  # High importance to prevent drift
            )

            # Add constraints as protected memories
            self._constraint_memory_ids = []
            for i, constraint in enumerate(constraints):
                constraint_id = self._memory_system.add_note(
                    content=f"CONSTRAINT {i+1}: {constraint}",
                    tags=["protected", "constraint", "critical"],
                    keywords=["constraint", "requirement", "rule"],
                    context=f"Hard constraint that must be followed: {constraint}",
                    importance_score=9.0,  # High importance
                )
                self._constraint_memory_ids.append(constraint_id)

        self.log(f"Initialized with goal: {original_goal}")
        self.log(f"Added {len(constraints)} constraints as protected memories")

    def update_goal(self, new_goal: str, rationale: str = "") -> None:
        """
        Update the goal with version tracking.

        A-MEM tracks goal evolution explicitly by adding new goal memories
        and linking them to previous versions.

        Args:
            new_goal: The updated goal statement
            rationale: Why the goal changed
        """
        if self._memory_system:
            # Add new goal memory, linked to previous
            evolution_note = f"GOAL UPDATE: {new_goal}"
            if rationale:
                evolution_note += f" (Rationale: {rationale})"

            new_goal_id = self._memory_system.add_note(
                content=evolution_note,
                tags=["protected", "goal", "goal-evolution"],
                keywords=["goal", "update", "evolution"],
                context=f"Goal evolved from: {self._original_goal}",
                importance_score=10.0,
                links={self._goal_memory_id: "evolved_from"} if self._goal_memory_id else None,
            )

            self._goal_memory_id = new_goal_id

        self.log(f"Goal updated: {new_goal}")

    def compress(
        self,
        context: List[Dict[str, Any]],
        trigger_point: int,
    ) -> str:
        """
        Compress context using A-MEM's memory retrieval.

        Steps:
        1. Convert conversation turns to memory notes (if not already)
        2. Consolidate memories
        3. Generate query from current context
        4. Retrieve most relevant memories
        5. Format as compressed context

        Args:
            context: List of conversation turns
            trigger_point: Which turn index to compress up to

        Returns:
            Compressed context string with goal, constraints, and relevant memories
        """
        self.log(f"Compressing {len(context)} turns up to point {trigger_point}")

        if self._memory_system is None:
            self.log("Memory system not available, using fallback")
            return self._fallback_compress(context, trigger_point)

        # Get turns to compress
        to_compress = context[:trigger_point]

        if not to_compress:
            return self._format_empty_context()

        # Add each turn as a memory note (if not already added)
        for turn in to_compress:
            turn_content = self._format_turn_for_memory(turn)
            self._memory_system.add_note(
                content=turn_content,
                tags=["conversation", f"turn_{turn.get('id', 'unknown')}"],
            )

        # Consolidate memories
        self._memory_system.consolidate_memories()

        # Generate query from recent context
        query = self._generate_retrieval_query(to_compress)

        # Retrieve relevant memories
        retrieved = self._memory_system.find_related_memories_raw(
            query, k=self._retrieve_k
        )

        # Format compressed context
        compressed = self._format_compressed_context(retrieved)

        original_chars = sum(len(str(t)) for t in to_compress)
        compressed_chars = len(compressed)
        self.log(f"Compressed {original_chars} chars -> {compressed_chars} chars")

        return compressed

    def name(self) -> str:
        return "Strategy D - A-MEM Style Agentic Memory"

    def _format_turn_for_memory(self, turn: Dict[str, Any]) -> str:
        """Format a conversation turn for memory storage."""
        turn_id = turn.get("id", "?")
        role = turn.get("role", "unknown")
        content = turn.get("content", "")

        formatted = f"[Turn {turn_id}] {role}: {content}"

        # Include tool call info if present
        if "tool_call" in turn and turn["tool_call"]:
            tool = turn["tool_call"]
            tool_name = tool.get("name", "unknown")
            formatted += f" [Tool: {tool_name}]"

        return formatted

    def _generate_retrieval_query(self, turns: List[Dict[str, Any]]) -> str:
        """Generate a query for memory retrieval from recent turns."""
        # Use the last few turns to generate context
        recent_turns = turns[-3:] if len(turns) > 3 else turns

        query_parts = []

        # Include goal in query for relevance
        if self._original_goal:
            query_parts.append(f"Goal: {self._original_goal}")

        # Include recent conversation
        for turn in recent_turns:
            content = turn.get("content", "")[:200]  # Truncate long content
            query_parts.append(content)

        return " ".join(query_parts)

    def _format_compressed_context(self, retrieved_memories: str) -> str:
        """Format the compressed context with goal, constraints, and memories."""
        parts = []

        # Always include goal first
        if self._original_goal:
            parts.append("=== CURRENT GOAL ===")
            parts.append(self._original_goal)
            parts.append("")

        # Include constraints
        if self._constraints:
            parts.append("=== CONSTRAINTS ===")
            for i, constraint in enumerate(self._constraints, 1):
                parts.append(f"{i}. {constraint}")
            parts.append("")

        # Include retrieved memories
        parts.append("=== RELEVANT CONTEXT (from memory) ===")
        if retrieved_memories:
            parts.append(retrieved_memories)
        else:
            parts.append("(No relevant memories retrieved)")

        return "\n".join(parts)

    def _format_empty_context(self) -> str:
        """Return context when there's nothing to compress."""
        parts = []

        if self._original_goal:
            parts.append(f"Goal: {self._original_goal}")

        if self._constraints:
            parts.append(f"Constraints: {', '.join(self._constraints)}")

        parts.append("(No previous conversation)")

        return "\n".join(parts)

    def _fallback_compress(
        self, context: List[Dict[str, Any]], trigger_point: int
    ) -> str:
        """Fallback compression when memory system is unavailable."""
        to_compress = context[:trigger_point]

        parts = []

        if self._original_goal:
            parts.append(f"=== GOAL ===\n{self._original_goal}\n")

        if self._constraints:
            parts.append("=== CONSTRAINTS ===")
            for c in self._constraints:
                parts.append(f"- {c}")
            parts.append("")

        parts.append("=== CONVERSATION SUMMARY ===")
        # Simple truncation as fallback
        for turn in to_compress[-5:]:  # Keep last 5 turns
            parts.append(f"{turn.get('role', '?')}: {turn.get('content', '')[:100]}...")

        return "\n".join(parts)

    def get_memory_count(self) -> int:
        """Get the current number of memories stored."""
        if self._memory_system:
            return len(self._memory_system.memories)
        return 0

    def reset(self) -> None:
        """Reset the memory system for a new evaluation."""
        self._initialize_memory_system()
        self._goal_memory_id = None
        self._constraint_memory_ids = []


# Convenience function for quick testing
def create_amem_strategy(
    retrieve_k: int = 10,
    llm_backend: str = "openai",
) -> StrategyD_AMemStyle:
    """Create an A-MEM style compression strategy."""
    return StrategyD_AMemStyle(
        retrieve_k=retrieve_k,
        llm_backend=llm_backend,
    )
