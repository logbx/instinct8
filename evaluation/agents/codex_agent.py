"""
Codex Agent Adapter

This module provides a Codex-style coding agent that wraps compression strategies
for evaluation on coding tasks.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import AgentConfig, BaseAgent


@dataclass
class ShellCommand:
    """A simulated shell command and its result."""

    command: str
    output: str
    exit_code: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FileOperation:
    """A simulated file operation."""

    operation: str  # "read", "write", "patch"
    path: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CodexAgent(BaseAgent):
    """
    Codex-style coding agent with tool simulation.

    This agent simulates a Codex-like coding assistant that:
    - Maintains a conversation context
    - Executes simulated tool calls (shell, file operations)
    - Auto-compacts when context exceeds threshold
    - Supports different compression strategies

    Usage:
        from strategies import StrategyB_CodexCheckpoint, StrategyD_AMemStyle

        config = AgentConfig(model="claude-sonnet-4-20250514")
        strategy = StrategyB_CodexCheckpoint()

        agent = CodexAgent(config, strategy, compaction_threshold=80000)
        agent.initialize_goal("Build a FastAPI auth system", ["Use JWT", "Hash passwords"])

        for turn in conversation:
            agent.ingest_turn(turn)

        answer = agent.answer_question("What is the current task?")
    """

    def __init__(
        self,
        config: AgentConfig,
        strategy: Optional[Any] = None,  # CompressionStrategy
        compaction_threshold: int = 80000,
    ):
        """
        Initialize the Codex agent.

        Args:
            config: Agent configuration
            strategy: Compression strategy to use (None = no compression)
            compaction_threshold: Token count at which to auto-compact
        """
        self._config = config
        self._strategy = strategy
        self._compaction_threshold = compaction_threshold

        # Context management
        self._context: List[Dict[str, Any]] = []
        self._total_tokens: int = 0
        self._compression_count: int = 0

        # Simulated environment
        self._file_system: Dict[str, str] = {}
        self._shell_history: List[ShellCommand] = []
        self._file_operations: List[FileOperation] = []

        # Goal tracking
        self._original_goal: Optional[str] = None
        self._constraints: List[str] = []

        # Generated code tracking
        self._generated_code: Dict[str, str] = {}

    @property
    def name(self) -> str:
        strategy_name = self._strategy.name() if self._strategy else "NoCompression"
        return f"CodexAgent({strategy_name})"

    def initialize_goal(self, goal: str, constraints: List[str]) -> None:
        """
        Initialize the task goal and constraints.

        Args:
            goal: The coding task goal
            constraints: List of requirements/constraints
        """
        self._original_goal = goal
        self._constraints = constraints

        if self._strategy:
            self._strategy.initialize(goal, constraints)

        # Add system turn with goal
        system_turn = {
            "id": 0,
            "role": "system",
            "content": self._format_system_prompt(goal, constraints),
        }
        self._context.append(system_turn)
        self._total_tokens += self._estimate_tokens(system_turn)

    def ingest_turn(self, turn: Dict[str, Any]) -> None:
        """
        Ingest a conversation turn, processing any tool calls.

        Args:
            turn: Turn dictionary with content and optional tool_calls
        """
        # Add turn to context
        self._context.append(turn)
        self._total_tokens += self._estimate_tokens(turn)

        # Execute tool calls if present
        tool_calls = turn.get("tool_calls") or turn.get("tool_call")
        if tool_calls:
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            elif isinstance(tool_calls, dict) and "tool_calls" in tool_calls:
                tool_calls = tool_calls["tool_calls"]

            for tool_call in tool_calls:
                self._execute_tool(tool_call)

        # Auto-compact if threshold exceeded
        if self._total_tokens > self._compaction_threshold:
            self.compress()

    def answer_question(
        self,
        question: str,
        category: Optional[int] = None,
        reference_answer: Optional[str] = None,
    ) -> str:
        """
        Answer a question based on current context.

        This is a mock implementation that extracts relevant information
        from the context. In production, this would call an LLM.

        Args:
            question: The question to answer
            category: Optional category (unused for coding)
            reference_answer: Optional reference (unused)

        Returns:
            Answer string
        """
        question_lower = question.lower()

        # Goal-related questions
        if "goal" in question_lower or "task" in question_lower:
            if self._original_goal:
                return f"The current goal is: {self._original_goal}"
            return "No explicit goal has been set."

        # Constraint-related questions
        if "constraint" in question_lower or "requirement" in question_lower:
            if self._constraints:
                return "Constraints: " + "; ".join(self._constraints)
            return "No constraints have been specified."

        # Next step questions
        if "next" in question_lower or "should" in question_lower:
            return self._generate_next_step_answer()

        # Progress questions
        if "progress" in question_lower or "done" in question_lower:
            return self._generate_progress_answer()

        # Code-related questions
        if "code" in question_lower or "file" in question_lower:
            return self._generate_code_answer()

        # Default: summarize recent context
        return self._generate_context_summary()

    def get_context_size(self) -> int:
        """Get current context size in tokens."""
        return self._total_tokens

    def compress(self, trigger_point: Optional[int] = None) -> None:
        """
        Compress the context using the configured strategy.

        Args:
            trigger_point: Optional turn ID to compress up to
        """
        if not self._strategy:
            return

        if trigger_point is None:
            trigger_point = len(self._context)

        # Get compressed context from strategy
        compressed = self._strategy.compress(self._context, trigger_point)

        # Replace context with compressed version
        self._context = [
            {
                "id": 0,
                "role": "system",
                "content": compressed,
                "is_compressed": True,
            }
        ]

        # Update token count
        old_tokens = self._total_tokens
        self._total_tokens = self._estimate_tokens({"content": compressed})
        self._compression_count += 1

        # Log compression
        print(
            f"[{self.name}] Compressed: {old_tokens} -> {self._total_tokens} tokens "
            f"(compression #{self._compression_count})"
        )

    def reset(self) -> None:
        """Reset agent state for new evaluation."""
        self._context = []
        self._total_tokens = 0
        self._compression_count = 0
        self._file_system = {}
        self._shell_history = []
        self._file_operations = []
        self._generated_code = {}

        if self._strategy and hasattr(self._strategy, "reset"):
            self._strategy.reset()

    def _format_system_prompt(self, goal: str, constraints: List[str]) -> str:
        """Format the system prompt with goal and constraints."""
        parts = [
            "You are a coding assistant helping with the following task:",
            "",
            f"GOAL: {goal}",
            "",
        ]

        if constraints:
            parts.append("CONSTRAINTS:")
            for i, c in enumerate(constraints, 1):
                parts.append(f"  {i}. {c}")
            parts.append("")

        parts.extend(
            [
                "You have access to the following tools:",
                "- shell: Execute shell commands",
                "- read_file: Read file contents",
                "- write_file: Write/create files",
                "- apply_patch: Apply code patches",
                "",
                "Work step by step to complete the task.",
            ]
        )

        return "\n".join(parts)

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a simulated tool call.

        Args:
            tool_call: Dictionary with 'name' and 'input' keys

        Returns:
            Tool execution result
        """
        name = tool_call.get("name", "")
        input_data = tool_call.get("input", "")

        if name == "shell":
            return self._simulate_shell(input_data)
        elif name == "read_file":
            return self._simulate_read_file(input_data)
        elif name == "write_file":
            return self._simulate_write_file(input_data)
        elif name == "apply_patch":
            return self._simulate_apply_patch(input_data)
        else:
            return {"error": f"Unknown tool: {name}"}

    def _simulate_shell(self, command: str) -> Dict[str, Any]:
        """Simulate shell command execution."""
        # Common commands and their simulated outputs
        output = ""
        exit_code = 0

        if command.startswith("ls"):
            output = "\n".join(self._file_system.keys()) or "."
        elif command.startswith("cat"):
            file_path = command.split()[-1] if len(command.split()) > 1 else ""
            output = self._file_system.get(file_path, f"cat: {file_path}: No such file")
        elif command.startswith("mkdir"):
            output = ""  # Success
        elif command.startswith("pip install"):
            output = "Successfully installed packages"
        elif command.startswith("pytest"):
            output = "===== X passed in Y.YYs ====="
        elif command.startswith("python"):
            output = "(execution output)"
        else:
            output = "(command executed)"

        shell_cmd = ShellCommand(command=command, output=output, exit_code=exit_code)
        self._shell_history.append(shell_cmd)

        return {"output": output, "exit_code": exit_code}

    def _simulate_read_file(self, path: str) -> Dict[str, Any]:
        """Simulate file read."""
        content = self._file_system.get(path)

        if content is not None:
            self._file_operations.append(
                FileOperation(operation="read", path=path, content=content)
            )
            return {"content": content}
        else:
            return {"error": f"File not found: {path}"}

    def _simulate_write_file(self, input_data: str) -> Dict[str, Any]:
        """Simulate file write."""
        # Parse path from input
        if isinstance(input_data, dict):
            path = input_data.get("path", "")
            content = input_data.get("content", "")
        else:
            # Assume path is the input
            path = input_data
            content = "(file content)"

        self._file_system[path] = content
        self._generated_code[path] = content
        self._file_operations.append(
            FileOperation(operation="write", path=path, content=content)
        )

        return {"success": True, "path": path}

    def _simulate_apply_patch(self, patch: str) -> Dict[str, Any]:
        """Simulate applying a code patch."""
        # Simple patch simulation
        self._file_operations.append(
            FileOperation(operation="patch", path="(patched)", content=patch)
        )
        return {"success": True, "message": "Patch applied"}

    def _estimate_tokens(self, turn: Dict[str, Any]) -> int:
        """Estimate token count for a turn."""
        content = turn.get("content", "")
        if isinstance(content, str):
            # Rough estimate: ~4 chars per token
            return len(content) // 4
        return 0

    def _generate_next_step_answer(self) -> str:
        """Generate answer about next steps."""
        if not self._context:
            return "Start by understanding the task requirements."

        # Look at recent context for guidance
        recent_turns = self._context[-3:]
        for turn in reversed(recent_turns):
            content = turn.get("content", "")
            if "TODO" in content or "next" in content.lower():
                return f"Based on recent context: {content[:200]}..."

        return "Continue implementing the current feature."

    def _generate_progress_answer(self) -> str:
        """Generate answer about progress."""
        files_created = len(self._generated_code)
        commands_run = len(self._shell_history)
        compressions = self._compression_count

        return (
            f"Progress: {files_created} files created, "
            f"{commands_run} commands executed, "
            f"{compressions} compressions performed."
        )

    def _generate_code_answer(self) -> str:
        """Generate answer about generated code."""
        if self._generated_code:
            files = list(self._generated_code.keys())
            return f"Generated files: {', '.join(files)}"
        return "No code has been generated yet."

    def _generate_context_summary(self) -> str:
        """Generate a summary of the current context."""
        if not self._context:
            return "No context available."

        summary_parts = []

        if self._original_goal:
            summary_parts.append(f"Goal: {self._original_goal}")

        if self._constraints:
            summary_parts.append(f"Constraints: {len(self._constraints)} defined")

        summary_parts.append(f"Context: {len(self._context)} turns")
        summary_parts.append(f"Tokens: {self._total_tokens}")

        return " | ".join(summary_parts)

    def get_generated_code(self) -> Dict[str, str]:
        """Get all generated code files."""
        return self._generated_code.copy()

    def get_file_system(self) -> Dict[str, str]:
        """Get the simulated file system."""
        return self._file_system.copy()

    def set_initial_files(self, files: Dict[str, str]) -> None:
        """Set initial files for the simulated file system."""
        self._file_system.update(files)


# Convenience function
def create_codex_agent(
    strategy_name: str = "codex",
    compaction_threshold: int = 80000,
    **kwargs,
) -> CodexAgent:
    """
    Create a CodexAgent with a specified strategy.

    Args:
        strategy_name: One of "codex", "amem", "hybrid", or "none"
        compaction_threshold: Token threshold for auto-compaction
        **kwargs: Additional strategy configuration

    Returns:
        Configured CodexAgent
    """
    config = AgentConfig()
    strategy = None

    if strategy_name == "codex":
        from strategies.strategy_b_codex import StrategyB_CodexCheckpoint

        strategy = StrategyB_CodexCheckpoint(**kwargs)
    elif strategy_name == "amem":
        from strategies.strategy_d_amem import StrategyD_AMemStyle

        strategy = StrategyD_AMemStyle(**kwargs)
    # else: no strategy (no compression)

    return CodexAgent(
        config=config,
        strategy=strategy,
        compaction_threshold=compaction_threshold,
    )
