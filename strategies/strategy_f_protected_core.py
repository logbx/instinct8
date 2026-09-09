"""
Strategy F: Protected Core + Goal Re-assertion (Novel)

This is the key innovation: explicit goal protection via a first-class ProtectedCore object.

Key Features:
- ProtectedCore stores goal/constraints separately from conversation history
- ProtectedCore is NEVER compressed, only RE-ASSERTED after compression
- Goal evolution is tracked explicitly via update_goal()
- Key decisions are preserved in the ProtectedCore

Algorithm:
1. Initialize ProtectedCore with original goal and constraints
2. On compression trigger:
   - Compress conversation "halo" (everything except ProtectedCore)
   - Rebuild context as: PROTECTED_CORE + COMPRESSED_HALO + RECENT_TURNS
3. ProtectedCore always appears first and is marked as AUTHORITATIVE

This differs from Strategy B (Codex) which embeds goal preservation in the prompt.
Strategy F makes goal protection explicit and first-class.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING
from datetime import datetime

from .strategy_base import CompressionStrategy

if TYPE_CHECKING:
    from evaluation.token_budget import TokenBudget


@dataclass
class Decision:
    """Represents a key decision made during the task."""
    decision: str
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProtectedCore:
    """
    First-class object that stores protected goal state.
    
    This is NEVER compressed - only re-asserted after compression.
    """
    original_goal: str
    current_goal: str
    hard_constraints: List[str]
    key_decisions: List[Decision] = field(default_factory=list)
    timestamp_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        ...


class OpenAISummarizer:
    """OpenAI API client for summarization."""
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        return content.strip() if content else "(summarization returned empty)"


class AnthropicSummarizer:
    """Anthropic API client for summarization."""
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import Anthropic
            self.client = Anthropic()
            self.model = model
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


def _create_llm_client(backend: str = "auto", model: Optional[str] = None) -> LLMClient:
    """Create LLM client based on backend preference and available API keys."""
    if backend == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAISummarizer(model or "gpt-4o-mini")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicSummarizer(model or "claude-sonnet-4-20250514")
        else:
            raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    elif backend == "openai":
        return OpenAISummarizer(model or "gpt-4o-mini")
    elif backend == "anthropic":
        return AnthropicSummarizer(model or "claude-sonnet-4-20250514")
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Summarization prompt for the conversation "halo" (excludes goal/constraints)
HALO_SUMMARIZATION_PROMPT = """Summarize this conversation history, focusing on:
- What progress has been made
- Key decisions and their outcomes
- Important context and information discovered
- What remains to be done

Do NOT include the original goal or constraints in the summary - those are handled separately.
Be concise and structured."""


class StrategyF_ProtectedCore(CompressionStrategy):
    """
    Protected Core + Goal Re-assertion strategy.
    
    Core insight: Store goal/constraints in a first-class ProtectedCore object
    that is NEVER compressed, only re-asserted after compression.
    """
    
    def __init__(
        self,
        system_prompt: str = "",
        model: Optional[str] = None,
        backend: str = "auto",
        token_budget: Optional["TokenBudget"] = None,
        keep_recent_turns: int = 3,
    ):
        """
        Initialize the Protected Core strategy.

        Args:
            system_prompt: The system prompt to preserve
            model: Model to use for summarization (auto-selected based on backend if None)
            backend: LLM backend - "auto", "openai", or "anthropic"
            token_budget: Artificial context window budget for testing compaction.
                         Defaults to 8K budget if not provided.
            keep_recent_turns: Number of recent turns to keep raw (default: 3)
        """
        from evaluation.token_budget import BUDGET_8K
        
        self._client: Optional[LLMClient] = None
        self._backend = backend
        self._model = model
        self.system_prompt = system_prompt
        self.token_budget = token_budget or BUDGET_8K
        self.keep_recent_turns = keep_recent_turns
        self.protected_core: Optional[ProtectedCore] = None
    
    @property
    def client(self) -> LLMClient:
        """Lazy client property - only creates LLM client when first accessed."""
        if self._client is None:
            self._client = _create_llm_client(backend=self._backend, model=self._model)
        return self._client
    
    @client.setter
    def client(self, value: LLMClient) -> None:
        """Allow setting client directly (for testing with mocks)."""
        self._client = value
    
    def initialize(self, original_goal: str, constraints: List[str]) -> None:
        """
        Initialize the ProtectedCore with original goal and constraints.
        
        This is called at task start and creates the protected state object.
        """
        self.protected_core = ProtectedCore(
            original_goal=original_goal,
            current_goal=original_goal,
            hard_constraints=constraints,
            key_decisions=[],
        )
        self.log(f"Protected Core initialized with goal: {original_goal}")
        self.log(f"Constraints: {constraints}")
    
    def update_goal(self, new_goal: str, rationale: str = "") -> None:
        """
        Update the current goal and record the decision.
        
        This is called when the goal evolves mid-task.
        The decision is tracked in the ProtectedCore for later reference.
        """
        if self.protected_core is None:
            raise ValueError("ProtectedCore not initialized. Call initialize() first.")
        
        decision = Decision(
            decision=f"Goal updated to: {new_goal}",
            rationale=rationale or "Goal evolution during task execution",
        )
        self.protected_core.key_decisions.append(decision)
        self.protected_core.current_goal = new_goal
        self.protected_core.timestamp_updated = datetime.now().isoformat()
        self.log(f"Goal updated to: {new_goal} (rationale: {rationale})")
    
    def update_constraints(self, new_constraints: List[str], rationale: str = "") -> None:
        """
        Update constraints and record the change.
        
        This is called when constraints change (e.g., budget shift from $5K to $1K).
        The change is tracked in the ProtectedCore.
        """
        if self.protected_core is None:
            raise ValueError("ProtectedCore not initialized. Call initialize() first.")
        
        old_constraints = self.protected_core.hard_constraints.copy()
        self.protected_core.hard_constraints = new_constraints
        
        decision = Decision(
            decision=f"Constraints updated: {', '.join(new_constraints)}",
            rationale=rationale or f"Constraint change from: {', '.join(old_constraints)}",
        )
        self.protected_core.key_decisions.append(decision)
        self.protected_core.timestamp_updated = datetime.now().isoformat()
        self.log(f"Constraints updated: {new_constraints}")
    
    def add_constraint(self, new_constraint: str, rationale: str = "") -> None:
        """
        Add a new constraint to the ProtectedCore.
        
        Useful when a new constraint is introduced mid-conversation.
        """
        if self.protected_core is None:
            raise ValueError("ProtectedCore not initialized. Call initialize() first.")
        
        if new_constraint not in self.protected_core.hard_constraints:
            self.protected_core.hard_constraints.append(new_constraint)
            
            decision = Decision(
                decision=f"New constraint added: {new_constraint}",
                rationale=rationale or "New constraint introduced during conversation",
            )
            self.protected_core.key_decisions.append(decision)
            self.protected_core.timestamp_updated = datetime.now().isoformat()
            self.log(f"Added constraint: {new_constraint}")
    
    def _detect_constraint_changes(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Detect constraint changes in a message (e.g., budget shifts).
        
        Returns:
            Dict with 'type' ('budget', 'timeline', 'team', etc.) and 'new_value' if detected
        """
        if self.protected_core is None:
            return None
        
        message_lower = message.lower()
        
        # Budget changes
        budget_patterns = [
            r'(?:budget|cost|spending).*?(?:is|now|changed?|reduced?|increased?|cut).*?\$?(\d+[km]?)',
            r'\$?(\d+[km]?).*?(?:budget|monthly|cost)',
            r'(?:reduce|cut|increase|change).*?(?:budget|cost).*?(?:to|from).*?\$?(\d+[km]?)',
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, message_lower)
            if match:
                value = match.group(1)
                # Check if this is different from current constraints
                current_budget = None
                for constraint in self.protected_core.hard_constraints:
                    if 'budget' in constraint.lower() or '$' in constraint:
                        # Extract current budget value
                        budget_match = re.search(r'\$?(\d+[km]?)', constraint.lower())
                        if budget_match:
                            current_budget = budget_match.group(1)
                            break
                
                if current_budget and value != current_budget:
                    return {
                        'type': 'budget',
                        'old_value': f'${current_budget}',
                        'new_value': f'${value}',
                    }
        
        # Timeline changes
        if any(word in message_lower for word in ['timeline', 'deadline', 'weeks', 'months', 'days']):
            # Could extract specific timeline changes here
            pass
        
        # Team size changes
        if any(word in message_lower for word in ['team', 'engineers', 'developers', 'people']):
            # Could extract team size changes here
            pass
        
        return None
    
    def _is_important_detail(self, turn: Dict[str, Any]) -> bool:
        """
        Determine if a turn contains important details that should be preserved.
        
        Heuristics:
        1. Has explicit "decision" field (template-specific)
        2. Contains specific technical choices (schema formats, thresholds, etc.)
        3. Contains constraint-like information (budgets, timelines, requirements)
        4. Contains tool call outputs with specific recommendations
        """
        # Method 1: Explicit decision field (template-specific)
        if turn.get("decision"):
            return True
        
        # Method 2: Look for specific technical details
        content = turn.get("content", "").lower()
        
        # Technical choices that are important
        important_indicators = [
            r'\b(?:schema|format|protocol|standard)\s+(?:is|will be|chosen|selected|using)\s+(\w+)',
            r'\b(?:threshold|limit|max|min)\s+(?:is|set to|of)\s+([\d>]+)',
            r'\b(?:instance|server|compute)\s+(?:is|will be|using)\s+([\w\.]+)',
            r'\b(?:architecture|stack|platform)\s+(?:is|will be|chosen|selected)\s+([\w\s]+)',
            r'\b(?:decision|chose|selected|recommend)\s+([^\.]+)',
        ]
        
        for pattern in important_indicators:
            if re.search(pattern, content):
                return True
        
        # Method 3: Tool calls with specific outputs
        tool_call = turn.get("tool_call")
        if tool_call and isinstance(tool_call, dict):
            output = tool_call.get("output", "")
            # If tool output contains specific recommendations or numbers
            if any(char.isdigit() for char in output) or len(output) > 50:
                return True
        
        return False
    
    def _extract_important_detail(self, turn: Dict[str, Any]) -> Optional[str]:
        """
        Extract the important detail from a turn as a decision-like string.
        
        Returns:
            A concise decision string, or None if nothing important found
        """
        # If explicit decision field exists, use it
        if turn.get("decision"):
            return turn.get("decision")
        
        content = turn.get("content", "")
        
        # Try to extract decision-like statements
        # Pattern: "We chose X" or "Decision: X" or "Selected: X"
        decision_patterns = [
            r'(?:decision|chose|selected|recommend|using|will use):\s*([^\.\n]+)',
            r'(?:architecture|stack|platform|schema|format)\s+(?:is|will be)\s+([^\.\n]+)',
        ]
        
        for pattern in decision_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 10 and len(extracted) < 200:
                    return extracted
        
        # Fallback: extract first sentence if it looks like a decision
        first_sentence = content.split('.')[0] if '.' in content else content[:100]
        if any(word in first_sentence.lower() for word in ['recommend', 'chose', 'selected', 'decision', 'using']):
            return first_sentence.strip()
        
        return None
    
    def _detect_and_update_goal_shifts(self, context: List[Dict[str, Any]]) -> None:
        """
        Scan context for goal shifts, constraint changes, and key decisions.
        
        Strategy F should:
        1. Detect shifts in user messages and update goal
        2. Detect constraint changes (budget, timeline, etc.) and update ProtectedCore
        3. Extract important technical decisions from turns and add to ProtectedCore
        """
        from evaluation.goal_tracking import detect_goal_shift_in_message, extract_new_goal_from_message
        
        if self.protected_core is None:
            return
        
        current_goal = self.protected_core.current_goal
        existing_decisions = {d.decision for d in self.protected_core.key_decisions}
        
        # Scan all turns for goal shifts, constraint changes, and key decisions
        for turn in context:
            # Check for goal shifts and constraint changes in user messages
            if turn.get("role") == "user":
                message = turn.get("content", "")
                
                # Check for goal shifts
                shift_detected = detect_goal_shift_in_message(message)
                if shift_detected:
                    new_goal = extract_new_goal_from_message(message, current_goal)
                    if new_goal and new_goal != current_goal:
                        turn_id = turn.get("id", "?")
                        rationale = f"Goal shift detected in user message at turn {turn_id}"
                        self.update_goal(new_goal, rationale=rationale)
                        current_goal = new_goal
                
                # Check for constraint changes
                constraint_change = self._detect_constraint_changes(message)
                if constraint_change:
                    # Update the relevant constraint
                    updated_constraints = []
                    for constraint in self.protected_core.hard_constraints:
                        if constraint_change['type'] == 'budget' and ('budget' in constraint.lower() or '$' in constraint):
                            # Replace old budget with new budget
                            updated_constraints.append(f"Budget: Maximum {constraint_change['new_value']} monthly cloud infrastructure cost")
                        else:
                            updated_constraints.append(constraint)
                    
                    if updated_constraints != self.protected_core.hard_constraints:
                        rationale = f"Constraint change detected: {constraint_change['old_value']} -> {constraint_change['new_value']}"
                        self.update_constraints(updated_constraints, rationale=rationale)
            
            # Extract important details from all turns
            if self._is_important_detail(turn):
                important_detail = self._extract_important_detail(turn)
                if important_detail and important_detail not in existing_decisions:
                    turn_content = turn.get("content", "")
                    rationale = turn_content[:200] + "..." if len(turn_content) > 200 else turn_content
                    
                    decision = Decision(
                        decision=important_detail,
                        rationale=rationale,
                    )
                    self.protected_core.key_decisions.append(decision)
                    existing_decisions.add(important_detail)
                    self.log(f"Added important detail to ProtectedCore: {important_detail[:50]}...")
    
    def compress(
        self,
        context: List[Dict[str, Any]],
        trigger_point: int,
    ) -> str:
        """
        Compress context using Protected Core strategy.
        
        Key difference from other strategies:
        - ProtectedCore is NEVER compressed
        - Only the conversation "halo" is compressed
        - ProtectedCore is re-asserted at the top of the compressed context
        
        Steps:
        1. Check if compression is needed based on token budget
        2. Separate: halo (to compress) vs recent turns (keep raw)
        3. Compress only the halo
        4. Rebuild: PROTECTED_CORE + COMPRESSED_HALO + RECENT_TURNS
        
        Args:
            context: List of conversation turns
            trigger_point: Which turn to compress up to

        Returns:
            Compressed context string with ProtectedCore re-asserted
        """
        if self.protected_core is None:
            raise ValueError("ProtectedCore not initialized. Call initialize() first.")
        
        self.log(f"Considering compression of {len(context)} turns up to point {trigger_point}")

        to_compress = context[:trigger_point]

        if not to_compress:
            self.log("Nothing to compress")
            return self._format_context_with_protected_core("", [])

        self._detect_and_update_goal_shifts(to_compress)

        reconstructed = self.render_reconstructed_prompt(to_compress)
        from evaluation.token_budget import estimate_tokens, should_compact
        estimated_tokens = estimate_tokens(reconstructed)
        
        if not should_compact(reconstructed, self.token_budget):
            self.log(f"Skipping compression - prompt tokens ({estimated_tokens} estimated) below budget ({self.token_budget.trigger_tokens})")
            return self._format_context_with_protected_core("", to_compress)

        self.log(f"Compressing - prompt tokens ({estimated_tokens} estimated) exceed budget ({self.token_budget.trigger_tokens})")

        split_point = max(0, trigger_point - self.keep_recent_turns)
        halo_to_compress = to_compress[:split_point]
        recent_turns = to_compress[split_point:]

        if halo_to_compress:
            halo_text = self.format_context(halo_to_compress)
            halo_summary = self._summarize_halo(halo_text)
        else:
            halo_summary = ""

        compressed = self._format_context_with_protected_core(halo_summary, recent_turns)

        original_chars = len(self.format_context(to_compress))
        compressed_chars = len(compressed)
        self.log(f"Compressed {original_chars} chars -> {compressed_chars} chars (ProtectedCore preserved)")

        return compressed
    
    def _summarize_halo(self, halo_text: str) -> str:
        """
        Summarize the conversation halo (everything except ProtectedCore).
        
        The prompt explicitly excludes goal/constraints since those are
        handled by the ProtectedCore.
        """
        prompt = f"{HALO_SUMMARIZATION_PROMPT}\n\nConversation history:\n{halo_text}"
        return self.client.complete(prompt, max_tokens=500)
    
    def _format_context_with_protected_core(
        self,
        halo_summary: str,
        recent_turns: List[Dict[str, Any]],
    ) -> str:
        """
        Format context with ProtectedCore ALWAYS front and center.
        
        Structure:
        1. System prompt (if present)
        2. PROTECTED CORE (authoritative, never compressed)
        3. Compressed conversation summary (halo)
        4. Recent turns (raw)
        """
        parts = []
        
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        
        decisions_str = "\n".join([
            f"  - {d.decision} (Rationale: {d.rationale})"
            for d in self.protected_core.key_decisions
        ]) if self.protected_core.key_decisions else "  (none yet)"
        
        protected_core_section = f"""PROTECTED CORE (AUTHORITATIVE - Never forget these):
================================================
Original Goal: {self.protected_core.original_goal}
Current Goal: {self.protected_core.current_goal}

Hard Constraints (MUST FOLLOW):
{chr(10).join(f'  - {c}' for c in self.protected_core.hard_constraints)}

Key Decisions Made:
{decisions_str}

Last Updated: {self.protected_core.timestamp_updated}
================================================

INSTRUCTION: Always prioritize the CURRENT GOAL and HARD CONSTRAINTS above all else.
If there's any ambiguity, refer back to this Protected Core as the source of truth."""
        
        parts.append(protected_core_section)
        
        if halo_summary:
            parts.append(f"\n--- Previous Conversation Summary ---\n{halo_summary}")
        
        if recent_turns:
            parts.append("\n--- Recent Turns (Raw) ---")
            for turn in recent_turns:
                turn_id = turn.get("id", "?")
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                parts.append(f"Turn {turn_id} ({role}): {content}")
        
        return "\n\n".join(parts)
    
    def render_reconstructed_prompt(self, context: List[Dict[str, Any]]) -> str:
        """
        Render the full prompt as it would appear after compression.
        
        This is used to check token budgets before actually compressing.
        """
        split_point = max(0, len(context) - self.keep_recent_turns)
        halo = context[:split_point]
        recent = context[split_point:]
        halo_summary = "[Compressed conversation summary would go here]"
        
        return self._format_context_with_protected_core(halo_summary, recent)
    
    def name(self) -> str:
        return "Strategy F - Protected Core + Goal Re-assertion"


# Convenience function for quick testing
def create_protected_core_strategy(system_prompt: str = "") -> StrategyF_ProtectedCore:
    """Create a Protected Core compression strategy."""
    return StrategyF_ProtectedCore(system_prompt=system_prompt)

