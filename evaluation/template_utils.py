"""
Template-based evaluation utilities.

Extracted from harness.py to support legacy template-based tests
while unified_harness.py handles dataset-based evaluation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from strategies.strategy_base import CompressionStrategy
from evaluation.metrics import MetricsCollector
from evaluation.goal_tracking import track_goal_evolution


@dataclass
class TrialResult:
    """Results from a single trial of a strategy on a template."""
    trial_id: int
    strategy_name: str
    template_id: str
    compression_points: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    granular_constraint_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "trial_id": self.trial_id,
            "strategy_name": self.strategy_name,
            "template_id": self.template_id,
            "compression_points": self.compression_points,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }
        if self.granular_constraint_metrics:
            result["granular_constraint_metrics"] = self.granular_constraint_metrics
        return result


class MockAgent:
    """
    A mock agent for running evaluation trials.
    
    This simulates an agent with a context that can be compressed.
    It uses OpenAI to generate responses based on its current context.
    """
    
    def __init__(
        self,
        strategy: CompressionStrategy,
        system_prompt: str,
        original_goal: str,
        constraints: List[str],
        model: str = "gpt-4o",
    ):
        self.client = OpenAI()
        self.model = model
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.original_goal = original_goal
        self.constraints = constraints
        
        self.strategy.initialize(original_goal, constraints)
        self.context: List[Dict[str, Any]] = []
        self.total_tokens = 0
    
    def add_turn(self, turn: Dict[str, Any]) -> None:
        self.context.append(turn)
        # Approximate: ~4 chars per token
        self.total_tokens += len(turn.get("content", "")) // 4
    
    def compress(self, trigger_point: int) -> str:
        """
        Compress the context using the configured strategy.
        
        Returns the compressed context string.
        """
        compressed = self.strategy.compress(self.context, trigger_point)
        
        self.total_tokens = len(compressed) // 4
        self.context = [{
            "id": 0,
            "role": "system",
            "content": compressed,
            "is_compression_point": False,
        }]
        
        return compressed
    
    def call(self, prompt: str) -> str:
        """
        Call the agent with a prompt and get a response.
        
        The agent responds based on its current context (which may be compressed).
        """
        context_text = self._format_current_context()
        messages = [{
            "role": "user",
            "content": f"Context:\n{context_text}\n\n---\n\nQuestion: {prompt}"
        }]
        
        try:
            api_messages = []
            if self.system_prompt:
                api_messages.append({"role": "system", "content": self.system_prompt})
            api_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                messages=api_messages,
            )
            content = response.choices[0].message.content
            return content if content else "(error: empty response)"
        except Exception as e:
            print(f"[agent] Error calling OpenAI: {e}")
            return f"(error: {e})"
    
    def _format_current_context(self) -> str:
        """Format the current context for prompting."""
        parts = []
        for turn in self.context:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)
    
    def get_token_count(self) -> int:
        """Get approximate token count of current context."""
        return self.total_tokens


def load_template(template_path: str) -> Dict[str, Any]:
    """Load a conversation template from JSON file."""
    with open(template_path, "r") as f:
        return json.load(f)


def run_single_trial(
    strategy: CompressionStrategy,
    template: Dict[str, Any],
    trial_id: int,
    use_granular_metrics: bool = False,
) -> TrialResult:
    """
    Run a single trial of a strategy on a template.
    
    Args:
        strategy: The compression strategy to test
        template: The evaluation template
        trial_id: Trial identifier
        use_granular_metrics: If True, enables granular constraint metrics with category breakdowns
    
    Returns TrialResult with metrics at each compression point.
    """
    print(f"\n=== Trial {trial_id} ===")
    
    # Extract template info
    initial_setup = template["initial_setup"]
    original_goal = initial_setup["original_goal"]
    constraints = initial_setup["hard_constraints"]
    system_prompt = initial_setup["system_prompt"]
    turns = template["turns"]
    probing_tasks = template.get("probing_tasks", {})
    
    # Create agent with strategy
    agent = MockAgent(
        strategy=strategy,
        system_prompt=system_prompt,
        original_goal=original_goal,
        constraints=constraints,
    )
    
    goal_timeline = track_goal_evolution(turns, original_goal, constraints)
    collector = MetricsCollector(
        original_goal=original_goal,
        constraints=constraints,
        use_granular_metrics=use_granular_metrics,
        goal_timeline=goal_timeline,
    )
    
    compression_point_counter = 0
    
    for turn in turns:
        turn_id = turn["turn_id"]
        print(f"  Turn {turn_id}...", end="")
        
        agent.add_turn({
            "id": turn_id,
            "role": turn["role"],
            "content": turn["content"],
            "is_compression_point": turn.get("is_compression_point", False),
            "tool_call": turn.get("tool_call"),
            "decision": turn.get("decision"),
        })
        
        if turn.get("is_compression_point", False):
            compression_point_counter += 1
            print(f" [COMPRESSION POINT {compression_point_counter}]")
            
            tokens_before = agent.get_token_count()
            goal_before = agent.call(probing_tasks.get(
                "goal_probe",
                "In one sentence, what is your current goal?"
            ))
            constraints_before = agent.call(probing_tasks.get(
                "constraint_probe",
                "What constraints are you operating under?"
            ))
            
            agent.compress(turn_id)
            tokens_after = agent.get_token_count()
            
            goal_after = agent.call(probing_tasks.get(
                "goal_probe",
                "In one sentence, what is your current goal?"
            ))
            constraints_after = agent.call(probing_tasks.get(
                "constraint_probe",
                "What constraints are you operating under?"
            ))
            
            if compression_point_counter == 1:
                print(f"    [DEBUG CP1] Goal after: {goal_after[:100]}...")
                print(f"    [DEBUG CP1] Constraints after: {constraints_after[:200]}...")
            
            behavioral_test = probing_tasks.get("behavioral_test", {})
            behavioral_prompt = behavioral_test.get(
                "prompt",
                "What should we do next?"
            )
            behavioral_after = agent.call(behavioral_prompt)
            
            # Collect salience metrics if using Strategy H
            extracted_salience = None
            ground_truth_salience = None
            
            # Lazy import to avoid circular dependency
            try:
                from strategies.strategy_h_selective_salience import SelectiveSalienceStrategy
                if isinstance(strategy, SelectiveSalienceStrategy):
                    # Get extracted salience from strategy's salience_set
                    extracted_salience = strategy.salience_set.copy() if hasattr(strategy, 'salience_set') else None
                    
                    # Get ground truth salience from template if available
                    ground_truth_salience = template.get("ground_truth_salience", {}).get(
                        f"compression_point_{compression_point_counter}",
                        None
                    )
            except ImportError:
                pass
            
            metrics = collector.collect_at_compression_point(
                compression_point_id=compression_point_counter,
                turn_id=turn_id,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                goal_stated_before=goal_before,
                goal_stated_after=goal_after,
                constraints_stated_before=constraints_before,
                constraints_stated_after=constraints_after,
                behavioral_response_after=behavioral_after,
                behavioral_test_context=behavioral_prompt,
                extracted_salience=extracted_salience,
                ground_truth_salience=ground_truth_salience,
            )
            
            print(f"    Goal drift: {metrics.goal_drift:.2f}")
            print(f"    Constraint loss: {metrics.constraint_loss:.2f}")
            print(f"    Compression ratio: {metrics.compression_ratio:.2f}")
        else:
            print(" ok")
    
    results = collector.get_results()
    granular_metrics = results.get("granular_constraint_metrics")
    
    trial_result = TrialResult(
        trial_id=trial_id,
        strategy_name=strategy.name(),
        template_id=template["template_id"],
        compression_points=results["compression_points"],
        summary=results["summary"],
    )
    
    if granular_metrics:
        trial_result.granular_constraint_metrics = granular_metrics
    
    return trial_result
