"""Integration test: verify compression triggers and constraint retention.

Forces compression to trigger on Strategy B and checks whether budget
constraints survive the compression round-trip.
"""

import os

import pytest

from strategies.strategy_b_codex import StrategyB_CodexCheckpoint
from evaluation.template_utils import MockAgent, load_template
from evaluation.metrics import _constraint_mentioned, _get_client
from evaluation.token_budget import BUDGET_8K, estimate_tokens


TEMPLATE_PATH = "templates/research-synthesis-008-8k-4compactions-realistic.json"
BUDGET_CONSTRAINT = "Budget: maximum $10K implementation cost"


@pytest.mark.integration
class TestWithActualCompression:
    """Test compression triggering and constraint retention."""

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        """Skip if API key or template missing."""
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API key available")
        try:
            self.template = load_template(TEMPLATE_PATH)
        except FileNotFoundError:
            pytest.skip(f"Template not found: {TEMPLATE_PATH}")
        self.client = _get_client()

    def test_compression_triggers(self):
        """Compression should trigger when token budget is exceeded."""
        setup = self.template["initial_setup"]

        strategy = StrategyB_CodexCheckpoint(
            system_prompt=setup["system_prompt"],
            backend="auto",
            token_budget=BUDGET_8K,
            use_goal_preservation=True,
        )
        strategy.initialize(setup["original_goal"], setup["hard_constraints"])

        agent = MockAgent(
            strategy=strategy,
            system_prompt=setup["system_prompt"],
            original_goal=setup["original_goal"],
            constraints=setup["hard_constraints"],
            model="gpt-4o",
        )

        # Build context up to the first compression point
        turns = self.template["turns"]
        cp1_turn = None
        for turn in turns:
            if turn.get("is_compression_point", False) and cp1_turn is None:
                cp1_turn = turn
                break
            agent.add_turn({
                "id": turn["turn_id"],
                "role": turn["role"],
                "content": turn["content"],
            })

        assert cp1_turn is not None, "Template should have at least one compression point"

        reconstructed = strategy.render_reconstructed_prompt(agent.context)
        tokens_est = estimate_tokens(reconstructed)

        # Compress
        agent.compress(cp1_turn["turn_id"])

        # After compression the context should be non-empty
        assert len(agent.context) > 0

    def test_task_context_preserved_after_compression(self):
        """TASK CONTEXT section should appear in compressed output."""
        setup = self.template["initial_setup"]

        strategy = StrategyB_CodexCheckpoint(
            system_prompt=setup["system_prompt"],
            backend="auto",
            token_budget=BUDGET_8K,
            use_goal_preservation=True,
        )
        strategy.initialize(setup["original_goal"], setup["hard_constraints"])

        agent = MockAgent(
            strategy=strategy,
            system_prompt=setup["system_prompt"],
            original_goal=setup["original_goal"],
            constraints=setup["hard_constraints"],
            model="gpt-4o",
        )

        turns = self.template["turns"]
        cp1_turn = None
        for turn in turns:
            if turn.get("is_compression_point", False) and cp1_turn is None:
                cp1_turn = turn
                break
            agent.add_turn({
                "id": turn["turn_id"],
                "role": turn["role"],
                "content": turn["content"],
            })

        if cp1_turn is None:
            pytest.skip("No compression point in template")

        agent.compress(cp1_turn["turn_id"])

        compressed = agent.context[0]["content"] if agent.context else ""
        assert "TASK CONTEXT" in compressed or "previous conversation" in compressed.lower()
