"""Integration test: Strategy I (A-MEM + Protected Core) on product-design template.

Runs Strategy I on the healthcare product-design template to verify
the hybrid strategy works correctly.
"""

import pytest
from pathlib import Path

from evaluation.template_utils import load_template, run_single_trial, TrialResult
from strategies.strategy_i_hybrid_amem_protected import StrategyI_AMemProtectedCore


PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "product-design-009-healthcare-8k-4compactions.json"


@pytest.mark.integration
class TestStrategyIProductDesign:
    """Test Strategy I on the healthcare product-design template."""

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        """Skip if template or API keys are missing."""
        import os

        if not TEMPLATE_PATH.exists():
            pytest.skip(f"Template not found: {TEMPLATE_PATH.name}")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API key available")

    def test_strategy_i_initializes(self):
        """Strategy I should initialise with both subsystems active."""
        template = load_template(str(TEMPLATE_PATH))
        setup = template["initial_setup"]

        strategy = StrategyI_AMemProtectedCore(
            system_prompt=setup["system_prompt"],
            backend="auto",
        )
        strategy.initialize(setup["original_goal"], setup["hard_constraints"])

        assert strategy is not None
        assert strategy.name() == "Strategy I - A-MEM + Protected Core Hybrid"

    def test_strategy_i_single_trial(self):
        """Strategy I should complete a single trial without error."""
        template = load_template(str(TEMPLATE_PATH))
        setup = template["initial_setup"]

        strategy = StrategyI_AMemProtectedCore(
            system_prompt=setup["system_prompt"],
            backend="auto",
        )
        strategy.initialize(setup["original_goal"], setup["hard_constraints"])

        result = run_single_trial(
            strategy=strategy,
            template=template,
            trial_id=1,
            use_granular_metrics=True,
        )

        assert result.summary is not None
        assert len(result.compression_points) > 0
        assert isinstance(result.summary.get("avg_goal_coherence_after", 0), (int, float))
