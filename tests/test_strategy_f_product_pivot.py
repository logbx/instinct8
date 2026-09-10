"""Integration test: Strategy F (Protected Core) on product-pivot template.

Runs Strategy F on the multi-shift product-pivot template to verify
that Protected Core handles multiple goal shifts correctly.
"""

import pytest
from pathlib import Path

from evaluation.template_utils import load_template, run_single_trial, TrialResult
from strategies.strategy_f_protected_core import StrategyF_ProtectedCore


PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "product-pivot-015-multi-shift-8k-4compactions.json"


@pytest.mark.integration
class TestStrategyFProductPivot:
    """Test Strategy F on the multi-shift product-pivot template."""

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        """Skip if template or API keys are missing."""
        import os

        if not TEMPLATE_PATH.exists():
            pytest.skip(f"Template not found: {TEMPLATE_PATH.name}")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API key available")

    def test_strategy_f_initializes(self):
        """Strategy F should initialise with Protected Core active."""
        template = load_template(str(TEMPLATE_PATH))
        setup = template["initial_setup"]

        strategy = StrategyF_ProtectedCore(
            system_prompt=setup["system_prompt"],
            backend="auto",
        )
        strategy.initialize(setup["original_goal"], setup["hard_constraints"])

        assert strategy.protected_core.original_goal == setup["original_goal"]
        assert len(strategy.protected_core.hard_constraints) == len(setup["hard_constraints"])

    def test_strategy_f_single_trial(self):
        """Strategy F should complete a single trial without error."""
        template = load_template(str(TEMPLATE_PATH))
        setup = template["initial_setup"]

        strategy = StrategyF_ProtectedCore(
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

        # Goal coherence scores should be numeric
        assert isinstance(result.summary.get("avg_goal_coherence_after", 0), (int, float))
