# instinct8 - Tests and Evaluations
# Run 'make help' to see available commands

.PHONY: help test test-quick test-integration eval-quick eval-full eval-hierarchical eval-compare eval-rigorous eval-binary eval-all clean

# Default target
help:
	@echo ""
	@echo "instinct8 - Tests and Evaluations"
	@echo "=================================="
	@echo ""
	@echo "Unit Tests:"
	@echo "  make test              Run all tests (~10s)"
	@echo "  make test-quick        Fast unit tests only, no API keys needed"
	@echo "  make test-integration  API-dependent integration tests only"
	@echo ""
	@echo "Quick Evaluations:"
	@echo "  make eval-quick        Quick eval - 5 samples (~2m)"
	@echo "  make eval-hierarchical Hierarchical depth eval (~5m)"
	@echo ""
	@echo "Full Evaluations:"
	@echo "  make eval-full         Full LoCoMo benchmark (~30m)"
	@echo "  make eval-compare      Compare all strategies (~15m)"
	@echo "  make eval-rigorous     Publication-ready eval (~1hr)"
	@echo ""
	@echo "Combined:"
	@echo "  make eval-all          Run test + eval-quick + eval-hierarchical"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean             Remove cached results"
	@echo ""

# =============================================================================
# UNIT TESTS
# =============================================================================

test:
	@echo "Running all tests..."
	python3 -m pytest tests/ -v

test-quick:
	@echo "Running quick tests (skipping integration)..."
	python3 -m pytest tests/ -v -x --tb=short -m "not integration"

test-integration:
	@echo "Running integration tests (requires API keys)..."
	python3 -m pytest tests/ -v -m "integration"

# =============================================================================
# QUICK EVALUATIONS (< 5 minutes)
# =============================================================================

eval-quick:
	@echo "Running quick evaluation (5 samples)..."
	python scripts/run_eval.py --dataset locomo --max-samples 5

eval-hierarchical:
	@echo "Running hierarchical compression evaluation..."
	python scripts/run_hierarchical_eval.py --strategy amem

eval-hierarchical-compare:
	@echo "Comparing hierarchical strategies..."
	python scripts/run_hierarchical_eval.py --compare

# =============================================================================
# FULL EVALUATIONS (15-60 minutes)
# =============================================================================

eval-full:
	@echo "Running full LoCoMo evaluation..."
	python scripts/run_eval.py --dataset locomo

eval-compare:
	@echo "Comparing all compression strategies..."
	python scripts/run_eval.py --compare

eval-rigorous:
	@echo "Running publication-ready evaluation (3 runs per sample)..."
	python scripts/run_eval.py --rigorous --n-runs 3

# =============================================================================
# BINARY TESTING (requires Codex CLI installed)
# =============================================================================
# Binary testing scripts removed in cleanup - use unified evaluation harness

# =============================================================================
# COMBINED RUNS
# =============================================================================

eval-all: test eval-quick eval-hierarchical
	@echo ""
	@echo "All evaluations complete!"
	@echo "Results saved to: results/"

# =============================================================================
# UTILITIES
# =============================================================================

clean:
	@echo "Cleaning cached results..."
	rm -rf results/*.json
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."

# Show results
results:
	@echo "Recent results:"
	@ls -lt results/*.json 2>/dev/null | head -10 || echo "No results found"
