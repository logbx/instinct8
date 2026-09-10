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
	@echo "Binary Testing:"
	@echo "  make eval-binary       Test real Codex binary (~10m)"
	@echo "  make eval-appserver    App-server compaction eval (~5m)"
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

eval-binary:
	@echo "Testing real Codex binary..."
	python scripts/run_codex_cli_eval.py

eval-binary-compare:
	@echo "A/B testing Codex variants..."
	@echo "Usage: python scripts/run_comparison_eval.py --name <variant-name>"

eval-appserver:
	@echo "Running app-server compaction evaluation..."
	python scripts/run_appserver_eval.py

eval-appserver-verbose:
	@echo "Running app-server eval with verbose output..."
	python scripts/run_appserver_eval.py --verbose --turns 15

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
