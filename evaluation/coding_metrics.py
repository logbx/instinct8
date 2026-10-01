"""
Coding Metrics Calculator

This module provides metrics for evaluating code generation, bug fixing,
refactoring, and research synthesis tasks.
"""

import ast
import difflib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .metric_interfaces import MetricResult, MetricType


# Metric names for coding evaluation
CODING_METRICS = [
    "syntax_validity",
    "test_pass_rate",
    "ast_similarity",
    "code_embedding_similarity",
    "requirements_met",
    "token_efficiency",
    "compression_retention",
    "diff_accuracy",
]


@dataclass
class TestCase:
    """A test case for code evaluation."""

    name: str
    input: str
    expected_output: str
    description: Optional[str] = None


@dataclass
class CodingEvalInput:
    """Input for coding evaluation."""

    generated_code: str
    expected_code: Optional[str] = None
    expected_files: Optional[Dict[str, str]] = None
    test_cases: Optional[List[TestCase]] = None
    acceptance_criteria: Optional[List[str]] = None
    language: str = "python"
    task_type: str = "code_generation"


class CodingMetricCalculator:
    """
    Calculator for code-specific metrics.

    Implements the MetricCalculator protocol for coding tasks:
    - Syntax validity
    - Test pass rate
    - AST similarity
    - Code embedding similarity
    - Requirements coverage
    - Token efficiency
    - Compression retention
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the coding metric calculator.

        Args:
            use_embeddings: Whether to use embedding-based similarity
            embedding_model: Model name for code embeddings
        """
        self._use_embeddings = use_embeddings
        self._embedding_model = embedding_model
        self._encoder = None

    @property
    def metric_names(self) -> List[str]:
        """Return names of all metrics this calculator produces."""
        return CODING_METRICS

    def calculate(
        self,
        generated_code: str,
        expected_code: Optional[str] = None,
        expected_files: Optional[Dict[str, str]] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        language: str = "python",
        task_type: str = "code_generation",
        tokens_before_compression: Optional[int] = None,
        tokens_after_compression: Optional[int] = None,
        **kwargs,
    ) -> List[MetricResult]:
        """
        Calculate all coding metrics.

        Args:
            generated_code: The generated code to evaluate
            expected_code: Expected/reference code
            expected_files: Expected file contents (for multi-file tasks)
            test_cases: List of test cases
            acceptance_criteria: List of acceptance criteria strings
            language: Programming language
            task_type: Type of coding task
            tokens_before_compression: Token count before compression
            tokens_after_compression: Token count after compression

        Returns:
            List of MetricResult objects
        """
        results = []

        # Syntax validity
        syntax_valid, syntax_errors = self.check_syntax_validity(
            generated_code, language
        )
        results.append(
            MetricResult(
                name="syntax_validity",
                value=1.0 if syntax_valid else 0.0,
                metric_type=MetricType.SCORE_0_1,
                metadata={"errors": syntax_errors} if syntax_errors else None,
            )
        )

        # Test pass rate (if test cases provided)
        if test_cases:
            converted_cases = [
                TestCase(
                    name=tc.get("name", ""),
                    input=tc.get("input", ""),
                    expected_output=tc.get("expected_output", tc.get("expected", "")),
                    description=tc.get("description"),
                )
                for tc in test_cases
            ]
            pass_rate, test_results = self.calculate_test_pass_rate(
                generated_code, converted_cases, language
            )
            results.append(
                MetricResult(
                    name="test_pass_rate",
                    value=pass_rate,
                    metric_type=MetricType.SCORE_0_1,
                    metadata={"test_results": test_results},
                )
            )

        # AST similarity (if expected code provided)
        if expected_code:
            ast_sim = self.calculate_ast_similarity(
                generated_code, expected_code, language
            )
            results.append(
                MetricResult(
                    name="ast_similarity",
                    value=ast_sim,
                    metric_type=MetricType.SCORE_0_1,
                )
            )

            # Code embedding similarity
            if self._use_embeddings:
                embed_sim = self.calculate_code_embedding_similarity(
                    generated_code, expected_code
                )
                results.append(
                    MetricResult(
                        name="code_embedding_similarity",
                        value=embed_sim,
                        metric_type=MetricType.SCORE_0_1,
                    )
                )

        # Requirements coverage (if acceptance criteria provided)
        if acceptance_criteria:
            req_met = self.calculate_requirements_met(
                generated_code, acceptance_criteria
            )
            results.append(
                MetricResult(
                    name="requirements_met",
                    value=req_met,
                    metric_type=MetricType.SCORE_0_1,
                )
            )

        # Token efficiency (compression metrics)
        if tokens_before_compression is not None and tokens_after_compression is not None:
            efficiency = self.calculate_token_efficiency(
                tokens_before_compression, tokens_after_compression
            )
            results.append(
                MetricResult(
                    name="token_efficiency",
                    value=efficiency,
                    metric_type=MetricType.PERCENTAGE,
                )
            )

            retention = self.calculate_compression_retention(
                generated_code, expected_code, tokens_before_compression, tokens_after_compression
            )
            results.append(
                MetricResult(
                    name="compression_retention",
                    value=retention,
                    metric_type=MetricType.SCORE_0_1,
                )
            )

        # Diff accuracy for refactoring/bug fixing
        if task_type in ("refactoring", "bug_fixing") and expected_files:
            diff_acc = self.calculate_diff_accuracy(generated_code, expected_files)
            results.append(
                MetricResult(
                    name="diff_accuracy",
                    value=diff_acc,
                    metric_type=MetricType.SCORE_0_1,
                )
            )

        return results

    def check_syntax_validity(
        self, code: str, language: str = "python"
    ) -> Tuple[bool, List[str]]:
        """
        Check if code has valid syntax.

        Args:
            code: Source code to check
            language: Programming language

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if language == "python":
            try:
                ast.parse(code)
                return True, []
            except SyntaxError as e:
                return False, [f"Line {e.lineno}: {e.msg}"]
            except ValueError as e:
                # Python < 3.11.4 raises ValueError (not SyntaxError) for null bytes.
                return False, [str(e)]
        elif language in ("javascript", "typescript"):
            # Try using node to check syntax
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".js", delete=False
                ) as f:
                    f.write(code)
                    f.flush()
                    result = subprocess.run(
                        ["node", "--check", f.name],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        return True, []
                    return False, [result.stderr]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fall back to basic check
                return self._basic_syntax_check(code), []
        else:
            # Basic heuristic for other languages
            return self._basic_syntax_check(code), []

    def _basic_syntax_check(self, code: str) -> bool:
        """Basic syntax check using bracket matching."""
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        in_string = False
        string_char = None

        for char in code:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif not in_string:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack.pop() != char:
                        return False

        return len(stack) == 0

    def calculate_test_pass_rate(
        self,
        code: str,
        test_cases: List[TestCase],
        language: str = "python",
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate the percentage of test cases that pass.

        Note: This is a mock implementation. In production, tests would
        be executed in a sandboxed environment.

        Args:
            code: Generated code
            test_cases: List of test cases
            language: Programming language

        Returns:
            Tuple of (pass_rate, test_results)
        """
        if not test_cases:
            return 1.0, []

        # Mock implementation - in production, execute tests in sandbox
        test_results = []
        passed = 0

        for tc in test_cases:
            # Simple heuristic: check if expected patterns are in the code
            result = {
                "name": tc.name,
                "passed": False,
                "reason": "",
            }

            # Check if code contains expected elements
            if tc.expected_output:
                # Simple keyword matching (placeholder for real test execution)
                keywords = re.findall(r"\b\w+\b", tc.expected_output.lower())
                code_lower = code.lower()
                matches = sum(1 for kw in keywords if kw in code_lower)
                result["passed"] = matches > len(keywords) * 0.5
                result["reason"] = (
                    "Code contains expected patterns"
                    if result["passed"]
                    else "Missing expected patterns"
                )

            if result["passed"]:
                passed += 1
            test_results.append(result)

        pass_rate = passed / len(test_cases) if test_cases else 1.0
        return pass_rate, test_results

    def calculate_ast_similarity(
        self,
        generated: str,
        expected: str,
        language: str = "python",
    ) -> float:
        """
        Calculate structural similarity between code using AST comparison.

        Args:
            generated: Generated code
            expected: Expected code
            language: Programming language

        Returns:
            Similarity score 0.0-1.0
        """
        if language != "python":
            # Fall back to text similarity for non-Python
            return self._text_similarity(generated, expected)

        try:
            gen_ast = ast.parse(generated)
            exp_ast = ast.parse(expected)

            gen_nodes = self._get_ast_node_types(gen_ast)
            exp_nodes = self._get_ast_node_types(exp_ast)

            if not gen_nodes and not exp_nodes:
                return 1.0
            if not gen_nodes or not exp_nodes:
                return 0.0

            # Jaccard similarity of node types
            intersection = len(gen_nodes & exp_nodes)
            union = len(gen_nodes | exp_nodes)

            return intersection / union if union > 0 else 0.0

        except (SyntaxError, ValueError):
            # Fall back to text similarity
            return self._text_similarity(generated, expected)

    def _get_ast_node_types(self, tree: ast.AST) -> set:
        """Extract all node types from an AST."""
        nodes = set()
        for node in ast.walk(tree):
            nodes.add(type(node).__name__)
        return nodes

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching."""
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def calculate_code_embedding_similarity(
        self, generated: str, expected: str
    ) -> float:
        """
        Calculate semantic similarity using code embeddings.

        Args:
            generated: Generated code
            expected: Expected code

        Returns:
            Cosine similarity score 0.0-1.0
        """
        try:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._embedding_model)

            gen_embedding = self._encoder.encode(generated, convert_to_tensor=True)
            exp_embedding = self._encoder.encode(expected, convert_to_tensor=True)

            # Cosine similarity
            from sentence_transformers.util import cos_sim
            similarity = cos_sim(gen_embedding, exp_embedding).item()

            return max(0.0, min(1.0, similarity))

        except ImportError:
            # Fall back to text similarity
            return self._text_similarity(generated, expected)

    def calculate_requirements_met(
        self, code: str, acceptance_criteria: List[str]
    ) -> float:
        """
        Calculate what fraction of acceptance criteria are likely met.

        Uses keyword matching as a heuristic. In production, this could
        use LLM-based evaluation.

        Args:
            code: Generated code
            acceptance_criteria: List of requirement strings

        Returns:
            Fraction of requirements met 0.0-1.0
        """
        if not acceptance_criteria:
            return 1.0

        code_lower = code.lower()
        met_count = 0

        for criterion in acceptance_criteria:
            # Extract keywords from criterion
            keywords = re.findall(r"\b\w+\b", criterion.lower())
            # Filter out common words
            stopwords = {"the", "a", "an", "is", "are", "should", "must", "be"}
            keywords = [kw for kw in keywords if kw not in stopwords]

            if keywords:
                # Check if most keywords appear in code
                matches = sum(1 for kw in keywords if kw in code_lower)
                if matches >= len(keywords) * 0.5:
                    met_count += 1

        return met_count / len(acceptance_criteria)

    def calculate_token_efficiency(
        self, tokens_before: int, tokens_after: int
    ) -> float:
        """
        Calculate token efficiency after compression.

        Args:
            tokens_before: Token count before compression
            tokens_after: Token count after compression

        Returns:
            Compression ratio (0.0-1.0, lower means more compression)
        """
        if tokens_before == 0:
            return 1.0
        return tokens_after / tokens_before

    def calculate_compression_retention(
        self,
        generated_after: str,
        expected: Optional[str],
        tokens_before: int,
        tokens_after: int,
    ) -> float:
        """
        Calculate how well information is retained after compression.

        Combines compression ratio with code quality preservation.

        Args:
            generated_after: Code generated after compression
            expected: Expected code (if available)
            tokens_before: Token count before compression
            tokens_after: Token count after compression

        Returns:
            Retention score 0.0-1.0
        """
        compression_ratio = self.calculate_token_efficiency(tokens_before, tokens_after)

        if expected:
            quality = self.calculate_ast_similarity(generated_after, expected, "python")
        else:
            # Assume syntax validity as quality proxy
            valid, _ = self.check_syntax_validity(generated_after)
            quality = 1.0 if valid else 0.5

        # Balance compression with quality retention
        # Higher is better: compress a lot while keeping quality
        if compression_ratio == 0:
            return quality

        # Score = quality * (1 - compression_ratio) + bonus for maintaining quality
        score = quality * (2 - compression_ratio) / 2
        return max(0.0, min(1.0, score))

    def calculate_diff_accuracy(
        self, generated: str, expected_files: Dict[str, str]
    ) -> float:
        """
        Calculate accuracy of code changes for refactoring/bug fixing.

        Args:
            generated: Generated code (may be multiple files concatenated)
            expected_files: Dict of filename -> expected content

        Returns:
            Average similarity across expected files 0.0-1.0
        """
        if not expected_files:
            return 1.0

        total_similarity = 0.0

        for filename, expected_content in expected_files.items():
            # Try to find matching section in generated code
            # This is a simplified heuristic
            similarity = self._text_similarity(generated, expected_content)
            total_similarity += similarity

        return total_similarity / len(expected_files)
