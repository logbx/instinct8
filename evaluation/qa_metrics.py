"""
QA Metrics Adapter

This module wraps the A-mem utils.py metric functions to conform
to the unified MetricCalculator interface.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metric_interfaces import MetricCalculator, MetricResult, MetricType

_amem_path = Path(__file__).resolve().parent.parent / "data" / "A-mem"

# Loaded under a private name so neither an unrelated top-level `utils` module
# nor A-mem's own utils.py can shadow the other in sys.modules.
_AMEM_UTILS_MODULE = "_amem_utils"

# Lazy imports to avoid loading heavy dependencies until needed
_utils_loaded = False
_calculate_rouge_scores = None
_calculate_bleu_scores = None
_calculate_bert_scores = None
_calculate_meteor_score = None
_calculate_sentence_similarity = None
_simple_tokenize = None


def _load_amem_utils():
    """Lazily load A-mem utils to avoid heavy imports at module load time."""
    global _utils_loaded, _calculate_rouge_scores, _calculate_bleu_scores
    global _calculate_bert_scores, _calculate_meteor_score
    global _calculate_sentence_similarity, _simple_tokenize

    if _utils_loaded:
        return

    utils_file = _amem_path / "utils.py"
    try:
        if not utils_file.is_file():
            raise ImportError(f"{utils_file} not found")

        # utils.py does `from load_dataset import ...`, so its directory must be
        # importable. Appended so it cannot shadow same-named modules elsewhere.
        if str(_amem_path) not in sys.path:
            sys.path.append(str(_amem_path))

        spec = importlib.util.spec_from_file_location(_AMEM_UTILS_MODULE, utils_file)
        utils = importlib.util.module_from_spec(spec)
        sys.modules[_AMEM_UTILS_MODULE] = utils
        try:
            spec.loader.exec_module(utils)
        except BaseException:
            sys.modules.pop(_AMEM_UTILS_MODULE, None)
            raise

        _calculate_rouge_scores = utils.calculate_rouge_scores
        _calculate_bleu_scores = utils.calculate_bleu_scores
        _calculate_bert_scores = utils.calculate_bert_scores
        _calculate_meteor_score = utils.calculate_meteor_score
        _calculate_sentence_similarity = utils.calculate_sentence_similarity
        _simple_tokenize = utils.simple_tokenize
        _utils_loaded = True
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import A-mem utils. Ensure A-mem dependencies are installed: {e}"
        ) from e


class QAMetricCalculator:
    """
    Unified calculator for all QA/retrieval metrics from A-mem.

    Wraps the A-mem utils metric functions and converts output to
    MetricResult objects for unified aggregation.

    Metrics calculated:
    - exact_match: Binary match (0 or 1)
    - f1: Token-level F1 score
    - rouge1_f, rouge2_f, rougeL_f: ROUGE F-measure scores
    - bleu1, bleu2, bleu3, bleu4: BLEU n-gram scores
    - bert_precision, bert_recall, bert_f1: BERTScore metrics
    - meteor: METEOR score
    - sbert_similarity: Sentence-BERT cosine similarity
    """

    QA_METRIC_NAMES = [
        "exact_match",
        "f1",
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "bert_precision",
        "bert_recall",
        "bert_f1",
        "meteor",
        "sbert_similarity",
    ]

    @property
    def metric_names(self) -> List[str]:
        return self.QA_METRIC_NAMES

    def calculate(
        self,
        prediction: str,
        reference: str,
        category: Optional[int] = None,
        **kwargs,
    ) -> List[MetricResult]:
        """
        Calculate all QA metrics for a prediction-reference pair.

        Args:
            prediction: Model's predicted answer
            reference: Ground truth answer
            category: Question category (1-5) for A-mem datasets
            **kwargs: Additional arguments (ignored)

        Returns:
            List of MetricResult objects, one per metric
        """
        _load_amem_utils()

        results = []
        category_str = str(category) if category is not None else None

        # Handle empty inputs
        if not prediction or not reference:
            for name in self.QA_METRIC_NAMES:
                results.append(
                    MetricResult(
                        name=name,
                        value=0.0,
                        metric_type=MetricType.SCORE_0_1,
                        category=category_str,
                    )
                )
            return results

        prediction = str(prediction).strip()
        reference = str(reference).strip()

        # Exact match
        exact_match = int(prediction.lower() == reference.lower())
        results.append(
            MetricResult(
                name="exact_match",
                value=exact_match,
                metric_type=MetricType.SCORE_0_1,
                category=category_str,
            )
        )

        # Token F1
        pred_tokens = set(_simple_tokenize(prediction))
        ref_tokens = set(_simple_tokenize(reference))
        common = pred_tokens & ref_tokens

        if pred_tokens and ref_tokens:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            f1 = 0.0

        results.append(
            MetricResult(
                name="f1",
                value=f1,
                metric_type=MetricType.SCORE_0_1,
                category=category_str,
            )
        )

        # ROUGE scores
        rouge_scores = _calculate_rouge_scores(prediction, reference)
        for key, value in rouge_scores.items():
            results.append(
                MetricResult(
                    name=key,
                    value=value,
                    metric_type=MetricType.SCORE_0_1,
                    category=category_str,
                )
            )

        # BLEU scores
        bleu_scores = _calculate_bleu_scores(prediction, reference)
        for key, value in bleu_scores.items():
            results.append(
                MetricResult(
                    name=key,
                    value=value,
                    metric_type=MetricType.SCORE_0_1,
                    category=category_str,
                )
            )

        # BERT scores
        bert_scores = _calculate_bert_scores(prediction, reference)
        for key, value in bert_scores.items():
            results.append(
                MetricResult(
                    name=key,
                    value=value,
                    metric_type=MetricType.SCORE_0_1,
                    category=category_str,
                )
            )

        # METEOR
        meteor = _calculate_meteor_score(prediction, reference)
        results.append(
            MetricResult(
                name="meteor",
                value=meteor,
                metric_type=MetricType.SCORE_0_1,
                category=category_str,
            )
        )

        # Sentence similarity
        sbert = _calculate_sentence_similarity(prediction, reference)
        results.append(
            MetricResult(
                name="sbert_similarity",
                value=sbert,
                metric_type=MetricType.SCORE_0_1,
                category=category_str,
            )
        )

        return results

    def calculate_batch(
        self,
        predictions: List[str],
        references: List[str],
        categories: Optional[List[int]] = None,
    ) -> List[List[MetricResult]]:
        """
        Calculate metrics for multiple prediction-reference pairs.

        Args:
            predictions: List of model predictions
            references: List of ground truth answers
            categories: Optional list of question categories

        Returns:
            List of metric result lists, one per pair
        """
        if categories is None:
            categories = [None] * len(predictions)

        results = []
        for pred, ref, cat in zip(predictions, references, categories):
            results.append(self.calculate(prediction=pred, reference=ref, category=cat))

        return results
