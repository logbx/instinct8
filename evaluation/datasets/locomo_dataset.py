"""
LoComo Dataset Adapter

This module provides a dataset adapter for the LoComo (Long-form Conversation Memory)
dataset used in A-mem evaluation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseDataset, EvalQuestion, EvalSample, EvalTurn

_amem_path = Path(__file__).resolve().parent.parent.parent / "data" / "A-mem"


class LoCoMoDataset(BaseDataset):
    """
    Dataset adapter for the LoComo dataset.

    The LoComo dataset contains multi-session conversations with associated
    QA pairs for evaluating memory-based agents.

    Each sample contains:
    - Multiple conversation sessions with timestamped turns
    - QA pairs with categories (1-5) including adversarial questions
    - Event summaries and observations
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        ratio: float = 1.0,
        categories: Optional[List[int]] = None,
    ):
        """
        Initialize the LoComo dataset.

        Args:
            dataset_path: Path to the LoComo JSON file
            ratio: Fraction of dataset to use (0.0 to 1.0)
            categories: Optional list of question categories to include (1-5)
        """
        self._path = Path(dataset_path)
        self._ratio = ratio
        self._categories = categories or [1, 2, 3, 4, 5]
        self._samples: List[EvalSample] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load and convert the LoComo dataset."""
        # Lazy import to avoid loading dependencies until needed. Appended so
        # data/A-mem cannot shadow same-named modules elsewhere on sys.path.
        if str(_amem_path) not in sys.path:
            sys.path.append(str(_amem_path))
        from load_dataset import load_locomo_dataset

        raw_samples = load_locomo_dataset(self._path)

        # Apply ratio
        if self._ratio < 1.0:
            num_samples = max(1, int(len(raw_samples) * self._ratio))
            raw_samples = raw_samples[:num_samples]

        # Convert samples
        for idx, sample in enumerate(raw_samples):
            eval_sample = self._convert_sample(sample, idx)
            self._samples.append(eval_sample)

    def _convert_sample(self, sample, idx: int) -> EvalSample:
        """Convert a LoCoMoSample to EvalSample."""
        # Convert conversation turns
        turns = []
        turn_id = 0

        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]
            for turn in session.turns:
                turns.append(
                    EvalTurn(
                        id=turn_id,
                        role=turn.speaker.lower(),
                        content=turn.text,
                        timestamp=session.date_time,
                        speaker=turn.speaker,
                    )
                )
                turn_id += 1

        # Convert QA pairs (filtering by category if specified)
        questions = []
        for qa in sample.qa:
            if qa.category in self._categories:
                questions.append(
                    EvalQuestion(
                        question=qa.question,
                        reference_answer=qa.answer or "",
                        category=qa.category,
                        evidence=qa.evidence,
                        adversarial_answer=qa.adversarial_answer,
                    )
                )

        return EvalSample(
            sample_id=str(idx),
            turns=turns,
            questions=questions,
            metadata={
                "session_summary": sample.session_summary,
                "speaker_a": sample.conversation.speaker_a,
                "speaker_b": sample.conversation.speaker_b,
                "num_sessions": len(sample.conversation.sessions),
            },
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self._samples[idx]

    @property
    def name(self) -> str:
        return f"LoComo({self._path.stem}, ratio={self._ratio})"

    @property
    def evaluation_type(self) -> str:
        return "qa"

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the dataset."""
        base_stats = super().get_statistics()

        # Category distribution
        category_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for sample in self._samples:
            for q in sample.questions:
                if q.category:
                    category_counts[q.category] = (
                        category_counts.get(q.category, 0) + 1
                    )

        base_stats["category_distribution"] = category_counts
        base_stats["num_sessions"] = sum(
            s.metadata.get("num_sessions", 0) for s in self._samples
        )

        return base_stats
