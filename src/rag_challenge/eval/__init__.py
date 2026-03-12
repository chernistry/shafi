from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_challenge.eval.golden import GoldenCase, load_golden_dataset
    from rag_challenge.eval.harness import EvalResult, run_evaluation
    from rag_challenge.eval.metrics import AnswerTypeFormatCompliance, CitationCoverage, GoldChunkRecallAtK

__all__ = [
    "AnswerTypeFormatCompliance",
    "CitationCoverage",
    "EvalResult",
    "GoldChunkRecallAtK",
    "GoldenCase",
    "load_golden_dataset",
    "run_evaluation",
]


def __getattr__(name: str) -> Any:
    if name in {"GoldenCase", "load_golden_dataset"}:
        module = import_module("rag_challenge.eval.golden")
        return getattr(module, name)
    if name in {"EvalResult", "run_evaluation"}:
        module = import_module("rag_challenge.eval.harness")
        return getattr(module, name)
    if name in {"AnswerTypeFormatCompliance", "CitationCoverage", "GoldChunkRecallAtK"}:
        module = import_module("rag_challenge.eval.metrics")
        return getattr(module, name)
    raise AttributeError(f"module 'rag_challenge.eval' has no attribute {name!r}")
