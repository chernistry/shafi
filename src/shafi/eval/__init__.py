from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shafi.eval.golden import GoldenCase, load_golden_dataset
    from shafi.eval.harness import EvalResult, run_evaluation
    from shafi.eval.metrics import AnswerTypeFormatCompliance, CitationCoverage, GoldChunkRecallAtK

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
        module = import_module("shafi.eval.golden")
        return getattr(module, name)
    if name in {"EvalResult", "run_evaluation"}:
        module = import_module("shafi.eval.harness")
        return getattr(module, name)
    if name in {"AnswerTypeFormatCompliance", "CitationCoverage", "GoldChunkRecallAtK"}:
        module = import_module("shafi.eval.metrics")
        return getattr(module, name)
    raise AttributeError(f"module 'shafi.eval' has no attribute {name!r}")
