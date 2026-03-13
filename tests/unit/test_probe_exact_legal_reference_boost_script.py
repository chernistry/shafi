from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "scripts" / "probe_exact_legal_reference_boost.py"
    spec = importlib.util.spec_from_file_location("probe_exact_legal_reference_boost", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_best_target_doc_rank_uses_unique_doc_order() -> None:
    module = _load_module()

    class Chunk:
        def __init__(self, doc_id: str) -> None:
            self.doc_id = doc_id

    chunks = [Chunk("docA"), Chunk("docA"), Chunk("docB"), Chunk("docC"), Chunk("docB")]
    assert module._best_target_doc_rank(chunks, ("docB",)) == 2
    assert module._best_target_doc_rank(chunks, ("docC",)) == 3
    assert module._best_target_doc_rank(chunks, ("missing",)) is None


def test_select_target_cases_filters_to_exact_non_case_law_questions(tmp_path: Path) -> None:
    module = _load_module()
    miss_pack = tmp_path / "miss_pack.json"
    questions = {
        "q1": "According to Article 10 of the Employment Law 2019, what is the filing deadline?",
        "q2": "What is the Date of Issue in CFI 057/2025?",
        "q3": "Who won the case?",
    }
    miss_pack.write_text(
        json.dumps(
            {
                "cases": [
                    {"qid": "q1", "target_doc_ids": ["doc-employment"], "miss_family": "same_doc", "route": "strict"},
                    {"qid": "q2", "target_doc_ids": ["doc-case"], "miss_family": "same_doc", "route": "strict"},
                    {"qid": "q3", "target_doc_ids": ["doc-other"], "miss_family": "same_doc", "route": "strict"},
                ]
            }
        ),
        encoding="utf-8",
    )

    def exact_refs(question: str) -> list[str]:
        if "Article 10" in question:
            return ["Article 10", "Employment Law 2019"]
        return []

    def doc_refs(question: str) -> list[str]:
        if "CFI 057/2025" in question:
            return ["CFI 057/2025"]
        return []

    cases = module._select_target_cases(
        miss_pack_path=miss_pack,
        questions_by_id=questions,
        exact_refs_fn=exact_refs,
        doc_refs_fn=doc_refs,
    )

    assert [case.qid for case in cases] == ["q1"]
