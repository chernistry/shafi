from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


def _gold_chunk_ids_factory() -> list[str]:
    return []


@dataclass(frozen=True)
class GoldenCase:
    """Single eval case loaded from JSON golden dataset."""

    case_id: str
    question: str
    answer_type: str = "free_text"
    gold_chunk_ids: list[str] = field(default_factory=_gold_chunk_ids_factory)
    expected_short_answer: str = ""
    doc_ids: list[str] | None = None
    tags: list[str] | None = None

    def to_expected_output(self) -> str:
        parts = [f"answer_type={self.answer_type}"]
        if self.gold_chunk_ids:
            parts.append(f"gold_chunk_ids={','.join(self.gold_chunk_ids)}")
        return "; ".join(parts)


def load_golden_dataset(path: str | Path) -> list[GoldenCase]:
    """Load golden dataset JSON.

    Format:
    [
      {
        "id": "q-001",
        "question": "...",
        "answer_type": "boolean",
        "gold_chunk_ids": ["doc1:0:0:abc"],  # optional until annotations exist
        "expected_short_answer": "...",  # optional
        "doc_ids": ["doc1"],             # optional
        "tags": ["statute"]              # optional
      }
    ]
    """
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a JSON array")

    cases: list[GoldenCase] = []
    for index, item in enumerate(cast("list[object]", data)):
        if not isinstance(item, dict):
            raise ValueError(f"Golden dataset item #{index} must be an object")
        cases.append(_parse_case(cast("dict[str, object]", item), index=index))
    return cases


def _parse_case(item: dict[str, object], *, index: int) -> GoldenCase:
    question_obj = item.get("question")
    case_id_obj = item.get("id", item.get("case_id", ""))
    answer_type_obj = item.get("answer_type", "free_text")
    gold_ids_obj = item.get("gold_chunk_ids", [])
    if not isinstance(question_obj, str) or not question_obj.strip():
        raise ValueError(f"Golden dataset item #{index} missing non-empty 'question'")
    if not isinstance(case_id_obj, str) or not case_id_obj.strip():
        case_id = f"case-{index + 1:04d}"
    else:
        case_id = case_id_obj.strip()
    if not isinstance(answer_type_obj, str) or not answer_type_obj.strip():
        answer_type = "free_text"
    else:
        answer_type = answer_type_obj.strip()
    if not isinstance(gold_ids_obj, list):
        raise ValueError(f"Golden dataset item #{index} field 'gold_chunk_ids' must be a list when present")

    gold_ids_list = cast("list[object]", gold_ids_obj)
    gold_chunk_ids = [text for chunk_id in gold_ids_list if (text := str(chunk_id).strip())]

    doc_ids = _optional_str_list(item.get("doc_ids"))
    tags = _optional_str_list(item.get("tags"))
    expected_short_answer_obj = item.get("expected_short_answer", "")
    expected_short_answer = (
        expected_short_answer_obj if isinstance(expected_short_answer_obj, str) else str(expected_short_answer_obj)
    )

    return GoldenCase(
        case_id=case_id,
        question=question_obj.strip(),
        answer_type=answer_type,
        gold_chunk_ids=gold_chunk_ids,
        expected_short_answer=expected_short_answer.strip(),
        doc_ids=doc_ids,
        tags=tags,
    )


def _optional_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Optional list field must be a JSON array when present")
    items = cast("list[object]", value)
    cleaned = [text for item in items if (text := str(item).strip())]
    return cleaned or None
