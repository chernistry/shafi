from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import fitz

from rag_challenge.config.settings import IngestionSettings

JsonDict = dict[str, object]


@dataclass(frozen=True)
class OcrRiskDoc:
    doc_id: str
    path: str
    page_count: int
    fast_text_chars: int
    fast_text_words: int
    fallback_reason: str
    anchor_sensitive: bool
    anchor_sensitive_question_ids: list[str]


def _load_scaffold(path: Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    payload = cast("JsonDict", payload_obj)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", records_obj):
        if not isinstance(raw, dict):
            continue
        record = cast("JsonDict", raw)
        question_id = str(record.get("question_id") or "").strip()
        if question_id:
            out[question_id] = record
    return out


def _parse_pdf_pymupdf_pages(path: Path) -> list[str]:
    pages: list[str] = []
    fitz_any = cast("Any", fitz)
    with fitz_any.open(str(path)) as pdf_obj:
        for page_obj in cast("list[object]", list(pdf_obj)):
            page_any = cast("Any", page_obj)
            page_text_obj = page_any.get_text("text")
            if not isinstance(page_text_obj, str):
                continue
            page_text = page_text_obj.strip()
            if page_text:
                pages.append(page_text)
    return pages


def _is_pdf_text_sufficient(text: str, *, min_chars: int, min_words: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    word_count = len(stripped.split())
    if len(stripped) < min_chars and word_count < min_words:
        return False
    visible_chars = sum(1 for char in stripped if char.isalnum())
    density = visible_chars / max(len(stripped), 1)
    return density >= 0.2


def _load_seed_qids(path: Path | None) -> list[str]:
    if path is None:
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            out.append(text)
    return out


def _record_doc_ids(record: JsonDict) -> set[str]:
    doc_ids: set[str] = set()
    for raw in cast("list[object]", record.get("retrieved_chunk_pages") or []):
        if not isinstance(raw, dict):
            continue
        doc_id = str(cast("JsonDict", raw).get("doc_id") or "").strip()
        if doc_id:
            doc_ids.add(doc_id)
    for raw in cast("list[object]", record.get("minimal_required_support_pages") or []):
        text = str(raw).strip()
        if not text:
            continue
        doc_id = text.split("_", 1)[0]
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def _anchor_sensitive_doc_index(records: dict[str, JsonDict], seed_qids: list[str]) -> dict[str, list[str]]:
    qid_filter = set(seed_qids)
    out: dict[str, list[str]] = {}
    for question_id, record in records.items():
        if qid_filter and question_id not in qid_filter:
            continue
        if not qid_filter:
            required_page_anchor = cast("JsonDict", record.get("required_page_anchor") or {})
            failure_class = str(record.get("failure_class") or "").strip().lower()
            if not required_page_anchor and failure_class != "support_undercoverage":
                continue
        for doc_id in _record_doc_ids(record):
            out.setdefault(doc_id, []).append(question_id)
    for qids in out.values():
        qids.sort()
    return out


def audit_docs(
    *,
    docs_dir: Path,
    scaffold_path: Path | None,
    seed_qids_path: Path | None,
) -> tuple[list[OcrRiskDoc], dict[str, object]]:
    ingestion_settings = IngestionSettings()
    records = _load_scaffold(scaffold_path)
    seed_qids = _load_seed_qids(seed_qids_path)
    anchor_doc_index = _anchor_sensitive_doc_index(records, seed_qids)
    risk_docs: list[OcrRiskDoc] = []
    pdf_paths = sorted(path for path in docs_dir.rglob("*.pdf") if path.is_file())
    fallback_reason_counts: Counter[str] = Counter()

    for path in pdf_paths:
        pages = _parse_pdf_pymupdf_pages(path)
        fast_text = "\n\n".join(text for text in pages if text).strip()
        if _is_pdf_text_sufficient(
            fast_text,
            min_chars=ingestion_settings.parser_pdf_text_min_chars,
            min_words=ingestion_settings.parser_pdf_text_min_words,
        ):
            continue
        page_count = len(pages)
        if page_count <= 1:
            continue
        fast_text_chars = len(fast_text)
        fast_text_words = len(fast_text.split())
        fallback_reason = "empty_fast_text" if not fast_text else "insufficient_fast_text"
        fallback_reason_counts[fallback_reason] += 1
        doc_id = path.stem
        anchor_qids = anchor_doc_index.get(doc_id, [])
        risk_docs.append(
            OcrRiskDoc(
                doc_id=doc_id,
                path=str(path),
                page_count=page_count,
                fast_text_chars=fast_text_chars,
                fast_text_words=fast_text_words,
                fallback_reason=fallback_reason,
                anchor_sensitive=bool(anchor_qids),
                anchor_sensitive_question_ids=anchor_qids,
            )
        )

    risk_docs.sort(key=lambda item: (not item.anchor_sensitive, -item.page_count, item.doc_id))
    summary: dict[str, object] = {
        "pdf_count": len(pdf_paths),
        "ocr_single_page_risk_count": len(risk_docs),
        "anchor_sensitive_risk_count": sum(1 for item in risk_docs if item.anchor_sensitive),
        "fallback_reason_counts": dict(fallback_reason_counts),
    }
    return risk_docs, summary


def render_report(*, risk_docs: list[OcrRiskDoc], summary: dict[str, object]) -> str:
    lines = [
        "# OCR Page Boundary Audit",
        "",
        f"- PDF count: `{summary['pdf_count']}`",
        f"- OCR single-page risk count: `{summary['ocr_single_page_risk_count']}`",
        f"- Anchor-sensitive risk count: `{summary['anchor_sensitive_risk_count']}`",
        f"- Fallback reason counts: `{summary['fallback_reason_counts']}`",
        "",
    ]
    for item in risk_docs[:50]:
        lines.extend(
            [
                f"## {item.doc_id}",
                f"- page_count: `{item.page_count}`",
                f"- fast_text_chars: `{item.fast_text_chars}`",
                f"- fast_text_words: `{item.fast_text_words}`",
                f"- fallback_reason: `{item.fallback_reason}`",
                f"- anchor_sensitive: `{item.anchor_sensitive}`",
                f"- anchor_sensitive_question_ids: `{', '.join(item.anchor_sensitive_question_ids) if item.anchor_sensitive_question_ids else '(none)'}`",
                f"- path: `{item.path}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multi-page PDFs that would collapse into a single synthetic page on OCR fallback.")
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, default=None)
    parser.add_argument("--seed-qids-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    risk_docs, summary = audit_docs(
        docs_dir=args.docs_dir,
        scaffold_path=args.scaffold,
        seed_qids_path=args.seed_qids_file,
    )
    report = render_report(risk_docs=risk_docs, summary=summary)
    payload = {
        "summary": summary,
        "risk_docs": [asdict(item) for item in risk_docs],
    }

    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
