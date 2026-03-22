from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedFunction=false
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import fitz

try:
    from probe_local_embedding_relevance import _parse_page_id
except ModuleNotFoundError:  # pragma: no cover
    from scripts.probe_local_embedding_relevance import _parse_page_id

if TYPE_CHECKING:
    import argparse

    from rag_challenge.core.local_page_reranker import PageRerankScore

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class QidSelection:
    question_id: str
    question: str
    source_doc_ids: list[str]
    baseline_page_ids: list[str]
    candidate_page_ids: list[str]
    selected_page_ids: list[str]
    top_scored_pages: list[PageRerankScore]


def _deepcopy_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _qid_set(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in getattr(args, "qid", []) or []:
        qid = str(raw).strip()
        if qid and qid not in seen:
            out.append(qid)
            seen.add(qid)
    qids_file = getattr(args, "qids_file", None)
    if isinstance(qids_file, Path):
        for line in qids_file.read_text(encoding="utf-8").splitlines():
            qid = line.strip()
            if qid and not qid.startswith("#") and qid not in seen:
                out.append(qid)
                seen.add(qid)
    if not out:
        raise ValueError("No QIDs provided")
    return out


def _baseline_doc_pages(page_ids: list[str]) -> dict[str, list[int]]:
    by_doc: dict[str, list[int]] = {}
    for page_id in page_ids:
        doc_id, page_number = _parse_page_id(page_id)
        pages = by_doc.setdefault(doc_id, [])
        if page_number not in pages:
            pages.append(page_number)
    return {doc_id: sorted(pages) for doc_id, pages in by_doc.items()}


def _pdf_page_count(pdf_path: Path, cache: dict[str, int]) -> int:
    key = str(pdf_path.resolve())
    cached = cache.get(key)
    if cached is not None:
        return cached
    doc = cast("Any", fitz.open(str(pdf_path)))
    try:
        page_count = int(doc.page_count)
    finally:
        doc.close()
    cache[key] = page_count
    return page_count


def _candidate_pages_for_doc(
    *,
    doc_id: str,
    baseline_pages: list[int],
    dataset_dir: Path,
    page_count_cache: dict[str, int],
    include_page_one: bool,
    include_page_two: bool,
    include_last_page: bool,
    neighbor_radius: int,
    max_pages_per_doc: int,
) -> list[str]:
    pdf_path = dataset_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return []
    page_count = _pdf_page_count(pdf_path, page_count_cache)

    candidate_pages: list[int] = []
    seen_pages: set[int] = set()

    def add_page(page_number: int) -> None:
        if page_number <= 0 or page_number > page_count or page_number in seen_pages:
            return
        seen_pages.add(page_number)
        candidate_pages.append(page_number)

    if include_page_one:
        add_page(1)
    if include_page_two:
        add_page(2)

    for page_number in baseline_pages:
        add_page(page_number)
        radius = max(0, int(neighbor_radius))
        for delta in range(1, radius + 1):
            add_page(page_number - delta)
            add_page(page_number + delta)

    if include_last_page:
        add_page(page_count)

    if max_pages_per_doc > 0:
        candidate_pages = candidate_pages[: max_pages_per_doc]

    return [f"{doc_id}_{page_number}" for page_number in candidate_pages]


def _select_pages(
    *,
    scored_pages: list[PageRerankScore],
    per_doc_pages: int,
    extra_global_pages: int,
) -> list[str]:
    by_doc: dict[str, list[PageRerankScore]] = {}
    for row in scored_pages:
        doc_id, _ = _parse_page_id(row.page_id)
        by_doc.setdefault(doc_id, []).append(row)

    selected: list[str] = []
    seen: set[str] = set()
    for doc_id in sorted(by_doc):
        doc_rows = sorted(by_doc[doc_id], key=lambda item: (-item.score, item.page_id))
        for row in doc_rows[: max(0, per_doc_pages)]:
            if row.page_id not in seen:
                selected.append(row.page_id)
                seen.add(row.page_id)

    if extra_global_pages > 0:
        total_limit = len(selected) + extra_global_pages
        for row in sorted(scored_pages, key=lambda item: (-item.score, item.page_id)):
            if row.page_id in seen:
                continue
            selected.append(row.page_id)
            seen.add(row.page_id)
            if len(selected) >= total_limit:
                break

    return selected


def _render_markdown(*, label: str, model: str, selections: list[QidSelection]) -> str:
    lines = [
        "# Local Page Reranker Candidate",
        "",
        f"- label: `{label}`",
        f"- model: `{model}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    for selection in selections:
        lines.extend(
            [
                f"## {selection.question_id}",
                "",
                f"- question: {selection.question}",
                f"- source_doc_ids: `{selection.source_doc_ids}`",
                f"- baseline_page_ids: `{selection.baseline_page_ids}`",
                f"- candidate_page_ids: `{selection.candidate_page_ids}`",
                f"- selected_page_ids: `{selection.selected_page_ids}`",
                "",
                "| Rank | Page | Score |",
                "| --- | --- | ---: |",
            ]
        )
        for index, row in enumerate(selection.top_scored_pages, start=1):
            lines.append(f"| {index} | `{row.page_id}` | {row.score:.4f} |")
        lines.append("")
    return "\n".join(lines)
