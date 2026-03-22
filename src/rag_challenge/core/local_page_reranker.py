from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from rag_challenge.models import RankedChunk

_PAGE_RE = re.compile(r"page:(\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PageRerankScore:
    page_id: str
    score: float
    best_chunk_id: str = ""
    chunk_ids: tuple[str, ...] = ()


def _page_num(section_path: str | None) -> int:
    match = _PAGE_RE.search(section_path or "")
    if match is None:
        return 10_000
    try:
        return int(match.group(1))
    except ValueError:
        return 10_000


def score_pages_from_chunk_scores(
    *,
    chunks: Sequence[RankedChunk],
    doc_ids: Collection[str],
    page_one_bias: float = 0.0,
    early_page_bias: float = 0.0,
) -> list[PageRerankScore]:
    selected_doc_ids = {str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()}
    if not selected_doc_ids:
        return []

    grouped: dict[tuple[str, int], list[tuple[int, RankedChunk]]] = {}
    for index, chunk in enumerate(chunks):
        doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        if doc_id not in selected_doc_ids:
            continue
        page_num = _page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num <= 0 or page_num >= 10_000:
            continue
        grouped.setdefault((doc_id, page_num), []).append((index, chunk))

    scored: list[PageRerankScore] = []
    for (doc_id, page_num), entries in grouped.items():
        ranked_entries = sorted(
            entries,
            key=lambda item: (
                float(getattr(item[1], "rerank_score", 0.0) or 0.0),
                float(getattr(item[1], "retrieval_score", 0.0) or 0.0),
                -item[0],
            ),
            reverse=True,
        )
        best_index, best_chunk = ranked_entries[0]
        top_rerank = float(getattr(best_chunk, "rerank_score", 0.0) or 0.0)
        top_retrieval = float(getattr(best_chunk, "retrieval_score", 0.0) or 0.0)
        second_rerank = (
            float(getattr(ranked_entries[1][1], "rerank_score", 0.0) or 0.0) if len(ranked_entries) > 1 else 0.0
        )
        chunk_count_bonus = min(len(ranked_entries), 3) * 0.01
        rank_bonus = max(0, 16 - best_index) * 0.0005
        position_bias = 0.0
        if page_num == 1:
            position_bias += max(0.0, page_one_bias)
        elif page_num == 2:
            position_bias += max(0.0, early_page_bias)
        score = (
            top_rerank
            + (0.15 * top_retrieval)
            + (0.05 * second_rerank)
            + chunk_count_bonus
            + rank_bonus
            + position_bias
        )
        page_id = f"{doc_id}_{page_num}"
        scored.append(
            PageRerankScore(
                page_id=page_id,
                score=score,
                best_chunk_id=str(getattr(best_chunk, "chunk_id", "") or ""),
                chunk_ids=tuple(str(getattr(chunk, "chunk_id", "") or "") for _, chunk in ranked_entries),
            )
        )

    return sorted(scored, key=lambda item: (-item.score, item.page_id))


def select_top_pages_per_doc(
    *,
    scored_pages: Sequence[PageRerankScore],
    doc_order: Sequence[str] | None = None,
    per_doc_pages: int = 1,
) -> list[PageRerankScore]:
    if per_doc_pages <= 0 or not scored_pages:
        return []

    by_doc: dict[str, list[PageRerankScore]] = {}
    for row in scored_pages:
        doc_id, _, _page = row.page_id.rpartition("_")
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(row)

    ordered_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    for doc_id in doc_order or ():
        normalized = str(doc_id).strip()
        if normalized and normalized in by_doc and normalized not in seen_doc_ids:
            ordered_doc_ids.append(normalized)
            seen_doc_ids.add(normalized)
    for doc_id in sorted(by_doc):
        if doc_id not in seen_doc_ids:
            ordered_doc_ids.append(doc_id)
            seen_doc_ids.add(doc_id)

    selected: list[PageRerankScore] = []
    for doc_id in ordered_doc_ids:
        doc_rows = sorted(by_doc.get(doc_id, []), key=lambda item: (-item.score, item.page_id))
        selected.extend(doc_rows[:per_doc_pages])
    return selected
