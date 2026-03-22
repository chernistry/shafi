"""Teacher-label builders for dense retriever and reranker training."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from shafi.ml.hard_negative_miner import HardNegativeMiner, NegativeCandidate, NegativeStrategy
from shafi.models.legal_objects import CaseObject, CorpusRegistry, LawObject

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from shafi.ml.grounding_dataset import GroundingMlRow
    from shafi.ml.synthetic_qa_factory import BridgeFactRecord, SyntheticQAExample


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class TrainingTriple:
    """One query/positive/negative training triple."""

    query: str
    positive_text: str
    positive_page_id: str
    negative_text: str
    negative_page_id: str
    source: str
    difficulty: str
    strategy: str


@dataclass(frozen=True, slots=True)
class TeacherLabelStats:
    """Aggregate teacher-label statistics."""

    total_triples: int
    by_source: dict[str, int]
    by_difficulty: dict[str, int]
    by_strategy: dict[str, int]


class TeacherLabelBuilder:
    """Build trainable triples from offline label sources."""

    def __init__(
        self,
        *,
        miner: HardNegativeMiner,
        page_texts: Mapping[str, str],
        page_doc_ids: Mapping[str, str],
    ) -> None:
        """Initialize the builder.

        Args:
            miner: Hard-negative miner.
            page_texts: Page text keyed by page ID.
            page_doc_ids: Parent doc IDs keyed by page ID.
        """
        self._miner = miner
        self._page_texts = dict(page_texts)
        self._page_doc_ids = dict(page_doc_ids)

    def build_from_grounding_rows(self, rows: Sequence[GroundingMlRow]) -> list[TrainingTriple]:
        """Build triples from reviewed/golden grounding rows."""
        triples: list[TrainingTriple] = []
        for row in rows:
            positive_page_ids = list(row.label_page_ids or row.sidecar_selected_pages or row.legacy_selected_pages)
            if not positive_page_ids:
                continue
            positive_text_by_page = {
                candidate.page_id: candidate.snippet_excerpt
                for candidate in row.page_candidates
                if candidate.page_id in positive_page_ids and candidate.snippet_excerpt
            }
            gold_doc_ids = [
                self._page_doc_ids.get(page_id, page_id.rpartition("_")[0]) for page_id in positive_page_ids
            ]
            negatives = self._miner.mine(query=row.question, gold_page_ids=positive_page_ids, gold_doc_ids=gold_doc_ids)
            triples.extend(
                self._triples_from_positive_pages(
                    query=row.question,
                    positive_page_ids=positive_page_ids,
                    positive_text_by_page=positive_text_by_page,
                    negatives=negatives,
                    source=f"grounding:{row.label_source}",
                    difficulty=_difficulty_from_scope(row.scope_mode),
                )
            )
        return triples

    def build_from_synthetic_qa(self, examples: Sequence[SyntheticQAExample]) -> list[TrainingTriple]:
        """Build triples from synthetic QA examples."""
        triples: list[TrainingTriple] = []
        for example in examples:
            if not example.gold_page_ids:
                continue
            positive_text_by_page = {
                page_id: self._page_texts.get(page_id, example.answer) for page_id in example.gold_page_ids
            }
            negatives = [
                NegativeCandidate(
                    page_id=page_id,
                    text=self._page_texts.get(page_id, ""),
                    strategy=NegativeStrategy.LEXICAL_NEAR_MISS,
                    similarity_score=1.0,
                    doc_id=self._page_doc_ids.get(page_id, page_id.rpartition("_")[0]),
                )
                for page_id in example.hard_negative_page_ids
                if self._page_texts.get(page_id, "")
            ]
            if not negatives:
                negatives = self._miner.mine(
                    query=example.question,
                    gold_page_ids=example.gold_page_ids,
                    gold_doc_ids=example.gold_doc_ids,
                )
            triples.extend(
                self._triples_from_positive_pages(
                    query=example.question,
                    positive_page_ids=example.gold_page_ids,
                    positive_text_by_page=positive_text_by_page,
                    negatives=negatives,
                    source=f"synthetic:{example.question_family.value}",
                    difficulty=example.difficulty,
                )
            )
        return triples

    def build_from_compiled_registry(self, registry: CorpusRegistry) -> list[TrainingTriple]:
        """Build triples from compiled field values in the corpus registry."""
        triples: list[TrainingTriple] = []
        for law in sorted(registry.laws.values(), key=lambda item: item.doc_id):
            triples.extend(self._triples_from_law(law))
        for case in sorted(registry.cases.values(), key=lambda item: item.doc_id):
            triples.extend(self._triples_from_case(case))
        return triples

    def build_from_bridge_facts(self, bridge_facts: Sequence[BridgeFactRecord]) -> list[TrainingTriple]:
        """Build triples from external bridge facts."""
        triples: list[TrainingTriple] = []
        for record in bridge_facts:
            if not record.page_ids or not record.answer:
                continue
            negatives = self._miner.mine(
                query=record.question,
                gold_page_ids=record.page_ids,
                gold_doc_ids=record.doc_ids,
            )
            positive_text_by_page = {
                page_id: self._page_texts.get(page_id, record.answer) for page_id in record.page_ids
            }
            triples.extend(
                self._triples_from_positive_pages(
                    query=record.question,
                    positive_page_ids=record.page_ids,
                    positive_text_by_page=positive_text_by_page,
                    negatives=negatives,
                    source="bridge_fact",
                    difficulty="medium",
                )
            )
        return triples

    def combine_and_deduplicate(self, groups: Sequence[Sequence[TrainingTriple]]) -> list[TrainingTriple]:
        """Combine multiple triple groups and remove duplicates."""
        deduped: dict[tuple[str, str, str], TrainingTriple] = {}
        for triple in [item for group in groups for item in group]:
            key = (triple.query.casefold(), triple.positive_page_id, triple.negative_page_id)
            deduped.setdefault(key, triple)
        return sorted(deduped.values(), key=lambda item: (item.query, item.positive_page_id, item.negative_page_id))

    def denoise_false_negatives(self, triples: Sequence[TrainingTriple]) -> list[TrainingTriple]:
        """Remove triples whose negative text appears to contain the positive answer."""
        denoised: list[TrainingTriple] = []
        for triple in triples:
            positive_tokens = set(_tokenize(triple.positive_text))
            negative_tokens = set(_tokenize(triple.negative_text))
            if positive_tokens and len(positive_tokens & negative_tokens) >= max(3, len(positive_tokens) // 2):
                continue
            denoised.append(triple)
        return denoised

    def statistics(self, triples: Sequence[TrainingTriple]) -> TeacherLabelStats:
        """Compute aggregate triple statistics."""
        by_source = Counter(triple.source for triple in triples)
        by_difficulty = Counter(triple.difficulty for triple in triples)
        by_strategy = Counter(triple.strategy for triple in triples)
        return TeacherLabelStats(
            total_triples=len(triples),
            by_source=dict(sorted(by_source.items())),
            by_difficulty=dict(sorted(by_difficulty.items())),
            by_strategy=dict(sorted(by_strategy.items())),
        )

    def _triples_from_law(self, law: LawObject) -> list[TrainingTriple]:
        if not law.page_ids:
            return []
        prompts: list[tuple[str, str]] = []
        if law.title:
            prompts.append((f"What is the title of {law.doc_id}?", law.title))
        if law.issuing_authority:
            prompts.append((f"Who issued {law.short_title or law.title}?", law.issuing_authority))
        if law.commencement_date:
            prompts.append((f"When did {law.short_title or law.title} come into force?", law.commencement_date))
        triples: list[TrainingTriple] = []
        for query, answer in prompts:
            negatives = self._miner.mine(query=query, gold_page_ids=law.page_ids, gold_doc_ids=[law.doc_id])
            triples.extend(
                self._triples_from_positive_pages(
                    query=query,
                    positive_page_ids=law.page_ids[:1],
                    positive_text_by_page={law.page_ids[0]: self._page_texts.get(law.page_ids[0], answer)},
                    negatives=negatives,
                    source="compiled:law",
                    difficulty="easy",
                )
            )
        return triples

    def _triples_from_case(self, case: CaseObject) -> list[TrainingTriple]:
        if not case.page_ids:
            return []
        prompts: list[tuple[str, str]] = []
        if case.judges:
            prompts.append((f"Which judge decided {case.case_number or case.title}?", case.judges[0]))
        if case.parties:
            prompts.append((f"Who is a party in {case.case_number or case.title}?", case.parties[0].name))
        triples: list[TrainingTriple] = []
        for query, answer in prompts:
            negatives = self._miner.mine(query=query, gold_page_ids=case.page_ids, gold_doc_ids=[case.doc_id])
            triples.extend(
                self._triples_from_positive_pages(
                    query=query,
                    positive_page_ids=case.page_ids[:1],
                    positive_text_by_page={case.page_ids[0]: self._page_texts.get(case.page_ids[0], answer)},
                    negatives=negatives,
                    source="compiled:case",
                    difficulty="medium",
                )
            )
        return triples

    def _triples_from_positive_pages(
        self,
        *,
        query: str,
        positive_page_ids: Sequence[str],
        positive_text_by_page: Mapping[str, str],
        negatives: Sequence[NegativeCandidate],
        source: str,
        difficulty: str,
    ) -> list[TrainingTriple]:
        triples: list[TrainingTriple] = []
        for positive_page_id in positive_page_ids:
            positive_text = positive_text_by_page.get(
                positive_page_id, self._page_texts.get(positive_page_id, "")
            ).strip()
            if not positive_text:
                continue
            for negative in negatives:
                if not negative.text.strip():
                    continue
                triples.append(
                    TrainingTriple(
                        query=query,
                        positive_text=positive_text,
                        positive_page_id=positive_page_id,
                        negative_text=negative.text,
                        negative_page_id=negative.page_id,
                        source=source,
                        difficulty=difficulty,
                        strategy=negative.strategy.value,
                    )
                )
        return triples


def build_page_texts_from_registry(
    registry: CorpusRegistry,
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Build page text, page-doc, and alias maps from a corpus registry."""
    page_texts: dict[str, str] = {}
    page_doc_ids: dict[str, str] = {}
    aliases_by_doc_id: dict[str, list[str]] = {}
    for bucket in (
        registry.laws.values(),
        registry.cases.values(),
        registry.orders.values(),
        registry.practice_directions.values(),
        registry.amendments.values(),
        registry.other_documents.values(),
    ):
        for obj in bucket:
            aliases: list[str] = [obj.title]
            if isinstance(obj, LawObject):
                aliases.extend([obj.short_title, obj.law_number])
            if isinstance(obj, CaseObject):
                aliases.extend([obj.case_number, *[party.name for party in obj.parties]])
            aliases_by_doc_id[obj.doc_id] = [alias for alias in aliases if alias]
            for page_id, text in obj.page_texts.items():
                if page_id and text:
                    page_texts[page_id] = text
                    page_doc_ids[page_id] = obj.doc_id
    return page_texts, page_doc_ids, aliases_by_doc_id


def write_training_triples_jsonl(path: Path, triples: Sequence[TrainingTriple]) -> None:
    """Write training triples as deterministic JSONL."""
    lines = [json.dumps(asdict(triple), ensure_ascii=True, sort_keys=True) for triple in triples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_pairwise_labels_jsonl(path: Path, triples: Sequence[TrainingTriple]) -> None:
    """Write pairwise labels derived from training triples."""
    rows: list[str] = []
    for triple in triples:
        rows.append(
            json.dumps(
                {
                    "query": triple.query,
                    "text": triple.positive_text,
                    "page_id": triple.positive_page_id,
                    "label": 1,
                    "source": triple.source,
                    "difficulty": triple.difficulty,
                    "strategy": triple.strategy,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        rows.append(
            json.dumps(
                {
                    "query": triple.query,
                    "text": triple.negative_text,
                    "page_id": triple.negative_page_id,
                    "label": 0,
                    "source": triple.source,
                    "difficulty": triple.difficulty,
                    "strategy": triple.strategy,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _difficulty_from_scope(scope_mode: str) -> str:
    if "negative" in scope_mode:
        return "hard"
    if "multi" in scope_mode:
        return "medium"
    return "easy"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]
