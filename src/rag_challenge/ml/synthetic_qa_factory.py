"""Deterministic synthetic QA generation over compiled corpus artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from rag_challenge.models.applicability import ApplicabilityGraph
from rag_challenge.models.legal_objects import CaseObject, CorpusRegistry, LawObject
from rag_challenge.models.schemas import LegalSegment, SegmentType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path


type JsonObject = dict[str, object]
type ParaphraseFn = Callable[[str], Sequence[str]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_COUNTERFACTUAL_TOPIC_HINTS = ("registration", "licensing", "authority", "court", "procedure")


class QuestionFamily(StrEnum):
    """Supported synthetic question families."""

    FACTOID_TITLE = "factoid_title"
    FACTOID_DATE = "factoid_date"
    FACTOID_PARTY = "factoid_party"
    FACTOID_JUDGE = "factoid_judge"
    FACTOID_AUTHORITY = "factoid_authority"
    PROVISION_ARTICLE = "provision_article"
    PROVISION_SCHEDULE = "provision_schedule"
    COMPARE_PARTY = "compare_party"
    COMPARE_AUTHORITY = "compare_authority"
    COMPARE_TOPIC = "compare_topic"
    TEMPORAL_AMENDMENT = "temporal_amendment"
    TEMPORAL_COMMENCEMENT = "temporal_commencement"
    TEMPORAL_SUPERSESSION = "temporal_supersession"
    COUNTERFACTUAL_UNSUPPORTED = "counterfactual_unsupported"
    COUNTERFACTUAL_NEGATIVE = "counterfactual_negative"
    ADVERSARIAL_ALIAS = "adversarial_alias"
    ADVERSARIAL_NEAR_MISS = "adversarial_near_miss"


class BridgeFactRecord(BaseModel):
    """External bridge-fact record used to seed synthetic questions."""

    fact_id: str
    question: str
    answer: str
    page_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class SyntheticQAExample(BaseModel):
    """One deterministic synthetic QA example."""

    question_id: str
    question: str
    answer: str
    answer_type: str
    gold_page_ids: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    question_family: QuestionFamily
    difficulty: str
    hard_negative_page_ids: list[str] = Field(default_factory=list)
    source_object_ids: list[str] = Field(default_factory=list)
    generation_method: str


class SyntheticQACorpusManifest(BaseModel):
    """Summary manifest for one synthetic QA export."""

    total_examples: int
    family_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    bridge_fact_count: int
    llm_paraphrase_enabled: bool
    source_paths: dict[str, str]


class SyntheticQAFactory:
    """Factory for deterministic synthetic QA and counterfactual examples."""

    def __init__(
        self,
        *,
        paraphrase_fn: ParaphraseFn | None = None,
        hard_negative_limit: int = 5,
    ) -> None:
        """Initialize the factory.

        Args:
            paraphrase_fn: Optional callback that returns paraphrase variants for
                a question.
            hard_negative_limit: Max hard negatives per example.
        """
        self._paraphrase_fn = paraphrase_fn
        self._hard_negative_limit = max(1, hard_negative_limit)

    def generate_all(
        self,
        *,
        registry: CorpusRegistry,
        segments: Sequence[LegalSegment],
        graph: ApplicabilityGraph,
        bridge_facts: Sequence[BridgeFactRecord] = (),
        max_per_family: int = 0,
    ) -> list[SyntheticQAExample]:
        """Generate the full synthetic QA corpus.

        Args:
            registry: Compiled corpus registry.
            segments: Compiler-derived legal segments.
            graph: Applicability graph.
            bridge_facts: Optional bridge-fact inputs.
            max_per_family: Optional cap per question family.

        Returns:
            Deterministically ordered synthetic examples.
        """
        page_texts = _build_page_texts(registry=registry, segments=segments)
        page_doc_ids = {page_id: page_id.rpartition("_")[0] for page_id in page_texts}
        examples = [
            *self.generate_factoid_questions(registry=registry, page_texts=page_texts, page_doc_ids=page_doc_ids),
            *self.generate_provision_questions(segments=segments, page_texts=page_texts, page_doc_ids=page_doc_ids),
            *self.generate_compare_questions(registry=registry, page_texts=page_texts, page_doc_ids=page_doc_ids),
            *self.generate_temporal_questions(registry=registry, graph=graph, page_texts=page_texts, page_doc_ids=page_doc_ids),
            *self.generate_counterfactual_questions(registry=registry),
            *self.generate_adversarial_questions(registry=registry, page_texts=page_texts, page_doc_ids=page_doc_ids),
            *self.generate_bridge_fact_questions(bridge_facts=bridge_facts, page_texts=page_texts, page_doc_ids=page_doc_ids),
        ]
        deduped = _deduplicate_examples(examples)
        if max_per_family > 0:
            deduped = _limit_per_family(deduped, limit=max_per_family)
        return sorted(deduped, key=lambda item: (item.question_family.value, item.question_id))

    def generate_factoid_questions(
        self,
        *,
        registry: CorpusRegistry,
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Generate factoid questions from compiled laws and cases."""
        examples: list[SyntheticQAExample] = []
        for law in sorted(registry.laws.values(), key=lambda item: item.doc_id):
            if law.title:
                examples.append(
                    self._build_example(
                        question=f"What is the title of {law.doc_id}?",
                        answer=law.title,
                        answer_type="name",
                        family=QuestionFamily.FACTOID_TITLE,
                        difficulty="easy",
                        gold_page_ids=list(law.page_ids[:1]),
                        source_object_ids=[law.object_id],
                        generation_method="law_title",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
            if law.issuing_authority:
                examples.append(
                    self._build_example(
                        question=f"Who issued {law.short_title or law.title}?",
                        answer=law.issuing_authority,
                        answer_type="name",
                        family=QuestionFamily.FACTOID_AUTHORITY,
                        difficulty="easy",
                        gold_page_ids=list(law.page_ids[:1]),
                        source_object_ids=[law.object_id],
                        generation_method="law_authority",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
            if law.commencement_date:
                examples.append(
                    self._build_example(
                        question=f"When did {law.short_title or law.title} come into force?",
                        answer=law.commencement_date,
                        answer_type="date",
                        family=QuestionFamily.FACTOID_DATE,
                        difficulty="easy",
                        gold_page_ids=list(law.page_ids[:1]),
                        source_object_ids=[law.object_id],
                        generation_method="law_commencement",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
        for case in sorted(registry.cases.values(), key=lambda item: item.doc_id):
            if case.parties:
                examples.append(
                    self._build_example(
                        question=f"Who is a party in {case.case_number or case.title}?",
                        answer=case.parties[0].name,
                        answer_type="name",
                        family=QuestionFamily.FACTOID_PARTY,
                        difficulty="easy",
                        gold_page_ids=list(case.page_ids[:1]),
                        source_object_ids=[case.object_id],
                        generation_method="case_party",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
            if case.judges:
                examples.append(
                    self._build_example(
                        question=f"Which judge decided {case.case_number or case.title}?",
                        answer=case.judges[0],
                        answer_type="name",
                        family=QuestionFamily.FACTOID_JUDGE,
                        difficulty="medium",
                        gold_page_ids=list(case.page_ids[:1]),
                        source_object_ids=[case.object_id],
                        generation_method="case_judge",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
        return examples

    def generate_provision_questions(
        self,
        *,
        segments: Sequence[LegalSegment],
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Generate provision lookup questions from legal segments."""
        examples: list[SyntheticQAExample] = []
        for segment in sorted(segments, key=lambda item: item.segment_id):
            if segment.segment_type not in {SegmentType.ARTICLE, SegmentType.SECTION, SegmentType.SCHEDULE}:
                continue
            answer = _extract_answer_span(segment.text)
            if not answer:
                continue
            family = (
                QuestionFamily.PROVISION_SCHEDULE
                if segment.segment_type is SegmentType.SCHEDULE
                else QuestionFamily.PROVISION_ARTICLE
            )
            question = f"What does {segment.legal_path or segment.segment_id} of {segment.doc_title} provide?"
            examples.append(
                self._build_example(
                    question=question,
                    answer=answer,
                    answer_type="free_text",
                    family=family,
                    difficulty="medium",
                    gold_page_ids=list(segment.page_ids),
                    source_object_ids=[segment.segment_id],
                    generation_method="segment_lookup",
                    page_texts=page_texts,
                    page_doc_ids=page_doc_ids,
                )
            )
        return examples

    def generate_compare_questions(
        self,
        *,
        registry: CorpusRegistry,
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Generate compare questions from repeated authorities and parties."""
        examples: list[SyntheticQAExample] = []
        laws_by_authority: dict[str, list[LawObject]] = defaultdict(list)
        for law in registry.laws.values():
            if law.issuing_authority:
                laws_by_authority[law.issuing_authority].append(law)
        for authority, laws in sorted(laws_by_authority.items()):
            if len(laws) < 2:
                continue
            examples.append(
                self._build_example(
                    question=f"Which laws were issued by {authority}?",
                    answer=", ".join(law.title for law in sorted(laws, key=lambda item: item.title)),
                    answer_type="names",
                    family=QuestionFamily.COMPARE_AUTHORITY,
                    difficulty="medium",
                    gold_page_ids=[page_id for law in laws for page_id in law.page_ids[:1]],
                    source_object_ids=[law.object_id for law in laws],
                    generation_method="authority_compare",
                    page_texts=page_texts,
                    page_doc_ids=page_doc_ids,
                )
            )
        cases_by_party: dict[str, list[CaseObject]] = defaultdict(list)
        for case in registry.cases.values():
            for party in case.parties:
                cases_by_party[party.name].append(case)
        for party_name, cases in sorted(cases_by_party.items()):
            if len(cases) < 2:
                continue
            examples.append(
                self._build_example(
                    question=f"Which cases involve {party_name}?",
                    answer=", ".join(case.title for case in sorted(cases, key=lambda item: item.title)),
                    answer_type="names",
                    family=QuestionFamily.COMPARE_PARTY,
                    difficulty="medium",
                    gold_page_ids=[page_id for case in cases for page_id in case.page_ids[:1]],
                    source_object_ids=[case.object_id for case in cases],
                    generation_method="party_compare",
                    page_texts=page_texts,
                    page_doc_ids=page_doc_ids,
                )
            )
        return examples

    def generate_temporal_questions(
        self,
        *,
        registry: CorpusRegistry,
        graph: ApplicabilityGraph,
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Generate temporal and amendment questions from the applicability graph."""
        examples: list[SyntheticQAExample] = []
        for law in sorted(registry.laws.values(), key=lambda item: item.doc_id):
            commencement = graph.get_commencement(law.doc_id)
            if commencement is not None:
                examples.append(
                    self._build_example(
                        question=f"When did {law.short_title or law.title} commence?",
                        answer=commencement.commencement_date,
                        answer_type="date",
                        family=QuestionFamily.TEMPORAL_COMMENCEMENT,
                        difficulty="medium",
                        gold_page_ids=[commencement.evidence_page_id] if commencement.evidence_page_id else list(law.page_ids[:1]),
                        source_object_ids=[law.object_id],
                        generation_method="graph_commencement",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
            amendments = graph.get_amendment_history(law.doc_id)
            if amendments:
                first_edge = amendments[0]
                answer = first_edge.effective_date or first_edge.source_doc_id
                gold_pages = [edge.evidence_page_id for edge in amendments if edge.evidence_page_id]
                examples.append(
                    self._build_example(
                        question=f"When was {law.short_title or law.title} amended?",
                        answer=answer,
                        answer_type="date" if first_edge.effective_date else "name",
                        family=QuestionFamily.TEMPORAL_AMENDMENT,
                        difficulty="hard",
                        gold_page_ids=gold_pages or list(law.page_ids[:1]),
                        source_object_ids=[law.object_id, *[edge.source_doc_id for edge in amendments]],
                        generation_method="graph_amendment",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
            current = graph.get_current_version(law.doc_id)
            if current is not None and current.doc_id != law.doc_id:
                examples.append(
                    self._build_example(
                        question=f"What is the current version of {law.short_title or law.title}?",
                        answer=current.title,
                        answer_type="name",
                        family=QuestionFamily.TEMPORAL_SUPERSESSION,
                        difficulty="hard",
                        gold_page_ids=list(current.page_ids[:1]),
                        source_object_ids=[law.object_id, current.object_id],
                        generation_method="graph_supersession",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
        return examples

    def generate_counterfactual_questions(self, *, registry: CorpusRegistry) -> list[SyntheticQAExample]:
        """Generate unsupported and negative counterfactual questions."""
        examples: list[SyntheticQAExample] = []
        for index, law in enumerate(sorted(registry.laws.values(), key=lambda item: item.doc_id)[:2]):
            fake_title = _mutate_title(law.title or law.short_title or law.doc_id, seed=index)
            topic = _COUNTERFACTUAL_TOPIC_HINTS[index % len(_COUNTERFACTUAL_TOPIC_HINTS)]
            examples.append(
                _make_example(
                    question=f"Does {fake_title} apply to {topic}?",
                    answer="N/A",
                    answer_type="unanswerable",
                    family=QuestionFamily.COUNTERFACTUAL_UNSUPPORTED,
                    difficulty="hard",
                    gold_page_ids=[],
                    gold_doc_ids=[],
                    hard_negative_page_ids=[],
                    source_object_ids=[law.object_id],
                    generation_method="counterfactual_fake_law",
                )
            )
        case_items = sorted(registry.cases.values(), key=lambda item: item.doc_id)
        if len(case_items) >= 2 and case_items[0].parties:
            examples.append(
                _make_example(
                    question=f"Did {case_items[0].parties[0].name} appear in {case_items[1].case_number or case_items[1].title}?",
                    answer="No",
                    answer_type="boolean",
                    family=QuestionFamily.COUNTERFACTUAL_NEGATIVE,
                    difficulty="hard",
                    gold_page_ids=[],
                    gold_doc_ids=[],
                    hard_negative_page_ids=[],
                    source_object_ids=[case_items[0].object_id, case_items[1].object_id],
                    generation_method="counterfactual_cross_case",
                )
            )
        return examples

    def generate_adversarial_questions(
        self,
        *,
        registry: CorpusRegistry,
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Generate adversarial alias and near-miss questions."""
        examples: list[SyntheticQAExample] = []
        for law in sorted(registry.laws.values(), key=lambda item: item.doc_id):
            if law.short_title and law.short_title != law.title:
                examples.append(
                    self._build_example(
                        question=f"What is the full title of {law.short_title}?",
                        answer=law.title,
                        answer_type="name",
                        family=QuestionFamily.ADVERSARIAL_ALIAS,
                        difficulty="hard",
                        gold_page_ids=list(law.page_ids[:1]),
                        source_object_ids=[law.object_id],
                        generation_method="alias_resolution",
                        page_texts=page_texts,
                        page_doc_ids=page_doc_ids,
                    )
                )
        laws = sorted(registry.laws.values(), key=lambda item: item.doc_id)
        for left, right in pairwise(laws):
            if not left.title or not right.title:
                continue
            overlap = _token_overlap(left.title, right.title)
            if overlap < 2:
                continue
            examples.append(
                self._build_example(
                    question=f"Which law has the title {left.title}?",
                    answer=left.title,
                    answer_type="name",
                    family=QuestionFamily.ADVERSARIAL_NEAR_MISS,
                    difficulty="hard",
                    gold_page_ids=list(left.page_ids[:1]),
                    source_object_ids=[left.object_id, right.object_id],
                    generation_method="near_miss_title",
                    page_texts=page_texts,
                    page_doc_ids=page_doc_ids,
                )
            )
        return examples

    def generate_bridge_fact_questions(
        self,
        *,
        bridge_facts: Sequence[BridgeFactRecord],
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> list[SyntheticQAExample]:
        """Project external bridge facts into synthetic QA examples."""
        examples: list[SyntheticQAExample] = []
        for record in sorted(bridge_facts, key=lambda item: item.fact_id):
            if not record.question or not record.answer:
                continue
            examples.append(
                self._build_example(
                    question=record.question,
                    answer=record.answer,
                    answer_type="free_text",
                    family=QuestionFamily.COMPARE_TOPIC,
                    difficulty="medium",
                    gold_page_ids=list(record.page_ids),
                    source_object_ids=[record.fact_id],
                    generation_method="bridge_fact",
                    page_texts=page_texts,
                    page_doc_ids=page_doc_ids,
                )
            )
        return examples

    def paraphrase_question(self, question: str) -> list[str]:
        """Return optional paraphrase variants for one question."""
        if self._paraphrase_fn is None:
            return []
        return [variant.strip() for variant in self._paraphrase_fn(question) if variant.strip() and variant.strip() != question]

    def validate_example_against_gold_pages(
        self,
        example: SyntheticQAExample,
        *,
        page_texts: dict[str, str],
    ) -> bool:
        """Validate that the answer is supported by the example's gold pages."""
        if not example.gold_page_ids:
            return example.answer in {"N/A", "No", "Unavailable", ""}
        gold_text = "\n".join(page_texts.get(page_id, "") for page_id in example.gold_page_ids)
        if not gold_text.strip():
            return False
        if example.answer in {"N/A", "No", "Unavailable", ""}:
            return True
        if example.answer in gold_text:
            return True
        return all(
            component in gold_text
            for component in [part.strip() for part in example.answer.split(",") if part.strip()]
        )

    def _build_example(
        self,
        *,
        question: str,
        answer: str,
        answer_type: str,
        family: QuestionFamily,
        difficulty: str,
        gold_page_ids: list[str],
        source_object_ids: list[str],
        generation_method: str,
        page_texts: dict[str, str],
        page_doc_ids: dict[str, str],
    ) -> SyntheticQAExample:
        """Build one example with hard negatives and optional paraphrases."""
        gold_doc_ids = _unique(page_doc_ids.get(page_id, page_id.rpartition("_")[0]) for page_id in gold_page_ids)
        negatives = mine_hard_negative_page_ids(
            question=question,
            gold_page_ids=gold_page_ids,
            gold_doc_ids=gold_doc_ids,
            page_texts=page_texts,
            page_doc_ids=page_doc_ids,
            limit=self._hard_negative_limit,
        )
        example = _make_example(
            question=question,
            answer=answer,
            answer_type=answer_type,
            family=family,
            difficulty=difficulty,
            gold_page_ids=gold_page_ids,
            gold_doc_ids=gold_doc_ids,
            hard_negative_page_ids=negatives,
            source_object_ids=source_object_ids,
            generation_method=generation_method,
        )
        variants = self.paraphrase_question(question)
        if not variants:
            return example
        return example.model_copy(update={"generation_method": f"{generation_method}+paraphrase", "question": variants[0]})


def load_corpus_registry(path: Path) -> CorpusRegistry:
    """Load a compiled corpus registry from JSON."""
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object payload at {path}")
    return CorpusRegistry.model_validate(cast("JsonObject", raw))


def load_legal_segments(path: Path) -> list[LegalSegment]:
    """Load legal segments from JSON or JSONL."""
    if path.suffix.casefold() == ".jsonl":
        return [
            LegalSegment.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if isinstance(raw, list):
        return [LegalSegment.model_validate(item) for item in cast("list[object]", raw)]
    if isinstance(raw, dict):
        items = cast("JsonObject", raw).get("segments", [])
        if isinstance(items, list):
            return [LegalSegment.model_validate(item) for item in cast("list[object]", items)]
    raise ValueError(f"Unsupported legal-segment payload at {path}")


def load_applicability_graph(path: Path) -> ApplicabilityGraph:
    """Load an applicability graph from JSON."""
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object payload at {path}")
    return ApplicabilityGraph.model_validate(cast("JsonObject", raw))


def load_bridge_fact_records(path: Path) -> list[BridgeFactRecord]:
    """Load bridge facts from JSON or JSONL."""
    if path.suffix.casefold() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if isinstance(raw, list):
            items = cast("list[object]", raw)
        elif isinstance(raw, dict):
            items = cast("list[object]", cast("JsonObject", raw).get("facts", []))
        else:
            raise ValueError(f"Unsupported bridge-fact payload at {path}")
    records: list[BridgeFactRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        row = cast("JsonObject", item)
        question = str(row.get("question") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("value") or "").strip()
        records.append(
            BridgeFactRecord(
                fact_id=str(row.get("fact_id") or f"bridge-fact-{index}"),
                question=question,
                answer=answer,
                page_ids=_string_list(row.get("page_ids") or row.get("source_page_ids")),
                doc_ids=_string_list(row.get("doc_ids") or row.get("source_doc_ids")),
                aliases=_string_list(row.get("aliases")),
            )
        )
    return records


def mine_hard_negative_page_ids(
    *,
    question: str,
    gold_page_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
    page_texts: dict[str, str],
    page_doc_ids: dict[str, str],
    limit: int,
) -> list[str]:
    """Mine deterministic hard-negative page IDs from available pages."""
    question_tokens = set(_tokenize(question))
    gold_page_set = set(gold_page_ids)
    gold_doc_set = set(gold_doc_ids)
    ranked = sorted(
        (
            (
                _negative_rank_key(
                    page_id=page_id,
                    question_tokens=question_tokens,
                    page_text=page_text,
                    page_doc_id=page_doc_ids.get(page_id, page_id.rpartition("_")[0]),
                    gold_doc_ids=gold_doc_set,
                ),
                page_id,
            )
            for page_id, page_text in page_texts.items()
            if page_id not in gold_page_set
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    return [page_id for _key, page_id in ranked[: max(1, limit)] if page_id]


def write_synthetic_qa_jsonl(path: Path, examples: Sequence[SyntheticQAExample]) -> None:
    """Write synthetic QA examples as deterministic JSONL."""
    lines = [json.dumps(example.model_dump(mode="json"), ensure_ascii=True, sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_manifest(
    *,
    examples: Sequence[SyntheticQAExample],
    bridge_fact_count: int,
    llm_paraphrase_enabled: bool,
    source_paths: dict[str, str],
) -> SyntheticQACorpusManifest:
    """Build a synthetic QA manifest."""
    family_counts = Counter(example.question_family.value for example in examples)
    difficulty_counts = Counter(example.difficulty for example in examples)
    return SyntheticQACorpusManifest(
        total_examples=len(examples),
        family_counts=dict(sorted(family_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        bridge_fact_count=bridge_fact_count,
        llm_paraphrase_enabled=llm_paraphrase_enabled,
        source_paths=source_paths,
    )


def _make_example(
    *,
    question: str,
    answer: str,
    answer_type: str,
    family: QuestionFamily,
    difficulty: str,
    gold_page_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
    hard_negative_page_ids: Sequence[str],
    source_object_ids: Sequence[str],
    generation_method: str,
) -> SyntheticQAExample:
    question_id = hashlib.sha1(f"{family.value}|{question}|{answer}".encode()).hexdigest()[:16]
    return SyntheticQAExample(
        question_id=question_id,
        question=question,
        answer=answer,
        answer_type=answer_type,
        gold_page_ids=list(_unique(gold_page_ids)),
        gold_doc_ids=list(_unique(gold_doc_ids)),
        question_family=family,
        difficulty=difficulty,
        hard_negative_page_ids=list(_unique(hard_negative_page_ids)),
        source_object_ids=list(_unique(source_object_ids)),
        generation_method=generation_method,
    )


def _build_page_texts(*, registry: CorpusRegistry, segments: Sequence[LegalSegment]) -> dict[str, str]:
    page_texts: dict[str, str] = {}
    for segment in segments:
        for page_id in segment.page_ids:
            if page_id and segment.text and page_id not in page_texts:
                page_texts[page_id] = segment.text
    for bucket in (
        registry.laws.values(),
        registry.cases.values(),
        registry.orders.values(),
        registry.practice_directions.values(),
        registry.amendments.values(),
        registry.other_documents.values(),
    ):
        for obj in bucket:
            page_texts.update({page_id: text for page_id, text in obj.page_texts.items() if page_id and text})
    return page_texts


def _extract_answer_span(text: str, *, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    first_sentence = _SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)[0]
    return first_sentence[:max_chars].rstrip()


def _mutate_title(title: str, *, seed: int) -> str:
    numbers = re.findall(r"\d+", title)
    if numbers:
        mutated = title.replace(numbers[-1], str(int(numbers[-1]) + seed + 7), 1)
        if mutated != title:
            return mutated
    return f"{title} Supplement {seed + 1}"


def _deduplicate_examples(examples: Sequence[SyntheticQAExample]) -> list[SyntheticQAExample]:
    deduped: dict[tuple[str, str], SyntheticQAExample] = {}
    for example in examples:
        deduped.setdefault((example.question_family.value, example.question.casefold()), example)
    return list(deduped.values())


def _limit_per_family(examples: Sequence[SyntheticQAExample], *, limit: int) -> list[SyntheticQAExample]:
    counts: dict[str, int] = defaultdict(int)
    limited: list[SyntheticQAExample] = []
    for example in sorted(examples, key=lambda item: (item.question_family.value, item.question_id)):
        family = example.question_family.value
        if counts[family] >= limit:
            continue
        counts[family] += 1
        limited.append(example)
    return limited


def _negative_rank_key(
    *,
    page_id: str,
    question_tokens: set[str],
    page_text: str,
    page_doc_id: str,
    gold_doc_ids: set[str],
) -> tuple[int, int, int]:
    tokens = set(_tokenize(page_text))
    overlap = len(question_tokens & tokens)
    same_doc_penalty = 1 if page_doc_id in gold_doc_ids else 0
    early_page_bonus = 1 if page_id.endswith("_1") else 0
    return same_doc_penalty, overlap, early_page_bonus


def _token_overlap(left: str, right: str) -> int:
    return len(set(_tokenize(left)) & set(_tokenize(right)))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique(str(item) for item in cast("list[object]", value))
