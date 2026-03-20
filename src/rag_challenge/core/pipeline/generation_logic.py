# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from rag_challenge.core.premise_guard import check_query_premise
from rag_challenge.models import Citation, QueryComplexity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag_challenge.models import RankedChunk
    from rag_challenge.telemetry import TelemetryCollector

from .constants import _STRICT_REPAIR_HINT_TEMPLATE
from .query_rules import (
    _extract_question_title_refs,
    _is_broad_enumeration_query,
    _is_citation_title_query,
    _is_common_elements_query,
    _is_enumeration_query,
    _is_multi_criteria_enumeration_query,
    _is_named_amendment_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
    _is_registrar_enumeration_query,
    _is_ruler_enactment_query,
)
from .state import RAGState  # noqa: TC001

logger = logging.getLogger(__name__)


class GenerationLogicMixin:
    async def _generate(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.set_models(llm=state["model"])

        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        strict_types = {"boolean", "number", "date", "name", "names"}
        strict_non_stream_types = {"boolean", "number", "date", "name", "names"}
        prompt_hint = str(state.get("conflict_prompt_context") or "").strip()
        strict_repair_enabled = bool(getattr(self._settings.pipeline, "strict_repair_enabled", True))

        # Pre-generation anti-hallucination guardrails for free_text.
        if answer_type == "free_text":
            context_for_guard = state.get("context_chunks", [])
            entity_scope_hint = self._build_entity_scope(context_for_guard)
            if entity_scope_hint:
                prompt_hint = f"{prompt_hint}\n\n{entity_scope_hint}".strip() if prompt_hint else entity_scope_hint
            query_text = str(state["query"] or "")
            query_lower = query_text.lower()
            common_elements_query = _is_common_elements_query(query_text)
            if common_elements_query:
                common_elements_hint = (
                    "For common-elements questions, answer concisely without IRAC or Issue/Rule/Application/"
                    "Conclusion headings. Every claimed common element must be supported by at least one citation "
                    "from every referenced document in the question; each numbered common element must cite that "
                    "support in the same item; if you cannot cite every referenced document for an element, omit "
                    "that element. Output ONLY a numbered list of common elements; do not add an explanatory "
                    "preamble, postscript, or cross-document caveat outside the numbered list. Each numbered "
                    "item must end with one parenthetical citation that includes at least one chunk ID from "
                    "every referenced document, for example (cite: id_a, id_b, id_c). Keep the list compact "
                    "and merge closely related interpretative rules into one item when they belong to the same "
                    "clause family. A structural "
                    "overlap counts as a valid common element "
                    "when each referenced document explicitly states the same structure, such as that Schedule 1 "
                    "contains interpretative provisions or a list of defined terms. List a more specific sub-item "
                    "only if that same sub-item is explicitly shown in every referenced document. If one "
                    "referenced document only shows the Schedule 1 structure but does not quote the substantive "
                    "interpretative rules, do not infer those sub-items as common from the other documents. "
                    "Do not end with "
                    "\"There is no information on this question.\" if you have already identified one or more "
                    "supported common elements. If no explicit common element remains after this check, say "
                    "exactly: \"There is no information on this question.\""
                )
                prompt_hint = f"{prompt_hint}\n\n{common_elements_hint}".strip() if prompt_hint else common_elements_hint
                named_docs = [ref for ref in (state.get("doc_refs") or []) if str(ref).strip()]
                if len(named_docs) >= 2:
                    docs_hint = (
                        "The referenced documents for this common-elements question are: "
                        f"{'; '.join(named_docs)}. If a claimed element is not explicitly supported in each of "
                        "those referenced documents, omit it."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{docs_hint}".strip() if prompt_hint else docs_hint
            if _is_broad_enumeration_query(query_text):
                source_block_count = len([chunk for chunk in context_for_guard if getattr(chunk, "chunk_id", "").strip()])
                enumeration_hint = (
                    "For this broad enumeration question, answer as a compact numbered list. Each item should "
                    "start with the exact matching law or document title supported by the sources. Use the "
                    "minimum number of citations needed to support each item. If separate source blocks are "
                    "needed to prove all criteria for the same law or to supply the exact citation title "
                    "requested, cite each needed block in that same item. Inspect every source block provided "
                    f"({source_block_count} total) before stopping. Output ONLY the numbered list items; do "
                    "not add caveats, comparative commentary, or summary text after the list."
                )
                prompt_hint = f"{prompt_hint}\n\n{enumeration_hint}".strip() if prompt_hint else enumeration_hint
                if (
                    query_lower.startswith("which laws")
                    and not _is_citation_title_query(query_text)
                    and not _is_registrar_enumeration_query(query_text)
                ):
                    titles_only_hint = (
                        "This question asks only which laws match. Each numbered item should mainly give the "
                        "law title itself. Do not add article-level, schedule-level, or explanatory detail "
                        "unless the question explicitly asks for that detail."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{titles_only_hint}".strip() if prompt_hint else titles_only_hint
                if _is_multi_criteria_enumeration_query(query_text):
                    multi_criteria_hint = (
                        "Review every source block before stopping. When multiple source blocks clearly refer to "
                        "the same law title, you may combine those blocks for that law only. Deduplicate by law "
                        "title, but do not stop after the first few matches if later blocks also satisfy all criteria. "
                        "List a law only if the source block or combined source blocks for that same law explicitly "
                        "support every criterion in the question; if one criterion is missing for that law, exclude it. "
                        "Each listed item's citations must collectively support every criterion for that same law."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{multi_criteria_hint}".strip() if prompt_hint else multi_criteria_hint
                if _is_multi_criteria_enumeration_query(query_text) and re.search(r"\b(19|20)\d{2}\b", query_text):
                    year_hint = (
                        "Use the exact year shown in each source block, such as in a law title like "
                        "\"DIFC Law No. 4 of 2018\" or \"Limited Partnership Law 2006\". Do not infer the "
                        "year from consolidated-version dates or amendment dates, and check every source block "
                        "before you finish the list. If the year/title evidence and the substantive matching "
                        "criterion appear in different source blocks for the same law, cite both blocks in that "
                        "same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{year_hint}".strip() if prompt_hint else year_hint
                if "administered by the registrar" in query_lower:
                    registrar_hint = (
                        "For this question, include a law only if the source block or combined source blocks for "
                        "that law explicitly state that the law is administered by the Registrar. A title, "
                        "enactment reference, or year alone is not enough. A definitional passage in another "
                        "document about 'legislation administered by the Registrar' or 'Prescribed Laws' does "
                        "NOT make that current document itself a match unless that same document explicitly "
                        "says this law is administered by the Registrar. Use one separate numbered item per "
                        "matching law, and do not merge two different laws into the same numbered item even if "
                        "their supporting clauses are similar."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{registrar_hint}".strip() if prompt_hint else registrar_hint
                if _is_citation_title_query(query_text):
                    citation_title_hint = (
                        "For each listed law, state the exact citation title as written in the source, usually "
                        "from a clause like \"This Law may be cited as ...\". Do not paraphrase, shorten, or "
                        "replace it with generic uppercase document headers. If the title clause and another "
                        "matching criterion appear in different source blocks for the same law, cite both blocks "
                        "in that same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{citation_title_hint}".strip() if prompt_hint else citation_title_hint
                question_title_refs = _extract_question_title_refs(query_text)
                if len(question_title_refs) >= 2 and any(term in query_lower for term in ("mention", "mentions", "reference", "references")):
                    named_refs_hint = (
                        "The named law references in this question are: "
                        f"{'; '.join(question_title_refs)}. List a document only if that same document explicitly "
                        "mentions every named law reference above. If a document mentions only one of them, "
                        "exclude it silently. If multiple documents in the sources satisfy all named references, "
                        "list all of them before stopping."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{named_refs_hint}".strip() if prompt_hint else named_refs_hint
                if _is_ruler_enactment_query(query_text):
                    enactment_hint = (
                        "For this question, a law matches if the Enactment Notice itself states both that the "
                        "Ruler of Dubai enacted the law and when it comes into force. A commencement rule written "
                        "as a relative period, such as 'on the 5th business day after enactment' or '90 days after "
                        "enactment', still counts as the commencement being specified in the Enactment Notice. "
                        "Do not exclude a law only because the notice gives a relative commencement period instead "
                        "of a calendar date."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{enactment_hint}".strip() if prompt_hint else enactment_hint

        streamed = False
        streamed_raw = ""
        answer = ""
        extracted = False
        strict_cited_ids: list[str] = []
        context_chunks = list(state.get("context_chunks", []))
        if answer_type in strict_types:
            context_chunks, strict_context_augmented = self._augment_strict_context_chunks(
                query=state["query"],
                answer_type=answer_type,
                context_chunks=context_chunks,
                retrieved=list(state.get("retrieved", [])),
            )
            if strict_context_augmented:
                collector.set_context_ids([chunk.chunk_id for chunk in context_chunks])
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        retrieved_chunks = list(state.get("retrieved", []))
        if retrieved_chunks:
            collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved_chunks])
            collector.set_chunk_snippets(self._build_chunk_snippet_map(retrieved_chunks))
            collector.set_chunk_page_hints(self._build_chunk_page_hint_map(retrieved_chunks))
        if context_chunks:
            collector.set_chunk_snippets(self._build_chunk_snippet_map(context_chunks))
            collector.set_chunk_page_hints(self._build_chunk_page_hint_map(context_chunks))
        collector.set_context_ids(context_chunk_ids)
        get_context_debug_stats = getattr(self._generator, "get_context_debug_stats", None)
        if callable(get_context_debug_stats):
            context_stats_obj = get_context_debug_stats(
                question=state["query"],
                chunks=context_chunks,
                complexity=state.get("complexity", QueryComplexity.SIMPLE),
                answer_type=answer_type,
            )
            if isinstance(context_stats_obj, tuple):
                context_stats_items = cast("tuple[object, ...]", context_stats_obj)
                if len(context_stats_items) != 2:
                    context_stats_items = ()
            else:
                context_stats_items = ()
            if len(context_stats_items) == 2:
                chunk_count_obj, budget_obj = context_stats_items
                if isinstance(chunk_count_obj, int) and isinstance(budget_obj, int):
                    collector.set_context_stats(chunk_count=chunk_count_obj, budget_tokens=budget_obj)
        if answer_type == "free_text" and not context_chunks:
            answer = self._insufficient_sources_answer(())
            collector.set_generation_mode("single_shot")
            collector.set_models(llm="insufficient-sources")
            logger.info(
                "free_text_no_context_fallback",
                extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
            )
        if (
            answer_type == "free_text"
            and bool(getattr(self._settings.pipeline, "premise_guard_enabled", True))
        ):
            terms_obj = getattr(self._settings.pipeline, "premise_guard_terms", [])
            disallowed_terms: list[str] = []
            if isinstance(terms_obj, list):
                terms = cast("list[object]", terms_obj)
                disallowed_terms = [text for term in terms if (text := str(term).strip())]
            guard = check_query_premise(
                query=state["query"],
                context_chunks=context_chunks,
                disallowed_terms=disallowed_terms,
            )
            if guard.triggered:
                answer = self._insufficient_sources_answer(tuple(context_chunk_ids[:1]))
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="premise-guard")
                logger.warning(
                    "premise_guard_triggered",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "term": guard.term,
                        "answer_type": answer_type,
                    },
                )
        context_chunks = self._prune_boolean_context_for_single_doc_article(
            query=state["query"],
            answer_type=answer_type,
            doc_refs=state.get("doc_refs"),
            context_chunks=context_chunks,
        )
        context_chunks = self._boost_family_context_chunks(
            query=state["query"],
            answer_type=answer_type,
            context_chunks=context_chunks,
        )
        if answer_type in strict_types and bool(getattr(self._settings.pipeline, "strict_types_extraction_enabled", True)):
            strict_result = self._strict_answerer.answer(
                answer_type=answer_type,
                query=state["query"],
                context_chunks=context_chunks,
                max_chunks=int(getattr(self._settings.pipeline, "strict_types_extraction_max_chunks", 4)),
            )
            if strict_result is not None and strict_result.confident:
                answer = strict_result.answer.strip()
                extracted = True
                strict_cited_ids = list(strict_result.cited_chunk_ids)
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="strict-extractor")

        if answer_type in strict_non_stream_types:
            if not answer:
                collector.set_generation_mode("single_shot")
                # Strict-types path fallback: non-stream LLM answer.
                with collector.timed("llm"):
                    generated_text, _citations = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=int(state["max_tokens"]),
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    )
                answer = generated_text.strip()
        else:
            if not answer:
                build_structured_answer = getattr(self._generator, "build_structured_free_text_answer", None)
                if callable(build_structured_answer):
                    built_obj = build_structured_answer(
                        question=state["query"],
                        chunks=context_chunks,
                        doc_refs=state.get("doc_refs"),
                    )
                    if isinstance(built_obj, str) and built_obj.strip():
                        answer = built_obj.strip()
                        collector.set_generation_mode("single_shot")
                        collector.set_models(llm="structured-extractor")
                        collector.mark_first_token()
                        writer({"type": "token", "text": answer})
            if not answer:
                collector.set_generation_mode("stream")
                answer_parts: list[str] = []
                first_token = True
                # Adaptive max_tokens: boost for multi-entity queries to prevent truncation.
                effective_max_tokens = int(state["max_tokens"])
                doc_ref_count = len([r for r in (state.get("doc_refs") or []) if str(r).strip()])
                if doc_ref_count >= 2:
                    effective_max_tokens = min(int(effective_max_tokens * 1.5), 1800)
                with collector.timed("llm"):
                    async for token in self._generator.generate_stream(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=effective_max_tokens,
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    ):
                        if first_token:
                            collector.mark_first_token()
                        if first_token:
                            first_token = False
                        writer({"type": "token", "text": token})
                        answer_parts.append(token)
                streamed = True
                answer = "".join(answer_parts).strip()
                if answer:
                    streamed_raw = answer
                    cleanup_obj = getattr(self._generator, "cleanup_truncated_answer", None)
                    if callable(cleanup_obj):
                        cleaned_obj = cleanup_obj(answer)
                        if isinstance(cleaned_obj, str):
                            malformed_tail_detected = cleaned_obj.strip() != answer.strip()
                            if malformed_tail_detected:
                                collector.set_llm_diagnostics(malformed_tail_detected=True)
                        if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                            answer = cleaned_obj.strip()
                    strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
                    should_strip_neg = answer_type == "free_text" and (
                        _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
                    )
                    if callable(strip_neg) and should_strip_neg:
                        stripped_obj = strip_neg(answer)
                        if isinstance(stripped_obj, str) and stripped_obj.strip():
                            answer = stripped_obj.strip()
                    if streamed_raw and not answer.strip():
                        answer = streamed_raw
                else:
                    # Rare provider anomaly: stream completes without tokens. Fall back once to non-stream generation.
                    collector.set_generation_mode("single_shot")
                    with collector.timed("llm"):
                        generated_text, _citations = await self._generator.generate(
                            state["query"],
                            context_chunks,
                            model=state["model"],
                            max_tokens=effective_max_tokens,
                            collector=collector,
                            complexity=state.get("complexity", QueryComplexity.SIMPLE),
                            answer_type=answer_type,
                            prompt_hint=prompt_hint,
                        )
                    answer = generated_text.strip()

        if answer_type in strict_types:
            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            # Skip coercion when strict_answerer already validated the answer with confidence.
            cited_ids_raw = strict_cited_ids or list(context_chunk_ids)
            if extracted:
                extracted_ok = True
            else:
                coerced, extracted_ok = self._coerce_strict_type_format(answer, answer_type, cited_ids_raw)
                answer = coerced.strip()

            # Rare second-pass "repair" if the first LLM output was not parseable.
            if (
                not extracted_ok
                and strict_repair_enabled
                and not extracted
                and not self._is_unanswerable_strict_answer(answer)
                and answer_type in {"boolean", "number", "date", "name", "names"}
            ):
                repair_hint = _STRICT_REPAIR_HINT_TEMPLATE.format(answer_type=answer_type)
                with collector.timed("llm"):
                    repaired_text, _ = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=min(int(state["max_tokens"]), 64),
                        collector=collector,
                        complexity=QueryComplexity.SIMPLE,
                        answer_type=answer_type,
                        prompt_hint=repair_hint,
                    )
                repaired, extracted_ok_2 = self._coerce_strict_type_format(repaired_text, answer_type, cited_ids_raw)
                if extracted_ok_2:
                    answer = repaired.strip()
                    logger.info(
                        "strict_repair_succeeded",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )
                else:
                    logger.warning(
                        "strict_repair_failed",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )

            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            if self._is_unanswerable_strict_answer(answer):
                used_ids = []
                cited_ids = []
            else:
                cited_ids = (
                    list(strict_cited_ids)
                    if (extracted and strict_cited_ids)
                    else self._localize_strict_support_chunk_ids(
                        answer_type=answer_type,
                        answer=answer,
                        query=state["query"],
                        context_chunks=context_chunks,
                    )
                )
                used_ids = self._expand_page_spanning_support_chunk_ids(
                    chunk_ids=cited_ids,
                    context_chunks=context_chunks,
                )
                if not cited_ids:
                    logger.warning(
                        "strict_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=[],
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            collector.set_cited_ids(cited_ids)
            _q_lower_fam = re.sub(r"\s+", " ", str(state["query"] or "").strip()).lower()
            _family_recall_priority = (
                self._ENACTMENT_QUERY_RE.search(_q_lower_fam)
                or self._ADMIN_QUERY_RE.search(_q_lower_fam)
                or self._OUTCOME_QUERY_RE.search(_q_lower_fam)
                or _is_broad_enumeration_query(state["query"])
                or _is_named_commencement_query(state["query"])
                or _is_named_multi_title_lookup_query(state["query"])
            )
            if _family_recall_priority:
                final_used_ids = list(shaped_used_ids if shaped_used_ids else used_ids)
            else:
                final_used_ids = self._rerank_support_pages_within_selected_docs(
                    query=state["query"],
                    answer_type=answer_type,
                    context_chunks=context_chunks,
                    used_ids=shaped_used_ids if shaped_used_ids else used_ids,
                )
            await self._set_final_used_pages(
                collector=collector,
                query=state["query"],
                answer=answer,
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_used_ids=final_used_ids,
            )
            citations: list[Citation] = []
            cited_ids = list(cited_ids)
        else:
            # Sanitize citations: remove any chunk IDs not present in context
            sanitize_citations = getattr(self._generator, "sanitize_citations", None)
            if callable(sanitize_citations):
                sanitized_obj = sanitize_citations(answer, context_chunk_ids)
                if isinstance(sanitized_obj, str):
                    answer = sanitized_obj
            cleanup_list_preamble = getattr(self._generator, "cleanup_list_answer_preamble", None)
            if callable(cleanup_list_preamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_preamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_items = getattr(self._generator, "cleanup_numbered_list_items", None)
            if callable(cleanup_list_items) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_items(
                    answer,
                    question=state["query"],
                    common_elements=_is_common_elements_query(state["query"]),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_titles_only = getattr(self._generator, "cleanup_broad_enumeration_titles_only", None)
            query_title_refs = _extract_question_title_refs(state["query"])
            named_ref_query = len(query_title_refs) >= 2 and any(
                term in str(state["query"]).lower() for term in ("mention", "mentions", "reference", "references")
            )
            if (
                callable(cleanup_titles_only)
                and _is_broad_enumeration_query(state["query"])
                and not _is_registrar_enumeration_query(state["query"])
                and not named_ref_query
            ):
                cleaned_obj = cleanup_titles_only(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_interpretative_items = getattr(self._generator, "cleanup_interpretative_provisions_enumeration_items", None)
            if (
                callable(cleanup_interpretative_items)
                and _is_broad_enumeration_query(state["query"])
                and "interpretative provisions" in str(state["query"]).lower()
            ):
                cleaned_obj = cleanup_interpretative_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_ref_items = getattr(self._generator, "cleanup_named_ref_enumeration_items", None)
            if callable(cleanup_named_ref_items) and named_ref_query and _is_broad_enumeration_query(state["query"]):
                cleaned_obj = cleanup_named_ref_items(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_ruler_enactment_items = getattr(self._generator, "cleanup_ruler_enactment_enumeration_items", None)
            if callable(cleanup_ruler_enactment_items) and _is_ruler_enactment_query(state["query"]):
                cleaned_obj = cleanup_ruler_enactment_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_registrar_items = getattr(self._generator, "cleanup_registrar_enumeration_items", None)
            if callable(cleanup_registrar_items) and (
                _is_registrar_enumeration_query(state["query"])
                or (_is_citation_title_query(state["query"]) and _is_enumeration_query(state["query"]))
            ):
                cleaned_obj = cleanup_registrar_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_postamble = getattr(self._generator, "cleanup_list_answer_postamble", None)
            if callable(cleanup_list_postamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_postamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
            if callable(strip_neg) and _is_common_elements_query(state["query"]):
                stripped_obj = strip_neg(answer)
                if isinstance(stripped_obj, str) and stripped_obj.strip():
                    answer = stripped_obj.strip()
            cleanup_common_elements_answer = getattr(self._generator, "cleanup_common_elements_canonical_answer", None)
            if callable(cleanup_common_elements_answer) and _is_common_elements_query(state["query"]):
                cleaned_obj = cleanup_common_elements_answer(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_commencement = getattr(self._generator, "cleanup_named_commencement_answer", None)
            if callable(cleanup_named_commencement):
                cleaned_obj = cleanup_named_commencement(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_administration = getattr(self._generator, "cleanup_named_administration_answer", None)
            if callable(cleanup_named_administration):
                cleaned_obj = cleanup_named_administration(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_penalty = getattr(self._generator, "cleanup_named_penalty_answer", None)
            if callable(cleanup_named_penalty):
                cleaned_obj = cleanup_named_penalty(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_multi_title_lookup = getattr(self._generator, "cleanup_named_multi_title_lookup_answer", None)
            if callable(cleanup_named_multi_title_lookup):
                cleaned_obj = cleanup_named_multi_title_lookup(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_amendment = getattr(self._generator, "cleanup_named_amendment_answer", None)
            if callable(cleanup_named_amendment):
                cleaned_obj = cleanup_named_amendment(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_enactment_date = getattr(self._generator, "cleanup_named_enactment_date_answer", None)
            if callable(cleanup_named_enactment_date):
                cleaned_obj = cleanup_named_enactment_date(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_made_by = getattr(self._generator, "cleanup_named_made_by_answer", None)
            if callable(cleanup_named_made_by):
                cleaned_obj = cleanup_named_made_by(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_registrar_authority = getattr(self._generator, "cleanup_named_registrar_authority_answer", None)
            if callable(cleanup_named_registrar_authority):
                cleaned_obj = cleanup_named_registrar_authority(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_translation_requirement = getattr(self._generator, "cleanup_named_translation_requirement_answer", None)
            if callable(cleanup_named_translation_requirement):
                cleaned_obj = cleanup_named_translation_requirement(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_account_effective_dates = getattr(self._generator, "cleanup_account_effective_dates_answer", None)
            if callable(cleanup_account_effective_dates):
                cleaned_obj = cleanup_account_effective_dates(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_final_answer = getattr(self._generator, "cleanup_final_answer", None)
            if callable(cleanup_final_answer):
                cleaned_obj = cleanup_final_answer(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    if cleaned_obj.strip() != answer.strip():
                        collector.set_llm_diagnostics(malformed_tail_detected=True)
                    answer = cleaned_obj.strip()

            citations = self._generator.extract_citations(answer, context_chunks)
            cited_ids = self._generator.extract_cited_chunk_ids(answer)
            is_unanswerable_free_text = self._is_unanswerable_free_text_answer(answer)
            support_ids: list[str] = []
            prefer_citation_trace = _is_named_multi_title_lookup_query(state["query"]) or _is_named_amendment_query(
                state["query"]
            )
            if not cited_ids and not is_unanswerable_free_text:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
                cited_ids = list(support_ids)
                if support_ids and not citations:
                    citations = self._citations_from_chunk_ids(
                        chunk_ids=support_ids,
                        context_chunks=context_chunks,
                    )
                elif not support_ids:
                    logger.warning(
                        "free_text_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            elif not is_unanswerable_free_text and not prefer_citation_trace:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
            if is_unanswerable_free_text:
                citations = []
                cited_ids = []
                support_ids = []
                collector.set_retrieved_ids([])
                collector.set_context_ids([])
            elif support_ids:
                support_ids = self._suppress_named_administration_family_orphan_support_ids(
                    query=state["query"],
                    cited_ids=cited_ids,
                    support_ids=support_ids,
                    context_chunks=context_chunks,
                )
            collector.set_cited_ids(cited_ids)
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=support_ids,
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            _q_lower_fam2 = re.sub(r"\s+", " ", str(state["query"] or "").strip()).lower()
            _family_recall_priority2 = (
                self._ENACTMENT_QUERY_RE.search(_q_lower_fam2)
                or self._ADMIN_QUERY_RE.search(_q_lower_fam2)
                or self._OUTCOME_QUERY_RE.search(_q_lower_fam2)
                or _is_broad_enumeration_query(state["query"])
                or _is_named_commencement_query(state["query"])
                or _is_named_multi_title_lookup_query(state["query"])
            )
            if _family_recall_priority2:
                final_used_ids = list(shaped_used_ids)
            else:
                final_used_ids = self._rerank_support_pages_within_selected_docs(
                    query=state["query"],
                    answer_type=answer_type,
                    context_chunks=context_chunks,
                    used_ids=shaped_used_ids,
                )
            await self._set_final_used_pages(
                collector=collector,
                query=state["query"],
                answer=answer,
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_used_ids=final_used_ids,
            )
            if answer_type == "free_text" and streamed and answer.strip():
                writer({"type": "answer_final", "text": answer})

        logger.info(
            "Generated answer %d chars with %d citations (strict_extracted=%s)",
            len(answer),
            len(citations),
            extracted,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "answer": answer,
            "citations": citations,
            "cited_chunk_ids": cited_ids,
            "streamed": streamed,
        }

    def _get_grounding_selector(self) -> object | None:
        """Lazily create grounding evidence selector if sidecar is enabled."""
        if not bool(getattr(self._settings.pipeline, "enable_grounding_sidecar", False)):
            return None
        cached = getattr(self, "_grounding_selector_cached", None)
        if cached is not None:
            return cached
        try:
            from rag_challenge.core.grounding.evidence_selector import GroundingEvidenceSelector

            retriever = self._retriever
            selector = GroundingEvidenceSelector(
                retriever=retriever,
                store=retriever._store,
                embedder=retriever._embedder,
                sparse_encoder=retriever._sparse_encoder,
                pipeline_settings=self._settings.pipeline,
            )
            self._grounding_selector_cached = selector  # type: ignore[attr-defined]
            return selector
        except Exception:
            logger.warning("Failed to create grounding selector", exc_info=True)
            return None

    async def _set_final_used_pages(
        self,
        *,
        collector: TelemetryCollector,
        query: str,
        answer: str,
        answer_type: str,
        context_chunks: Sequence[RankedChunk],
        current_used_ids: Sequence[str],
    ) -> None:
        """Finalize used-page telemetry with terminal anchor constraints.

        When grounding sidecar is enabled, delegates page selection to the
        evidence selector. Falls back to the legacy path if the selector
        returns None or raises.

        Args:
            collector: Telemetry collector for the current request.
            query: Raw user question.
            answer: Final answer text.
            answer_type: Normalized answer type.
            context_chunks: Ranked context chunks used for answering.
            current_used_ids: Support chunk IDs selected before late page shaping.
        """
        from rag_challenge.core.grounding.evidence_selector import answer_requires_empty_grounding

        if answer_requires_empty_grounding(answer):
            collector.set_used_ids(list(current_used_ids))
            collector.set_used_page_ids_override([])
            return

        # Try grounding sidecar first
        selector = self._get_grounding_selector()
        if selector is not None:
            try:
                from rag_challenge.core.grounding.evidence_selector import GroundingEvidenceSelector

                assert isinstance(selector, GroundingEvidenceSelector)
                sidecar_page_ids = await selector.select_page_ids(
                    query=query,
                    answer=answer,
                    answer_type=answer_type,
                    context_chunks=context_chunks,
                    current_used_ids=current_used_ids,
                    collector=collector,
                )
                if sidecar_page_ids is not None:
                    collector.set_used_ids(list(current_used_ids))
                    collector.set_used_page_ids_override(sidecar_page_ids)
                    return
            except Exception:
                logger.warning("Grounding sidecar failed; falling back to legacy", exc_info=True)

        # Legacy path
        anchor_page_ids = self._explicit_anchor_page_ids(
            query=query,
            context_chunks=context_chunks,
            preferred_chunk_ids=current_used_ids,
        )
        final_used_ids = list(current_used_ids)
        if not anchor_page_ids:
            final_used_ids = self._enhance_page_recall(
                query=query,
                answer_type=answer_type,
                context_chunks=context_chunks,
                current_used_ids=final_used_ids,
            )
        collector.set_used_ids(final_used_ids)

        final_page_ids = self._chunk_ids_to_page_ids(
            chunk_ids=final_used_ids,
            context_chunks=context_chunks,
        )
        citation_pages = self._extract_citation_pages(
            question=query,
            answer=answer,
            answer_type=answer_type,
            context_chunks=context_chunks,
        )
        if citation_pages:
            final_page_ids = citation_pages

        if anchor_page_ids:
            final_page_ids = self._constrain_page_ids_to_anchor(
                page_ids=final_page_ids,
                anchor_page_ids=anchor_page_ids,
            )

        trimmed = self._trim_to_article_page(
            question=query,
            answer_type=answer_type,
            context_chunks=context_chunks,
            current_page_ids=final_page_ids,
        )
        if trimmed:
            final_page_ids = (
                self._constrain_page_ids_to_anchor(
                    page_ids=trimmed,
                    anchor_page_ids=anchor_page_ids,
                )
                if anchor_page_ids
                else trimmed
            )

        if final_page_ids:
            collector.set_used_page_ids_override(final_page_ids)

    def _chunk_ids_to_page_ids(
        self,
        *,
        chunk_ids: Sequence[str],
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        """Convert support chunk IDs into ordered page IDs.

        Args:
            chunk_ids: Candidate support chunk IDs.
            context_chunks: Ranked context chunks that define page identity.

        Returns:
            Ordered unique page IDs present in the provided context chunks.
        """
        context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
        page_ids: list[str] = []
        seen_page_ids: set[str] = set()
        for raw_chunk_id in chunk_ids:
            chunk = context_by_id.get(str(raw_chunk_id).strip())
            if chunk is None:
                continue
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
            page_num = self._page_num(str(getattr(chunk, "section_path", "") or ""))
            if not doc_id or page_num <= 0:
                continue
            page_id = f"{doc_id}_{page_num}"
            if page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)
            page_ids.append(page_id)
        return page_ids

    @staticmethod
    def _constrain_page_ids_to_anchor(
        *,
        page_ids: Sequence[str],
        anchor_page_ids: Sequence[str],
    ) -> list[str]:
        """Clamp used-page output to the explicit anchor set.

        Args:
            page_ids: Candidate used page IDs.
            anchor_page_ids: Allowed page IDs derived from the explicit anchor.

        Returns:
            Ordered page IDs that stay within the anchor set.
        """
        ordered_anchor_page_ids = [str(page_id).strip() for page_id in anchor_page_ids if str(page_id).strip()]
        if not ordered_anchor_page_ids:
            return [str(page_id).strip() for page_id in page_ids if str(page_id).strip()]

        allowed_page_ids = set(ordered_anchor_page_ids)
        constrained = [
            str(page_id).strip()
            for page_id in page_ids
            if str(page_id).strip() and str(page_id).strip() in allowed_page_ids
        ]
        return constrained or ordered_anchor_page_ids

    async def _verify(self, state: RAGState) -> dict[str, object]:
        verifier_settings = getattr(self._settings, "verifier", None)
        if verifier_settings is not None and not bool(getattr(verifier_settings, "enabled", True)):
            return {}
        if self._verifier is None:
            return {}

        answer = str(state.get("answer") or "").strip()
        cited_ids = list(state.get("cited_chunk_ids", []))

        # Skip LLM verifier entirely for strict types — deterministic coerce already ran in _generate
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        if answer_type in {"boolean", "number", "date", "name", "names"}:
            return {}
        should_verify = self._verifier.should_verify(answer, cited_ids, force=False)
        if not should_verify:
            return {}

        collector = state["collector"]
        context_chunks = state.get("context_chunks", [])
        with collector.timed("verify"):
            verification = await self._verifier.verify(
                state["query"],
                answer,
                context_chunks,
            )

        # For free_text, verifier is audit-only (post-hoc revision cannot affect already-streamed output).
        if answer_type == "free_text":
            logger.info(
                "Verifier audit grounded=%s unsupported_claims=%d",
                verification.is_grounded,
                len(verification.unsupported_claims),
            )
            return {}

        next_answer = answer
        if not verification.is_grounded:
            if verification.revised_answer:
                next_answer = verification.revised_answer.strip()
                logger.info(
                    "Verifier revised answer grounded=%s unsupported_claims=%d",
                    verification.is_grounded,
                    len(verification.unsupported_claims),
                )
            else:
                # Fail-safe: keep strict-type output format deterministic even when sources are insufficient.
                next_answer = self._strict_type_fallback(answer_type, cited_ids)

        effective_cited_ids = self._generator.extract_cited_chunk_ids(next_answer) or cited_ids
        next_answer, _ = self._coerce_strict_type_format(next_answer, answer_type, effective_cited_ids)
        next_answer = next_answer.strip()

        # Sanitize citations after coercion
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        sanitize_citations = getattr(self._generator, "sanitize_citations", None)
        if callable(sanitize_citations):
            sanitized_obj = sanitize_citations(next_answer, context_chunk_ids)
            if isinstance(sanitized_obj, str):
                next_answer = sanitized_obj

        if next_answer == answer:
            return {}

        citations = self._generator.extract_citations(next_answer, context_chunks)
        next_cited_ids = self._generator.extract_cited_chunk_ids(next_answer)
        if self._is_unanswerable_strict_answer(next_answer):
            citations = []
            next_cited_ids = []
            collector.set_retrieved_ids([])
            collector.set_context_ids([])
        collector.set_cited_ids(next_cited_ids)
        return {
            "answer": next_answer,
            "citations": citations,
            "cited_chunk_ids": next_cited_ids,
        }
    async def _emit(self, state: RAGState) -> dict[str, object]:
        """Emit answer tokens for strict-types flows that generated non-streaming output."""
        if bool(state.get("streamed", False)):
            return {}

        answer = str(state.get("answer") or "")
        if not answer.strip():
            return {}

        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.mark_first_token()
        writer({"type": "token", "text": answer})
        writer({"type": "answer_final", "text": answer})
        return {"streamed": True}
    async def _finalize(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        telemetry = collector.finalize()
        writer = self._get_stream_writer_or_noop()
        writer({"type": "telemetry", "payload": telemetry.model_dump()})
        return {"telemetry": telemetry}
    @staticmethod
    def _get_stream_writer_or_noop() -> Callable[[dict[str, object]], None]:
        from rag_challenge.core import pipeline as pipeline_module

        try:
            writer = pipeline_module.get_stream_writer()
        except RuntimeError:
            return lambda _: None
        return cast("Callable[[dict[str, object]], None]", writer)
