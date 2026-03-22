"""Thin facade that composes typed support helper functions."""

from __future__ import annotations

from .support_formatting import (
    citation_suffix,
    coerce_strict_type_format,
    insufficient_sources_answer,
    is_unanswerable_free_text_answer,
    is_unanswerable_strict_answer,
    raw_ranked,
    strict_type_citation_suffix,
    strict_type_fallback,
)
from .support_free_text import (
    best_support_chunk_id_for_doc_page,
    best_title_support_chunk_id,
    context_family_chunk_ids,
    doc_ids_for_chunk_ids,
    extract_free_text_item_slots,
    free_text_doc_group_match_score,
    free_text_item_candidate_chunks,
    free_text_item_title_slot,
    free_text_slot_full_context_priority,
    group_context_chunks_by_doc,
    split_free_text_items,
    split_free_text_support_fragments,
    split_names,
)
from .support_localization import (
    citations_from_chunk_ids,
    localize_boolean_compare_support_chunk_ids,
    localize_boolean_support_chunk_ids,
    localize_free_text_support_chunk_ids,
    localize_strict_support_chunk_ids,
    suppress_named_administration_family_orphan_support_ids,
)
from .support_page_policy import (
    ADMIN_QUERY_RE,
    ENACTMENT_QUERY_RE,
    OUTCOME_QUERY_RE,
    apply_support_shape_policy,
    boost_family_context_chunks,
    enhance_page_recall,
    explicit_anchor_page_ids,
    explicit_page_reference_support_chunk_ids,
    extract_citation_pages,
    is_metadata_page_family_query,
    is_named_metadata_support_query,
    named_metadata_requires_support_union,
    rerank_support_pages_within_selected_docs,
    trim_to_article_page,
)
from .support_query_primitives import (
    build_chunk_page_hint_map,
    build_chunk_snippet,
    build_chunk_snippet_map,
    combined_named_refs,
    date_fragment_variants,
    expand_page_spanning_support_chunk_ids,
    matched_doc_chunks_for_ref,
    missing_named_ref_targets,
    normalize_support_text,
    ordinal_suffix,
    page_num,
    page_text_looks_like_continuation_head,
    page_text_looks_like_continuation_tail,
    page_text_looks_like_new_section,
    paired_support_question_refs,
    ref_has_criterion_support,
    should_apply_doc_shortlist_gating,
    support_question_refs,
    support_terms,
    targeted_named_ref_query,
)
from .support_scoring import (
    account_effective_clause_score,
    account_enactment_clause_score,
    apply_doc_shortlist_gating,
    best_support_chunk_id,
    boolean_year_compare_chunk_score,
    chunk_support_blob,
    chunk_support_score,
    doc_shortlist_score,
    normalize_numeric_text,
    restriction_effectiveness_clause_score,
)
from .types import CompatClassOrInstanceMethod


class SupportLogicMixin:
    """Aggregate support logic for the pipeline builder."""

    ADMIN_QUERY_RE = ADMIN_QUERY_RE
    ENACTMENT_QUERY_RE = ENACTMENT_QUERY_RE
    OUTCOME_QUERY_RE = OUTCOME_QUERY_RE
    account_effective_clause_score = account_effective_clause_score
    account_enactment_clause_score = account_enactment_clause_score
    apply_doc_shortlist_gating = apply_doc_shortlist_gating
    apply_support_shape_policy = apply_support_shape_policy
    best_support_chunk_id = best_support_chunk_id
    best_support_chunk_id_for_doc_page = best_support_chunk_id_for_doc_page
    best_title_support_chunk_id = best_title_support_chunk_id
    boolean_year_compare_chunk_score = boolean_year_compare_chunk_score
    boost_family_context_chunks = boost_family_context_chunks
    build_chunk_page_hint_map = build_chunk_page_hint_map
    build_chunk_snippet = staticmethod(build_chunk_snippet)
    build_chunk_snippet_map = build_chunk_snippet_map
    chunk_support_blob = chunk_support_blob
    chunk_support_score = chunk_support_score
    citation_suffix = staticmethod(citation_suffix)
    citations_from_chunk_ids = citations_from_chunk_ids
    coerce_strict_type_format = coerce_strict_type_format
    combined_named_refs = combined_named_refs
    context_family_chunk_ids = context_family_chunk_ids
    date_fragment_variants = date_fragment_variants
    doc_ids_for_chunk_ids = doc_ids_for_chunk_ids
    doc_shortlist_score = doc_shortlist_score
    enhance_page_recall = enhance_page_recall
    explicit_anchor_page_ids = explicit_anchor_page_ids
    expand_page_spanning_support_chunk_ids = expand_page_spanning_support_chunk_ids
    explicit_page_reference_support_chunk_ids = explicit_page_reference_support_chunk_ids
    extract_citation_pages = staticmethod(extract_citation_pages)
    extract_free_text_item_slots = extract_free_text_item_slots
    free_text_doc_group_match_score = free_text_doc_group_match_score
    free_text_item_candidate_chunks = free_text_item_candidate_chunks
    free_text_item_title_slot = free_text_item_title_slot
    free_text_slot_full_context_priority = free_text_slot_full_context_priority
    group_context_chunks_by_doc = group_context_chunks_by_doc
    insufficient_sources_answer = insufficient_sources_answer
    is_metadata_page_family_query = is_metadata_page_family_query
    is_named_metadata_support_query = is_named_metadata_support_query
    is_unanswerable_free_text_answer = staticmethod(is_unanswerable_free_text_answer)
    is_unanswerable_strict_answer = staticmethod(is_unanswerable_strict_answer)
    localize_boolean_compare_support_chunk_ids = localize_boolean_compare_support_chunk_ids
    localize_boolean_support_chunk_ids = localize_boolean_support_chunk_ids
    localize_free_text_support_chunk_ids = localize_free_text_support_chunk_ids
    localize_strict_support_chunk_ids = localize_strict_support_chunk_ids
    matched_doc_chunks_for_ref = matched_doc_chunks_for_ref
    missing_named_ref_targets = missing_named_ref_targets
    named_metadata_requires_support_union = named_metadata_requires_support_union
    normalize_numeric_text = staticmethod(normalize_numeric_text)
    normalize_support_text = staticmethod(normalize_support_text)
    ordinal_suffix = staticmethod(ordinal_suffix)
    page_num = staticmethod(page_num)
    page_text_looks_like_continuation_head = staticmethod(page_text_looks_like_continuation_head)
    page_text_looks_like_continuation_tail = staticmethod(page_text_looks_like_continuation_tail)
    page_text_looks_like_new_section = staticmethod(page_text_looks_like_new_section)
    paired_support_question_refs = paired_support_question_refs
    raw_ranked = staticmethod(raw_ranked)
    ref_has_criterion_support = ref_has_criterion_support
    rerank_support_pages_within_selected_docs = rerank_support_pages_within_selected_docs
    restriction_effectiveness_clause_score = restriction_effectiveness_clause_score
    should_apply_doc_shortlist_gating = should_apply_doc_shortlist_gating
    split_free_text_items = split_free_text_items
    split_free_text_support_fragments = split_free_text_support_fragments
    split_names = staticmethod(split_names)
    strict_type_citation_suffix = strict_type_citation_suffix
    strict_type_fallback = strict_type_fallback
    support_question_refs = support_question_refs
    support_terms = support_terms
    suppress_named_administration_family_orphan_support_ids = suppress_named_administration_family_orphan_support_ids
    targeted_named_ref_query = targeted_named_ref_query
    trim_to_article_page = staticmethod(trim_to_article_page)


_SUPPORT_COMPAT_ALIASES: dict[str, str] = {
    "_ADMIN_QUERY_RE": "ADMIN_QUERY_RE",
    "_ENACTMENT_QUERY_RE": "ENACTMENT_QUERY_RE",
    "_OUTCOME_QUERY_RE": "OUTCOME_QUERY_RE",
    "_account_effective_clause_score": "account_effective_clause_score",
    "_account_enactment_clause_score": "account_enactment_clause_score",
    "_apply_doc_shortlist_gating": "apply_doc_shortlist_gating",
    "_apply_support_shape_policy": "apply_support_shape_policy",
    "_best_support_chunk_id": "best_support_chunk_id",
    "_best_support_chunk_id_for_doc_page": "best_support_chunk_id_for_doc_page",
    "_best_title_support_chunk_id": "best_title_support_chunk_id",
    "_boolean_year_compare_chunk_score": "boolean_year_compare_chunk_score",
    "_boost_family_context_chunks": "boost_family_context_chunks",
    "_build_chunk_page_hint_map": "build_chunk_page_hint_map",
    "_build_chunk_snippet": "build_chunk_snippet",
    "_build_chunk_snippet_map": "build_chunk_snippet_map",
    "_chunk_support_blob": "chunk_support_blob",
    "_chunk_support_score": "chunk_support_score",
    "_citation_suffix": "citation_suffix",
    "_citations_from_chunk_ids": "citations_from_chunk_ids",
    "_coerce_strict_type_format": "coerce_strict_type_format",
    "_combined_named_refs": "combined_named_refs",
    "_context_family_chunk_ids": "context_family_chunk_ids",
    "_date_fragment_variants": "date_fragment_variants",
    "_doc_ids_for_chunk_ids": "doc_ids_for_chunk_ids",
    "_doc_shortlist_score": "doc_shortlist_score",
    "_enhance_page_recall": "enhance_page_recall",
    "_explicit_anchor_page_ids": "explicit_anchor_page_ids",
    "_expand_page_spanning_support_chunk_ids": "expand_page_spanning_support_chunk_ids",
    "_explicit_page_reference_support_chunk_ids": "explicit_page_reference_support_chunk_ids",
    "_extract_citation_pages": "extract_citation_pages",
    "_extract_free_text_item_slots": "extract_free_text_item_slots",
    "_free_text_doc_group_match_score": "free_text_doc_group_match_score",
    "_free_text_item_candidate_chunks": "free_text_item_candidate_chunks",
    "_free_text_item_title_slot": "free_text_item_title_slot",
    "_free_text_slot_full_context_priority": "free_text_slot_full_context_priority",
    "_group_context_chunks_by_doc": "group_context_chunks_by_doc",
    "_insufficient_sources_answer": "insufficient_sources_answer",
    "_is_metadata_page_family_query": "is_metadata_page_family_query",
    "_is_named_metadata_support_query": "is_named_metadata_support_query",
    "_is_unanswerable_free_text_answer": "is_unanswerable_free_text_answer",
    "_is_unanswerable_strict_answer": "is_unanswerable_strict_answer",
    "_localize_boolean_compare_support_chunk_ids": "localize_boolean_compare_support_chunk_ids",
    "_localize_boolean_support_chunk_ids": "localize_boolean_support_chunk_ids",
    "_localize_free_text_support_chunk_ids": "localize_free_text_support_chunk_ids",
    "_localize_strict_support_chunk_ids": "localize_strict_support_chunk_ids",
    "_matched_doc_chunks_for_ref": "matched_doc_chunks_for_ref",
    "_missing_named_ref_targets": "missing_named_ref_targets",
    "_named_metadata_requires_support_union": "named_metadata_requires_support_union",
    "_normalize_numeric_text": "normalize_numeric_text",
    "_normalize_support_text": "normalize_support_text",
    "_ordinal_suffix": "ordinal_suffix",
    "_page_num": "page_num",
    "_page_text_looks_like_continuation_head": "page_text_looks_like_continuation_head",
    "_page_text_looks_like_continuation_tail": "page_text_looks_like_continuation_tail",
    "_page_text_looks_like_new_section": "page_text_looks_like_new_section",
    "_paired_support_question_refs": "paired_support_question_refs",
    "_raw_ranked": "raw_ranked",
    "_ref_has_criterion_support": "ref_has_criterion_support",
    "_rerank_support_pages_within_selected_docs": "rerank_support_pages_within_selected_docs",
    "_restriction_effectiveness_clause_score": "restriction_effectiveness_clause_score",
    "_should_apply_doc_shortlist_gating": "should_apply_doc_shortlist_gating",
    "_split_free_text_items": "split_free_text_items",
    "_split_free_text_support_fragments": "split_free_text_support_fragments",
    "_split_names": "split_names",
    "_strict_type_citation_suffix": "strict_type_citation_suffix",
    "_strict_type_fallback": "strict_type_fallback",
    "_support_question_refs": "support_question_refs",
    "_support_terms": "support_terms",
    "_suppress_named_administration_family_orphan_support_ids": "suppress_named_administration_family_orphan_support_ids",
    "_targeted_named_ref_query": "targeted_named_ref_query",
    "_trim_to_article_page": "trim_to_article_page",
}

for _alias, _target in _SUPPORT_COMPAT_ALIASES.items():
    setattr(SupportLogicMixin, _alias, CompatClassOrInstanceMethod(_target))
