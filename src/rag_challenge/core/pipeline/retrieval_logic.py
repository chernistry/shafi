"""Thin facade that composes typed retrieval helper functions."""

from __future__ import annotations

from .retrieval_boolean_handlers import (
    account_effective_support_family_seed_chunk_ids,
    administration_support_family_seed_chunk_ids,
    ensure_account_effective_dates_context,
    ensure_boolean_admin_compare_context,
    ensure_boolean_judge_compare_context,
    ensure_boolean_year_compare_context,
    ensure_notice_doc_context,
    prune_boolean_context_for_single_doc_article,
    remuneration_recordkeeping_clause_score,
)
from .retrieval_common_elements import (
    common_elements_evidence_score,
    common_elements_ref_tokens,
    common_elements_title_match_score,
    ensure_common_elements_context,
)
from .retrieval_context import (
    augment_strict_context_chunks,
    build_entity_scope,
    collapse_doc_family_crowding_context,
    detect_coverage_gaps,
    doc_family_collapse_candidate_score,
    ensure_must_include_context,
    ensure_page_one_context,
    raw_to_ranked,
)
from .retrieval_named_handlers import (
    chunk_has_named_administration_clause,
    chunk_has_self_registrar_clause,
    ensure_named_administration_context,
    ensure_named_amendment_context,
    ensure_named_commencement_context,
    ensure_named_multi_title_context,
    ensure_named_penalty_context,
    ensure_self_registrar_context,
    named_administration_clause_score,
    named_amendment_clause_score,
    named_commencement_clause_score,
    named_commencement_title_match_score,
    named_multi_title_clause_score,
    named_penalty_clause_score,
)
from .retrieval_primitives import (
    augment_query_for_sparse_retrieval,
    dedupe_chunk_ids,
    extract_provision_refs,
    merge_retrieved_preserving_chunk_ids,
    section_page_num,
    seed_terms_for_query,
    targeted_provision_ref_query,
)
from .retrieval_seed_selection import (
    best_named_administration_chunk,
    boolean_admin_seed_chunk_score,
    boolean_year_seed_chunk_score,
    case_issue_date_seed_chunk_score,
    case_judge_seed_chunk_score,
    case_outcome_seed_chunk_score,
    case_ref_identity_score,
    extract_title_ref_from_chunk_text,
    extract_title_refs_from_query,
    is_canonical_ref_family_chunk,
    is_consolidated_or_amended_family_chunk,
    is_notice_focus_query,
    notice_doc_score,
    ref_doc_family_consistency_adjustment,
    select_case_issue_date_seed_chunk_id,
    select_case_judge_seed_chunk_id,
    select_case_outcome_seed_chunk_id,
    select_seed_chunk_id,
    select_targeted_title_seed_chunk_id,
)
from .types import CompatClassOrInstanceMethod


class RetrievalLogicMixin:
    """Aggregate retrieval logic for the pipeline builder."""

    account_effective_support_family_seed_chunk_ids = account_effective_support_family_seed_chunk_ids
    administration_support_family_seed_chunk_ids = administration_support_family_seed_chunk_ids
    augment_query_for_sparse_retrieval = staticmethod(augment_query_for_sparse_retrieval)
    augment_strict_context_chunks = augment_strict_context_chunks
    best_named_administration_chunk = best_named_administration_chunk
    boolean_admin_seed_chunk_score = boolean_admin_seed_chunk_score
    boolean_year_seed_chunk_score = boolean_year_seed_chunk_score
    build_entity_scope = staticmethod(build_entity_scope)
    case_issue_date_seed_chunk_score = case_issue_date_seed_chunk_score
    case_judge_seed_chunk_score = case_judge_seed_chunk_score
    case_outcome_seed_chunk_score = case_outcome_seed_chunk_score
    case_ref_identity_score = case_ref_identity_score
    chunk_has_named_administration_clause = chunk_has_named_administration_clause
    chunk_has_self_registrar_clause = chunk_has_self_registrar_clause
    collapse_doc_family_crowding_context = collapse_doc_family_crowding_context
    common_elements_evidence_score = staticmethod(common_elements_evidence_score)
    common_elements_ref_tokens = common_elements_ref_tokens
    common_elements_title_match_score = common_elements_title_match_score
    dedupe_chunk_ids = staticmethod(dedupe_chunk_ids)
    detect_coverage_gaps = staticmethod(detect_coverage_gaps)
    doc_family_collapse_candidate_score = doc_family_collapse_candidate_score
    ensure_account_effective_dates_context = ensure_account_effective_dates_context
    ensure_boolean_admin_compare_context = ensure_boolean_admin_compare_context
    ensure_boolean_judge_compare_context = ensure_boolean_judge_compare_context
    ensure_boolean_year_compare_context = ensure_boolean_year_compare_context
    ensure_common_elements_context = ensure_common_elements_context
    ensure_must_include_context = staticmethod(ensure_must_include_context)
    ensure_named_administration_context = ensure_named_administration_context
    ensure_named_amendment_context = ensure_named_amendment_context
    ensure_named_commencement_context = ensure_named_commencement_context
    ensure_named_multi_title_context = ensure_named_multi_title_context
    ensure_named_penalty_context = ensure_named_penalty_context
    ensure_notice_doc_context = ensure_notice_doc_context
    ensure_page_one_context = staticmethod(ensure_page_one_context)
    ensure_self_registrar_context = ensure_self_registrar_context
    extract_provision_refs = staticmethod(extract_provision_refs)
    extract_title_ref_from_chunk_text = staticmethod(extract_title_ref_from_chunk_text)
    extract_title_refs_from_query = staticmethod(extract_title_refs_from_query)
    is_canonical_ref_family_chunk = is_canonical_ref_family_chunk
    is_consolidated_or_amended_family_chunk = is_consolidated_or_amended_family_chunk
    is_notice_focus_query = is_notice_focus_query
    merge_retrieved_preserving_chunk_ids = merge_retrieved_preserving_chunk_ids
    named_administration_clause_score = named_administration_clause_score
    named_amendment_clause_score = named_amendment_clause_score
    named_commencement_clause_score = staticmethod(named_commencement_clause_score)
    named_commencement_title_match_score = named_commencement_title_match_score
    named_multi_title_clause_score = named_multi_title_clause_score
    named_penalty_clause_score = named_penalty_clause_score
    notice_doc_score = notice_doc_score
    prune_boolean_context_for_single_doc_article = prune_boolean_context_for_single_doc_article
    raw_to_ranked = staticmethod(raw_to_ranked)
    ref_doc_family_consistency_adjustment = ref_doc_family_consistency_adjustment
    remuneration_recordkeeping_clause_score = remuneration_recordkeeping_clause_score
    section_page_num = staticmethod(section_page_num)
    seed_terms_for_query = staticmethod(seed_terms_for_query)
    select_case_issue_date_seed_chunk_id = select_case_issue_date_seed_chunk_id
    select_case_judge_seed_chunk_id = select_case_judge_seed_chunk_id
    select_case_outcome_seed_chunk_id = select_case_outcome_seed_chunk_id
    select_seed_chunk_id = select_seed_chunk_id
    select_targeted_title_seed_chunk_id = select_targeted_title_seed_chunk_id
    targeted_provision_ref_query = targeted_provision_ref_query



_RETRIEVAL_COMPAT_ALIASES: dict[str, str] = {
    "_account_effective_support_family_seed_chunk_ids": "account_effective_support_family_seed_chunk_ids",
    "_administration_support_family_seed_chunk_ids": "administration_support_family_seed_chunk_ids",
    "_augment_query_for_sparse_retrieval": "augment_query_for_sparse_retrieval",
    "_augment_strict_context_chunks": "augment_strict_context_chunks",
    "_best_named_administration_chunk": "best_named_administration_chunk",
    "_boolean_admin_seed_chunk_score": "boolean_admin_seed_chunk_score",
    "_boolean_year_seed_chunk_score": "boolean_year_seed_chunk_score",
    "_build_entity_scope": "build_entity_scope",
    "_case_issue_date_seed_chunk_score": "case_issue_date_seed_chunk_score",
    "_case_judge_seed_chunk_score": "case_judge_seed_chunk_score",
    "_case_outcome_seed_chunk_score": "case_outcome_seed_chunk_score",
    "_case_ref_identity_score": "case_ref_identity_score",
    "_chunk_has_named_administration_clause": "chunk_has_named_administration_clause",
    "_chunk_has_self_registrar_clause": "chunk_has_self_registrar_clause",
    "_collapse_doc_family_crowding_context": "collapse_doc_family_crowding_context",
    "_common_elements_evidence_score": "common_elements_evidence_score",
    "_common_elements_ref_tokens": "common_elements_ref_tokens",
    "_common_elements_title_match_score": "common_elements_title_match_score",
    "_dedupe_chunk_ids": "dedupe_chunk_ids",
    "_detect_coverage_gaps": "detect_coverage_gaps",
    "_doc_family_collapse_candidate_score": "doc_family_collapse_candidate_score",
    "_ensure_account_effective_dates_context": "ensure_account_effective_dates_context",
    "_ensure_boolean_admin_compare_context": "ensure_boolean_admin_compare_context",
    "_ensure_boolean_judge_compare_context": "ensure_boolean_judge_compare_context",
    "_ensure_boolean_year_compare_context": "ensure_boolean_year_compare_context",
    "_ensure_common_elements_context": "ensure_common_elements_context",
    "_ensure_must_include_context": "ensure_must_include_context",
    "_ensure_named_administration_context": "ensure_named_administration_context",
    "_ensure_named_amendment_context": "ensure_named_amendment_context",
    "_ensure_named_commencement_context": "ensure_named_commencement_context",
    "_ensure_named_multi_title_context": "ensure_named_multi_title_context",
    "_ensure_named_penalty_context": "ensure_named_penalty_context",
    "_ensure_notice_doc_context": "ensure_notice_doc_context",
    "_ensure_page_one_context": "ensure_page_one_context",
    "_ensure_self_registrar_context": "ensure_self_registrar_context",
    "_extract_provision_refs": "extract_provision_refs",
    "_extract_title_ref_from_chunk_text": "extract_title_ref_from_chunk_text",
    "_extract_title_refs_from_query": "extract_title_refs_from_query",
    "_is_canonical_ref_family_chunk": "is_canonical_ref_family_chunk",
    "_is_consolidated_or_amended_family_chunk": "is_consolidated_or_amended_family_chunk",
    "_is_notice_focus_query": "is_notice_focus_query",
    "_merge_retrieved_preserving_chunk_ids": "merge_retrieved_preserving_chunk_ids",
    "_named_administration_clause_score": "named_administration_clause_score",
    "_named_amendment_clause_score": "named_amendment_clause_score",
    "_named_commencement_clause_score": "named_commencement_clause_score",
    "_named_commencement_title_match_score": "named_commencement_title_match_score",
    "_named_multi_title_clause_score": "named_multi_title_clause_score",
    "_named_penalty_clause_score": "named_penalty_clause_score",
    "_notice_doc_score": "notice_doc_score",
    "_prune_boolean_context_for_single_doc_article": "prune_boolean_context_for_single_doc_article",
    "_raw_to_ranked": "raw_to_ranked",
    "_ref_doc_family_consistency_adjustment": "ref_doc_family_consistency_adjustment",
    "_remuneration_recordkeeping_clause_score": "remuneration_recordkeeping_clause_score",
    "_section_page_num": "section_page_num",
    "_seed_terms_for_query": "seed_terms_for_query",
    "_select_case_issue_date_seed_chunk_id": "select_case_issue_date_seed_chunk_id",
    "_select_case_judge_seed_chunk_id": "select_case_judge_seed_chunk_id",
    "_select_case_outcome_seed_chunk_id": "select_case_outcome_seed_chunk_id",
    "_select_seed_chunk_id": "select_seed_chunk_id",
    "_select_targeted_title_seed_chunk_id": "select_targeted_title_seed_chunk_id",
    "_targeted_provision_ref_query": "targeted_provision_ref_query",
}

for _alias, _target in _RETRIEVAL_COMPAT_ALIASES.items():
    setattr(RetrievalLogicMixin, _alias, CompatClassOrInstanceMethod(_target))
