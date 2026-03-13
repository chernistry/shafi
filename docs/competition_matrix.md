# Competition Matrix

## Current Status

- Current public best: `v6_public_exactness_champion` total=`0.741560` rank=`8`
- Current best offline candidate: `triad_f331_e0798_plus_dotted` paranoid=`0.741560`
- Warm-up submissions used/remaining: `9/10`, remaining=`1`
- Current default decision: `local_ceiling_reached_hold_budget`
- Current S: `0.888000`
- Current G: `0.800729`
- Required `G` to beat rank `1`: `0.936636` (ΔG `+0.135907`)
- Required `G` to beat rank `3`: `0.923223` (ΔG `+0.122494`)
- Required `G` to beat rank `5`: `0.858661` (ΔG `+0.057932`)
- #1 reachable through current small-diff path: `no`
- Current path locally ceilinged: `yes`

## Matrix

| label | date | status | branch_class | git_commit | baseline | lineage_confidence | answer_drift | page_drift | hidden_g_trusted | hidden_g_all | judge_pass_rate | judge_grounding | judge_accuracy | exactness_resolved_qids | exactness_unresolved_qids | external_det | external_asst | external_g | external_t | external_f | external_total | external_rank | platform_like_total_estimate | strict_total_estimate | paranoid_total_estimate | supervisor_action | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| submitted_unknown_01 | 2026-03-11 | submitted | unknown | unknown | unknown | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Warm-up submission counted on the public board, but no recoverable local lineage artifact was found. |
| submitted_unknown_02 | 2026-03-11 | submitted | unknown | unknown | unknown | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Warm-up submission counted on the public board, but no recoverable local lineage artifact was found. |
| submitted_unknown_03 | 2026-03-11 | submitted | unknown | unknown | unknown | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Warm-up submission counted on the public board, but no recoverable local lineage artifact was found. |
| submitted_unknown_04 | 2026-03-11 | submitted | unknown | unknown | unknown | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Warm-up submission counted on the public board, but no recoverable local lineage artifact was found. |
| v5_public_support_baseline | 2026-03-12 | submitted | support_calibrated_baseline | unknown | public | high | - | - | - | - | - | - | - | - | - | 0.9429 | 0.6667 | 0.8007 | 0.9960 | 1.0471 | 0.718177 | - | - | - | - | - | Last clean accepted lineage artifact available in repo. |
| v6_public_exactness_champion | 2026-03-12 | submitted | answer_only_exactness | unknown | v5_public_support_baseline | low | - | - | - | - | - | - | - | - | - | 0.9710 | 0.6930 | 0.8010 | 0.9960 | 1.0471 | 0.741560 | 8 | - | - | - | - | Best public result so far; local artifact lineage is ambiguous relative to the public score state. |
| v7_public_all_context_failure | 2026-03-12 | submitted | broad_page_inflation | unknown | v6_public_exactness_champion | low | - | - | - | - | - | - | - | - | - | 0.9710 | 0.6470 | 0.6080 | - | - | 0.554000 | - | - | - | - | - | All-context page broadening; catastrophic public grounding regression. |
| v8_public_onora_no_gain | 2026-03-12 | submitted | answer_only_exactness | unknown | v6_public_exactness_champion | low | - | - | - | - | - | - | - | - | - | 0.9710 | 0.6870 | 0.8010 | - | - | 0.740000 | - | - | - | - | - | ONORA casing tweak; no real public gain over v6. |
| v9_public_reranked_pages_failure | 2026-03-12 | submitted | mixed_page_rerank_failure | unknown | v6_public_exactness_champion | high | - | - | - | - | - | - | - | - | - | 0.9714 | 0.7000 | 0.6538 | 0.9940 | 1.0471 | 0.605673 | - | - | - | - | - | Reranked-page submission; public grounding crash. |
| v10_exactness_only | 2026-03-12 | candidate | exactness_only_fallback | unknown | submission_v4_anchor_lineage | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Exactness-only safety artifact; defensive fallback only because lineage is anchored to v4 lineage rather than clean public v6. |
| v5046_exactness_only_from_v6_context_seed | 2026-03-12 | candidate | exactness_only_fallback | unknown | submission_v6_context_seed | low | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Local page-stable exactness fallback relative to v6_context_seed only; not promoted for public submit. |
| triad_f331_e0798 | 2026-03-13 | ceiling | support_only_offense | 0343e02 | submission_v6_context_seed | high | 0 | 4 | 0.0425 | 0.0206 | - | - | - | - | - | - | - | - | - | - | - | - | 0.760607 | 0.760607 | - | - | Best pure-G small-diff ceiling candidate under local gates. [candidate_cycle=PROMISING] |
| triad_f331_e0798_plus_dotted | 2026-03-13 | ceiling | combined_small_diff_ceiling | 0343e02 | submission_v6_context_seed | high | 2 | 4 | 0.0425 | 0.0206 | 1.0000 | 5.0000 | 5.0000 | 43f77e,f95091 | 5046b4 | - | - | - | - | - | - | - | 0.757142 | 0.760607 | 0.741560 | local_ceiling_reached_hold_budget | Current best bounded combined candidate; local small-diff ceiling leader. [candidate_cycle=EXPERIMENTAL_NO_SUBMIT] |
| triad_f331_e0798_plus_dotted_5046 | 2026-03-13 | candidate | combined_small_diff_with_fallback_rider | 0343e02 | submission_v6_context_seed | high | 3 | 4 | 0.0425 | 0.0206 | - | - | - | 43f77e,5046b4,f95091 | - | - | - | - | - | - | - | - | 0.769172 | 0.769172 | - | - | Local exactness-max variant; resolves the full known incorrect scaffold tail but still not promoted. [candidate_cycle=EXPERIMENTAL_NO_SUBMIT] |
| v_embeddinggemma_fullcollection_iter14_invalid | 2026-03-13 | invalid | embedding_fullcollection_invalid_wiring | 3317825 | triad_f331_e0798_plus_dotted | n/a | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Initial full-collection embedding branch run invalidated by collection wiring mismatch; replaced by iter14c. |
| v_embeddinggemma_fullcollection_iter14c | 2026-03-13 | rejected | embedding_fullcollection_branch | 3317825 | triad_f331_e0798_plus_dotted | unknown | 14 | 50 | -0.0198 | -0.0218 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Full-collection embedding branch; bounded gate says NO_SUBMIT against current leader. [gate=NO_SUBMIT] |
| v_within_doc_rerank_surrogate_iter13 | 2026-03-13 | rejected | within_doc_rerank_surrogate | 0343e02 | triad_f331_e0798_plus_dotted | unknown | 0 | 4 | 0.0000 | -0.0100 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | Within-doc rerank/localization branch; best subsets are non-inferior only, not better. [gate=PROMISING] |
| v10_local_page_candidates_r1 | 2026-03-13 | rejected | page_candidate_generator_family_aware | a2ef637 | triad_f331_e0798_plus_dotted | low | 14 | 49 | 0.0029 | -0.0168 | - | - | - | 5046b4 | - | - | - | - | - | - | - | - | - | - | - | - | Ticket 23 family-aware page candidate generator over the current combined leader. Rejected: answer/page drift exploded, lineage confidence stayed low, all-case hidden-G regressed, and paranoid envelope fell well below the current ceiling. [gate=EXPERIMENTAL_NO_SUBMIT] |
