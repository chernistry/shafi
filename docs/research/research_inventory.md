# R008 -- Research Directory Inventory

Total: 807 entries in `.sdd/researches/`, organized into numbered research tickets (1002-1068), numbered tickets (00-203, 339, 600-649), domain hypotheses (H1-H19), candidate/experiment runs, and special directories.

## Special Directories

| Directory | Contents |
|-----------|----------|
| `_duplicates/` | 2 duplicate analysis MDs (public archive patterns) |
| `_external_inputs/` | Leaderboard snapshots subfolder |
| `_scratch/` | 1 temporary CSV |

---

## Research Series 1002-1068: Closed-World Compiler Program

All dated 2026-03-19. This was the major research closeout day.

| ID | Directory | Verdict | Summary |
|----|-----------|---------|---------|
| 1002 | reviewed_resurrection_promotion_doctrine_r1 | CLOSED | Triad resurrection candidate frozen for future replay-backed promotion |
| 1003 | instruction_conditioned_zerank_r1 | CLOSED | Deterministic instruction families for conditional reranking |
| 1003-1006 | authority_lane_replay | CLOSED | Combined authority lane replay -- regressed on reviewed grounding |
| 1004 | authoritative_page_priors_r1 | CLOSED | Deterministic authority helper for sidecar activation and page choice |
| 1005 | law_bundle_scope_graph_r1 | CLOSED | Deterministic law-family bundle helper for sidecar scope matching |
| 1006 | bounded_page_pair_contract_r1 | CLOSED | Bounded semantic pair activation and one-page/justified-pair selection |
| 1007 | bounded_visual_page_rerank_feasibility_r1 | NOT ACTIVATED | Conditional on 1003-1006 success -- they failed |
| 1008 | authoritative_evidence_layer_branch_gate_r1 | NOT ACTIVATED | Depended on 1007 which depended on failed 1003-1006 |
| 1010 | evidence_portfolio_selector_r1 | CLOSED | Deterministic portfolio construction in grounding (part of 1010-1013 lane) |
| 1010-1013 | evidence_lane | CLOSED | Combined evidence-set lane -- failed promotion gate against main |
| 1011 | counterfactual_necessity_pruner_r1 | CLOSED | Leave-one-out necessity pruning after portfolio selection |
| 1012 | compare_family_typed_panel_extractor_r1 | CLOSED | Typed compare-family panel extraction, one authoritative page per side |
| 1013 | typed_condition_audit_and_fallback_r1 | CLOSED | Typed condition audit before final grounding commit, fallback to safe sidecar |
| 1014 | segment_spanning_article_schedule_shadow_r1 | NOT ACTIVATED | 1010-1013 failed gate, no justification for deeper expansion |
| 1015 | pairwise_within_doc_authoritativeness_r1 | NOT ACTIVATED | Evidence-set lane failed outright against main |
| 1016 | retrieve_verify_retrieve_complement_r1 | NOT ACTIVATED | No evidence for complement retrieval after 1010-1013 loss |
| 1017 | visual_authority_micro_rerank_r1 | NOT ACTIVATED | Required non-negative earlier lane plus visual authority family matches |
| 1018 | synthetic_counter_question_sidecar_index_r1 | NOT ACTIVATED | Earlier lane did not earn promotion |
| 1019 | bounded_document_axis_micro_agents_r1 | NOT ACTIVATED | Last-resort research ticket, sequence terminated |
| 1020 | paper_source_closeout_r1 | CLOSED | Accepted: bounded fact/page relevance check and context-aware query rewrite from LawThinker. Rejected: multi-agent orchestration, memory blocks, Chinese-law code |
| 1021 | introspection_escalation_gate_r1 | IMPLEMENTED | Deterministic escalation logic in evidence selector |
| 1022 | context_aware_shadow_query_rewrite_r1 | IMPLEMENTED | Deterministic shadow rewrite builder for retrieval |
| 1023 | bounded_page_fact_relevance_verifier_r1 | IMPLEMENTED | Bounded verifier logic with prompts |
| 1024 | grounding_selector_integration_r1 | IMPLEMENTED | Integrated LawThinker/LRAS lane into evidence selector |
| 1025 | conflict_resolution | -- | Conflict resolution notes (no closeout MD) |
| 1025 | evaluation_and_promotion_gate_r1 | GATE | Ran ruff/pyright/pytest and platform evaluation commands |
| 1026 | triad_f331_e0798_rehydration_r2 | CLOSED | Scored triad replay artifact against reviewed golden |
| 1027 | t20_docfamily_collapse_control_r2 | CLOSED | Scored docfamily collapse surrogate against reviewed golden |
| 1041 | closed_world_failure_cartography_r1 | IMPLEMENTED | Failure cartography module: rules, models, build script |
| 1042 | corpus_compiler_skeleton_r1 | PROMOTE | Offline corpus compiler converting parsed legal docs to typed objects + corpus registry |
| 1043 | canonical_entity_alias_resolver_r1 | PROMOTE | Canonical entity alias resolution for law titles, case numbers, parties, judges |
| 1044 | legal_segment_compiler_r2 | NO_GO | Typed legal segments did not improve retrieval in structural ablation |
| 1044-1046 | activation | ABLATION | Structural retrieval ablation: chunks, shadow_chunks, pages, segments tested |
| 1045 | amendment_temporal_applicability_graph_r1 | CLOSED | Offline applicability graph for amendment/commencement/supersession edges |
| 1046 | bridge_fact_registry_r1 | NO_GO | Bridge facts did not flatten cross-doc reasoning as hoped |
| 1047 | multimodal_regionizer_r1 | NOT ACTIVATED | Hard-page detector, visual region extractor, region enricher. Ingestion-only |
| 1048 | synthetic_question_and_counterfactual_factory_r1 | CLOSED | Deterministic synthetic QA factory over compiled registry |
| 1049 | teacher_labels_and_hard_negative_miner_r1 | CLOSED | Hard-negative mining and teacher-label building for grounding |
| 1050 | corpus_tuned_dense_retriever_r1 | CLOSED | Offline corpus-tuned dense retriever training lane |
| 1051 | corpus_tuned_reranker_and_set_selector_r1 | CLOSED | Pairwise reranker and compact set-selector training helpers |
| 1052 | retrieval_utility_predictor_r2 | CLOSED | Bundle-sufficiency predictor, offline-capable, benchmarked on public reviewed |
| 1053 | query_contract_compiler_r2 | A/B ONLY | Deterministic query-contract compiler as pipeline node after classify |
| 1054 | database_first_exact_field_answerer_r1 | NO_GO | Structured data answerer bypassing retrieval -- not safe enough |
| 1055 | compare_join_engine_r1 | NO_GO | Compare families are join problems but engine not reliable enough |
| 1056 | temporal_applicability_engine_r1 | CLOSED | Structured temporal/applicability runtime engine from graph |
| 1057 | claim_to_span_grounding_graph_r2 | CLOSED | Narrow claim-graph layer wired into pipeline |
| 1058 | proof_carrying_answer_compiler_r2 | CLOSED | Proof-carrying answer compiler wired into pipeline |
| 1059 | committee_mbr_answer_selection_r1 | NO_GO | Committee agreement precondition not met -- not enough independent routes |
| 1060 | external_segment_payload_shadow_benchmark_r1 | CLOSED | External structured segment payload as offline shadow benchmark |
| 1061 | rich_segment_embedding_text_composer_r1 | ABLATION | External segment composer ablation -- title/caption/claimant families tested |
| 1062 | adversarial_near_miss_grounding_corpus_r1 | CLOSED | Offline adversarial grounding corpus with lineage-safe hard negatives |
| 1063 | closed_world_compiler_program_umbrella_r1 | NO_GO | Umbrella closeout: no pre-private champion replacement earned |
| 1064 | compiled_memory_kernel_r1 | A/B ONLY | Sparse memory entries from validated outcomes for route hints |
| 1065 | selective_icr_local_rerank_and_provider_exit_r1 | NO_GO | Shadow/eval lane built but loses legal accuracy on reviewed slices |
| 1066 | triad_submit_calibration_r1 | CLOSED | Calibrated replay candidate assembly for warmup phase |
| 1067 | kanon2_1792_dimension_ablation_r1 | PROMOTE_TO_CALIBRATION | 1792 dimensions materially improved reviewed grounding |
| 1068 | allowlist_tiny_page_localizer | CLOSED | Ticket 641 answer-stable grounding replay harness |
| 1068 | allowlist_tiny_page_localizer_overlay | CLOSED | Overlay variant of allowlist page localizer |
| 1068 | kanon2_1792_submit_calibration_r1 | PROMOTE_FINAL | 1792 artifact audit-safe, current best candidate path |

---

## Ticket Series 600-649: Grounding and ML Training Pipeline

All dated 2026-03-18 to 2026-03-19.

| ID | Directory | Summary |
|----|-----------|---------|
| 601 | gap_audit | Gap audit matrix for grounding pipeline |
| 602 | grounding_ml_testing_contract_refresh | ML testing contract for grounding subsystem |
| 604 | sidecar_scope_expansion_single_doc_r1 | Expand sidecar to single-doc queries |
| 605 | page_role_payload_audit_and_repair_r1 | Audit and repair page-role payload coverage |
| 606 | grounding_page_retrieval_rrf_audit_r1 | Reciprocal rank fusion audit for page retrieval |
| 607 | full_case_and_compare_scope_hardening_r1 | Harden grounding for full-case and compare scopes |
| 608 | negative_unanswerable_empty_grounding_gate_r1 | Empty grounding gate for unanswerable/negative queries |
| 609 | freeze_sidecar_ml_dataset_export_v1 | Freeze sidecar ML dataset export |
| 610 | ml_training_scaffold_bootstrap_r1 | Bootstrap ML training scaffold |
| 611 | obliqa_download | ObliQA raw dataset download and inventory |
| 612 | cuad_download | CUAD raw dataset download and inventory |
| 613 | contractnli_download | ContractNLI raw dataset download and inventory |
| 614 | ledgar_download | LEDGAR raw dataset download and inventory |
| 615 | normalize_external_legal_datasets_for_grounding_v1 | Normalize all 4 external datasets for grounding training |
| 616 | train_grounding_router_v1 | Train grounding router model |
| 617 | train_page_scorer_v1 | Train page scorer (LightGBM) v1 |
| 618 | trained_vs_heuristic_sidecar_ablation_r1 | A/B ablation: trained sidecar vs heuristic |
| 619 | import_reviewed_labels_v3_and_benchmark_slices | Import reviewed grounding labels and benchmark slices |
| 620 | public100_reviewed_gate_r1 | Reviewed gate on Public100 slice |
| 621 | rebuild_grounding_ml_export_v2_reviewed | Rebuild ML export with reviewed labels |
| 622 | reviewed_aware_router_and_page_scorer_retune_r1 | Retune router and scorer with reviewed labels |
| 623 | reviewed_heuristic_vs_trained_ablation_r1 | Reviewed heuristic vs trained ablation |
| 624 | reviewed_heuristic_grounding_repair_r1 | Repair heuristic grounding using reviewed labels |
| 625 | runtime_page_scorer_sidecar_challenger_r1 | Runtime page scorer as sidecar challenger |
| 626 | reviewed_public100_runtime_trained_page_gate_r1 | Runtime trained page gate on Public100 |
| 627 | runtime_safe_page_scorer_feature_hygiene_r1 | Feature hygiene for runtime-safe page scorer |
| 628 | runtime_trained_page_scorer_subset_rerank_r1 | Trained page scorer for subset reranking |
| 629 | reviewed_public100_runtime_trained_page_regate_r1 | Re-gate trained page scorer on Public100 |
| 630 | reviewed_single_doc_statute_final_page_selector_r1 | Same-day baseline for single-doc statute page selection |
| 631 | reviewed_sidecar_candidate_repair_r1 | Sidecar candidate repair against reviewed labels |
| 632 | reviewed_deterministic_page_role_heading_boost_r1 | Deterministic page-role heading boost |
| 633 | tiny_within_doc_page_reranker_r1 | Not started under deadline |
| 634 | clean_public_calibration_challenger_r1 | No eligible public calibration submit found |
| 635 | devops_env_lineage_and_compose_network_truth_r1 | DevOps env lineage and compose network truth |
| 636 | container_ingest_contract_and_smoke_gate_r1 | Container ingest contract and smoke gate |
| 637 | local_preflight_and_ops_self_check_r1 | Local preflight and ops self-check |
| 638 | ops_surface_and_repo_clarity_r1 | Ops surface and repo clarity |
| 639 | grounding_resume_after_devops_baseline_r1 | Grounding resume after devops baseline stop |
| 640 | calibrated_resurrection_matrix_r1 | Calibrated resurrection matrix |
| 640 | main_baseline_freeze | Main baseline freeze snapshot |
| 641 | answer_stable_grounding_replay_harness_r1 | Answer-stable grounding replay harness |
| 642 | page_toplines_headings_and_field_labels_at_ingest_r1 | Page toplines, headings, and field labels at ingest |
| 643 | document_template_and_officialness_catalogizer_r1 | Document template and officialness catalogizer |
| 644 | heading_only_and_field_only_shadow_retrieval_lane_r1 | Shadow retrieval lane for heading-only and field-only |
| 645 | law_title_alias_and_amendment_link_graph_r1 | Law title alias and amendment link graph |
| 646 | offline_llm_document_interrogation_enrichment_r1 | Offline LLM document interrogation for enrichment |
| 647 | duplicate_reference_page_suppressor_r1 | Suppress duplicate reference pages |
| 648 | page_pair_and_neighborhood_selector_r1 | Page pair and neighborhood selector |
| 649 | real_workload_embedder_and_representation_bakeoff_r1 | Embedder and representation bakeoff on real workload |

---

## Domain Hypotheses (H1-H19)

All dated 2026-03-19.

| ID | Directory | Summary |
|----|-----------|---------|
| H1 | domain_h1_law_title_family | Law title family normalization and grouping |
| H2 | domain_h2_enactment_commencement | Enactment and commencement date extraction |
| H3 | domain_h3_case_number_normalization | Case number normalization rules |
| H4 | domain_h4_case_party_caption | Case party caption extraction |
| H5 | domain_h5_issuing_authority | Issuing authority identification |
| H6 | domain_h6_article_schedule_retrieval | Article and schedule targeted retrieval |
| H7 | domain_h7_rich_segment_text | Rich segment text composition for embeddings |
| H8 | domain_h8_jurisdiction_rewrite | Jurisdiction-aware query rewrite and filtering |
| H9 | domain_h9_strict_formatting_uae_titles_case_refs | Strict formatting for UAE titles and case references |
| H10 | domain_h10_page_semantics_panels | Page semantics panels for layout understanding |
| H11 | domain_h11_document_interrogation | LLM-based document interrogation for enrichment |
| H13 | domain_h13_commencement_and_enactment_dates | Commencement and enactment date closeout |
| H14 | domain_h14_title_alias_and_abbreviation_family | Title alias and abbreviation family resolution |
| H15 | domain_h15_uae_legal_field_intent_rules | UAE/DIFC legal-field intent rules |
| H16 | domain_h16_page_region_caption_and_signature | Page region caption and signature extraction |
| H17 | domain_h17_amendment_repeal_replace_relationships | Amendment, repeal, and replace relationships |
| H18 | domain_h18_case_party_noise_cleanup | Case party noise cleanup |
| H19 | domain_h19_case_judge_and_panel_normalization | Case judge and panel normalization |

---

## Ticket Series 00-203: Warmup-Phase Competition Tickets

All dated 2026-03-13 to 2026-03-14. 203 tickets covering the full warmup research sprint.

| ID Range | Theme | Count | Summary |
|----------|-------|-------|---------|
| 00-10 | Page trace and fingerprint | 11 | Page trace ledger, hidden-G blindspots, production mimic, run manifest, candidate fingerprint dedup, impact router, cheap-to-expensive racing, judge cache, citation hard floor, gap-to-rank branch freeze, champion lineage |
| 11-20 | Grounding candidates | 10 | Bounded miss pack, Cohere rerank falsifier, explicit page reference audit, page forcing impl, caselaw page1 baseline, caselaw bounded impl, support precision, support shape gate v2, doc-family collapse |
| 21-33 | Local page rerankers | 13 | Page rerank core (phase 1+2), scorer selection, benchmark, integration A/B, within-doc defensive fallback, Det exactness audit, local ColBERT sidecar, Grok 4.20 beta, exact legal reference boosting, bounded page candidate v2, embeddinggemma, BGE-M3, Qwen3 pair |
| 34-45 | Infrastructure and probes | 12 | Answer-to-page attribution, narrow legal query expansion, summary-augmented chunks, Jina late chunking, legislation article boundary, concurrency safety audit, stress-test ingestion, session pool, parallel eval, telemetry dashboard, local embedding probe, Grok single-agent |
| 46-65 | Readiness and audits | 20 | Final readiness, BM25 probe, exact legal reference probe, leaderboard truth, stale work prune, single-doc miss pack, rerank gate, strict null audit, metadata rescue, weird tail source audit, no-result guardrails, synthetic OCR audit, page-preserving docling fallback, private run rehearsal, Cohere rerank falsifier |
| 66-85 | Telemetry and external data | 20 | Final readiness review, exact provenance telemetry, support snippet telemetry, structured doc stress pack, structured retrieval gap audit, external framework triage, same-doc page selector, structure-aware retrieval patch, manifest backfill, parser truth reconciliation, LegalBench-RAG, exact-ref fail-open, public legal synthetic pack, named-ref audit, weird tail, strict null reset, open RAG bench, public-only rerank |
| 86-97 | Warmup decision | 12 | Freeze warmup offense, private-only queue reorder, production mimic reporting, trusted page trace, OCR screenshot stress, board mimic, board best semantics, dry-run archive, warmup decision one-pager, final warmup decision, warmup outcome interpretation |
| 100-203 | Advanced candidates | ~70 | LegalBench exact-ref subset, exact-ref fail-open analysis, same-doc selector refresh/falsifier/gate, public structured pack expansion, CourtListener pack, named-ref lanes, unsupported pack, strict null telemetry, OCR stress wave, model route regression, hosted model route A/B, private doctor preflight, artifact freezes, scanner manifests, docling competitor triage, warmup slot semantics, calibration submit verification, human warmup decision, QID diff, model route decomposition, metadata page family, shadow extractive free-text, structured pack metadata, citation title law number, metadata slot completeness, frontier judge error taxonomy, TOC metadata, hard private artifact order, tiny local model fallback, boolean admin guard, residual boolean, anchor sensitive title audit, case caption residual, helper support overbreadth, minimal support boolean hardening, A/B/C readiness, artifact reconciliation, private-day doctrine dry run, ABC changed-case source packet, final private-day runbook |

---

## Candidate and Experiment Runs

| Directory Pattern | Count | Summary |
|-------------------|-------|---------|
| `candidate_ceiling_cycle_*` | 4 | Ceiling cycle estimation with different strategies |
| `candidate_debug_*` | 7 | Debug signals for specific candidate configurations |
| `candidate_family_compare_*` | 4 | Family comparison and frontier analysis |
| `comparison_targeted_judge_*` | 1 | Targeted judge comparison for comparison family |
| `comparison_title_cycle_*` | 1 | Comparison title-page candidate audit |
| `embedding_*` | 3 | Embedding doc-family and full-collection experiments |
| `embedding_support_opportunities_*` | 1 | Mining embedding support opportunities |
| `exactness_*` | 5 | Exactness candidate audits and rider subset searches |
| `explicit_anchor_gap_audit_*` | 1 | Explicit anchor gap audit |
| `explicit_profile_baseline_*` | 1 | Explicit profile baseline for comparison |
| `hidden_g_blindspots_*` | 1 | Hidden-G benchmark blindspot audit |
| `kanon2_1792_*` | 2 | Kanon-2 1792-dimension domain clean manual variants |
| `matrix_fill_*` | 3 | Matrix fill experiments (embedding, exactness, within-doc) |
| `page_localizer_*` | 1 | Page localizer anchor filter evaluation |
| `page_scorer_627_*` | 3 | Page scorer 627 on main baseline evaluations |
| `party_title_combo_search_*` | 2 | Party/title combo candidate searches |
| `pipeline_quality_hardening_*` | 1 | Pipeline quality hardening evaluation |
| `portfolio_combo_search_*` | 1 | Portfolio combo candidate search |
| `production_mimic_*` | 7 | Production-mimic local evaluations with various configs |
| `projection_gap_riders_*` | 1 | Projection gap rider opportunity mining |
| `projectiongap_*` | 2 | Projection gap family and new-family debug |
| `proof_path_audit_*` | 1 | Proof path audit for answer lineage |
| `reranker_audit_*` | 1 | Reranker audit: shadow ICR truncation/tokenizer |
| `retrieval_first_shadow_anchor_*` | 1 | Anchor-sensitive canary pack for retrieval-first approach |
| `scan_single_support_swaps_*` | 1 | Single support swap scanning |
| `search_portfolio_*` | 1 | Portfolio frontier search |
| `single_swap_scan_*` | 5 | Single-swap scans with various configs (party title, explicit page, blanket) |
| `support_combo_search_*` | 5 | Support combo searches (comparison, explicit, core, judge wins) |
| `tiny_donor_probe_*` | 2 | Tiny donor probe for page localizer subsets |
| `v14_v6_context_seed_resubmit_*` | 1 | V14 v6 context seed resubmit experiment |
| `v6_exactness_reaudit_*` | 1 | V6 exactness re-audit (standalone MD) |
| `v6_restore_public_gold_eval_*` | 1 | Restore public gold evaluation for v6 |
| `within_doc_rerank_*` | 3 | Within-doc rerank opportunities and subset search |
| `anchor_page_hardening_*` | 2 | Anchor page hardening evaluations |
| `clean_env_baseline_*` | 1 | Clean environment baseline scoring report |
| `doc_resolver_*` | 1 | Document resolver support units evaluation |
| `docref_dense_rerank_*` | 1 | Docref dense rerank localization evaluation |
| `question_families` | 1 | Question family taxonomy (no MD) |
| `2026-03-13-hypothesis-coverage-matrix.md` | 1 | Standalone hypothesis coverage matrix |
| `43f77_caption_equivalence_*` | 1 | Caption equivalence analysis |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| 1002-1068 series (compiler program) | 61 directories |
| 600-649 series (grounding/ML) | 50 directories |
| Domain hypotheses (H1-H19) | 18 directories |
| Ticket series (00-203) | ~200 directories |
| Candidate/experiment runs | ~70 directories |
| Special directories | 3 |
| Standalone files in researches/ | 2 |
| **Total entries** | **~807** |

### Verdict Distribution (1002-1068 series)
| Verdict | Count |
|---------|-------|
| CLOSED (not activated by gate) | 25 |
| IMPLEMENTED / PROMOTE | 8 |
| NO_GO | 8 |
| A/B ONLY | 2 |
| ABLATION | 2 |
| Other | 4 |
