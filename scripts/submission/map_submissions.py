#!/usr/bin/env python3
"""Map all 15 platform submissions (v1-v15) to their local eval artifacts.

Cross-references submission_status files, preflight summaries, submission
files, raw_results, eval directories, and .sdd/researches closeout files
to produce a comprehensive mapping table.
"""

import json
import os
import glob
import hashlib
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
# Handle worktree: the actual data lives at the repo root
# Walk up until we find platform_runs/warmup
candidate = REPO
while candidate != candidate.parent:
    if (candidate / "platform_runs" / "warmup").exists():
        break
    candidate = candidate.parent
DATA_ROOT = candidate
WARMUP = DATA_ROOT / "platform_runs" / "warmup"
SDD = DATA_ROOT / ".sdd" / "researches"

# Known actual platform scores
PLATFORM_SCORES = {
    "v1":  {"det": 0.814, "asst": 0.607, "g": 0.638, "t": 0.996, "f": 1.041, "total": 0.497},
    "v2":  {"det": 0.943, "asst": 0.673, "g": 0.772, "t": 0.996, "f": 1.046, "total": 0.694},
    "v3":  {"det": 0.943, "asst": 0.707, "g": 0.766, "t": 0.996, "f": 1.046, "total": 0.696},
    "v4":  {"det": 0.943, "asst": 0.667, "g": 0.758, "t": 0.996, "f": 1.047, "total": 0.680},
    "v5":  {"det": 0.943, "asst": 0.667, "g": 0.801, "t": 0.996, "f": 1.047, "total": 0.718},
    "v6":  {"det": 0.971, "asst": 0.693, "g": 0.801, "t": 0.996, "f": 1.047, "total": 0.742},
    "v7":  {"det": 0.971, "asst": 0.647, "g": 0.608, "t": 0.996, "f": 1.047, "total": 0.554},
    "v8":  {"det": 0.971, "asst": 0.687, "g": 0.801, "t": 0.996, "f": 1.047, "total": 0.740},
    "v9":  {"det": 0.971, "asst": 0.700, "g": 0.654, "t": 0.994, "f": 1.047, "total": 0.606},
    "v10": {"det": 0.943, "asst": 0.693, "g": 0.772, "t": 0.995, "f": 1.047, "total": 0.698},
    "v11": {"det": 0.957, "asst": 0.680, "g": 0.689, "t": 0.995, "f": 1.043, "total": 0.625},
    "v12": {"det": 0.829, "asst": 0.667, "g": 0.613, "t": 0.993, "f": 1.034, "total": 0.491},
    "v13": {"det": 0.943, "asst": 0.620, "g": 0.650, "t": 0.993, "f": 1.029, "total": 0.562},
    "v14": {"det": 0.829, "asst": 0.673, "g": 0.613, "t": 0.993, "f": 1.034, "total": 0.492},
    "v15": {"det": 0.929, "asst": 0.607, "g": 0.593, "t": 0.996, "f": 1.045, "total": 0.514},
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def scan_submission_status_files():
    """Scan submission_status_*.json for version tags and metrics."""
    results = {}
    for path in sorted(WARMUP.glob("submission_status*.json")):
        data = load_json(path)
        if not data:
            continue
        version = data.get("version")
        if not version:
            continue
        metrics = data.get("metrics", {})
        entry = {
            "status_file": path.name,
            "uuid": data.get("uuid", ""),
            "version": version,
            "status": data.get("status", ""),
            "det": metrics.get("deterministic"),
            "asst": metrics.get("assistant"),
            "g": metrics.get("grounding"),
            "t": metrics.get("telemetry"),
            "f": metrics.get("ttft_multiplier"),
            "total": metrics.get("total_score"),
            "created_at": data.get("created_at", ""),
        }
        # If this version not seen or this is a better match, keep it
        if version not in results:
            results[version] = []
        results[version].append(entry)
    return results


def scan_preflight_summaries():
    """Scan all preflight_summary_*.json files for SHA hashes and metadata."""
    results = {}
    for path in sorted(WARMUP.glob("preflight_summary_*.json")):
        data = load_json(path)
        if not data:
            continue
        tag = path.name.replace("preflight_summary_", "").replace(".json", "")
        results[tag] = {
            "preflight_file": path.name,
            "submission_sha256": data.get("submission_sha256", ""),
            "code_archive_sha256": data.get("code_archive_sha256", ""),
            "questions_count": data.get("questions_count"),
            "qdrant_point_count": data.get("qdrant_point_count"),
            "raw_results_path": data.get("raw_results_path", ""),
            "null_answers": sum(data.get("null_answer_counts_by_type", {}).values()),
            "anomaly_count": len(data.get("anomaly_report", {}).get("anomaly_case_ids", [])),
        }
    return results


def scan_submission_files():
    """Scan all submission_*.json files (the actual answer payloads)."""
    results = {}
    for path in sorted(WARMUP.glob("submission_*.json")):
        bn = path.name
        # Skip status files and code_archive files
        if "submission_status" in bn or "code_archive" in bn:
            continue
        tag = bn.replace("submission_", "").replace(".json", "")
        data = load_json(path)
        if not data:
            continue
        entry = {"submission_file": bn}
        if isinstance(data, dict):
            entry["keys"] = list(data.keys())[:5]
            if "answers" in data:
                answers = data["answers"]
                entry["answer_count"] = len(answers) if isinstance(answers, list) else "N/A"
            if "architecture_summary" in data:
                entry["arch"] = str(data["architecture_summary"])[:80]
        elif isinstance(data, list):
            entry["answer_count"] = len(data)
        results[tag] = entry
    return results


def scan_raw_results():
    """Scan raw_results_*.json files."""
    results = {}
    for path in sorted(WARMUP.glob("raw_results_*.json")):
        tag = path.name.replace("raw_results_", "").replace(".json", "")
        data = load_json(path)
        if not data:
            continue
        entry = {"raw_results_file": path.name}
        if isinstance(data, list):
            entry["case_count"] = len(data)
        elif isinstance(data, dict):
            entry["keys"] = list(data.keys())[:5]
        results[tag] = entry
    return results


def scan_code_archive_audits():
    """Scan code_archive_audit_*.json files for SHA info."""
    results = {}
    for path in sorted(WARMUP.glob("code_archive_audit_*.json")):
        tag = path.name.replace("code_archive_audit_", "").replace(".json", "")
        data = load_json(path)
        if not data:
            continue
        results[tag] = {
            "audit_file": path.name,
            "archive_path": data.get("archive_path", ""),
            "archive_size": data.get("archive_size_bytes"),
            "files": len(data.get("files", [])),
            "issues": data.get("issues", []),
        }
    return results


def scan_eval_dirs():
    """Scan eval_*/ directories for comparison data."""
    results = {}
    for d in sorted(WARMUP.glob("eval_*")):
        if not d.is_dir():
            continue
        tag = d.name
        entry = {"eval_dir": tag, "files": [f.name for f in sorted(d.iterdir())]}
        # Try to load the compare JSON
        compare_files = list(d.glob("candidate_debug_compare_*.json"))
        if compare_files:
            cdata = load_json(compare_files[0])
            if cdata:
                entry["baseline_label"] = cdata.get("baseline_label", "")
                entry["candidate_label"] = cdata.get("candidate_label", "")
        # Try to load eval candidate debug for scores
        for ef in d.glob("eval_candidate_debug_*.json"):
            edata = load_json(ef)
            if edata and isinstance(edata, dict):
                label = edata.get("label", ef.stem)
                entry.setdefault("eval_labels", []).append(label)
        results[tag] = entry
    return results


def scan_closeout_files():
    """Scan .sdd/researches/*/closeout.md for version references."""
    results = {}
    if not SDD.exists():
        return results
    for path in sorted(SDD.glob("*/closeout.md")):
        try:
            text = path.read_text()
        except Exception:
            continue
        research_name = path.parent.name
        # Look for version references like v1, v2, ..., v15
        import re
        versions_found = set()
        for m in re.finditer(r'\bv(\d{1,2})\b', text):
            vn = int(m.group(1))
            if 1 <= vn <= 15:
                versions_found.add(f"v{vn}")
        # Also look for submission references
        submission_refs = set()
        for m in re.finditer(r'submission_(\w+)', text):
            submission_refs.add(m.group(1))
        if versions_found:
            for v in versions_found:
                results.setdefault(v, []).append({
                    "research": research_name,
                    "versions_mentioned": sorted(versions_found),
                })
    return results


def match_status_to_known_scores(status_entries):
    """Match submission_status entries to known platform scores by metrics."""
    matches = {}
    for version, entries in status_entries.items():
        for entry in entries:
            known = PLATFORM_SCORES.get(version)
            if known:
                # Check if metrics match
                det_match = entry["det"] is not None and abs(entry["det"] - known["det"]) < 0.002
                total_match = entry["total"] is not None and abs(entry["total"] - known["total"]) < 0.002
                if det_match and total_match:
                    matches[version] = entry
    return matches


def try_match_by_sha(preflights, code_audits):
    """Group local runs by code_archive_sha256 to find families."""
    sha_groups = {}
    for tag, pf in preflights.items():
        code_sha = pf.get("code_archive_sha256", "")
        if code_sha:
            sha_groups.setdefault(code_sha[:16], []).append(tag)
    return sha_groups


def try_match_by_scores():
    """Try to match platform v1-v15 to local runs by looking at preflight
    deterministic counts, answer patterns, etc. We use the submission_status
    files as the primary link since they have explicit version tags."""

    # Strategy: status files give us explicit version->uuid mappings.
    # We also know the platform scores. For versions NOT in status files,
    # we need to match by SHA or score patterns from raw results.
    pass


def compute_submission_sha256(path):
    """Compute SHA256 of a submission file to match against preflight."""
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return ""


def build_submission_sha_index(submissions, preflights):
    """Build index: compute SHA of each submission file, match to preflight."""
    sha_to_tag = {}
    for tag, pf in preflights.items():
        sub_sha = pf.get("submission_sha256", "")
        if sub_sha:
            sha_to_tag[sub_sha] = tag

    # Now compute SHA of each submission file
    matches = {}
    for tag, sub_info in submissions.items():
        path = WARMUP / f"submission_{tag}.json"
        if path.exists():
            sha = compute_submission_sha256(path)
            if sha in sha_to_tag:
                pf_tag = sha_to_tag[sha]
                matches[tag] = {"preflight_tag": pf_tag, "sha_match": True}
            # Also check if tag == preflight tag (same naming)
            if tag in preflights:
                if tag not in matches:
                    matches[tag] = {"preflight_tag": tag, "sha_match": False, "name_match": True}
    return matches


def main():
    print("=" * 120)
    print("PLATFORM SUBMISSION v1-v15 MAPPING TO LOCAL ARTIFACTS")
    print("=" * 120)
    print(f"\nData root: {DATA_ROOT}")
    print(f"Warmup dir: {WARMUP}")
    print(f"SDD dir: {SDD}")
    print()

    # 1. Scan all data sources
    print("--- Scanning data sources ---")
    status_entries = scan_submission_status_files()
    preflights = scan_preflight_summaries()
    submissions = scan_submission_files()
    raw_results = scan_raw_results()
    code_audits = scan_code_archive_audits()
    eval_dirs = scan_eval_dirs()
    closeout_refs = scan_closeout_files()

    print(f"  submission_status files: {sum(len(v) for v in status_entries.values())} entries across {len(status_entries)} versions")
    print(f"  preflight_summary files: {len(preflights)}")
    print(f"  submission files: {len(submissions)}")
    print(f"  raw_results files: {len(raw_results)}")
    print(f"  code_archive_audit files: {len(code_audits)}")
    print(f"  eval directories: {len(eval_dirs)}")
    print(f"  closeout version refs: {sum(len(v) for v in closeout_refs.values())} refs across {len(closeout_refs)} versions")

    # 2. Build SHA index
    print("\n--- Building SHA cross-reference index ---")
    sha_matches = build_submission_sha_index(submissions, preflights)
    sha_groups = try_match_by_sha(preflights, code_audits)
    print(f"  SHA-matched submission->preflight pairs: {sum(1 for m in sha_matches.values() if m.get('sha_match'))}")
    print(f"  Name-matched pairs: {sum(1 for m in sha_matches.values() if m.get('name_match'))}")
    print(f"  Unique code_archive SHAs: {len(sha_groups)}")

    # 3. Status file analysis - these are our primary source of truth
    print("\n--- Submission Status Files (explicit version tags) ---")
    status_matches = match_status_to_known_scores(status_entries)
    for version in sorted(status_entries.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99):
        entries = status_entries[version]
        known = PLATFORM_SCORES.get(version, {})
        for e in entries:
            score_match = "MATCH" if version in status_matches and status_matches[version] == e else "NO MATCH"
            print(f"  {e['status_file']}: version={version} total={e['total']:.4f} "
                  f"(known={known.get('total', 'N/A')}) [{score_match}] uuid={e['uuid'][:12]}")

    # 4. Now try to build the full v1-v15 mapping
    # Strategy:
    #   a) Use name patterns from local v-tagged files as primary candidates
    #   b) Cross-reference with submission_status files for confirmation
    #   c) For v1/v2, use hybrid series inference
    #   d) Link status files to their local artifact counterparts
    print("\n" + "=" * 120)
    print("BUILDING v1-v15 MAPPING")
    print("=" * 120)

    # Primary mapping: version -> local artifact tag
    mapping = {}

    # Build a lookup: for each local tag that starts with 'v' followed by digits
    import re as re_mod
    v_tagged_locals = {}
    for tag in sorted(set(list(preflights.keys()) + list(submissions.keys()) + list(raw_results.keys()))):
        m = re_mod.match(r'^v(\d+)_', tag)
        if m:
            vnum = int(m.group(1))
            if 1 <= vnum <= 15:
                v_tagged_locals.setdefault(f"v{vnum}", []).append(tag)
        m2 = re_mod.match(r'^v(\d+)$', tag)
        if m2:
            vnum = int(m2.group(1))
            if 1 <= vnum <= 15:
                v_tagged_locals.setdefault(f"v{vnum}", []).append(tag)

    print("\n--- Local tags with v-number prefixes ---")
    for v in sorted(v_tagged_locals.keys(), key=lambda x: int(x[1:])):
        tags = v_tagged_locals[v]
        print(f"  {v}: {tags}")

    # Step A: Map v3-v15 from name patterns first (these are the actual submitted artifacts)
    for version in sorted(PLATFORM_SCORES.keys(), key=lambda x: int(x[1:])):
        candidates = v_tagged_locals.get(version, [])
        if len(candidates) == 1:
            tag = candidates[0]
            mapping[version] = {
                "source": "name_pattern_unique",
                "tag": tag,
            }
        elif len(candidates) > 1:
            # Multiple candidates - check which has the most matching artifacts
            best = None
            for tag in candidates:
                has_pf = tag in preflights
                has_rr = tag in raw_results
                has_sub = tag in submissions
                score = (has_pf + has_rr + has_sub)
                if best is None or score > best[1]:
                    best = (tag, score)
            if best:
                mapping[version] = {
                    "source": "name_pattern_best",
                    "tag": best[0],
                    "candidates": candidates,
                }

    # Step B: For v1 and v2, use hybrid series inference
    if "v1" not in mapping:
        mapping["v1"] = {
            "source": "inference_hybrid_v1",
            "tag": "hybrid_v1",
            "note": "hybrid_v1 is the earliest local run; Det=0.814 is uniquely low, suggesting first submission",
        }

    if "v2" not in mapping:
        mapping["v2"] = {
            "source": "inference_hybrid_series",
            "tag": "hybrid_v1.1",
            "note": "v2 is second submission; hybrid_v1.1 is first iteration improvement",
        }

    # Step C: Enrich with submission_status data (uuid, created_at) where available
    # Map status version tags to mapping entries
    for version, entries in status_entries.items():
        if version in mapping:
            for entry in entries:
                known = PLATFORM_SCORES.get(version, {})
                if entry["total"] is not None and known and abs(entry["total"] - known["total"]) < 0.002:
                    mapping[version]["uuid"] = entry["uuid"]
                    mapping[version]["created_at"] = entry["created_at"]
                    mapping[version]["status_file"] = entry["status_file"]
                    mapping[version]["status_det"] = entry["det"]
                    mapping[version]["status_total"] = entry["total"]
                    break

    # Step D: For still-unmapped versions, mark as unmapped
    for version in sorted(PLATFORM_SCORES.keys(), key=lambda x: int(x[1:])):
        if version in mapping:
            continue
        mapping[version] = {
            "source": "UNMAPPED",
            "note": f"No direct match found for {version}",
        }

    # Step E: For versions mapped to v-tagged locals, also try to find
    # additional artifact files that correspond to the same submission
    # (e.g., ticket501_current -> v10, 1068_allowlist -> v13, etc.)
    # Map known status->artifact connections
    STATUS_TO_ARTIFACT_HINTS = {
        "v5": {"hint_tag": "v5_context_seed",
               "note": "status_v4_anchor_lineage.json reports v5; v5_context_seed is the local artifact"},
        "v10": {"hint_tag": "ticket501_current",
                "note": "status_ticket501_current reports v10; ticket501_current has matching artifacts"},
        "v13": {"hint_tag": "v13_doc_family_context",
                "note": "status 1068_allowlist reports v13; v13_doc_family_context is the local v13 artifact"},
        "v14": {"hint_tag": "v14_onora_fix",
                "note": "status v6_context_seed_resubmit reports v14; v14_onora_fix is the local v14 artifact"},
        "v15": {"hint_tag": "v15_rerank_booleans",
                "note": "status replay_1068_live_v15 reports v15; v15_rerank_booleans is the local v15 artifact"},
    }
    for version, hints in STATUS_TO_ARTIFACT_HINTS.items():
        if version in mapping:
            hint_tag = hints["hint_tag"]
            # Check if the hint tag has more artifacts than current
            current_tag = mapping[version].get("tag", "")
            hint_has_pf = hint_tag in preflights
            hint_has_rr = hint_tag in raw_results
            hint_has_sub = hint_tag in submissions
            cur_has_pf = current_tag in preflights
            cur_has_rr = current_tag in raw_results
            cur_has_sub = current_tag in submissions
            # If hint has more artifacts, add it as additional_tag
            if (hint_has_pf + hint_has_rr + hint_has_sub) > (cur_has_pf + cur_has_rr + cur_has_sub):
                mapping[version]["additional_tag"] = hint_tag
                mapping[version]["additional_note"] = hints["note"]
            elif hint_tag != current_tag:
                mapping[version]["additional_tag"] = hint_tag
                mapping[version]["additional_note"] = hints["note"]

    # 5. Enrich mapping with artifact paths
    print("\n--- Enriching mapping with artifact paths ---")
    for version in sorted(mapping.keys(), key=lambda x: int(x[1:])):
        m = mapping[version]
        tag = m.get("tag", "")
        additional_tag = m.get("additional_tag", "")

        # Check for matching artifacts on primary tag
        if tag:
            m["has_preflight"] = tag in preflights
            m["has_submission"] = tag in submissions
            m["has_raw_results"] = tag in raw_results
            m["has_code_audit"] = tag in code_audits

            if tag in preflights:
                pf = preflights[tag]
                m["submission_sha256"] = pf["submission_sha256"][:16]
                m["code_archive_sha256"] = pf["code_archive_sha256"][:16]
                m["qdrant_points"] = pf.get("qdrant_point_count")
                m["null_answers"] = pf.get("null_answers")
                m["anomaly_count"] = pf.get("anomaly_count")

            if tag in raw_results:
                m["case_count"] = raw_results[tag].get("case_count")

        # Also check additional_tag artifacts
        if additional_tag and additional_tag != tag:
            m["add_has_preflight"] = additional_tag in preflights
            m["add_has_submission"] = additional_tag in submissions
            m["add_has_raw_results"] = additional_tag in raw_results
            m["add_has_code_audit"] = additional_tag in code_audits

            if additional_tag in preflights and not m.get("submission_sha256"):
                pf = preflights[additional_tag]
                m["submission_sha256"] = pf["submission_sha256"][:16]
                m["code_archive_sha256"] = pf["code_archive_sha256"][:16]
                m["qdrant_points"] = pf.get("qdrant_point_count")
                m["null_answers"] = pf.get("null_answers")
                m["anomaly_count"] = pf.get("anomaly_count")

            if additional_tag in raw_results and not m.get("case_count"):
                m["case_count"] = raw_results[additional_tag].get("case_count")

        # Check eval dirs
        m["eval_dirs"] = []
        for edir, einfo in eval_dirs.items():
            if tag and (tag in edir or tag.replace("_", "") in edir):
                m["eval_dirs"].append(edir)
            elif additional_tag and (additional_tag in edir):
                m["eval_dirs"].append(edir)

        # Check closeout refs
        m["closeout_refs"] = closeout_refs.get(version, [])

    # 6. Print the final mapping table
    print("\n" + "=" * 120)
    print("FINAL MAPPING TABLE: Platform v1-v15 -> Local Artifacts")
    print("=" * 120)

    for version in sorted(mapping.keys(), key=lambda x: int(x[1:])):
        m = mapping[version]
        known = PLATFORM_SCORES[version]
        tag = m.get("tag", m.get("status_file", "???"))

        print(f"\n{'─' * 100}")
        print(f"  PLATFORM {version.upper()} | Total={known['total']:.3f} "
              f"(Det={known['det']:.3f} Asst={known['asst']:.3f} G={known['g']:.3f} "
              f"T={known['t']:.3f} F={known['f']:.3f})")
        print(f"  Source: {m['source']}")
        print(f"  Local tag: {tag}")

        if m.get("uuid"):
            print(f"  UUID: {m['uuid']}")
        if m.get("created_at"):
            print(f"  Created: {m['created_at']}")
        if m.get("status_file"):
            print(f"  Status file: {m['status_file']} (det={m.get('status_det')}, total={m.get('status_total')})")
        if m.get("note"):
            print(f"  Note: {m['note']}")
        if m.get("additional_tag"):
            print(f"  Additional tag: {m['additional_tag']}")
        if m.get("additional_note"):
            print(f"  Additional note: {m['additional_note']}")
        if m.get("candidates"):
            print(f"  Other candidates: {m['candidates']}")

        # Artifacts - primary tag
        has_pf = m.get("has_preflight", False)
        has_sub = m.get("has_submission", False)
        has_rr = m.get("has_raw_results", False)
        has_ca = m.get("has_code_audit", False)

        actual_tag = m.get("tag", "")
        additional_tag = m.get("additional_tag", "")
        print(f"  Artifacts (primary tag: {actual_tag or 'NONE'}):")
        if has_pf:
            print(f"    preflight_summary_{actual_tag}.json  [submission_sha={m.get('submission_sha256', 'N/A')}]")
        else:
            print(f"    preflight: NONE")
        if has_sub:
            print(f"    submission_{actual_tag}.json")
        else:
            print(f"    submission: NONE")
        if has_rr:
            print(f"    raw_results_{actual_tag}.json  [cases={m.get('case_count', 'N/A')}]")
        else:
            print(f"    raw_results: NONE")
        if has_ca:
            print(f"    code_archive_audit_{actual_tag}.json")
        else:
            print(f"    code_audit: NONE")

        # Additional tag artifacts
        if additional_tag and additional_tag != actual_tag:
            add_pf = m.get("add_has_preflight", False)
            add_sub = m.get("add_has_submission", False)
            add_rr = m.get("add_has_raw_results", False)
            add_ca = m.get("add_has_code_audit", False)
            if any([add_pf, add_sub, add_rr, add_ca]):
                print(f"  Additional artifacts (tag: {additional_tag}):")
                if add_pf:
                    print(f"    preflight_summary_{additional_tag}.json")
                if add_sub:
                    print(f"    submission_{additional_tag}.json")
                if add_rr:
                    print(f"    raw_results_{additional_tag}.json")
                if add_ca:
                    print(f"    code_archive_audit_{additional_tag}.json")

        if m.get("qdrant_points"):
            print(f"  Qdrant points: {m['qdrant_points']}")
        if m.get("null_answers") is not None:
            print(f"  Null answers: {m['null_answers']}")
        if m.get("anomaly_count") is not None:
            print(f"  Anomalies: {m['anomaly_count']}")

        if m.get("eval_dirs"):
            print(f"  Eval dirs: {m['eval_dirs']}")
        if m.get("closeout_refs"):
            research_names = [r["research"] for r in m["closeout_refs"][:5]]
            print(f"  Closeout refs ({len(m['closeout_refs'])} total): {research_names}")

    # 7. SHA family analysis
    print("\n" + "=" * 120)
    print("CODE ARCHIVE SHA256 FAMILIES (runs sharing same code)")
    print("=" * 120)
    for sha, tags in sorted(sha_groups.items(), key=lambda x: -len(x[1])):
        if len(tags) > 1:
            # Check if any of these tags are in our mapping
            mapped_versions = []
            for v, m in mapping.items():
                if m.get("tag") in tags:
                    mapped_versions.append(v)
            print(f"\n  SHA {sha}... ({len(tags)} runs) {f'-> {mapped_versions}' if mapped_versions else ''}")
            for t in tags[:10]:
                print(f"    {t}")
            if len(tags) > 10:
                print(f"    ... and {len(tags) - 10} more")

    # 8. Local eval scores extraction
    print("\n" + "=" * 120)
    print("LOCAL EVAL SCORES (from eval_*/ directories)")
    print("=" * 120)
    for edir, einfo in sorted(eval_dirs.items()):
        print(f"\n  {edir}/")
        if einfo.get("baseline_label"):
            print(f"    Baseline: {einfo['baseline_label']}")
        if einfo.get("candidate_label"):
            print(f"    Candidate: {einfo['candidate_label']}")
        if einfo.get("eval_labels"):
            print(f"    Eval labels: {einfo['eval_labels']}")

        # Try to load judge scores
        for jf in (WARMUP / edir).glob("judge_candidate_debug_*.jsonl"):
            scores = []
            try:
                with open(jf) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            s = rec.get("score", rec.get("judge_score"))
                            if s is not None:
                                scores.append(s)
                        except:
                            pass
                if scores:
                    avg = sum(scores) / len(scores)
                    print(f"    {jf.name}: {len(scores)} scores, avg={avg:.3f}, "
                          f"min={min(scores):.3f}, max={max(scores):.3f}")
            except:
                pass

    # 9. Summary of unmapped versions
    print("\n" + "=" * 120)
    print("MAPPING CONFIDENCE SUMMARY")
    print("=" * 120)
    print(f"\n  {'Ver':<5} {'Total':<7} {'Primary Tag':<40} {'Additional Tag':<35} {'Source':<25} {'Artifacts'}")
    print(f"  {'─'*4:<5} {'─'*6:<7} {'─'*39:<40} {'─'*34:<35} {'─'*24:<25} {'─'*25}")
    for version in sorted(mapping.keys(), key=lambda x: int(x[1:])):
        m = mapping[version]
        known = PLATFORM_SCORES[version]
        tag = m.get("tag", "???")
        additional_tag = m.get("additional_tag", "")
        source = m["source"]
        artifacts = []
        if m.get("has_preflight") or m.get("add_has_preflight"):
            artifacts.append("PF")
        if m.get("has_submission") or m.get("add_has_submission"):
            artifacts.append("SUB")
        if m.get("has_raw_results") or m.get("add_has_raw_results"):
            artifacts.append("RR")
        if m.get("has_code_audit") or m.get("add_has_code_audit"):
            artifacts.append("CA")
        if m.get("eval_dirs"):
            artifacts.append(f"EVAL({len(m['eval_dirs'])})")
        art_str = ",".join(artifacts) if artifacts else "NONE"
        print(f"  {version:<5} {known['total']:<7.3f} {tag:<40} {additional_tag:<35} {source:<25} {art_str}")

    # 10. Attempt deeper matching for v1/v2 by computing actual submission file SHAs
    print("\n" + "=" * 120)
    print("SHA CROSS-REFERENCE: submission file SHA vs preflight submission_sha256")
    print("=" * 120)
    print("\n  Checking if submission file SHA matches its preflight counterpart...")
    verified = 0
    mismatched = 0
    for tag in sorted(set(submissions.keys()) & set(preflights.keys())):
        sub_path = WARMUP / f"submission_{tag}.json"
        if sub_path.exists():
            actual_sha = compute_submission_sha256(sub_path)
            expected_sha = preflights[tag]["submission_sha256"]
            if actual_sha == expected_sha:
                verified += 1
            else:
                mismatched += 1
                print(f"    MISMATCH: {tag} actual={actual_sha[:16]} expected={expected_sha[:16]}")
    print(f"\n  Verified: {verified}, Mismatched: {mismatched}")


if __name__ == "__main__":
    main()
