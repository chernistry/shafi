"""Build a per-QID drift ledger comparing champion control, v6 seed, and v10 current.

Usage:
    python scripts/build_champion_drift_ledger.py --out-json .sdd/researches/drift_ledger_2026-03-16.json \
        --out-md .sdd/researches/drift_ledger_2026-03-16.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]

CONTROL_SUBMISSION = ROOT / ".sdd/researches/ticket121_freeze_private_artifact_2026-03-14/submission.json"
CONTROL_RAW = ROOT / ".sdd/researches/ticket121_freeze_private_artifact_2026-03-14/raw_results.json"

V6_SUBMISSION = ROOT / "platform_runs/warmup/submission_v6_context_seed.json"
V6_RAW = ROOT / "platform_runs/warmup/raw_results_v6_context_seed.json"

V10_SUBMISSION = ROOT / "platform_runs/warmup/submission_ticket501_current.json"
V10_RAW = ROOT / "platform_runs/warmup/raw_results_ticket501_current.json"

V10_PREFLIGHT = ROOT / "platform_runs/warmup/preflight_summary_ticket501_current.json"


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _build_answer_map(submission: dict) -> dict[str, str | None]:
    return {
        a["question_id"]: a.get("answer")
        for a in submission.get("answers", [])
    }


def _build_telemetry_map(raw_results: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in raw_results:
        tel = r.get("telemetry", {})
        qid = tel.get("question_id") or r.get("case", {}).get("case_id", "")
        out[qid] = tel
    return out


def _build_case_map(raw_results: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in raw_results:
        case = r.get("case", {})
        cid = case.get("case_id", "")
        out[cid] = case
    return out


def _pages_str(pages: list | None) -> str:
    if not pages:
        return ""
    return ",".join(str(p) for p in sorted(pages))


def _short(text: str | None, limit: int = 120) -> str:
    if text is None:
        return "<null>"
    t = text.replace("\n", " ").strip()
    if len(t) > limit:
        return t[:limit] + "…"
    return t


def build_ledger(
    *,
    control_sub: dict,
    control_raw: list[dict],
    v6_sub: dict,
    v6_raw: list[dict],
    v10_sub: dict,
    v10_raw: list[dict],
    v10_preflight: dict,
) -> list[dict]:
    ctrl_answers = _build_answer_map(control_sub)
    v6_answers = _build_answer_map(v6_sub)
    v10_answers = _build_answer_map(v10_sub)

    ctrl_tel = _build_telemetry_map(control_raw)
    v6_tel = _build_telemetry_map(v6_raw)
    v10_tel = _build_telemetry_map(v10_raw)

    ctrl_cases = _build_case_map(control_raw)
    v10_cases = _build_case_map(v10_raw)
    v6_cases = _build_case_map(v6_raw)

    anomaly_flags = {}
    anomaly_report = v10_preflight.get("anomaly_report", {})
    for cid, flags in (anomaly_report.get("anomaly_flags_by_case") or {}).items():
        anomaly_flags[cid] = flags

    support_flags = {}
    support_report = v10_preflight.get("support_shape_report", {})
    for cid, flags in (support_report.get("flags_by_case") or {}).items():
        support_flags[cid] = flags

    all_qids = sorted(set(ctrl_answers) | set(v6_answers) | set(v10_answers))

    ledger: list[dict] = []
    for qid in all_qids:
        ca = ctrl_answers.get(qid)
        v6a = v6_answers.get(qid)
        v10a = v10_answers.get(qid)

        ct = ctrl_tel.get(qid, {})
        v6t = v6_tel.get(qid, {})
        v10t = v10_tel.get(qid, {})

        case_info = ctrl_cases.get(qid) or v10_cases.get(qid) or v6_cases.get(qid) or {}

        answer_changed_ctrl_v10 = (ca != v10a)
        answer_changed_ctrl_v6 = (ca != v6a)
        answer_changed_v6_v10 = (v6a != v10a)

        ctrl_used = ct.get("used_page_ids") or []
        v6_used = v6t.get("used_page_ids") or []
        v10_used = v10t.get("used_page_ids") or []
        pages_changed_ctrl_v10 = set(map(str, ctrl_used)) != set(map(str, v10_used))
        pages_changed_ctrl_v6 = set(map(str, ctrl_used)) != set(map(str, v6_used))

        ctrl_docs = ct.get("doc_refs") or []
        v6_docs = v6t.get("doc_refs") or []
        v10_docs = v10t.get("doc_refs") or []

        row = {
            "qid": qid,
            "question": case_info.get("question", ""),
            "answer_type": case_info.get("answer_type") or ct.get("answer_type") or v10t.get("answer_type", ""),
            "ctrl_answer": ca,
            "v6_answer": v6a,
            "v10_answer": v10a,
            "answer_changed_ctrl_v10": answer_changed_ctrl_v10,
            "answer_changed_ctrl_v6": answer_changed_ctrl_v6,
            "answer_changed_v6_v10": answer_changed_v6_v10,
            "ctrl_model": ct.get("model_llm", ""),
            "v6_model": v6t.get("model_llm", ""),
            "v10_model": v10t.get("model_llm", ""),
            "ctrl_gen_mode": ct.get("generation_mode", ""),
            "v6_gen_mode": v6t.get("generation_mode", ""),
            "v10_gen_mode": v10t.get("generation_mode", ""),
            "ctrl_used_pages": _pages_str(ctrl_used),
            "v6_used_pages": _pages_str(v6_used),
            "v10_used_pages": _pages_str(v10_used),
            "pages_changed_ctrl_v10": pages_changed_ctrl_v10,
            "pages_changed_ctrl_v6": pages_changed_ctrl_v6,
            "ctrl_doc_refs": ctrl_docs,
            "v6_doc_refs": v6_docs,
            "v10_doc_refs": v10_docs,
            "ctrl_context_chunks": len(ct.get("context_chunk_ids") or []),
            "v6_context_chunks": len(v6t.get("context_chunk_ids") or []),
            "v10_context_chunks": len(v10t.get("context_chunk_ids") or []),
            "v10_anomaly_flags": anomaly_flags.get(qid, []),
            "v10_support_flags": support_flags.get(qid, []),
            "v10_malformed_tail": v10t.get("malformed_tail_detected", False),
            "v10_model_upgraded": v10t.get("model_upgraded", False),
        }
        ledger.append(row)

    return ledger


def render_markdown(ledger: list[dict]) -> str:
    changed_answer = [r for r in ledger if r["answer_changed_ctrl_v10"]]
    changed_pages = [r for r in ledger if r["pages_changed_ctrl_v10"]]
    changed_route = [r for r in ledger if r["ctrl_gen_mode"] != r["v10_gen_mode"]]
    anomaly = [r for r in ledger if r["v10_anomaly_flags"]]
    support = [r for r in ledger if r["v10_support_flags"]]

    lines = [
        "# Champion vs V10 Drift Ledger (2026-03-16)",
        "",
        "## Summary",
        "",
        f"- Total QIDs: {len(ledger)}",
        f"- Answer changed (ctrl vs v10): {len(changed_answer)}",
        f"- Pages changed (ctrl vs v10): {len(changed_pages)}",
        f"- Generation route changed (ctrl vs v10): {len(changed_route)}",
        f"- V10 anomaly-flagged: {len(anomaly)}",
        f"- V10 support-shape-flagged: {len(support)}",
        "",
        "## Answer Drift Details",
        "",
    ]

    if changed_answer:
        lines.append("| QID (short) | Type | Ctrl Answer | V10 Answer | Route Change | Pages Change |")
        lines.append("|---|---|---|---|---|---|")
        for r in changed_answer:
            qshort = r["qid"][:12] + "…"
            route_chg = "YES" if r["ctrl_gen_mode"] != r["v10_gen_mode"] else ""
            page_chg = "YES" if r["pages_changed_ctrl_v10"] else ""
            lines.append(
                f"| `{qshort}` | {r['answer_type']} | {_short(r['ctrl_answer'], 60)} | {_short(r['v10_answer'], 60)} | {route_chg} | {page_chg} |"
            )
    else:
        lines.append("No answer drift detected.")

    lines.extend([
        "",
        "## Route Changes (generation_mode drift)",
        "",
    ])
    if changed_route:
        lines.append("| QID (short) | Ctrl Route | V10 Route | Answer Changed |")
        lines.append("|---|---|---|---|")
        for r in changed_route:
            qshort = r["qid"][:12] + "…"
            ans_chg = "YES" if r["answer_changed_ctrl_v10"] else ""
            lines.append(f"| `{qshort}` | {r['ctrl_gen_mode']} | {r['v10_gen_mode']} | {ans_chg} |")
    else:
        lines.append("No route changes detected.")

    lines.extend([
        "",
        "## Page Drift (used_page_ids changed without answer change)",
        "",
    ])
    page_only = [r for r in changed_pages if not r["answer_changed_ctrl_v10"]]
    if page_only:
        lines.append("| QID (short) | Ctrl Pages | V10 Pages |")
        lines.append("|---|---|---|")
        for r in page_only:
            qshort = r["qid"][:12] + "…"
            lines.append(f"| `{qshort}` | {r['ctrl_used_pages']} | {r['v10_used_pages']} |")
    else:
        lines.append("No page-only drift detected.")

    lines.extend([
        "",
        "## V10 Anomaly and Support-Shape Flags",
        "",
    ])
    flagged = [r for r in ledger if r["v10_anomaly_flags"] or r["v10_support_flags"]]
    if flagged:
        lines.append("| QID (short) | Anomaly Flags | Support Flags |")
        lines.append("|---|---|---|")
        for r in flagged:
            qshort = r["qid"][:12] + "…"
            af = ", ".join(r["v10_anomaly_flags"]) if r["v10_anomaly_flags"] else ""
            sf = ", ".join(r["v10_support_flags"]) if r["v10_support_flags"] else ""
            lines.append(f"| `{qshort}` | {af} | {sf} |")
    else:
        lines.append("No flagged cases.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    control_sub = _load_json(CONTROL_SUBMISSION)
    control_raw = _load_json(CONTROL_RAW)
    v6_sub = _load_json(V6_SUBMISSION)
    v6_raw = _load_json(V6_RAW)
    v10_sub = _load_json(V10_SUBMISSION)
    v10_raw = _load_json(V10_RAW)
    v10_preflight = _load_json(V10_PREFLIGHT)

    ledger = build_ledger(
        control_sub=control_sub,
        control_raw=control_raw,
        v6_sub=v6_sub,
        v6_raw=v6_raw,
        v10_sub=v10_sub,
        v10_raw=v10_raw,
        v10_preflight=v10_preflight,
    )

    Path(args.out_json).write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.out_md).write_text(render_markdown(ledger), encoding="utf-8")

    changed = sum(1 for r in ledger if r["answer_changed_ctrl_v10"])
    pages = sum(1 for r in ledger if r["pages_changed_ctrl_v10"])
    routes = sum(1 for r in ledger if r["ctrl_gen_mode"] != r["v10_gen_mode"])
    print(f"Drift ledger: {len(ledger)} QIDs, {changed} answer changes, {pages} page changes, {routes} route changes")
    print(f"  -> {args.out_json}")
    print(f"  -> {args.out_md}")


if __name__ == "__main__":
    main()
