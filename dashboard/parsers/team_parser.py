"""Team parser — reads agent STATUS.json, TASK_QUEUE.jsonl, and BULLETIN.jsonl."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _read_latest_local_eval(root: Path) -> dict:
    """Parse the most recent warmup_score_*.md file for G and Det numbers."""
    eval_dir = root / "data" / "eval"
    if not eval_dir.exists():
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}

    # Try JSON score files first (more reliable), sorted by mtime newest last
    json_files = sorted(eval_dir.glob("warmup_score_*.json"), key=lambda f: f.stat().st_mtime)
    if json_files:
        try:
            data = json.loads(json_files[-1].read_text())
            s = data.get("summary", {})
            g = (s.get("overall_grounding_f_beta") or s.get("grounding_fbeta")
                 or s.get("g_score") or s.get("overall_g"))
            det_pct = (s.get("overall_exact_match_rate") or s.get("exact_match_rate")
                       or s.get("det_pct"))
            det_n = s.get("exact_match_correct") or s.get("exact_match_n") or s.get("det_correct")
            det_total = (s.get("exact_match_evaluated") or s.get("exact_match_total")
                         or s.get("det_total") or 70)
            if g is not None:
                return {
                    "g": round(float(g), 4),
                    "det_correct": int(det_n) if det_n is not None else int(round(float(det_pct or 0) * det_total)),
                    "det_total": int(det_total),
                    "det_pct": round(float(det_pct or 0), 4),
                    "source": json_files[-1].name,
                }
        except Exception:
            pass

    # Fall back to parsing the latest markdown file
    md_files = sorted(eval_dir.glob("warmup_score_*.md"), key=lambda f: f.stat().st_mtime)
    if not md_files:
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}
    try:
        text = md_files[-1].read_text()
        g_m = re.search(r"Overall Grounding F-beta.*?:\s*([\d.]+)", text)
        det_m = re.search(r"Exact Match Rate.*?:\s*([\d.]+)\s*\((\d+)/(\d+)\)", text)
        g = float(g_m.group(1)) if g_m else 0.379
        det_correct = int(det_m.group(2)) if det_m else 54
        det_total = int(det_m.group(3)) if det_m else 70
        return {
            "g": g,
            "det_correct": det_correct,
            "det_total": det_total,
            "det_pct": round(det_correct / det_total, 4),
            "source": md_files[-1].name,
        }
    except Exception:
        return {"g": 0.379, "det_correct": 54, "det_total": 70, "det_pct": 54 / 70, "source": "hardcoded"}


def parse_team(root: Path) -> dict:
    agents_dir = root / ".sdd" / "agents"
    agents = ["franky", "smarty", "alby", "sissy", "rocky", "muffy", "papa", "benny", "cocky"]
    result = []

    for name in agents:
        agent_dir = agents_dir / name
        status_file = agent_dir / "STATUS.json"
        queue_file = agent_dir / "TASK_QUEUE.jsonl"

        # Read status
        status_raw: dict = {}
        if status_file.exists():
            try:
                status_raw = json.loads(status_file.read_text())
            except Exception:
                pass

        # Read queue
        pending, active, done = 0, 0, 0
        next_task = ""
        if queue_file.exists():
            try:
                for line in queue_file.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        s = t.get("status", "")
                        if s == "pending":
                            pending += 1
                            if not next_task:
                                next_task = f"{t.get('task_id', '')} — {t.get('description', '')[:60]}"
                        elif s == "active":
                            active += 1
                            if not next_task:
                                next_task = f"[active] {t.get('task_id', '')} — {t.get('description', '')[:50]}"
                        elif s == "done":
                            done += 1
                    except Exception:
                        pass
            except Exception:
                pass

        agent_status = status_raw.get("status", "unknown")
        is_idle = agent_status in ("idle", "pending_activation", "unknown") and pending == 0
        has_work_idle = agent_status in ("idle", "standby") and pending > 0

        result.append({
            "name": name,
            "status": agent_status,
            "is_idle": is_idle,
            "has_work_idle": has_work_idle,
            "current_task": status_raw.get("current_task") or status_raw.get("last_completed") or status_raw.get("last_action") or "",
            "progress": status_raw.get("progress") or "",
            "recent_findings": status_raw.get("recent_findings") or [],
            "timestamp": status_raw.get("timestamp") or status_raw.get("last_updated") or "",
            "pending": pending,
            "active": active,
            "done": done,
            "next_task": next_task,
            "tests_passing": status_raw.get("tests_passing"),
            "cycle": status_raw.get("cycle"),  # for papa
        })

    return result


def parse_bulletin(root: Path, limit: int = 30) -> list[dict]:
    bulletin_file = root / ".sdd" / "agents" / "BULLETIN.jsonl"
    if not bulletin_file.exists():
        return []

    entries = []
    try:
        lines = bulletin_file.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                entries.append({
                    "from": e.get("from", "?"),
                    "timestamp": e.get("timestamp", ""),
                    "type": e.get("type", ""),
                    "message": e.get("message", ""),
                })
            except Exception:
                pass
    except Exception:
        pass

    return entries[-limit:]


def parse_recent_commits(root: Path, limit: int = 12) -> list[dict]:
    try:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{limit}"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        commits = []
        for line in result.stdout.splitlines():
            if line.strip():
                sha, _, msg = line.partition(" ")
                commits.append({"sha": sha, "message": msg})
        return commits
    except Exception:
        return []


# Reference scoring thresholds for internal dashboard rank estimation
_LEADERBOARD: list[dict] = [
    {"rank": 1,  "team": "Leader",      "total": 0.982, "det": 1.000, "asst": 0.833, "g": 0.990, "t": 0.995, "f": 1.050, "latency_ms": 900},
    {"rank": 10, "team": "Top-10",      "total": 0.920, "det": 0.986, "asst": 0.820, "g": 0.957, "t": 0.995, "f": 1.033, "latency_ms": 847},
    {"rank": 51, "team": "Tzur Labs",   "total": 0.742, "det": 0.971, "asst": 0.693, "g": 0.801, "t": 0.996, "f": 1.047, "latency_ms": 347, "is_us": True},
]


# Approximate score distribution for rank estimation (warmup leaderboard, anonymized)
_ALL_TEAM_SCORES: list[float] = [
    0.982, 0.963, 0.954, 0.953, 0.950,
    0.937, 0.936, 0.924, 0.922, 0.920,
    0.905, 0.887, 0.850, 0.787, 0.744,
    0.742, 0.740, 0.720, 0.714,
]


def _score_formula(det: float, asst: float, g: float, t: float = 0.996, f: float = 1.047) -> float:
    """Competition score formula: (0.7*Det + 0.3*Asst) * G * T * F."""
    return (0.7 * det + 0.3 * asst) * g * t * f


def _estimate_rank(projected_score: float) -> tuple[int, str]:
    """Find where projected_score would rank among all 61 teams."""
    all_scores = sorted([s for s in _ALL_TEAM_SCORES if abs(s - 0.742) > 0.0005], reverse=True)
    # Count how many teams we'd beat
    rank = 1
    for s in all_scores:
        if projected_score < s:
            rank += 1
    if rank == 1:
        return 1, "🥇 1st place!"
    elif rank <= 3:
        return rank, f"🥈 Top 3 (rank {rank})"
    elif rank <= 10:
        return rank, f"Top 10 (rank {rank})"
    elif rank <= 20:
        return rank, f"Top 20 (rank {rank})"
    else:
        return rank, f"Rank ~{rank}"


def _read_live_eval_progress(root: Path) -> dict:
    """Read live eval progress from rocky_private1_checkpoint.jsonl."""
    checkpoint = root / "data" / "rocky_private1_checkpoint.jsonl"
    if not checkpoint.exists():
        return {"answered": 0, "total": 900, "pct": 0, "ttft_avg_ms": 0, "types": {}}
    try:
        ids: set[str] = set()
        ttft_sum = 0.0
        ttft_count = 0
        types: dict[str, int] = {}
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ids.add(d["id"])
                t = d.get("answer_type", "?")
                types[t] = types.get(t, 0) + 1
                if "ttft_ms" in d:
                    ttft_sum += d["ttft_ms"]
                    ttft_count += 1
            except Exception:
                pass
        answered = len(ids)
        return {
            "answered": answered,
            "total": 900,
            "pct": round(answered / 900 * 100, 1),
            "ttft_avg_ms": round(ttft_sum / ttft_count) if ttft_count else 0,
            "types": types,
        }
    except Exception:
        return {"answered": 0, "total": 900, "pct": 0, "ttft_avg_ms": 0, "types": {}}


def _read_best_eval(root: Path) -> dict:
    """Read the best eval result file and return metrics."""
    candidates = [
        ("V2 (RECOMMENDED)", root / "data" / "private_submission_V2.json"),
        ("FINAL_SUBMISSION", root / "data" / "private_submission_FINAL_SUBMISSION.json"),
        ("V17_DOI", root / "data" / "private_submission_V17_SUPER_HYBRID_DOI.json"),
        ("V17_SUPER", root / "data" / "private_submission_V17_SUPER_HYBRID.json"),
        ("V16_BEST", root / "data" / "private_submission_V16_BEST.json"),
        ("V16_HYBRID", root / "data" / "private_submission_V16_HYBRID.json"),
        ("ULTIMATE_FINAL", root / "data" / "private_submission_V15_ULTIMATE_FINAL.json"),
        ("FINAL_BEST", root / "data" / "private_submission_V15_FINAL_BEST.json"),
        ("ULTIMATE", root / "data" / "private_submission_ULTIMATE.json"),
        ("V16", root / "data" / "private_submission_V16.json"),
        ("V15_REC", root / "data" / "private_submission_V15_ENRICHED_recovered.json"),
        ("V15_PATCH", root / "data" / "private_submission_V15_ENRICHED_patched.json"),
        ("V15_FINAL", root / "data" / "private_submission_V15_ENRICHED_FINAL.json"),
        ("V10.1", root / "data" / "rocky_v10_1_full900.json"),
        ("V9.1", root / "data" / "rocky_v9_1_full900.json"),
        ("V10", root / "data" / "rocky_v10_full900.json"),
    ]
    for label, path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                results = data.get("results") or data.get("answers") or []
                total_q = data.get("total", len(results))
                nulls = data.get("null_count", sum(1 for r in results if r.get("answer") is None))

                no_pages = 0
                ttfts = []
                for r in results:
                    # Handle different structures for page_ids and ttft
                    tel = r.get("telemetry", {})
                    if not r.get("used_page_ids") and not tel.get("retrieval", {}).get("retrieved_chunk_pages"):
                        no_pages += 1

                    ttft = r.get("ttft_ms") or tel.get("timing", {}).get("ttft_ms")
                    if ttft:
                        ttfts.append(ttft)

                g_proxy = data.get("g_proxy", round(1 - no_pages / max(total_q, 1), 4))
                ttft_avg = data.get("ttft_avg_ms") or (sum(ttfts) / len(ttfts) if ttfts else 0)
                over_5s = sum(1 for t in ttfts if t > 5000)

                # Compute F coefficient per-question then average (more accurate than from avg TTFT)
                if ttfts:
                    f_per_q: list[float] = []
                    for _t in ttfts:
                        if _t < 1000: f_per_q.append(1.05)
                        elif _t < 2000: f_per_q.append(1.02)
                        elif _t < 3000: f_per_q.append(1.00)
                        elif _t < 5000: f_per_q.append(0.99 - (_t - 3000) * 0.14 / 2000)
                        else: f_per_q.append(0.85)
                    f_coeff = sum(f_per_q) / len(f_per_q)
                else:
                    if ttft_avg < 1000: f_coeff = 1.05
                    elif ttft_avg < 2000: f_coeff = 1.02
                    elif ttft_avg < 3000: f_coeff = 1.00
                    elif ttft_avg < 5000: f_coeff = 0.99 - (ttft_avg - 3000) * 0.14 / 2000
                    else: f_coeff = 0.85

                return {
                    "version": label,
                    "total_q": total_q,
                    "nulls": nulls,
                    "no_pages": no_pages,
                    "g_proxy": round(g_proxy, 4),
                    "ttft_avg_ms": round(ttft_avg),
                    "over_5s": over_5s,
                    "f_coeff": round(f_coeff, 4),
                    "file": path.name,
                    "head": data.get("code_version", "?"),
                }
            except Exception:
                continue
    return {"version": "?", "total_q": 0, "nulls": 0, "no_pages": 0, "g_proxy": 0,
            "ttft_avg_ms": 0, "over_5s": 0, "f_coeff": 1.0, "file": "none", "head": "?"}


def _read_version_history(root: Path) -> list[dict]:
    """Read metrics from all eval versions for the progression chart."""
    versions = []
    files = [
        ("V2", "rocky_private1_v2_full900.json", "2026-03-21 08:00"),
        ("V5", "rocky_private1_full900.json", "2026-03-21 10:00"),
        ("V6", "private_submission_v6.json", "2026-03-21 11:00"),
        ("V8", "rocky_v8_full900.json", "2026-03-21 14:00"),
        ("V8.1", "rocky_v8_1_full900.json", "2026-03-21 16:00"),
        ("V9", "rocky_v9_full900.json", "2026-03-21 19:24"),
        ("V9.1", "rocky_v9_1_full900.json", "2026-03-21 19:27"),
        ("V10", "rocky_v10_full900.json", "2026-03-21 20:20"),
        ("V10.1", "rocky_v10_1_full900.json", "2026-03-21 20:40"),
    ]
    for label, fname, ts in files:
        path = root / "data" / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            g = data.get("g_proxy", 0)
            nulls = data.get("null_count", 0)
            ttft = data.get("ttft_avg_ms", 0)
            total_q = data.get("total", 0)
            # For submission files, parse differently
            if "answers" in data and not g:
                answers = data["answers"]
                total_q = len(answers)
                nulls = sum(1 for a in answers if a.get("answer") is None)
                ttfts = [a["telemetry"]["timing"]["ttft_ms"] for a in answers
                         if a.get("telemetry", {}).get("timing", {}).get("ttft_ms")]
                ttft = round(sum(ttfts) / len(ttfts)) if ttfts else 0
                pages_counts = [len(a["telemetry"]["retrieval"].get("retrieved_chunk_pages", []))
                                for a in answers]
                no_pages = sum(1 for c in pages_counts if c == 0)
                g = round(1 - no_pages / max(total_q, 1), 4)  # rough proxy
            versions.append({
                "label": label, "date": ts, "g_proxy": round(g, 4),
                "nulls": nulls, "ttft_avg_ms": round(ttft), "total_q": total_q,
            })
        except Exception:
            continue
    return versions


def parse_score_state(root: Path | None = None) -> dict:
    """Актуальное состояние конкурса с реальными данными из eval-файлов."""
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    
    deadline_dt = datetime(2026, 3, 22, 15, 0, tzinfo=timezone.utc)
    diff = deadline_dt - now_dt
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    time_to_deadline = f"{hours}h {minutes}m" if diff.total_seconds() > 0 else "DEADLINE REACHED"

    if root is None:
        root = Path(__file__).resolve().parent.parent.parent

    # Читаем лучший eval
    best = _read_best_eval(root)
    eval_progress = _read_live_eval_progress(root)
    version_history = _read_version_history(root)

    # Вычисляем проекцию Total
    est_det = 0.96  # 103+ corrections: 27 booleans, 56 DOI dates, 12 registry fixes
    est_asst = 0.73  # gpt-4.1-mini + 3 law-name prefixes
    est_t = 0.999  # 171 warmup doc_ids stripped → T 0.986→0.999
    proj_total_low = _score_formula(est_det, 0.70, best["g_proxy"], est_t, best["f_coeff"])
    proj_total_mid = _score_formula(est_det, est_asst, best["g_proxy"], est_t, best["f_coeff"])
    proj_total_high = _score_formula(1.000, 0.80, best["g_proxy"], est_t, best["f_coeff"])

    leader = next(t for t in _LEADERBOARD if t["rank"] == 1)
    rank_low, label_low = _estimate_rank(proj_total_low)
    rank_mid, label_mid = _estimate_rank(proj_total_mid)
    rank_high, label_high = _estimate_rank(proj_total_high)

    # Шансы
    if proj_total_mid >= leader["total"]:
        chances_1st = "60-80%"
        chances_top5 = "90%+"
    elif proj_total_high >= leader["total"]:
        chances_1st = "30-50%"
        chances_top5 = "70-85%"
    elif proj_total_mid >= 0.95:
        chances_1st = "10-25%"
        chances_top5 = "50-70%"
    else:
        chances_1st = "5-15%"
        chances_top5 = "30-50%"

    return {
        "last_updated": now,
        "time_to_deadline": time_to_deadline,
        "current_best": {
            "total": round(proj_total_mid, 3),
            "total_range": f"{proj_total_low:.3f} — {proj_total_high:.3f}",
            "g_proxy": best["g_proxy"],
            "det": est_det,
            "asst": est_asst,
            "f": best["f_coeff"],
            "t": est_t,
            "rank": rank_mid,
            "rank_label": label_mid,
            "source": f"{best['version']} ({best['file']})",
            "version": best["version"],
        },
        "platform": {
            "total": 0.741,
            "g": 0.801,
            "det": 0.971,
            "asst": 0.693,
            "rank": 51,
            "rank_label": "Ранг 51 / 61 (SLOT1, устаревший warmup)",
            "leader": leader["total"],
            "leader_note": "Warmup leader 0.982. Private scores expected lower.",
        },
        "local_warmup": _read_latest_local_eval(root),
        "server_status": {
            "healthy": True,
            "port": 8000,
            "collection": "legal_chunks_private_1792",
            "points": 10307,
            "dimensions": 1792,
            "model": "kanon-2-embedder",
            "profile": "private_v9_rerank12.env",
            "docs_indexed": "300/300 (100%)",
            "ingest_complete": True,
            "note": "V2 ГОТОВ К САБМИТУ: V2.json (null=3, nopg=3, 10109 pages, F=1.032). "
                    "103+ коррекций, +6846 pages enrichment, 171 warmup doc_id удалены. "
                    "Fallback: FINAL_SUBMISSION.json. АГЕНТЫ ЗАВЕРШАЮТ РАБОТУ.",
        },
        "eval_progress": {
            "answered": best["total_q"],
            "total": 900,
            "pct": round(best["total_q"] / 900 * 100, 1),
            "ttft_avg_ms": best["ttft_avg_ms"],
            "types": {},
        },
        "run_900q": {
            "status": f"{best['version']} — {best['total_q']}/900 завершено",
            "g_proxy": best["g_proxy"],
            "nulls": best["nulls"],
            "no_pages": best["no_pages"],
            "over_5s": best["over_5s"],
            "ttft_avg_ms": best["ttft_avg_ms"],
            "f_coeff": best["f_coeff"],
        },
        "honest_assessment": {
            "projected_total": round(proj_total_mid, 3),
            "projected_range": f"{proj_total_low:.3f} — {proj_total_high:.3f}",
            "leader_total": leader["total"],
            "gap_pp": round((leader["total"] - proj_total_mid) * 100, 1),
            "components": [
                {
                    "name": "G (граундинг)",
                    "value": best["g_proxy"],
                    "target": 1.000,
                    "verdict_ru": "ОТЛИЧНО" if best["g_proxy"] >= 0.99 else ("ХОРОШО" if best["g_proxy"] >= 0.95 else "СЛАБО"),
                    "status": "perfect" if best["g_proxy"] >= 0.99 else ("good" if best["g_proxy"] >= 0.95 else "warning"),
                    "note_ru": f"G={best['g_proxy']:.4f}. Nulls={best['nulls']}, no-pages={best['no_pages']}.",
                },
                {
                    "name": "F (TTFT скорость)",
                    "value": best["f_coeff"],
                    "target": 1.050,
                    "verdict_ru": "ХОРОШО" if best["f_coeff"] >= 1.0 else ("СРЕДНЕ" if best["f_coeff"] >= 0.99 else "ПЛОХО"),
                    "status": "good" if best["f_coeff"] >= 1.0 else ("warning" if best["f_coeff"] >= 0.99 else "critical"),
                    "note_ru": f"TTFT={best['ttft_avg_ms']}мс, >5с: {best['over_5s']} запросов. Коэф={best['f_coeff']:.4f}.",
                },
                {
                    "name": "Det (точный ответ)",
                    "value": est_det,
                    "target": 1.000,
                    "verdict_ru": "ОЦЕНКА" if est_det < 1.0 else "ИДЕАЛЬНО",
                    "status": "warning" if est_det < 1.0 else "perfect",
                    "note_ru": "103+ коррекций: 27 bool + 56 DOI + 12 registry + 10 format fixes. Оценка ~0.96.",
                },
                {
                    "name": "Asst (качество текста)",
                    "value": est_asst,
                    "target": 0.833,
                    "verdict_ru": "ОЦЕНКА",
                    "status": "warning",
                    "note_ru": "Нет golden для private. Warmup=0.693. gpt-4.1-mini + 3 law-name prefixes. Оценка ~0.73.",
                },
                {
                    "name": "T (телеметрия)",
                    "value": est_t,
                    "target": 1.000,
                    "verdict_ru": "ОТЛИЧНО" if est_t >= 0.999 else "ОК",
                    "status": "perfect" if est_t >= 0.999 else "good",
                    "note_ru": f"T={est_t}. 171 warmup doc_id удалены. Все telemetry поля валидны.",
                },
            ],
            "chances": {
                "first_place": chances_1st,
                "top5": chances_top5,
            },
        },
        "projections": [
            {
                "label": "SLOT1 (платформа, устаревший)",
                "det": 0.971, "asst": 0.693, "g": 0.801,
                "total": 0.741, "rank": 51, "rank_label": "Ранг ~51", "status": "old",
            },
            {
                "label": f"{best['version']} пессимистичная (Det=0.95, Asst=0.70)",
                "det": est_det, "asst": 0.70, "g": best["g_proxy"],
                "total": round(proj_total_low, 3), "rank": rank_low,
                "rank_label": label_low, "status": "pessimistic",
                "delta": f"+{round(proj_total_low - 0.741, 3)}",
            },
            {
                "label": f"{best['version']} реалистичная (Det=0.95, Asst=0.75)",
                "det": est_det, "asst": est_asst, "g": best["g_proxy"],
                "total": round(proj_total_mid, 3), "rank": rank_mid,
                "rank_label": label_mid, "status": "current",
                "delta": f"+{round(proj_total_mid - 0.741, 3)}",
            },
            {
                "label": f"{best['version']} оптимистичная (Det=1.0, Asst=0.80)",
                "det": 1.0, "asst": 0.80, "g": best["g_proxy"],
                "total": round(proj_total_high, 3), "rank": rank_high,
                "rank_label": label_high, "status": "stretch",
                "delta": f"+{round(proj_total_high - 0.741, 3)}",
            },
        ],
        "gap_to_leader": {
            "det": round(leader.get("det", 1.0) - est_det, 3),
            "asst": round(leader["asst"] - est_asst, 3),
            "g": round(leader["g"] - best["g_proxy"], 3),
            "f": round(leader.get("f", 1.050) - best["f_coeff"], 3),
        },
        "leaderboard": _LEADERBOARD,
        "targets": {
            "det_pct": 1.0,
            "g": 1.000,
            "asst": 0.833,
            "total": 0.982,
        },
        "pending_gains": [
            {
                "fix": "SUBMIT #1 (RECOMMENDED): V2.json — null=3, nopg=3, 10109 pages, F=1.032. "
                       "103+ коррекций (27 bool + 56 DOI + 12 registry) + 6846 enriched pages + 3 FT prefixes.",
                "det_delta": "+0.05-0.10 (all corrections)", "g_delta": "G=0.9967 (10109 pages)",
                "priority": "SUBMIT NOW",
            },
            {
                "fix": "SUBMIT #2 (fallback): FINAL_SUBMISSION.json — null=3, nopg=4, 3263 pages. "
                       "Same answers as V2, just fewer grounding pages.",
                "det_delta": "same", "g_delta": "G=0.9956 (3263 pages)", "priority": "BACKUP",
            },
            {
                "fix": "V18 eval — FAILED (null=9, nopg=41). DO NOT SUBMIT.",
                "det_delta": "REGRESSION", "g_delta": "G=0.954", "priority": "REJECTED",
            },
        ],
        "eval_history": version_history or [
            {"label": "Warmup initial", "total": 0.741, "g_proxy": 0.801, "nulls": 0, "date": "2026-03-19"},
            {"label": "SLOT2 (warmup)", "total": 0.990, "g_proxy": 1.043, "nulls": 0, "date": "2026-03-20"},
        ],
        "unexplored_directions": [
            {"name": "Predicted Outputs", "desc": "enable_predicted_outputs=False. Спекулятивное декодирование GPT-4.1.", "effort": "5 мин", "risk": "низкий"},
            {"name": "Multi-hop Decomposition", "desc": "PIPELINE_ENABLE_MULTI_HOP=False. Декомпозиция сложных вопросов.", "effort": "30 мин", "risk": "средний"},
            {"name": "Conflict Detection", "desc": "PIPELINE_ENABLE_CONFLICT_DETECTION=False. Обнаружение противоречий.", "effort": "20 мин", "risk": "низкий"},
            {"name": "Answer Consensus", "desc": "enable_answer_consensus=False. Ансамбль ответов.", "effort": "30 мин", "risk": "средний"},
            {"name": "Retrieval Escalation", "desc": "enable_retrieval_escalation=False. Расширение поиска при 0 результатов.", "effort": "20 мин", "risk": "низкий"},
            {"name": "Doc Title Boost", "desc": "enable_doc_title_boost=False. OREV коммитнул (c5fc987) но не включено в .env.", "effort": "5 мин", "risk": "низкий"},
            {"name": "Dedup Duplicate Docs", "desc": "dedup_duplicate_docs=False. OREV нашёл 1336 дупов (13%). Коммит c5fc987.", "effort": "5 мин", "risk": "низкий"},
        ],
        "rejected_experiments": [
            {"name": "BM25 Hybrid", "result": "+0pp G, +260мс TTFT", "reason": "Нет выигрыша"},
            {"name": "RAG Fusion", "result": "-7.8pp G", "reason": "Регрессия"},
            {"name": "HyDE", "result": "-0.65pp, +560мс", "reason": "Регрессия"},
            {"name": "Step-back", "result": "-0.65pp, +1542мс", "reason": "Регрессия"},
            {"name": "Interleaved Citations", "result": "+0pp", "reason": "Нет выигрыша"},
            {"name": "Citation Verifier", "result": "-0.63pp", "reason": "Регрессия"},
            {"name": "FlashRank Reranker", "result": "G=0.9667, 27% overlap, TTFT хуже", "reason": "Провалил gate — TZUF подтвердил"},
            {"name": "gpt-4.1-mini for all", "result": "хуже Asst на free_text", "reason": "gpt-4.1 даёт лучше качество"},
            {"name": "Isaacus EQA", "result": "V12: 873/900 nulls, G=0.03", "reason": "None path ломает pipeline"},
            {"name": "SHAI-DEEP-2", "result": "двойной TTFT", "reason": "F коэффициент падает, не окупается"},
            {"name": "ULTIMATE trick-Q wipe", "result": "nopg=32, G=0.9644", "reason": "Слишком агрессивное обнуление — 28 ложных срабатываний"},
        ],
        "ingest": {
            "docs_total": 300,
            "docs_indexed": 300,
            "points": 10307,
            "status": "ГОТОВО",
            "collection": "legal_chunks_private_1792",
            "dimensions": 1792,
        },
    }
