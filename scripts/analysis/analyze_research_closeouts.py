#!/usr/bin/env python3
"""Analyze all closeout.md files in .sdd/researches/ to extract patterns and insights."""

import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE = Path("/Users/sasha/IdeaProjects/personal_projects/shafi")
RESEARCH_DIR = BASE / ".sdd" / "researches"


@dataclass
class CloseoutRecord:
    dir_name: str
    file_path: str
    ticket_number: Optional[str] = None
    date: Optional[str] = None
    raw_text: str = ""
    status: Optional[str] = None  # promoted, killed, no_go, abandoned, etc.
    scores: list = field(default_factory=list)  # (label, value) tuples
    score_deltas: list = field(default_factory=list)  # (label, before, after, delta)
    submission_versions: list = field(default_factory=list)  # v1-v15 refs
    verdict_text: str = ""
    topics: list = field(default_factory=list)
    answer_changed_count: Optional[int] = None
    page_changed_count: Optional[int] = None
    platform_total_score: Optional[float] = None
    local_vs_platform: bool = False


def extract_date(dir_name: str) -> Optional[str]:
    m = re.search(r"(20\d{6}|\d{4}-\d{2}-\d{2})", dir_name)
    if m:
        d = m.group(1)
        if len(d) == 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return d
    return None


def extract_ticket(dir_name: str, text: str) -> Optional[str]:
    m = re.match(r"^(\d{3,4})_", dir_name)
    if m:
        return m.group(1)
    m = re.search(r"#\s*(\d{3,4})\b", text)
    if m:
        return m.group(1)
    m = re.search(r"Ticket\s+(\d{3,4})\b", text)
    if m:
        return m.group(1)
    return None


def extract_status(text: str) -> Optional[str]:
    t = text.lower()
    # Check for explicit status markers first
    if re.search(r"status:\s*`?no.?go`?", t):
        return "NO_GO"
    if re.search(r"outcome:\s*`?no.?go`?", t):
        return "NO_GO"
    if re.search(r"\bpromote\b", t) and not re.search(r"\bnot\s+promot", t):
        return "PROMOTED"
    if re.search(r"verdict.*kill|kill.*verdict|decision.*kill|\bkill\b.*lane|\bkilled\b", t):
        return "KILLED"
    if "measured no-go" in t or "measured no go" in t:
        return "NO_GO"
    if re.search(r"do.not.submit", t):
        return "NO_GO"
    if "abandoned" in t:
        return "ABANDONED"
    if re.search(r"verdict.*no.go|no.go.*verdict", t):
        return "NO_GO"
    if "regress" in t and ("kill" in t or "revert" in t or "no-go" in t):
        return "KILLED"
    if "completed" in t and "submit" not in t:
        return "COMPLETED"
    return None


def extract_scores(text: str) -> list:
    """Extract score-like numbers (0.xxx patterns) with their labels."""
    results = []
    # Pattern: label = 0.xxxx or label: 0.xxxx
    for m in re.finditer(r"[`*]*([a-zA-Z_0-9 ]+?)[`*]*\s*[=:]\s*`?(0\.\d{3,6})`?", text):
        label = m.group(1).strip()
        val = float(m.group(2))
        results.append((label, val))
    # Pattern: Det = 0.xxx, Asst = 0.xxx etc
    for m in re.finditer(r"(Det|Asst|G|F|Total)\s*=\s*`?(0\.\d+)`?", text):
        results.append((f"platform_{m.group(1)}", float(m.group(2))))
    return results


def extract_score_transitions(text: str) -> list:
    """Extract score transitions like 0.5123 -> 0.4757."""
    results = []
    for m in re.finditer(
        r"[`*]*([a-zA-Z_0-9 ]+?)[`*]*[:\s]*`?(0\.\d{3,6})`?\s*->\s*`?(0\.\d{3,6})`?",
        text,
    ):
        label = m.group(1).strip()
        before = float(m.group(2))
        after = float(m.group(3))
        delta = after - before
        results.append((label, before, after, delta))
    # Also match explicit delta patterns
    for m in re.finditer(
        r"[`*]*([a-zA-Z_0-9 ]+?)[`*]*\s*=\s*`?([+-]0\.\d{3,6})`?", text
    ):
        label = m.group(1).strip()
        val = float(m.group(2))
        if abs(val) < 1.0:  # likely a delta
            results.append((label, None, None, val))
    return results


def extract_submission_versions(text: str) -> list:
    return list(set(re.findall(r"\bv(\d{1,2})\b", text)))


def extract_verdict_section(text: str) -> str:
    """Extract text from verdict/conclusion/recommendation sections."""
    parts = []
    for m in re.finditer(
        r"##\s*(Verdict|Conclusion|Recommendation|Interpretation|Why.*Not.*Promoted|Next [Ss]tep).*?\n(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    ):
        parts.append(m.group(2).strip())
    return "\n".join(parts)


def extract_topics(dir_name: str) -> list:
    """Extract topic keywords from directory name."""
    # Remove ticket number and date
    name = re.sub(r"^\d{3,4}_", "", dir_name)
    name = re.sub(r"_?r\d+$", "", name)
    name = re.sub(r"_?20\d{6}.*$", "", name)
    name = re.sub(r"_?\d{4}-\d{2}-\d{2}.*$", "", name)
    # Split into words
    words = [w for w in name.split("_") if len(w) > 1]
    return words


TOPIC_CLUSTERS = {
    "rerank": ["rerank", "reranker", "zerank", "colbert"],
    "grounding": ["grounding", "evidence", "page", "pages", "ground"],
    "embedding": ["embed", "embedding", "kanon2", "1792", "dimension"],
    "authority": ["authority", "authoritative", "priors"],
    "scope": ["scope", "bundle", "law", "legal"],
    "replay": ["replay", "calibration", "calibrate"],
    "submission": ["submit", "submission", "platform", "resubmit"],
    "selector": ["selector", "evidence", "portfolio"],
    "support_facts": ["support", "surrogate", "scaffold"],
    "comparison": ["compare", "comparison", "diff"],
    "latency": ["latency", "ttft", "speed"],
    "extraction": ["extract", "extractor", "typed"],
    "pruning": ["prune", "pruner", "counterfactual"],
    "localization": ["localizer", "localize", "anchor"],
    "doc_family": ["family", "docfamily", "doc"],
}


def classify_topic(dir_name: str) -> list:
    name_lower = dir_name.lower()
    matched = []
    for cluster, keywords in TOPIC_CLUSTERS.items():
        if any(kw in name_lower for kw in keywords):
            matched.append(cluster)
    return matched if matched else ["other"]


def parse_closeout(file_path: str) -> CloseoutRecord:
    dir_name = os.path.basename(os.path.dirname(file_path))
    text = Path(file_path).read_text(errors="replace")

    rec = CloseoutRecord(
        dir_name=dir_name,
        file_path=file_path,
        raw_text=text,
        ticket_number=extract_ticket(dir_name, text),
        date=extract_date(dir_name),
        status=extract_status(text),
        scores=extract_scores(text),
        score_deltas=extract_score_transitions(text),
        submission_versions=extract_submission_versions(text),
        verdict_text=extract_verdict_section(text),
        topics=classify_topic(dir_name),
    )

    # answer changed count
    m = re.search(r"answer_changed_count\s*[=:]\s*`?(\d+)`?", text)
    if m:
        rec.answer_changed_count = int(m.group(1))

    m = re.search(r"page_changed_count\s*[=:]\s*`?(\d+)`?", text)
    if m:
        rec.page_changed_count = int(m.group(1))

    # Platform total score
    m = re.search(r"(?:total_score|Total)\s*[=:]\s*`?(0\.\d+)`?", text)
    if m:
        rec.platform_total_score = float(m.group(1))

    # Local vs platform comparison
    if re.search(
        r"local.*platform|platform.*local|local.*eval.*vs|replay.*did not transfer|overfit",
        text.lower(),
    ):
        rec.local_vs_platform = True

    return rec


def find_all_closeouts() -> list:
    results = []
    for root, dirs, files in os.walk(RESEARCH_DIR):
        for f in files:
            if f == "closeout.md":
                results.append(os.path.join(root, f))
    return sorted(results)


def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    closeout_files = find_all_closeouts()
    print(f"Found {len(closeout_files)} closeout.md files across {RESEARCH_DIR}")

    records = [parse_closeout(f) for f in closeout_files]

    # -------------------------------------------------------------------------
    # 1. STATUS DISTRIBUTION
    # -------------------------------------------------------------------------
    print_section("1. RESEARCH OUTCOME DISTRIBUTION")
    status_counts = Counter(r.status or "UNKNOWN" for r in records)
    for status, count in status_counts.most_common():
        pct = 100 * count / len(records)
        bar = "#" * int(pct / 2)
        print(f"  {status:<15} {count:>4}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Total: {len(records)}")

    # -------------------------------------------------------------------------
    # 2. TOPIC DISTRIBUTION
    # -------------------------------------------------------------------------
    print_section("2. RESEARCH TOPIC CLUSTERS")
    topic_counts = Counter()
    topic_statuses = defaultdict(Counter)
    for r in records:
        for t in r.topics:
            topic_counts[t] += 1
            topic_statuses[t][r.status or "UNKNOWN"] += 1

    print(f"  {'Topic':<20} {'Count':>5}  {'Promoted':>8}  {'NO_GO':>6}  {'Killed':>6}  {'Unknown':>7}")
    print(f"  {'-'*20} {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}")
    for topic, count in topic_counts.most_common(20):
        promoted = topic_statuses[topic].get("PROMOTED", 0)
        no_go = topic_statuses[topic].get("NO_GO", 0)
        killed = topic_statuses[topic].get("KILLED", 0)
        unknown = topic_statuses[topic].get("UNKNOWN", 0)
        print(f"  {topic:<20} {count:>5}  {promoted:>8}  {no_go:>6}  {killed:>6}  {unknown:>7}")

    # -------------------------------------------------------------------------
    # 3. TIMELINE OF RESEARCH ACTIVITY
    # -------------------------------------------------------------------------
    print_section("3. RESEARCH ACTIVITY BY DATE")
    date_counts = Counter()
    date_statuses = defaultdict(Counter)
    for r in records:
        d = r.date or "no-date"
        date_counts[d] += 1
        date_statuses[d][r.status or "UNKNOWN"] += 1

    for date in sorted(date_counts.keys()):
        count = date_counts[date]
        promoted = date_statuses[date].get("PROMOTED", 0)
        no_go = date_statuses[date].get("NO_GO", 0)
        killed = date_statuses[date].get("KILLED", 0)
        bar = "#" * min(count, 80)
        print(f"  {date}  {count:>4} researches  (P:{promoted} NG:{no_go} K:{killed})  {bar}")

    # -------------------------------------------------------------------------
    # 4. PROMOTED RESEARCH (SUCCESS STORIES)
    # -------------------------------------------------------------------------
    print_section("4. PROMOTED / SUCCESSFUL RESEARCH")
    promoted = [r for r in records if r.status == "PROMOTED"]
    if promoted:
        for r in promoted:
            print(f"  [{r.ticket_number or '?'}] {r.dir_name}")
            if r.score_deltas:
                for label, before, after, delta in r.score_deltas:
                    if before is not None:
                        print(f"    {label}: {before:.4f} -> {after:.4f} (delta: {delta:+.4f})")
                    else:
                        print(f"    {label}: delta = {delta:+.4f}")
            if r.verdict_text:
                for line in r.verdict_text.split("\n")[:5]:
                    if line.strip():
                        print(f"    > {line.strip()}")
            print()
    else:
        print("  No explicitly promoted research found.\n")

    # -------------------------------------------------------------------------
    # 5. KILLED / NO-GO RESEARCH WITH REASONS
    # -------------------------------------------------------------------------
    print_section("5. KILLED / NO-GO RESEARCH (TOP REASONS)")
    killed_nogo = [r for r in records if r.status in ("KILLED", "NO_GO")]
    # Show ones with verdict text
    shown = 0
    for r in killed_nogo:
        if r.verdict_text and shown < 25:
            print(f"  [{r.ticket_number or '?'}] {r.dir_name}")
            print(f"    Status: {r.status}")
            if r.score_deltas:
                for label, before, after, delta in r.score_deltas[:3]:
                    if before is not None:
                        print(f"    {label}: {before:.4f} -> {after:.4f} ({delta:+.4f})")
            # Show first 3 lines of verdict
            for line in r.verdict_text.split("\n")[:3]:
                if line.strip():
                    print(f"    > {line.strip()}")
            print()
            shown += 1

    # -------------------------------------------------------------------------
    # 6. SCORE TRANSITION ANALYSIS
    # -------------------------------------------------------------------------
    print_section("6. SCORE TRANSITIONS (ALL observed before->after)")
    all_deltas = []
    for r in records:
        for label, before, after, delta in r.score_deltas:
            if before is not None and after is not None:
                all_deltas.append((r.dir_name, r.ticket_number, label, before, after, delta))

    # Sort by delta magnitude
    all_deltas.sort(key=lambda x: x[5], reverse=True)

    print("  TOP 15 IMPROVEMENTS:")
    for dir_name, ticket, label, before, after, delta in all_deltas[:15]:
        print(f"    [{ticket or '?'}] {label}: {before:.4f} -> {after:.4f} ({delta:+.4f})")
        print(f"         {dir_name}")

    print("\n  TOP 15 REGRESSIONS:")
    for dir_name, ticket, label, before, after, delta in all_deltas[-15:]:
        print(f"    [{ticket or '?'}] {label}: {before:.4f} -> {after:.4f} ({delta:+.4f})")
        print(f"         {dir_name}")

    # -------------------------------------------------------------------------
    # 7. PLATFORM SUBMISSION SCORES
    # -------------------------------------------------------------------------
    print_section("7. PLATFORM SUBMISSION SCORES (where total_score found)")
    platform_records = [r for r in records if r.platform_total_score is not None]
    platform_records.sort(key=lambda r: r.platform_total_score or 0, reverse=True)
    if platform_records:
        print(f"  {'Score':>8}  {'Ticket':>6}  Directory")
        print(f"  {'-'*8}  {'-'*6}  {'-'*50}")
        for r in platform_records:
            print(f"  {r.platform_total_score:>8.4f}  {r.ticket_number or '?':>6}  {r.dir_name}")
    else:
        print("  No platform total scores found.\n")

    # -------------------------------------------------------------------------
    # 8. ANSWER DRIFT ANALYSIS
    # -------------------------------------------------------------------------
    print_section("8. ANSWER DRIFT ANALYSIS")
    drift_records = [r for r in records if r.answer_changed_count is not None]
    zero_drift = [r for r in drift_records if r.answer_changed_count == 0]
    nonzero_drift = [r for r in drift_records if r.answer_changed_count > 0]
    print(f"  Researches with answer drift data: {len(drift_records)}")
    print(f"  Zero answer drift: {len(zero_drift)}")
    print(f"  Non-zero answer drift: {len(nonzero_drift)}")
    if nonzero_drift:
        print("\n  Non-zero drift cases:")
        nonzero_drift.sort(key=lambda r: r.answer_changed_count or 0, reverse=True)
        for r in nonzero_drift[:15]:
            print(f"    [{r.ticket_number or '?'}] drift={r.answer_changed_count}  {r.dir_name}")

    # -------------------------------------------------------------------------
    # 9. LOCAL vs PLATFORM COMPARISON RESEARCH
    # -------------------------------------------------------------------------
    print_section("9. LOCAL EVAL vs PLATFORM SCORE RESEARCH")
    local_platform = [r for r in records if r.local_vs_platform]
    if local_platform:
        for r in local_platform:
            print(f"  [{r.ticket_number or '?'}] {r.dir_name}")
            if r.verdict_text:
                for line in r.verdict_text.split("\n")[:5]:
                    if line.strip():
                        print(f"    > {line.strip()}")
            print()
    else:
        print("  No explicit local-vs-platform comparison research found.")
        # Try broader search
        print("  Searching for implicit comparisons...")
        for r in records:
            t = r.raw_text.lower()
            if any(
                kw in t
                for kw in [
                    "did not transfer",
                    "did not reproduce",
                    "local replay uplift",
                    "overfit",
                    "leaderboard",
                    "public score",
                    "warmup score",
                ]
            ):
                print(f"    [{r.ticket_number or '?'}] {r.dir_name}")
                # Find the relevant line
                for line in r.raw_text.split("\n"):
                    ll = line.lower()
                    if any(
                        kw in ll
                        for kw in [
                            "did not transfer",
                            "did not reproduce",
                            "overfit",
                            "leaderboard",
                            "public score",
                        ]
                    ):
                        print(f"      > {line.strip()}")
                print()

    # -------------------------------------------------------------------------
    # 10. SUBMISSION VERSION REFERENCES
    # -------------------------------------------------------------------------
    print_section("10. SUBMISSION VERSION REFERENCES")
    version_mentions = Counter()
    version_details = defaultdict(list)
    for r in records:
        for v in r.submission_versions:
            version_mentions[v] += 1
            version_details[v].append(r.dir_name)

    for v in sorted(version_mentions.keys(), key=lambda x: int(x)):
        count = version_mentions[v]
        print(f"  v{v}: {count} mentions")
        for d in version_details[v][:3]:
            print(f"    - {d}")
        if len(version_details[v]) > 3:
            print(f"    ... and {len(version_details[v]) - 3} more")

    # -------------------------------------------------------------------------
    # 11. KEY PATTERNS AND INSIGHTS
    # -------------------------------------------------------------------------
    print_section("11. KEY PATTERNS AND ACTIONABLE INSIGHTS")

    # Regression patterns
    regression_topics = Counter()
    for r in records:
        if r.status in ("KILLED", "NO_GO"):
            for t in r.topics:
                regression_topics[t] += 1

    print("  TOPIC KILL RATES (topics that failed most often):")
    for topic, total in topic_counts.most_common(15):
        failures = topic_statuses[topic].get("NO_GO", 0) + topic_statuses[topic].get("KILLED", 0)
        if total >= 3:
            rate = 100 * failures / total
            print(f"    {topic:<20} {failures:>3}/{total:<3} killed/no-go ({rate:.0f}%)")

    print()

    # Score patterns - which score ranges correlated with promotion
    print("  REVIEWED SCORE RANGES IN PROMOTED vs KILLED:")
    for status_group, label in [("PROMOTED", "Promoted"), ("KILLED", "Killed"), ("NO_GO", "No-Go")]:
        group = [r for r in records if r.status == status_group]
        all_100_scores = []
        high_81_scores = []
        for r in group:
            for slabel, val in r.scores:
                if "all_100" in slabel or "all 100" in slabel:
                    all_100_scores.append(val)
                elif "high_81" in slabel or "high 81" in slabel or "high_confidence" in slabel:
                    high_81_scores.append(val)
        if all_100_scores:
            print(
                f"    {label} all_100: min={min(all_100_scores):.4f} "
                f"max={max(all_100_scores):.4f} avg={sum(all_100_scores)/len(all_100_scores):.4f} "
                f"(n={len(all_100_scores)})"
            )
        if high_81_scores:
            print(
                f"    {label} high_81: min={min(high_81_scores):.4f} "
                f"max={max(high_81_scores):.4f} avg={sum(high_81_scores)/len(high_81_scores):.4f} "
                f"(n={len(high_81_scores)})"
            )

    print()

    # Common kill reasons from verdict text
    print("  COMMON KILL/NO-GO REASON KEYWORDS:")
    kill_reason_keywords = Counter()
    reason_patterns = [
        ("regression on reviewed", r"regress.*review"),
        ("did not improve grounding", r"did not.*improv|no.*grounding.*lift"),
        ("answer drift", r"answer.*drift|answer.*changed"),
        ("overfit to local", r"overfit|did not transfer"),
        ("latency only win", r"latency.*win|speed.*win|only.*fast"),
        ("lineage debt", r"lineage.*debt|provenance.*incomplete"),
        ("no replay winner", r"no.*replay.*winner|replay.*no.*go"),
        ("precision drop", r"precision.*drop|precision.*loss"),
    ]
    for r in killed_nogo:
        t = r.raw_text.lower()
        for reason_label, pattern in reason_patterns:
            if re.search(pattern, t):
                kill_reason_keywords[reason_label] += 1

    for reason, count in kill_reason_keywords.most_common():
        print(f"    {reason:<35} {count:>3} occurrences")

    # -------------------------------------------------------------------------
    # 12. HIGHEST SCORING REVIEWED GROUNDING
    # -------------------------------------------------------------------------
    print_section("12. HIGHEST REVIEWED GROUNDING SCORES (all_100)")
    all_100_entries = []
    for r in records:
        for label, val in r.scores:
            if "all_100" in label or "all 100" in label:
                all_100_entries.append((val, r.dir_name, r.ticket_number, r.status))
    all_100_entries.sort(reverse=True)
    for val, dname, ticket, status in all_100_entries[:20]:
        print(f"  {val:.4f}  [{ticket or '?':>5}] ({status or '?':<10})  {dname}")

    # -------------------------------------------------------------------------
    # 13. TICKET NUMBER DENSITY
    # -------------------------------------------------------------------------
    print_section("13. TICKET DENSITY (tickets with most research)")
    ticket_counts = Counter()
    for r in records:
        if r.ticket_number:
            ticket_counts[r.ticket_number] += 1
    for ticket, count in ticket_counts.most_common(20):
        print(f"  Ticket {ticket}: {count} closeouts")

    # -------------------------------------------------------------------------
    # 14. EMBEDDING DIMENSION FINDINGS
    # -------------------------------------------------------------------------
    print_section("14. EMBEDDING DIMENSION & RETRIEVAL FINDINGS")
    for r in records:
        t = r.raw_text.lower()
        if "1792" in t or "dimension" in t or "embed" in dir_name.lower() if (dir_name := r.dir_name) else False:
            if r.verdict_text:
                print(f"  [{r.ticket_number or '?'}] {r.dir_name}")
                for line in r.verdict_text.split("\n")[:4]:
                    if line.strip():
                        print(f"    > {line.strip()}")
                print()


if __name__ == "__main__":
    main()
