# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    team_name: str
    total: float
    det: float
    asst: float
    g: float
    t: float
    f: float
    latency_ms: int
    submissions: int
    last_submission: str

    @property
    def s(self) -> float:
        return (0.7 * self.det) + (0.3 * self.asst)

    @property
    def total_recomputed(self) -> float:
        return self.s * self.g * self.t * self.f


@dataclass(frozen=True)
class GapTarget:
    rank: int
    team_name: str
    total: float
    target_g_at_current_s: float
    delta_g_at_current_s: float
    target_s_at_current_g: float
    delta_s_at_current_g: float


def load_rows(path: Path) -> list[LeaderboardRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[LeaderboardRow] = []
        for raw in reader:
            rows.append(
                LeaderboardRow(
                    rank=int(str(raw["Rank"]).strip()),
                    team_name=str(raw["Team name"]).strip(),
                    total=float(str(raw["Total score"]).strip()),
                    det=float(str(raw["Det"]).strip()),
                    asst=float(str(raw["Asst"]).strip()),
                    g=float(str(raw["G"]).strip()),
                    t=float(str(raw["T"]).strip()),
                    f=float(str(raw["F"]).strip()),
                    latency_ms=int(float(str(raw["Latency"]).strip())),
                    submissions=int(str(raw["Submissions"]).strip()),
                    last_submission=str(raw["Last submission"]).strip(),
                )
            )
    return rows


def _find_team(rows: list[LeaderboardRow], team_name: str) -> LeaderboardRow:
    for row in rows:
        if row.team_name == team_name:
            return row
    raise ValueError(f"Team not found in leaderboard: {team_name}")


def _max_total_error(rows: list[LeaderboardRow]) -> float:
    return max(abs(row.total - row.total_recomputed) for row in rows) if rows else 0.0


def _gap_targets(rows: list[LeaderboardRow], subject: LeaderboardRow) -> list[GapTarget]:
    targets: list[GapTarget] = []
    subject_stf = subject.s * subject.t * subject.f
    subject_gtf = subject.g * subject.t * subject.f
    for row in rows:
        if row.rank >= subject.rank:
            continue
        target_g = row.total / subject_stf if subject_stf > 0 else float("inf")
        target_s = row.total / subject_gtf if subject_gtf > 0 else float("inf")
        targets.append(
            GapTarget(
                rank=row.rank,
                team_name=row.team_name,
                total=row.total,
                target_g_at_current_s=target_g,
                delta_g_at_current_s=target_g - subject.g,
                target_s_at_current_g=target_s,
                delta_s_at_current_g=target_s - subject.s,
            )
        )
    return targets


def _sensitivity(row: LeaderboardRow) -> tuple[float, float]:
    delta_total_per_g_point = row.s * row.t * row.f
    delta_total_per_s_point = row.g * row.t * row.f
    return delta_total_per_g_point, delta_total_per_s_point


def _perfect_s_total(row: LeaderboardRow) -> float:
    return 1.0 * row.g * row.t * row.f


def build_report(rows: list[LeaderboardRow], *, team_name: str) -> str:
    subject = _find_team(rows, team_name)
    gaps = _gap_targets(rows, subject)
    max_error = _max_total_error(rows)
    d_total_per_g, d_total_per_s = _sensitivity(subject)
    perfect_s_total = _perfect_s_total(subject)

    lines = [
        "# Leaderboard Geometry Report",
        "",
        "## Formula Check",
        "",
        "- Assumed formula: `S = 0.7*Det + 0.3*Asst`",
        "- Assumed total: `Total = S * G * T * F`",
        f"- Max absolute recompute error on snapshot: `{max_error:.6f}`",
        "",
        "## Subject Team",
        "",
        f"- Team: `{subject.team_name}`",
        f"- Rank: `{subject.rank}`",
        f"- Total: `{subject.total:.6f}`",
        f"- S: `{subject.s:.6f}`",
        f"- Det: `{subject.det:.6f}`",
        f"- Asst: `{subject.asst:.6f}`",
        f"- G: `{subject.g:.6f}`",
        f"- T: `{subject.t:.6f}`",
        f"- F: `{subject.f:.6f}`",
        f"- Latency: `{subject.latency_ms} ms`",
        f"- Warm-up submissions used: `{subject.submissions}`",
        "",
        "## Sensitivity",
        "",
        f"- `+0.01 G` -> `+{d_total_per_g * 0.01:.6f}` total",
        f"- `+0.01 S` -> `+{d_total_per_s * 0.01:.6f}` total",
        f"- Perfect `S=1.0` at current `G/T/F` -> `{perfect_s_total:.6f}` total",
        "",
        "## Gap To Higher Ranks",
        "",
    ]
    if not gaps:
        lines.append("- Already rank 1")
    else:
        for gap in gaps:
            lines.extend(
                [
                    f"- Rank `{gap.rank}` `{gap.team_name}` total `{gap.total:.6f}`",
                    f"  - Need `G={gap.target_g_at_current_s:.6f}` at current `S/T/F` (`ΔG={gap.delta_g_at_current_s:+.6f}`)",
                    f"  - Need `S={gap.target_s_at_current_g:.6f}` at current `G/T/F` (`ΔS={gap.delta_s_at_current_g:+.6f}`)",
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `G` is the dominant lever from this state."
                if d_total_per_g >= d_total_per_s
                else "- `S` is the dominant lever from this state."
            ),
            (
                "- Public #1 is not reachable by exactness-only at the current `G/T/F`."
                if perfect_s_total + 1e-9 < rows[0].total
                else "- Public #1 is theoretically reachable without changing `G`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary(rows: list[LeaderboardRow], *, team_name: str) -> dict[str, object]:
    subject = _find_team(rows, team_name)
    d_total_per_g, d_total_per_s = _sensitivity(subject)
    return {
        "team_name": subject.team_name,
        "rank": subject.rank,
        "total": subject.total,
        "s": subject.s,
        "g": subject.g,
        "t": subject.t,
        "f": subject.f,
        "latency_ms": subject.latency_ms,
        "submissions": subject.submissions,
        "perfect_s_total": _perfect_s_total(subject),
        "delta_total_per_g_0_01": d_total_per_g * 0.01,
        "delta_total_per_s_0_01": d_total_per_s * 0.01,
        "gap_targets": [asdict(target) for target in _gap_targets(rows, subject)],
        "max_recompute_error": _max_total_error(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze leaderboard score geometry for a target team.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.leaderboard)
    report = build_report(rows, team_name=args.team)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)

    if args.json_out is not None:
        args.json_out.write_text(
            json_dumps(build_summary(rows, team_name=args.team)),
            encoding="utf-8",
        )


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


if __name__ == "__main__":
    main()
