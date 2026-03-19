# RAG Challenge Dashboard — Design Spec

**Date:** 2026-03-19
**Status:** Draft
**Audience:** Personal daily driver; later blog-exportable
**Approach:** Python FastAPI backend + Plotly.js frontend (no build step)

---

## 1. Problem

Three directories (`data/`, `platform_runs/`, `.sdd/researches/`) contain 20,000+ files spanning eval scores, judge results, page benchmarks, competition leaderboard data, ticket closeouts, and experiment comparisons. The existing `data/dash.html` only covers eval JSON latency/quality with Chart.js. There's no unified view for deciding what to work on next.

## 2. Design Principles

From user's BI storytelling philosophy:
- **One headline KPI, one supporting trend, one diagnostic layer, one action**
- Dashboards are decision surfaces, not data museums
- Every chart must answer: "what should I do differently?"
- EN/RU language toggle (personal use now, blog-ready later)

## 3. Architecture

```
dashboard/
├── server.py              # FastAPI app — mounts static + API routes
├── parsers/
│   ├── __init__.py
│   ├── eval_parser.py     # data/eval_*.json → score time series
│   ├── judge_parser.py    # data/judge_*.jsonl → per-case judge scores
│   ├── benchmark_parser.py # data/page_benchmark_*.md → F-beta series
│   ├── matrix_parser.py   # data/competition_matrix.json → leaderboard snapshot
│   └── research_parser.py # .sdd/researches/**/closeout.md + JSON → ticket impact
├── static/
│   ├── index.html         # SPA shell with tab navigation
│   ├── app.js             # Tab routing, Plotly chart rendering, i18n
│   ├── style.css          # Dark theme (GitHub-dark palette #0d1117)
│   └── i18n/
│       ├── en.json        # English labels
│       └── ru.json        # Russian labels
```

### Data flow

1. User runs `python -m dashboard.server` (or `python dashboard/server.py`)
2. FastAPI serves `static/index.html` + mounts `/api/*` routes
3. Each API call triggers the relevant parser to scan files on disk
4. Parsers cache results in memory keyed by `max(file.stat().st_mtime for file in glob)` — rescan when any file in the glob set changes (not directory mtime, which doesn't update on content-only changes on macOS)
5. Frontend fetches JSON, Plotly.js renders interactive charts
6. Locale toggle swaps all visible labels via `i18n/{lang}.json` without reload

### API Endpoints

| Endpoint | Source Files | Returns |
|----------|-------------|---------|
| `GET /api/eval/timeline` | `data/eval_*.json` | `{timestamp, label, citation_coverage, doc_ref_hit_rate, format_compliance, ttft_p50, ttft_p95, ttft_by_type: {type: {p50, p95, count}}, compliance_by_type: {type: float}}[]` |
| `GET /api/eval/latest` | Latest eval JSON | Same shape as one timeline entry, plus per-case detail from the JSON |
| `GET /api/judge/timeline` | `data/judge_*.jsonl` | `{timestamp, label, pass_rate, n_cases, avg_accuracy, avg_grounding, avg_clarity, avg_uncertainty}[]` |
| `GET /api/judge/latest` | Latest judge JSONL | Per-case array: `{case_id, question_id, answer_type, verdict, accuracy, grounding, clarity}[]` |
| `GET /api/benchmark/timeline` | `data/page_benchmark_*.md` | `{timestamp, label, f_beta, orphan_rate, slot_recall, overprune_violations, worst_cases: [...]}[]` |
| `GET /api/matrix` | `data/competition_matrix.json` | `{team, rank, total, s, g, t, f, latency_ms, submissions, gap_targets: [...]}` |
| `GET /api/scores/timeline` | `data/competition_matrix.json` (rows array) + `.sdd/researches/**/platform_scoring_*.json` | `{timestamp, label, total, s, g, t, f}[]` — G/S/T/F over time from platform scoring artifacts |
| `GET /api/research/tickets` | `.sdd/researches/*/closeout.md` | `{ticket_id, title, date, status, impact_summary}[]` |
| `GET /api/research/experiments` | `.sdd/researches/*.json` | Version comparison summaries, delta scores |

#### Data provenance notes

- **F-beta by answer type** is available in `data/page_benchmark_*.md` files (page-level F-beta) and in golden-label scoring reports (stored as `data/eval_*` or `/tmp/` artifacts). The eval JSON files do **not** contain F-beta — they contain `citation_coverage`, `doc_ref_hit_rate`, `format_compliance`, and latency. For the "Where to Invest" gap bars, the benchmark parser extracts F-beta from the markdown. For per-type breakdown, we join benchmark worst-case data (which includes answer type) with judge per-case data (which has answer_type per question).
- **G/S/T/F scores** live in `competition_matrix.json` (current snapshot) and in `.sdd/researches/**/platform_scoring_*.json` files (historical). They do **not** come from eval JSONs. The `/api/scores/timeline` endpoint aggregates these.
- **Judge scores** (accuracy, grounding, clarity, uncertainty_handling) come from `judge_*.jsonl` files, not eval JSONs. These are the per-case quality signals for the "System Health" tab.

### Timestamp parsing

Eval/judge filenames use `ddmmyyyy_hhmm` format (e.g., `eval_01032026_2221.json` → `2026-03-01T22:21`). Some have label suffixes after the timestamp (e.g., `_rule_alignment_lock`). Parser extracts the first `\d{8}_\d{4}` match.

Benchmark/projection MDs use `ddmmyyyy_hhmmss` (e.g., `page_benchmark_09032026_220628_hidden_g_lockdown.md`).

Research ticket dirs use ISO date suffix (e.g., `620_public100_reviewed_gate_r1_2026-03-19`).

## 4. Tab Design

### Tab 1: "Where to Invest" (hero page)

Three-row layout following the BI narrative layers:

**Row 1 — What changed + What to do:**
- 2 KPI cards: Total Score (with delta vs previous), G-Score (with rank)
- 1 wide action card: "Recommended next action" — computed as the answer type with highest ROI (gap × question_count × improvability heuristic)

**Row 2 — Why + Where:**
- Left panel (220px): ROI-ranked investment map — answer types sorted by ROI score, with progress bars and counts
- Right panel (flex): Horizontal bar chart of judge avg grounding score by answer type (from latest judge JSONL), color-coded red→orange→green. If a page benchmark is available with the same timestamp, overlay F-beta markers.

**Row 3 — Diagnostic detail:**
- Left: Ticket impact sparkline — bar chart of recent tickets colored green (improvement) / red (regression) by G-score delta
- Right: Worst grounding cases table — QID, type, F-beta, match status from latest eval

**ROI computation:** `roi = (1 - avg_grounding_score/5) * question_count * improvability` where `avg_grounding_score` is the judge grounding average (0-5 scale) for that answer type, and `improvability` is a static heuristic: 1.0 for boolean/number (discrete, fixable with better page selection), 0.7 for name/names (entity-dependent), 0.2 for free_text (structurally hard, LLM-quality bound).

### Tab 2: "Score Timeline"

**Main chart:** Multi-line Plotly time series from `/api/scores/timeline` (sourced from `competition_matrix.json` rows and `.sdd/researches/**/platform_scoring_*.json`):
- G-score (primary, bold line)
- S-score, T-score, F-score (secondary, thinner)
- Total score (dashed)

**Below:** Grouped bar chart of judge avg grounding by answer type over time (from `/api/judge/timeline`) — shows which types are improving/regressing.

**Annotations:** Vertical dashed lines at labeled eval runs parsed from filename suffixes (e.g., `_rule_alignment_lock`).

**Interactions:** Hover for exact values, click a point to see that run's full breakdown in a side panel.

### Tab 3: "System Health"

**Row 1:** Latency trend — P50 and P95 TTFT over time (Plotly line chart), broken down by answer type on hover.

**Row 2:** Quality gauges — format compliance rate, citation coverage, doc_ref_hit_rate as time series.

**Row 3:** Judge score trends — avg accuracy, grounding, clarity, uncertainty_handling as stacked area chart. Pass rate as overlay line.

### Tab 4: "Research Explorer"

**Left sidebar:** Scrollable list of ticket directories from `.sdd/researches/`, grouped by date, showing ticket number + short title extracted from directory name. Search/filter box at top.

**Main area:** When a ticket is selected, shows:
- Closeout summary (parsed from `closeout.md` if present)
- Key metrics from any JSON files in the directory
- Before/after comparison if version comparison data exists

This tab is lower priority — can be a simple file browser initially.

## 5. i18n Strategy

Two JSON locale files with identical keys:

```json
// en.json
{
  "tab.invest": "Where to Invest",
  "tab.timeline": "Score Timeline",
  "tab.health": "System Health",
  "tab.research": "Research Explorer",
  "kpi.total": "Total Score",
  "kpi.gscore": "G-Score",
  "kpi.rank": "Rank",
  "action.title": "Recommended Next Action",
  "invest.roi_map": "ROI-Ranked Investment Map",
  "invest.gap_bars": "F-beta by Answer Type",
  "invest.ticket_impact": "Ticket Impact on G-Score",
  "invest.worst_cases": "Worst Grounding Cases",
  ...
}
```

```json
// ru.json
{
  "tab.invest": "Куда инвестировать",
  "tab.timeline": "Динамика оценок",
  "tab.health": "Здоровье системы",
  "tab.research": "Исследования",
  "kpi.total": "Общий балл",
  "kpi.gscore": "G-балл",
  "kpi.rank": "Место",
  "action.title": "Рекомендуемое действие",
  ...
}
```

Toggle: `[EN | RU]` button in top-right header. Saves preference to `localStorage`. All Plotly axis labels and chart titles use locale keys. Data values (numbers, QIDs) stay untranslated.

## 6. Visual Design

- **Theme:** GitHub-dark palette (`#0d1117` background, `#161b22` cards, `#21262d` borders)
- **Colors:** Red `#f85149` (bad), Orange `#f0883e` (warning), Green `#3fb950` (good), Blue `#58a6ff` (accent/action)
- **Typography:** System monospace stack for data, system sans for labels
- **Layout:** Max-width 1200px centered, responsive flex rows that stack on narrow screens

## 7. Error Handling

- **Missing directories:** If `data/`, `platform_runs/`, or `.sdd/researches/` don't exist, the corresponding API endpoints return `{"data": [], "warning": "directory not found"}` with HTTP 200 (not 404 — the dashboard still loads, just with empty sections).
- **Malformed files:** Parsers wrap individual file parsing in try/except. A malformed eval JSON or benchmark MD is logged and skipped — never crashes the endpoint. The response includes a `skipped_files` count.
- **Empty data:** Frontend handles empty arrays gracefully — shows "No data yet" placeholder instead of broken charts.
- **Benchmark MD parsing:** These files have semi-structured format. Parser uses regex `r"F_beta\(2\.5\):\s*([\d.]+)"` and similar patterns. If the format drifts, the file is skipped (logged). Consider converting benchmarks to JSON at eval-generation time in the future.
- **Missing competition_matrix.json:** `/api/matrix` and `/api/scores/timeline` return empty data with a warning. KPI cards show "—" placeholders.

## 8. Startup & Usage

```bash
# From project root:
python dashboard/server.py
# Opens at http://localhost:8050
```

No build step, no npm, no dependencies beyond FastAPI + uvicorn (already in project).
Plotly.js loaded from CDN in index.html.

## 9. Blog Export Path (future)

Each Plotly chart supports `Plotly.toImage()` for static PNG/SVG export. The `/api/*` endpoints return clean JSON that can be re-rendered in any frontend. The eventual blog integration would either:
- Embed static chart images exported from the dashboard
- Or use the same API endpoints from a blog component that renders Plotly charts

Not in scope for v1 — just noting the path is clean.

## 10. Out of Scope

- Real-time WebSocket updates (page refresh is fine)
- User authentication
- Database storage (files on disk are the source of truth)
- Editing/modifying data through the dashboard
- The `data/external/` and `data/derived/` directories (ML training data, not measurement)
- The 558 `code_archive_*.json` files in `platform_runs/warmup/` (code snapshots, not metrics)

## 11. File Cleanup Context

The user also asked to "навести порядок" (tidy up) in data/, platform_runs/, .sdd/researches/. The dashboard itself is the primary deliverable. File cleanup (removing stale tmp files, organizing naming) can happen as a separate follow-up task after the dashboard is working, since the dashboard's parsers will define what file patterns matter.
