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
4. Parsers cache results in memory keyed by directory mtime — rescan only on change
5. Frontend fetches JSON, Plotly.js renders interactive charts
6. Locale toggle swaps all visible labels via `i18n/{lang}.json` without reload

### API Endpoints

| Endpoint | Source Files | Returns |
|----------|-------------|---------|
| `GET /api/eval/timeline` | `data/eval_*.json` | Array of `{timestamp, scores_by_type, latency, compliance}` |
| `GET /api/eval/latest` | Latest eval JSON | Full breakdown: KPI cards, gap bars, worst cases |
| `GET /api/judge/timeline` | `data/judge_*.jsonl` | `{timestamp, pass_rate, avg_accuracy, avg_grounding, avg_clarity}[]` |
| `GET /api/judge/latest` | Latest judge JSONL | Per-case detail with verdict, scores, worst grounding |
| `GET /api/benchmark/timeline` | `data/page_benchmark_*.md` | `{timestamp, f_beta, slot_recall, orphan_rate}[]` |
| `GET /api/matrix` | `data/competition_matrix.json` | Leaderboard state, gap targets, rank history |
| `GET /api/research/tickets` | `.sdd/researches/*/closeout.md` | `{ticket_id, title, date, status, impact_summary}[]` |
| `GET /api/research/experiments` | `.sdd/researches/*.json` | Version comparison summaries, delta scores |

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
- Right panel (flex): Horizontal bar chart of F-beta by answer type from latest eval, color-coded red→orange→green

**Row 3 — Diagnostic detail:**
- Left: Ticket impact sparkline — bar chart of recent tickets colored green (improvement) / red (regression) by G-score delta
- Right: Worst grounding cases table — QID, type, F-beta, match status from latest eval

**ROI computation:** `roi = (1 - avg_fbeta) * question_count * improvability` where improvability is 1.0 for types with answer match=True but F-beta=0 (pure grounding problem), 0.5 for types with some mismatches, 0.2 for free_text (structurally hard).

### Tab 2: "Score Timeline"

**Main chart:** Multi-line Plotly time series showing key metrics over time:
- G-score (primary, bold line)
- S-score, T-score, F-score (secondary, thinner)
- Total score (dashed)

**Below:** Stacked area or grouped bar chart of F-beta by answer type over time — shows which types are improving/regressing.

**Annotations:** Vertical dashed lines at significant events (labeled eval runs, ticket completions) parsed from filename suffixes.

**Interactions:** Hover for exact values, click a point to see that eval's full breakdown in a side panel.

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

## 7. Startup & Usage

```bash
# From project root:
python dashboard/server.py
# Opens at http://localhost:8050
```

No build step, no npm, no dependencies beyond FastAPI + uvicorn (already in project).
Plotly.js loaded from CDN in index.html.

## 8. Blog Export Path (future)

Each Plotly chart supports `Plotly.toImage()` for static PNG/SVG export. The `/api/*` endpoints return clean JSON that can be re-rendered in any frontend. The eventual blog integration would either:
- Embed static chart images exported from the dashboard
- Or use the same API endpoints from a blog component that renders Plotly charts

Not in scope for v1 — just noting the path is clean.

## 9. Out of Scope

- Real-time WebSocket updates (page refresh is fine)
- User authentication
- Database storage (files on disk are the source of truth)
- Editing/modifying data through the dashboard
- The `data/external/` and `data/derived/` directories (ML training data, not measurement)
- The 558 `code_archive_*.json` files in `platform_runs/warmup/` (code snapshots, not metrics)

## 10. File Cleanup Context

The user also asked to "навести порядок" (tidy up) in data/, platform_runs/, .sdd/researches/. The dashboard itself is the primary deliverable. File cleanup (removing stale tmp files, organizing naming) can happen as a separate follow-up task after the dashboard is working, since the dashboard's parsers will define what file patterns matter.
