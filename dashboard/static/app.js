/* Tzur Labs — RAG Challenge Dashboard */
"use strict";

let _allBulletin = [];
let _refreshTimer = null;

// ── Main Controller ───────────────────────────────────────────────────────────

async function loadAll() {
  try {
    const _j = r => r.json().then(d => d.data ?? d);
    const [agents, bulletin, scores, notes] = await Promise.all([
      fetch("/api/team/agents").then(_j),
      fetch("/api/team/bulletin").then(_j),
      fetch("/api/team/scores").then(_j),
      fetch("/static/status_notes.json").then(_j).catch(() => []),
    ]);

    _allBulletin = bulletin;

    // Header
    const cb = scores.current_best || {};
    const ttd = scores.time_to_deadline || "?";
    document.getElementById("current-best-label").textContent = `Best: ${cb.version || "Unknown"} (${cb.total || 0}) | Deadline: ${ttd}`;
    document.getElementById("global-ts").textContent = scores.last_updated || new Date().toISOString();

    renderSystemHealth(agents);
    renderKPIs(scores);
    renderDiagnostics(scores);
    renderAgents(agents);
    renderStatusNotes(notes);
    renderBulletin(bulletin);

  } catch (e) {
    console.error("loadAll error:", e);
  }
}

// ── System Health ─────────────────────────────────────────────────────────────

function renderSystemHealth(agents) {
  const el = document.getElementById("system-health");
  if (!el) return;

  const cocky = agents.find(a => a.name === "cocky");
  const isRegression = cocky && (cocky.current_task || "").includes("CRITICAL");

  el.style.display = "inline-block";
  if (isRegression) {
    el.className = "status-badge alert";
    el.textContent = "REGRESSION ALERT";
  } else {
    el.className = "status-badge stable";
    el.textContent = "PIPELINE STABLE";
  }
}

// ── KPIs (Top Strip) ──────────────────────────────────────────────────────────

function renderKPIs(scores) {
  const el = document.getElementById("kpi-strip");
  if (!el) return;

  const cb = scores.current_best || {};
  const run = scores.run_900q || {};

  const kpis = [
    {
      label: "Projected Total",
      value: (cb.total || 0).toFixed(3),
      sub: `vs Leader: ${scores.honest_assessment?.leader_total || 0}`,
    },
    {
      label: "G Proxy (Grounding)",
      value: (cb.g_proxy || 0).toFixed(4),
      sub: cb.g_proxy >= 0.99 ? "Excellent" : "Warning",
    },
    {
      label: "F Coefficient",
      value: (cb.f || 0).toFixed(3),
      sub: `Target: 1.050`,
    },
    {
      label: "Avg TTFT",
      value: `${run.ttft_avg_ms || 0} ms`,
      sub: `Over 5s: ${run.over_5s || 0}`,
    }
  ];

  el.innerHTML = kpis.map(k => `
    <div class="kpi-item">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-sub">${k.sub}</div>
    </div>
  `).join("");
}

// ── Diagnostics Table ─────────────────────────────────────────────────────────

function renderDiagnostics(scores) {
  const el = document.getElementById("diag-table-body");
  if (!el) return;

  const run = scores.run_900q || {};
  const ingest = scores.ingest || {};

  const rows = [
    { label: "Null Answers", value: run.nulls || 0, status: run.nulls === 0 ? "good" : (run.nulls < 5 ? "warn" : "crit") },
    { label: "Missing Pages (nopg)", value: run.no_pages || 0, status: run.no_pages < 5 ? "good" : "crit" },
    { label: "Evaluation Progress", value: `${scores.eval_progress?.answered || 0} / 900`, status: "good" },
    { label: "Documents Indexed", value: ingest.docs_indexed || "300/300", status: "good" },
    { label: "Det Proxy", value: scores.current_best?.det || "0.95", status: "warn" },
  ];

  el.innerHTML = rows.map(r => `
    <tr>
      <th>${r.label}</th>
      <td class="status-${r.status}">${r.value}</td>
    </tr>
  `).join("");
}

// ── Fleet Operations (Agents) ─────────────────────────────────────────────────

function renderAgents(agents) {
  const el = document.getElementById("agent-grid");
  if (!el) return;

  const now = new Date();

  el.innerHTML = agents.map(a => {
    const st = a.status || "unknown";
    const stClass = (st === "working" || st === "active") ? "active" : (st === "dead" || st === "idle") ? "error" : "";
    
    let timeAgo = "";
    if (a.timestamp) {
      try {
        const ts = new Date(a.timestamp);
        const diffSec = Math.floor((now - ts) / 1000);
        if (diffSec < 60) timeAgo = "now";
        else if (diffSec < 3600) timeAgo = `${Math.floor(diffSec / 60)}m`;
        else if (diffSec < 86400) timeAgo = `${Math.floor(diffSec / 3600)}h`;
      } catch(e) {}
    }

    const finding = (a.recent_findings && a.recent_findings.length > 0) ? a.recent_findings[0] : "";

    return `
      <div class="agent-row">
        <div class="agent-name">${a.name}</div>
        <div class="agent-status-text ${stClass}">${st} ${timeAgo ? `(${timeAgo})` : ""}</div>
        <div class="agent-metric">${a.done || 0} done</div>
        <div>
          <div class="agent-task">${a.current_task || a.next_task || "-"}</div>
          ${finding ? `<div class="agent-findings">${finding}</div>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

// ── Command Log (IGGY) ────────────────────────────────────────────────────────

function renderStatusNotes(notes) {
  const el = document.getElementById("iggy-status-body");
  if (!el) return;

  if (!notes || notes.length === 0) {
    el.innerHTML = `<div class="log-msg">No entries...</div>`;
    return;
  }

  el.innerHTML = notes.map(n => {
    const ts = (n.ts || "").replace("T", " ").replace("Z", "");
    return `
      <div class="log-entry log-iggy">
        <div class="log-ts">${ts.slice(-5)}</div>
        <div class="log-author">${n.author}</div>
        <div class="log-msg">${n.note}</div>
      </div>
    `;
  }).join("");
}

// ── System Bulletin ───────────────────────────────────────────────────────────

function renderBulletin(entries) {
  const el = document.getElementById("bulletin-body");
  if (!el) return;

  const reversed = [...entries].reverse();

  el.innerHTML = reversed.map(e => {
    const ts = (e.timestamp || "").slice(11, 16);
    return `
      <div class="log-entry">
        <div class="log-ts">${ts}</div>
        <div class="log-author">${e.from || "?"}</div>
        <div class="log-msg">${e.message || ""}</div>
      </div>
    `;
  }).join("");
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadAll();
setInterval(loadAll, 60000);
