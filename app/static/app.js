"use strict";

/* ---------- tiny helpers ---------- */
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

let toastTimer = null;
function toast(msg, kind) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast" + (kind === "err" ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4000);
}

const fmtMs = (ms) => ms == null ? "—" : (ms >= 10000 ? (ms / 1000).toFixed(1) + "s" : ms + "ms");
const fmtTokens = (t) => t == null ? "—" : (t >= 1000 ? (t / 1000).toFixed(1) + "k" : String(t));
const fmtCost = (c) => c == null ? "—" : (c === 0 ? "$0" : (c < 0.01 ? "$" + c.toFixed(5) : "$" + c.toFixed(3)));
const fmtPct = (r) => r == null ? "—" : (r * 100).toFixed(1) + "%";
const fmtNum = (n) => n == null ? "—" : (Math.round(n * 10) / 10);
const fmtDur = (a, b) => {
  if (!a || !b) return "—";
  const s = (new Date(b) - new Date(a)) / 1000;
  if (s < 60) return Math.round(s) + "s";
  return Math.round(s / 60) + "m" + Math.round(s % 60) + "s";
};

const CHECK_LABELS = { basic: "Chat", streaming: "Streaming", tools: "Tools", json: "JSON" };
const MODEL_STATUS = {
  pending: ["muted", "antre"], running: ["muted", "jalan"], pass: ["ok", "lulus"],
  warn: ["warn", "peringatan"], fail: ["err", "gagal"], na: ["muted", "n/a"],
  skipped: ["muted", "skip"], error: ["err", "error"], done: ["ok", "selesai"],
  interrupted: ["muted", "terputus"],
};
const CHECK_STATUS = { pass: ["ok", "Lulus"], warn: ["warn", "Warn"], fail: ["err", "Gagal"], na: ["muted", "N/A"] };
const OUTCOMES = {
  completed: ["ok", "Selesai"], completed_incomplete: ["warn", "Tidak lengkap"],
  max_turns: ["warn", "Max turn"], token_cap: ["warn", "Token cap"],
  request_error: ["err", "Error request"], aborted: ["muted", "Dibatalkan"],
  skipped: ["muted", "Skip"], error: ["err", "Error"],
};
const KIND_LABELS = { functional: "Fungsional", bench: "Benchmark", load: "Load probe", context: "Context scaling" };
const CATEGORY_LABELS = {
  invalid_json_args: "JSON args rusak", unknown_tool: "tool tak dikenal",
  schema_violation: "skema salah", path_escape: "akses path", file_not_found: "file tak ada",
  edit_not_found: "edit tak cocok", edit_ambiguous: "edit ambigu",
  tool_execution_error: "eksekusi gagal", command_not_found: "perintah tak ada",
  command_timeout: "perintah timeout", gateway_error: "error gateway", http_error: "HTTP error",
};

function badge(kind, label) {
  const cls = kind || "muted";
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}
function kvBadge(map, key) {
  const [cls, label] = map[key] || ["muted", key || "?"];
  return badge(cls, label);
}

function regBadge(metrics) {
  const reg = metrics && metrics.regression;
  if (!reg) return "";
  return `<span class="badge err" title="vs baseline ${reg.baseline} → ${reg.current} (${reg.delta_pct > 0 ? "+" : ""}${reg.delta_pct}%)">⚠ regresi ${reg.delta_pct > 0 ? "+" : ""}${reg.delta_pct}%</span>`;
}

function sparkline(values, w = 110, h = 26) {
  const vals = values.filter((v) => v != null);
  if (vals.length < 2) return "";
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = (max - min) || 1;
  const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * w},${h - ((v - min) / range) * h}`).join(" ");
  return `<svg class="spark" width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"/></svg>`;
}

/* ---------- state ---------- */
const state = {
  status: null,
  models: [],
  modelsByName: {},
  tasks: [],
  selected: { functional: new Set(), bench: new Set(), perf: new Set() },
  funcRunId: null,
  benchRunId: null,
  perfRunId: null,
  perfKind: "load",
  rmIndex: {},
  pollTimer: null,
  polling: false,
};

const PICKERS = {
  functional: { sel: "#fn-picker" },
  bench: { sel: "#bench-picker", estimate: updateBenchEstimate },
  perf: { sel: "#perf-picker", estimate: updatePerfEstimate },
};

/* ---------- status / models / pickers ---------- */
async function refreshStatus() {
  try { state.status = await api("/api/status"); } catch (e) { state.status = null; }
  renderPill();
}

function renderPill() {
  const el = $("#gw-pill");
  const s = state.status;
  if (!s) { el.className = "pill muted"; el.textContent = "memuat…"; return; }
  const keyOk = s.admin_key_set || s.test_key_set;
  const label = `${s.gateway_url} · admin:${s.admin_key_set ? "✓" : "—"} · test:${s.test_key_set ? "✓" : "—"}`;
  if (!keyOk) { el.className = "pill warn"; el.textContent = "gateway belum dikonfigurasi"; el.title = label; }
  else if (s.models_cached > 0) { el.className = "pill ok"; el.textContent = `${s.models_cached} model · ${s.gateway_url}`; }
  else { el.className = "pill muted"; el.textContent = label; }
}

async function refreshModels() {
  try {
    const data = await api("/api/models?refresh=1");
    state.models = data.models || [];
  } catch (e) {
    state.models = [];
    toast(e.message, "err");
  }
  state.modelsByName = {};
  for (const m of state.models) state.modelsByName[m.name] = m;
  renderPickerList("functional");
  renderPickerList("bench");
  renderPickerList("perf");
  updateBenchEstimate();
  updatePerfEstimate();
  refreshStatus();
}

function isChatCandidate(m) { return m.mode == null || m.mode === "chat"; }

function isSelectable(m, name) {
  if (!isChatCandidate(m)) return false;
  if (m.blocked) return name === "functional" && $("#fn-include-blocked").checked;
  return true;
}

function buildPicker(name) {
  const root = $(PICKERS[name].sel);
  root.innerHTML = `
    <div class="picker-toolbar">
      <input class="picker-search" placeholder="Cari model…">
      <button class="btn tiny" data-act="all">Semua chat</button>
      <button class="btn tiny" data-act="none">Kosongkan</button>
      <span class="picker-count"></span>
    </div>
    <div class="picker-list"></div>`;
  root.querySelector(".picker-search").addEventListener("input", () => renderPickerList(name));
  root.querySelector('[data-act="all"]').addEventListener("click", () => {
    for (const m of state.models) {
      if (isChatCandidate(m) && !m.blocked) state.selected[name].add(m.name);
    }
    renderPickerList(name);
    if (PICKERS[name].estimate) PICKERS[name].estimate();
  });
  root.querySelector('[data-act="none"]').addEventListener("click", () => {
    state.selected[name].clear();
    renderPickerList(name);
    if (PICKERS[name].estimate) PICKERS[name].estimate();
  });
  renderPickerList(name);
}

function renderPickerList(name) {
  const root = $(PICKERS[name].sel);
  const q = (root.querySelector(".picker-search").value || "").toLowerCase();
  const list = root.querySelector(".picker-list");
  const selected = state.selected[name];
  const models = state.models.filter((m) => m.name.toLowerCase().includes(q));
  list.innerHTML = "";
  if (!state.models.length) {
    list.innerHTML = `<div class="muted" style="padding:10px">Belum ada model — isi koneksi gateway di Settings lalu Refresh.</div>`;
    root.querySelector(".picker-count").textContent = "";
    return;
  }
  for (const m of models) {
    const label = document.createElement("label");
    label.className = "pick" + (m.blocked ? " blocked" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selected.has(m.name);
    cb.disabled = !isSelectable(m, name);
    cb.addEventListener("change", () => {
      if (cb.checked) selected.add(m.name); else selected.delete(m.name);
      renderPickerList(name);
      if (PICKERS[name].estimate) PICKERS[name].estimate();
    });
    const tags = [];
    if (!isChatCandidate(m)) tags.push(`<span class="tag">non-chat</span>`);
    if (m.supports_tools) tags.push(`<span class="tag tools">tools</span>`);
    if (m.supports_json) tags.push(`<span class="tag tools">json</span>`);
    if (m.blocked) tags.push(`<span class="tag blocked">blocked</span>`);
    if (m.provider) tags.push(`<span class="tag">${esc(m.provider)}</span>`);
    label.innerHTML = `<span class="pname">${esc(m.name)}</span><span class="ptags">${tags.join("")}</span>`;
    label.insertBefore(cb, label.firstChild);
    list.appendChild(label);
  }
  root.querySelector(".picker-count").textContent = `${selected.size} dipilih`;
}

function modelCost(m, inTokens, outTokens) {
  if (m && m.input_cost_per_token != null && m.output_cost_per_token != null) {
    return inTokens * m.input_cost_per_token + outTokens * m.output_cost_per_token;
  }
  return null;
}

function updateBenchEstimate() {
  let total = null, missing = 0;
  for (const name of state.selected.bench) {
    const m = state.modelsByName[name];
    if (!m) continue;
    const c = modelCost(m, 25000, 4000);
    if (c != null) total = (total || 0) + c; else missing++;
  }
  $("#bench-estimate").textContent = total == null
    ? (missing ? "— (harga model tidak tersedia)" : "—")
    : "$" + total.toFixed(3) + (missing ? ` + ${missing} model tanpa harga` : "");
}

function updatePerfEstimate() {
  const mode = $("#perf-mode").value;
  let total = null, missing = 0;
  if (mode === "load") {
    const levels = parseCsv($("#perf-levels").value);
    const rpl = parseInt($("#perf-rpl").value, 10) || 8;
    const reqs = levels.length * rpl;
    for (const name of state.selected.perf) {
      const m = state.modelsByName[name];
      if (!m) continue;
      const c = modelCost(m, reqs * 8, reqs * 16);
      if (c != null) total = (total || 0) + c; else missing++;
    }
  } else {
    const sizes = parseCsv($("#perf-sizes").value);
    const sumSizes = sizes.reduce((a, b) => a + b, 0);
    for (const name of state.selected.perf) {
      const m = state.modelsByName[name];
      if (!m) continue;
      const c = modelCost(m, sumSizes, 16);
      if (c != null) total = (total || 0) + c; else missing++;
    }
  }
  $("#perf-estimate").textContent = total == null
    ? (missing ? "— (harga model tidak tersedia)" : "—")
    : "$" + total.toFixed(4) + (missing ? ` + ${missing} model tanpa harga` : "");
}

function parseCsv(s) {
  return (s || "").split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => !isNaN(n));
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "history") loadHistory();
  });
});

/* ---------- functional ---------- */
async function startFunctional() {
  const checks = [...document.querySelectorAll("#fn-checks input:checked")].map((i) => i.value);
  if (!checks.length) return toast("Pilih minimal satu jenis tes", "err");
  const includeBlocked = $("#fn-include-blocked").checked;
  const models = [...state.selected.functional];
  const body = {
    checks,
    include_blocked: includeBlocked,
    models: models.length ? models : null,
    repeat: parseInt($("#fn-repeat").value, 10) || 1,
    warmup: $("#fn-warmup").checked,
  };
  const btn = $("#fn-start");
  btn.disabled = true;
  try {
    const res = await api("/api/runs/functional", { method: "POST", body: JSON.stringify(body) });
    state.funcRunId = res.run_id;
    ensurePoll();
    toast("Tes fungsional dimulai");
  } catch (e) { toast(e.message, "err"); btn.disabled = false; }
}

function renderFunctionalRun(data) {
  const { run, models } = data;
  for (const row of models) state.rmIndex[row.id] = row;

  const done = models.filter((r) => !["pending", "running"].includes(r.status)).length;
  $("#fn-progress").innerHTML = run.status === "running"
    ? `<div class="progress"><span style="width:${models.length ? (done / models.length) * 100 : 0}%"></span></div>
       <div class="muted">${done}/${models.length} selesai…</div>` : "";

  const counts = {};
  for (const r of models) counts[r.status] = (counts[r.status] || 0) + 1;
  let chips = "";
  for (const [k, v] of Object.entries(counts)) chips += `<span class="chip ${MODEL_STATUS[k] ? MODEL_STATUS[k][0] : "muted"}">${v} ${k}</span>`;
  const cost = models.reduce((s, r) => s + (((r.metrics || {}).cost_usd) || 0), 0);
  if (run.summary && run.summary.regressions) chips += `<span class="chip err">${run.summary.regressions} regresi</span>`;
  $("#fn-summary").innerHTML = `<div class="summary">${chips}<span class="chip">biaya ${fmtCost(cost)}</span></div>`;

  $("#fn-table").innerHTML = functionalTableHtml(models, true);
  $("#fn-table").querySelectorAll("[data-fn]").forEach((b) =>
    b.addEventListener("click", () => openFunctionalDetail(Number(b.dataset.fn))));
  $("#fn-start").disabled = run.status === "running";
}

function functionalTableHtml(models, actions) {
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Model</th><th>Status</th>`;
  for (const c of ["basic", "streaming", "tools", "json"]) html += `<th>${CHECK_LABELS[c]}</th>`;
  html += `<th>Latency</th><th>p95</th><th>tok/s</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const checks = row.checks || {};
    const mm = row.metrics || {};
    const sc = checks.streaming || {};
    html += `<tr><td class="mono">${esc(row.model)} ${regBadge(mm)}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>`;
    for (const c of ["basic", "streaming", "tools", "json"]) {
      const r = checks[c];
      html += `<td>${r ? kvBadge(CHECK_STATUS, r.status).replace(">", ` title="${esc(r.error || r.note || "")}">`) : '<span class="badge muted">—</span>'}</td>`;
    }
    html += `<td>${fmtMs(mm.avg_latency_ms)}</td><td>${fmtMs(mm.p95_latency_ms)}</td><td>${fmtNum(sc.tok_per_s)}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-fn="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function checkDetailRow(name, r) {
  let body = `<div class="step"><div class="stephead"><strong>${CHECK_LABELS[name]}</strong> ${kvBadge(CHECK_STATUS, r.status)}
    <span class="muted">latency ${fmtMs(r.latency_ms)}${r.ttft_ms != null ? " · ttft " + fmtMs(r.ttft_ms) : ""}${r.generation_ms != null ? " · gen " + fmtMs(r.generation_ms) : ""}${r.http_status ? " · HTTP " + r.http_status : ""}</span></div>`;
  const bits = [];
  if (r.tok_per_s != null) bits.push(`tok/s <b>${fmtNum(r.tok_per_s)}</b>`);
  if (r.itl_ms != null) bits.push(`ITL <b>${fmtNum(r.itl_ms)} ms</b>`);
  if (r.cached_tokens != null) bits.push(`cached <b>${r.cached_tokens}</b>`);
  if (r.reasoning_tokens != null) bits.push(`reasoning <b>${r.reasoning_tokens}</b>`);
  if (r.p95_latency_ms != null) bits.push(`p95 <b>${fmtMs(r.p95_latency_ms)}</b>`);
  if (bits.length) body += `<div class="muted">${bits.join(" · ")}</div>`;
  if (r.error) body += `<div class="code">${esc(r.error)}</div>`;
  else if (r.note) body += `<div class="muted">${esc(r.note)}</div>`;
  if (r.samples && r.samples.length) {
    body += `<div class="muted" style="margin-top:6px">Sampel (${r.samples.length}):</div><div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>status</th><th>latency</th><th>ttft</th><th>tok/s</th></tr></thead><tbody>`;
    r.samples.forEach((s, i) => {
      body += `<tr><td>${i + 1}</td><td>${kvBadge(CHECK_STATUS, s.status)}</td><td>${fmtMs(s.latency_ms)}</td><td>${fmtMs(s.ttft_ms)}</td><td>${fmtNum(s.tok_per_s)}</td></tr>`;
    });
    body += `</tbody></table></div>`;
  }
  body += `</div>`;
  return body;
}

async function openFunctionalDetail(rmId) {
  const row = state.rmIndex[rmId];
  if (!row) return;
  const meta = state.modelsByName[row.model] || {};
  const checks = row.checks || {};
  const mm = row.metrics || {};
  let body = `<div class="chips">
      <span class="chipbox">Model<b class="mono">${esc(row.model)}</b></span>
      <span class="chipbox">Status<b>${MODEL_STATUS[row.status] ? MODEL_STATUS[row.status][1] : row.status}</b></span>
      <span class="chipbox">Provider<b>${esc(meta.provider || "—")}</b></span>
      <span class="chipbox">Latency<b>${fmtMs(mm.avg_latency_ms)}</b></span>
      <span class="chipbox">p95<b>${fmtMs(mm.p95_latency_ms)}</b></span>
      <span class="chipbox">Biaya<b>${fmtCost(mm.cost_usd)}</b></span>
      ${regBadge(mm)}
    </div><div id="detail-spark"></div>`;
  openModal(row.model, body);
  for (const c of ["basic", "streaming", "tools", "json"]) {
    if (checks[c]) $("#modal-body").insertAdjacentHTML("beforeend", checkDetailRow(c, checks[c]));
  }
  try {
    const hist = await api(`/api/models/history?model=${encodeURIComponent(row.model)}&kind=functional`);
    const vals = (hist.points || []).map((p) => p.value).reverse();
    const svg = sparkline(vals);
    if (svg) $("#detail-spark").innerHTML = `<div class="muted">Tren ${hist.metric}: ${svg}</div>`;
  } catch (e) { /* ignore */ }
}

/* ---------- benchmark ---------- */
async function loadTasks() {
  try {
    const data = await api("/api/tasks");
    state.tasks = data.tasks || [];
  } catch (e) { state.tasks = []; }
  const sel = $("#bench-task");
  sel.innerHTML = state.tasks.map((t) => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join("");
  if (state.tasks[0]) $("#bench-max-turns").value = state.tasks[0].max_turns || 30;
}

async function startBench() {
  const taskId = $("#bench-task").value;
  if (!taskId) return toast("Pilih task", "err");
  const models = [...state.selected.bench];
  if (!models.length) return toast("Pilih minimal satu model untuk benchmark", "err");
  const body = {
    task_id: taskId,
    models,
    max_turns: parseInt($("#bench-max-turns").value, 10) || 30,
    token_cap: parseInt($("#bench-token-cap").value, 10) || 150000,
    stream_turns: true,
  };
  const btn = $("#bench-start");
  btn.disabled = true;
  try {
    const res = await api("/api/runs/bench", { method: "POST", body: JSON.stringify(body) });
    state.benchRunId = res.run_id;
    ensurePoll();
    toast("Benchmark dimulai");
  } catch (e) { toast(e.message, "err"); btn.disabled = false; }
}

function renderBenchRun(data) {
  const { run, models } = data;
  for (const row of models) state.rmIndex[row.id] = row;

  const done = models.filter((r) => !["pending", "running"].includes(r.status)).length;
  $("#bench-progress").innerHTML = run.status === "running"
    ? `<div class="progress"><span style="width:${models.length ? (done / models.length) * 100 : 0}%"></span></div>
       <div class="muted">${done}/${models.length} selesai…</div>` : "";

  const outcomes = {};
  for (const r of models) { const o = r.outcome || r.status; outcomes[o] = (outcomes[o] || 0) + 1; }
  let chips = "";
  for (const [k, v] of Object.entries(outcomes)) chips += `<span class="chip ${OUTCOMES[k] ? OUTCOMES[k][0] : "muted"}">${v} ${OUTCOMES[k] ? OUTCOMES[k][1] : k}</span>`;
  const avg = run.summary ? run.summary.avg_error_rate : null;
  const cost = models.reduce((s, r) => s + (((r.metrics || {}).cost_usd) || 0), 0);
  if (run.summary && run.summary.regressions) chips += `<span class="chip err">${run.summary.regressions} regresi</span>`;
  $("#bench-summary").innerHTML = `<div class="summary">${chips}
    <span class="chip">rata-rata error rate ${fmtPct(avg)}</span>
    <span class="chip">biaya ${fmtCost(cost)}</span></div>`;

  $("#bench-table").innerHTML = benchTableHtml(models, true);
  $("#bench-table").querySelectorAll("[data-bench]").forEach((b) =>
    b.addEventListener("click", () => openBenchDetail(Number(b.dataset.bench))));
  $("#bench-start").disabled = run.status === "running";
}

function benchTableHtml(models, actions) {
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Model</th><th>Outcome</th><th>Tool calls</th><th>Error</th><th>Error rate</th><th>Recovery</th><th>TTFT</th><th>tok/s</th><th>Turn</th><th>Token</th><th>Biaya</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const m = row.metrics || {};
    const rate = m.error_rate;
    let bar = "";
    if (rate != null) {
      const pct = Math.min(100, rate * 100);
      const color = rate === 0 ? "var(--ok)" : (rate <= 0.1 ? "var(--warn)" : "var(--err)");
      bar = `<div class="ratebar"><span style="width:${pct}%;background:${color}"></span></div>`;
    }
    html += `<tr><td class="mono">${esc(row.model)} ${regBadge(m)}</td>
      <td>${kvBadge(OUTCOMES, row.outcome || row.status)}</td>
      <td>${m.tool_calls ?? "—"}</td>
      <td>${m.tool_errors ?? "—"}</td>
      <td>${bar} ${fmtPct(rate)}</td>
      <td>${fmtPct(m.recovery_rate)}</td>
      <td>${fmtMs(m.avg_turn_ttft_ms)}</td>
      <td>${fmtNum(m.gen_tok_per_s)}</td>
      <td>${m.turns ?? "—"}</td>
      <td>${fmtTokens(m.tokens_in)}→${fmtTokens(m.tokens_out)}</td>
      <td>${fmtCost(m.cost_usd)}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-bench="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

async function openBenchDetail(rmId) {
  const row = state.rmIndex[rmId];
  if (!row) return;
  openModal(row.model, `<div class="muted">Memuat trace…</div>`);
  try {
    const [detail, files] = await Promise.all([
      api(`/api/runs/${row.run_id}/models/${rmId}`),
      api(`/api/bench/${rmId}/files`),
    ]);
    const m = detail.model.metrics || {};
    let body = `<div class="chips">
      <span class="chipbox">Outcome<b>${OUTCOMES[detail.model.outcome] ? OUTCOMES[detail.model.outcome][1] : (detail.model.outcome || "—")}</b></span>
      <span class="chipbox">Tool calls<b>${m.tool_calls ?? "—"}</b></span>
      <span class="chipbox">Error<b>${m.tool_errors ?? "—"}</b></span>
      <span class="chipbox">Error rate<b>${fmtPct(m.error_rate)}</b></span>
      <span class="chipbox">Recovery<b>${fmtPct(m.recovery_rate)}</b></span>
      <span class="chipbox">TTFT/turn<b>${fmtMs(m.avg_turn_ttft_ms)}</b></span>
      <span class="chipbox">Latency/turn<b>${fmtMs(m.avg_turn_latency_ms)}</b></span>
      <span class="chipbox">tok/s<b>${fmtNum(m.gen_tok_per_s)}</b></span>
      <span class="chipbox">Model vs tool<b>${m.model_tool_ratio ?? "—"}</b></span>
      <span class="chipbox">Token<b>${fmtTokens(m.tokens_in)}→${fmtTokens(m.tokens_out)}</b></span>
      <span class="chipbox">Biaya<b>${fmtCost(m.cost_usd)}</b></span>
      <span class="chipbox">Durasi<b>${fmtDur(detail.model.started_at, detail.model.finished_at)}</b></span>
      ${m.stream_fallback ? '<span class="chipbox">Stream<b>fallback ' + m.stream_fallback + '×</b></span>' : ""}
      ${regBadge(m)}
    </div><div id="detail-spark"></div>`;

    const cats = m.by_category || {};
    if (Object.keys(cats).length) {
      body += `<h3>Kategori error</h3><div class="summary">`;
      for (const [k, v] of Object.entries(cats)) body += `<span class="chip err">${v}× ${CATEGORY_LABELS[k] || k}</span>`;
      body += `</div>`;
    }

    body += `<h3>Artefak (${files.files.length})</h3>`;
    if (files.files.length) {
      body += `<div class="filelist">`;
      for (const f of files.files) body += `<button class="btn tiny" data-file="${esc(f.path)}">${esc(f.path)} <span class="muted">(${f.size}b)</span></button>`;
      body += `</div><div id="preview-host"></div>`;
    } else {
      body += `<div class="muted">Tidak ada file yang dihasilkan.</div>`;
    }

    body += `<h3>Trace (${detail.steps.length} langkah)</h3><div class="timeline">`;
    let turn = null;
    for (const s of detail.steps) {
      if (s.turn !== turn) { turn = s.turn; body += `<div class="turnsep">Turn ${turn ?? "—"}</div>`; }
      if (s.type === "assistant") {
        const content = (s.result || "").trim();
        body += `<div class="step assistant"><div class="stephead">assistant
          <span class="muted">${fmtMs(s.latency_ms)}${s.ttft_ms != null ? " · ttft " + fmtMs(s.ttft_ms) : ""} · ${fmtTokens(s.tokens_in)}→${fmtTokens(s.tokens_out)} tok</span></div>
          ${content ? `<div class="code">${esc(content)}</div>` : '<div class="muted">→ memanggil tool</div>'}</div>`;
      } else if (s.type === "tool_call") {
        body += `<div class="step call"><div class="stephead">tool call: <span class="mono">${esc(s.tool || "?")}</span></div>
          <div class="code">${esc(s.args || "")}</div></div>`;
      } else if (s.type === "tool_result") {
        const ok = s.ok;
        body += `<div class="step result ${ok ? "ok" : "err"}"><div class="stephead">hasil
          ${ok ? badge("ok", "ok") : badge("err", CATEGORY_LABELS[s.category] || s.category || "error")}
          ${s.flags ? badge("warn", s.flags) : ""}<span class="muted">${fmtMs(s.latency_ms)}</span></div>
          <div class="code">${esc(s.result || "")}</div></div>`;
      } else if (s.type === "error") {
        body += `<div class="step result err"><div class="stephead">error ${badge("err", CATEGORY_LABELS[s.category] || s.category || "error")}</div>
          <div class="code">${esc(s.result || "")}</div></div>`;
      }
    }
    body += `</div>`;
    $("#modal-body").innerHTML = body;
    $("#modal-title").textContent = row.model + " — trace benchmark";

    $("#modal-body").querySelectorAll("[data-file]").forEach((b) =>
      b.addEventListener("click", () => showPreview(b.dataset.file, rmId)));
    const htmlFile = files.files.find((f) => f.path.endsWith(".html"));
    if (htmlFile) showPreview(htmlFile.path, rmId);

    try {
      const hist = await api(`/api/models/history?model=${encodeURIComponent(row.model)}&kind=bench`);
      const svg = sparkline((hist.points || []).map((p) => p.value).reverse());
      if (svg) $("#detail-spark").innerHTML = `<div class="muted">Tren ${hist.metric}: ${svg}</div>`;
    } catch (e) { /* ignore */ }
  } catch (e) {
    $("#modal-body").innerHTML = `<div class="muted">Gagal memuat detail: ${esc(e.message)}</div>`;
  }
}

function showPreview(path, rmId) {
  const host = $("#preview-host");
  if (!host) return;
  const url = `/api/bench/${rmId}/file?path=${encodeURIComponent(path)}`;
  host.innerHTML = `<div class="muted">Preview: ${esc(path)}</div>
    <iframe class="preview-frame" sandbox="" src="${url}"></iframe>`;
}

/* ---------- performance (load / context) ---------- */
function perfMode() { return $("#perf-mode").value; }

async function startPerf() {
  const mode = perfMode();
  const models = [...state.selected.perf];
  if (!models.length) return toast("Pilih minimal satu model", "err");
  const btn = $("#perf-start");
  btn.disabled = true;
  let path, body;
  if (mode === "load") {
    path = "/api/runs/load";
    body = { models, levels: parseCsv($("#perf-levels").value), requests_per_level: parseInt($("#perf-rpl").value, 10) || 8 };
  } else {
    path = "/api/runs/context";
    body = { models, sizes: parseCsv($("#perf-sizes").value) };
  }
  try {
    const res = await api(path, { method: "POST", body: JSON.stringify(body) });
    state.perfRunId = res.run_id;
    state.perfKind = mode;
    ensurePoll();
    toast(mode === "load" ? "Load probe dimulai" : "Context scaling dimulai");
  } catch (e) { toast(e.message, "err"); btn.disabled = false; }
}

function renderPerfRun(data) {
  const { run, models } = data;
  const kind = run.kind || state.perfKind;
  for (const row of models) state.rmIndex[row.id] = row;
  const done = models.filter((r) => !["pending", "running"].includes(r.status)).length;
  $("#perf-progress").innerHTML = run.status === "running"
    ? `<div class="progress"><span style="width:${models.length ? (done / models.length) * 100 : 0}%"></span></div>
       <div class="muted">${done}/${models.length} selesai…</div>` : "";
  let chips = "";
  if (run.summary && run.summary.regressions) chips += `<span class="chip err">${run.summary.regressions} regresi</span>`;
  $("#perf-summary").innerHTML = `<div class="summary">${chips}</div>`;

  $("#perf-table").innerHTML = kind === "load" ? loadTableHtml(models, true) : contextTableHtml(models, true);
  $("#perf-table").querySelectorAll("[data-perf]").forEach((b) =>
    b.addEventListener("click", () => openPerfDetail(Number(b.dataset.perf), kind)));
  $("#perf-start").disabled = run.status === "running";
}

function loadTableHtml(models, actions) {
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Model</th><th>Status</th><th>Total req</th><th>429</th><th>Max throughput</th><th>p95</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const m = row.metrics || {};
    const levels = m.levels || [];
    const maxTp = m.max_throughput_rps;
    const p95s = levels.map((l) => l.p95_ms).filter((v) => v != null);
    html += `<tr><td class="mono">${esc(row.model)} ${regBadge(m)}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>
      <td>${m.total_requests ?? "—"}</td><td>${m.total_429 ?? "—"}</td>
      <td>${maxTp != null ? maxTp + " rps" : "—"}</td>
      <td>${p95s.length ? fmtMs(Math.max(...p95s)) : "—"}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-perf="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function contextTableHtml(models, actions) {
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Model</th><th>Status</th><th>Max TTFT</th><th>Max prefill tok/s</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const m = row.metrics || {};
    html += `<tr><td class="mono">${esc(row.model)} ${regBadge(m)}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>
      <td>${fmtMs(m.max_ttft_ms)}</td><td>${fmtNum(m.max_prefill_tok_per_s)}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-perf="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function openPerfDetail(rmId, kind) {
  const row = state.rmIndex[rmId];
  if (!row) return;
  const m = row.metrics || {};
  const levels = m.levels || [];
  let body = `<div class="chips"><span class="chipbox">Model<b class="mono">${esc(row.model)}</b></span>${regBadge(m)}</div>`;
  if (kind === "load") {
    body += `<div class="scroll"><table class="tbl"><thead><tr><th>Concurrency</th><th>Requests</th><th>Success</th><th>429</th><th>Error</th><th>p50</th><th>p95</th><th>Throughput</th><th>Wall</th></tr></thead><tbody>`;
    for (const l of levels) {
      body += `<tr><td>${l.concurrency}</td><td>${l.requests}</td><td>${l.success}</td><td>${l.http_429}</td><td>${l.errors}</td>
        <td>${fmtMs(l.p50_ms)}</td><td>${fmtMs(l.p95_ms)}</td><td>${l.throughput_rps ?? "—"} rps</td><td>${fmtMs(l.wall_ms)}</td></tr>`;
    }
    body += `</tbody></table></div>`;
  } else {
    body += `<div class="scroll"><table class="tbl"><thead><tr><th>Target tok</th><th>Actual tok</th><th>TTFT</th><th>Latency</th><th>Prefill tok/s</th><th>Status</th></tr></thead><tbody>`;
    for (const l of levels) {
      body += `<tr><td>${l.target_tokens}</td><td>${fmtTokens(l.tokens_in)}</td><td>${fmtMs(l.ttft_ms)}</td><td>${fmtMs(l.latency_ms)}</td>
        <td>${fmtNum(l.prefill_tok_per_s)}</td><td>${l.status === "pass" ? badge("ok", "ok") : l.status === "context_exceeded" ? badge("warn", "melebihi konteks") : badge("err", "error")}${l.error ? `<div class="muted">${esc(l.error)}</div>` : ""}</td></tr>`;
    }
    body += `</tbody></table></div>`;
  }
  openModal(row.model + (kind === "load" ? " — load probe" : " — context scaling"), body);
}

/* ---------- modal ---------- */
function openModal(title, body) {
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = body;
  $("#modal").classList.remove("hidden");
}
$("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });

/* ---------- polling ---------- */
function ensurePoll() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(poll, 2000);
}
async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    if (state.funcRunId) {
      const data = await api(`/api/runs/${state.funcRunId}`);
      renderFunctionalRun(data);
      if (data.run.status !== "running") state.funcRunId = null;
    }
    if (state.benchRunId) {
      const data = await api(`/api/runs/${state.benchRunId}`);
      renderBenchRun(data);
      if (data.run.status !== "running") state.benchRunId = null;
    }
    if (state.perfRunId) {
      const data = await api(`/api/runs/${state.perfRunId}`);
      renderPerfRun(data);
      if (data.run.status !== "running") state.perfRunId = null;
    }
    if (!state.funcRunId && !state.benchRunId && !state.perfRunId) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch (e) { /* transient */ }
  state.polling = false;
}

/* ---------- history ---------- */
async function loadHistory() {
  try {
    const data = await api("/api/runs");
    const runs = data.runs || [];
    let html = `<div class="scroll"><table class="tbl"><thead><tr>
      <th>#</th><th>Jenis</th><th>Mulai</th><th>Status</th><th>Ringkasan</th><th></th></tr></thead><tbody>`;
    for (const r of runs) {
      let summary = "";
      const s = r.summary || {};
      if (r.kind === "functional" && s.counts) summary = Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(" · ");
      else if (s.outcomes) summary = Object.entries(s.outcomes).map(([k, v]) => `${v} ${OUTCOMES[k] ? OUTCOMES[k][1] : k}`).join(" · ");
      else if (s.counts) summary = Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(" · ");
      if (s.regressions) summary += ` · ${s.regressions} regresi`;
      html += `<tr><td>${r.id}</td><td>${KIND_LABELS[r.kind] || r.kind}</td><td class="mono">${esc(r.started_at || "")}</td>
        <td>${kvBadge(MODEL_STATUS, r.status)}</td><td>${esc(summary)}</td>
        <td><button class="btn tiny" data-hist="${r.id}">Lihat</button></td></tr>`;
    }
    html += "</tbody></table></div>";
    $("#history-list").innerHTML = html || '<div class="muted">Belum ada run.</div>';
    $("#history-list").querySelectorAll("[data-hist]").forEach((b) =>
      b.addEventListener("click", () => viewHistoryRun(Number(b.dataset.hist))));
  } catch (e) { toast(e.message, "err"); }
}

async function viewHistoryRun(runId) {
  $("#history-detail").innerHTML = '<div class="muted">Memuat…</div>';
  try {
    const data = await api(`/api/runs/${runId}`);
    for (const row of data.models) state.rmIndex[row.id] = row;
    const header = `<div class="summary"><span class="chip">run #${runId}</span><span class="chip">${KIND_LABELS[data.run.kind] || data.run.kind}</span><span class="chip">${data.run.status}</span></div>`;
    const kind = data.run.kind;
    if (kind === "functional") {
      $("#history-detail").innerHTML = header + '<div id="hist-fn-table"></div>';
      $("#hist-fn-table").innerHTML = functionalTableHtml(data.models, true);
      $("#hist-fn-table").querySelectorAll("[data-fn]").forEach((b) =>
        b.addEventListener("click", () => openFunctionalDetail(Number(b.dataset.fn))));
    } else if (kind === "bench") {
      $("#history-detail").innerHTML = header + '<div id="hist-bench-table"></div>';
      $("#hist-bench-table").innerHTML = benchTableHtml(data.models, true);
      $("#hist-bench-table").querySelectorAll("[data-bench]").forEach((b) =>
        b.addEventListener("click", () => openBenchDetail(Number(b.dataset.bench))));
    } else {
      $("#history-detail").innerHTML = header + '<div id="hist-perf-table"></div>';
      $("#hist-perf-table").innerHTML = kind === "load" ? loadTableHtml(data.models, true) : contextTableHtml(data.models, true);
      $("#hist-perf-table").querySelectorAll("[data-perf]").forEach((b) =>
        b.addEventListener("click", () => openPerfDetail(Number(b.dataset.perf), kind)));
    }
  } catch (e) {
    $("#history-detail").innerHTML = `<div class="muted">Gagal memuat: ${esc(e.message)}</div>`;
  }
}

/* ---------- settings ---------- */
async function loadSettings() {
  try {
    const s = await api("/api/settings");
    $("#set-url").value = s.GATEWAY_URL || "";
    $("#set-func-conc").value = s.FUNCTIONAL_CONCURRENCY || "4";
    $("#set-bench-conc").value = s.BENCH_CONCURRENCY || "2";
    $("#set-timeout").value = s.DEFAULT_TIMEOUT_S || "60";
    $("#set-token-cap").value = s.BENCH_TOKEN_CAP || "150000";
    $("#set-admin-key").placeholder = s.admin_key_set ? "••• tersimpan (kosongkan jika tidak mengubah)" : "belum diisi";
    $("#set-test-key").placeholder = s.test_key_set ? "••• tersimpan (kosongkan jika tidak mengubah)" : "belum diisi";
  } catch (e) { /* ignore */ }
}

async function saveSettings() {
  const body = {
    GATEWAY_URL: $("#set-url").value.trim(),
    FUNCTIONAL_CONCURRENCY: $("#set-func-conc").value,
    BENCH_CONCURRENCY: $("#set-bench-conc").value,
    DEFAULT_TIMEOUT_S: $("#set-timeout").value,
    BENCH_TOKEN_CAP: $("#set-token-cap").value,
    GATEWAY_ADMIN_KEY: $("#set-admin-key").value,
    GATEWAY_TEST_KEY: $("#set-test-key").value,
  };
  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    toast("Settings tersimpan");
    $("#set-admin-key").value = "";
    $("#set-test-key").value = "";
    await Promise.all([refreshStatus(), loadSettings()]);
    await refreshModels();
  } catch (e) { toast(e.message, "err"); }
}

async function testConnection() {
  try {
    const data = await api("/api/models?refresh=1");
    state.models = data.models || [];
    state.modelsByName = {};
    for (const m of state.models) state.modelsByName[m.name] = m;
    renderPickerList("functional");
    renderPickerList("bench");
    renderPickerList("perf");
    updateBenchEstimate();
    updatePerfEstimate();
    refreshStatus();
    toast(`Koneksi OK — ${state.models.length} model terdaftar`);
  } catch (e) { toast(e.message, "err"); }
}

/* ---------- wiring ---------- */
$("#refresh-models").addEventListener("click", refreshModels);
$("#fn-start").addEventListener("click", startFunctional);
$("#bench-start").addEventListener("click", startBench);
$("#perf-start").addEventListener("click", startPerf);
$("#fn-include-blocked").addEventListener("change", () => renderPickerList("functional"));
$("#set-save").addEventListener("click", saveSettings);
$("#set-test").addEventListener("click", testConnection);
$("#bench-task").addEventListener("change", () => {
  const t = state.tasks.find((x) => x.id === $("#bench-task").value);
  if (t && t.max_turns) $("#bench-max-turns").value = t.max_turns;
});
$("#perf-mode").addEventListener("change", () => {
  const mode = perfMode();
  $("#perf-load-cfg").classList.toggle("hidden", mode !== "load");
  $("#perf-context-cfg").classList.toggle("hidden", mode !== "context");
  updatePerfEstimate();
});
$("#perf-levels").addEventListener("input", updatePerfEstimate);
$("#perf-rpl").addEventListener("input", updatePerfEstimate);
$("#perf-sizes").addEventListener("input", updatePerfEstimate);

(async function init() {
  buildPicker("functional");
  buildPicker("bench");
  buildPicker("perf");
  await Promise.allSettled([refreshStatus(), loadSettings()]);
  await loadTasks();
  await refreshModels();
  updatePerfEstimate();
})();
