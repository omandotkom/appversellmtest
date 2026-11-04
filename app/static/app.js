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
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

const fmtMs = (ms) => ms == null ? "—" : (ms >= 10000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms");
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
const keyOf = (providerId, model) => `p${providerId}:${model}`;
const targetOfKey = (key) => {
  const idx = key.indexOf(":");
  return { provider_id: Number(key.slice(1, idx)), model: key.slice(idx + 1) };
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
  command_timeout: "perintah timeout", provider_error: "error provider", gateway_error: "error gateway",
  http_error: "HTTP error",
};
const WEIGHT_LABELS = {
  functional_tok_per_s: "tok/s", ttft_ms: "TTFT", pass_rate: "pass rate",
  itl_ms: "ITL", avg_latency_ms: "latency", cost_usd: "biaya",
};

function badge(kind, label) {
  return `<span class="badge ${kind || "muted"}">${esc(label)}</span>`;
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

/* ---------- ranking ---------- */
function rankBadge(rank) {
  if (rank == null) return '<span class="rank na">—</span>';
  return `<span class="rank r${Math.min(rank, 4)}" title="peringkat ${rank}">${rank}</span>`;
}
function rankMap(models, getter, higherBetter) {
  const rows = models.map((r) => ({ id: r.id, v: getter(r) })).filter((r) => r.v != null);
  rows.sort((a, b) => higherBetter ? b.v - a.v : a.v - b.v);
  const map = new Map();
  rows.forEach((r, i) => map.set(r.id, i + 1));
  return map;
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
  providers: [],
  models: [],
  modelsByKey: {},
  tasks: [],
  selected: {
    functional: new Set(), bench: new Set(), perf: new Set(), comparison: new Set(),
  },
  funcRunId: null,
  benchRunId: null,
  perfRunId: null,
  cmpActiveRunId: null,
  perfKind: "load",
  rmIndex: {},
  pollTimer: null,
  polling: false,
};

const PICKERS = {
  functional: { sel: "#fn-picker" },
  bench: { sel: "#bench-picker", estimate: updateBenchEstimate },
  perf: { sel: "#perf-picker", estimate: updatePerfEstimate },
  comparison: { sel: "#cmp-picker" },
};

/* ---------- status / catalog / pickers ---------- */
async function refreshStatus() {
  try { state.status = await api("/api/status"); } catch { state.status = null; }
  renderPill();
}

function renderPill() {
  const el = $("#gw-pill");
  const s = state.status;
  if (!s) { el.className = "pill muted"; el.textContent = "memuat…"; return; }
  if (!s.providers) {
    el.className = "pill warn";
    el.textContent = "belum ada provider";
    return;
  }
  if (!s.providers_enabled) {
    el.className = "pill warn";
    el.textContent = `${s.providers} provider (semua disabled)`;
    return;
  }
  el.className = "pill ok";
  el.textContent = `${s.providers_enabled}/${s.providers} provider · ${s.models} model`;
  el.title = "Klik Providers untuk mengelola endpoint dan model.";
}

async function refreshCatalog() {
  let data;
  try {
    data = await api("/api/catalog");
  } catch (e) {
    state.providers = [];
    state.models = [];
    toast(e.message, "err");
    renderAllPickers();
    renderProviderList();
    return;
  }
  state.providers = data.providers || [];
  state.models = [];
  for (const provider of state.providers) {
    for (const model of provider.models || []) {
      state.models.push({
        ...model,
        provider_id: provider.id,
        provider_name: provider.name,
        provider_enabled: provider.enabled,
        provider_base_url: provider.base_url,
        _key: keyOf(provider.id, model.model_name),
      });
    }
  }
  state.modelsByKey = {};
  for (const model of state.models) state.modelsByKey[model._key] = model;
  for (const set of Object.values(state.selected)) {
    for (const key of [...set]) if (!state.modelsByKey[key]) set.delete(key);
  }
  renderAllPickers();
  renderProviderList();
  updateBenchEstimate();
  updatePerfEstimate();
  await refreshStatus();
}

function renderAllPickers() {
  for (const name of Object.keys(PICKERS)) renderPickerList(name);
}

function isChatCandidate(model) {
  return model.mode == null || model.mode === "chat";
}

function isSelectable(model, pickerName) {
  if (!model.provider_enabled || !model.enabled) return false;
  if (!isChatCandidate(model)) return false;
  if (model.blocked) {
    return (pickerName === "functional" || pickerName === "comparison")
      && $("#fn-include-blocked") && $("#fn-include-blocked").checked;
  }
  return true;
}

function buildPicker(name) {
  const root = $(PICKERS[name].sel);
  root.innerHTML = `
    <div class="picker-toolbar">
      <input class="picker-search" placeholder="Cari provider atau model…">
      <button class="btn tiny" data-act="all">Semua chat</button>
      <button class="btn tiny" data-act="none">Kosongkan</button>
      <span class="picker-count"></span>
    </div>
    <div class="picker-list"></div>`;
  root.querySelector(".picker-search").addEventListener("input", () => renderPickerList(name));
  root.querySelector('[data-act="all"]').addEventListener("click", () => {
    for (const model of state.models) {
      if (isSelectable(model, name) && !model.blocked) state.selected[name].add(model._key);
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
  if (!root) return;
  const q = (root.querySelector(".picker-search").value || "").toLowerCase();
  const list = root.querySelector(".picker-list");
  const selected = state.selected[name];
  const models = state.models.filter((model) =>
    model.model_name.toLowerCase().includes(q) || model.provider_name.toLowerCase().includes(q));
  list.innerHTML = "";
  if (!state.models.length) {
    list.innerHTML = `<div class="muted" style="padding:10px">Belum ada model. Buka tab Providers, tambahkan provider, lalu refresh modelnya.</div>`;
    root.querySelector(".picker-count").textContent = "";
    return;
  }

  let currentProvider = null;
  for (const model of models) {
    if (model.provider_id !== currentProvider) {
      currentProvider = model.provider_id;
      const header = document.createElement("div");
      header.className = "picker-group";
      header.innerHTML = `<span>${esc(model.provider_name)}</span>${model.provider_enabled ? "" : ' <span class="tag blocked">disabled</span>'}`;
      list.appendChild(header);
    }
    const label = document.createElement("label");
    label.className = "pick" + (model.blocked || !model.enabled ? " blocked" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selected.has(model._key);
    cb.disabled = !isSelectable(model, name);
    cb.addEventListener("change", () => {
      if (cb.checked) selected.add(model._key); else selected.delete(model._key);
      renderPickerList(name);
      if (PICKERS[name].estimate) PICKERS[name].estimate();
    });
    const tags = [];
    if (!isChatCandidate(model)) tags.push(`<span class="tag">non-chat</span>`);
    if (!model.enabled) tags.push(`<span class="tag blocked">off</span>`);
    if (model.supports_tools) tags.push(`<span class="tag tools">tools</span>`);
    if (model.supports_json) tags.push(`<span class="tag tools">json</span>`);
    if (model.blocked) tags.push(`<span class="tag blocked">blocked</span>`);
    label.innerHTML = `<span class="pname">${esc(model.model_name)}</span><span class="ptags">${tags.join("")}</span>`;
    label.insertBefore(cb, label.firstChild);
    list.appendChild(label);
  }
  root.querySelector(".picker-count").textContent = `${selected.size} dipilih`;
}

function selectedTargets(name) {
  return [...state.selected[name]].map(targetOfKey);
}

function modelCost(model, inTokens, outTokens) {
  if (model && model.input_cost_per_token != null && model.output_cost_per_token != null) {
    return inTokens * model.input_cost_per_token + outTokens * model.output_cost_per_token;
  }
  return null;
}

function updateBenchEstimate() {
  let total = null, missing = 0;
  for (const key of state.selected.bench) {
    const model = state.modelsByKey[key];
    if (!model) continue;
    const cost = modelCost(model, 25000, 4000);
    if (cost != null) total = (total || 0) + cost; else missing++;
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
    for (const key of state.selected.perf) {
      const model = state.modelsByKey[key];
      if (!model) continue;
      const cost = modelCost(model, reqs * 8, reqs * 16);
      if (cost != null) total = (total || 0) + cost; else missing++;
    }
  } else {
    const sumSizes = parseCsv($("#perf-sizes").value).reduce((a, b) => a + b, 0);
    for (const key of state.selected.perf) {
      const model = state.modelsByKey[key];
      if (!model) continue;
      const cost = modelCost(model, sumSizes, 16);
      if (cost != null) total = (total || 0) + cost; else missing++;
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
    if (btn.dataset.tab === "comparison") loadComparisonRuns().then(loadComparison);
    if (btn.dataset.tab === "providers") renderProviderList();
    if (btn.dataset.tab === "settings") loadSettings();
  });
});

/* ---------- functional ---------- */
async function startFunctional() {
  const checks = [...document.querySelectorAll("#fn-checks input:checked")].map((i) => i.value);
  if (!checks.length) return toast("Pilih minimal satu jenis tes", "err");
  const targets = selectedTargets("functional");
  const body = {
    checks,
    include_blocked: $("#fn-include-blocked").checked,
    targets: targets.length ? targets : null,
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
  for (const row of models) counts[row.status] = (counts[row.status] || 0) + 1;
  let chips = "";
  for (const [key, value] of Object.entries(counts)) {
    chips += `<span class="chip ${MODEL_STATUS[key] ? MODEL_STATUS[key][0] : "muted"}">${value} ${key}</span>`;
  }
  const cost = models.reduce((sum, row) => sum + (((row.metrics || {}).cost_usd) || 0), 0);
  if (run.summary && run.summary.regressions) chips += `<span class="chip err">${run.summary.regressions} regresi</span>`;
  $("#fn-summary").innerHTML = `<div class="summary">${chips}<span class="chip">biaya ${fmtCost(cost)}</span></div>`;

  $("#fn-table").innerHTML = functionalTableHtml(models, true);
  $("#fn-table").querySelectorAll("[data-fn]").forEach((b) =>
    b.addEventListener("click", () => openFunctionalDetail(Number(b.dataset.fn))));
  $("#fn-start").disabled = run.status === "running";
}

function functionalTableHtml(models, actions) {
  const ranks = rankMap(models, (row) => (row.metrics || {}).avg_latency_ms, false);
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Provider</th><th>Model</th><th>Rank</th><th>Status</th>`;
  for (const check of ["basic", "streaming", "tools", "json"]) html += `<th>${CHECK_LABELS[check]}</th>`;
  html += `<th>Latency</th><th>p95</th><th>tok/s</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const checks = row.checks || {};
    const metrics = row.metrics || {};
    const streaming = checks.streaming || {};
    html += `<tr><td>${esc(providerLabel(row))}</td><td class="mono">${esc(row.model)} ${regBadge(metrics)}</td><td>${rankBadge(ranks.get(row.id))}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>`;
    for (const check of ["basic", "streaming", "tools", "json"]) {
      const result = checks[check];
      html += `<td>${result ? kvBadge(CHECK_STATUS, result.status).replace(">", ` title="${esc(result.error || result.note || "")}">`) : '<span class="badge muted">—</span>'}</td>`;
    }
    html += `<td>${fmtMs(metrics.avg_latency_ms)}</td><td>${fmtMs(metrics.p95_latency_ms)}</td><td>${fmtNum(streaming.tok_per_s)}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-fn="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function checkDetailRow(name, result) {
  let body = `<div class="step"><div class="stephead"><strong>${CHECK_LABELS[name]}</strong> ${kvBadge(CHECK_STATUS, result.status)}
    <span class="muted">latency ${fmtMs(result.latency_ms)}${result.ttft_ms != null ? " · ttft " + fmtMs(result.ttft_ms) : ""}${result.generation_ms != null ? " · gen " + fmtMs(result.generation_ms) : ""}${result.http_status ? " · HTTP " + result.http_status : ""}</span></div>`;
  const bits = [];
  if (result.tok_per_s != null) bits.push(`tok/s <b>${fmtNum(result.tok_per_s)}</b>`);
  if (result.itl_ms != null) bits.push(`ITL <b>${fmtNum(result.itl_ms)} ms</b>`);
  if (result.cached_tokens != null) bits.push(`cached <b>${result.cached_tokens}</b>`);
  if (result.reasoning_tokens != null) bits.push(`reasoning <b>${result.reasoning_tokens}</b>`);
  if (result.p95_latency_ms != null) bits.push(`p95 <b>${fmtMs(result.p95_latency_ms)}</b>`);
  if (bits.length) body += `<div class="muted">${bits.join(" · ")}</div>`;
  if (result.error) body += `<div class="code">${esc(result.error)}</div>`;
  else if (result.note) body += `<div class="muted">${esc(result.note)}</div>`;
  if (result.samples && result.samples.length) {
    body += `<div class="muted" style="margin-top:6px">Sampel (${result.samples.length}):</div><div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>status</th><th>latency</th><th>ttft</th><th>tok/s</th></tr></thead><tbody>`;
    result.samples.forEach((sample, index) => {
      body += `<tr><td>${index + 1}</td><td>${kvBadge(CHECK_STATUS, sample.status)}</td><td>${fmtMs(sample.latency_ms)}</td><td>${fmtMs(sample.ttft_ms)}</td><td>${fmtNum(sample.tok_per_s)}</td></tr>`;
    });
    body += `</tbody></table></div>`;
  }
  return body + `</div>`;
}

function providerLabel(row) {
  return row.provider_name || "—";
}

async function openFunctionalDetail(rmId) {
  const row = state.rmIndex[rmId];
  if (!row) return;
  const checks = row.checks || {};
  const metrics = row.metrics || {};
  let body = `<div class="chips">
      <span class="chipbox">Provider<b>${esc(providerLabel(row))}</b></span>
      <span class="chipbox">Model<b class="mono">${esc(row.model)}</b></span>
      <span class="chipbox">Status<b>${MODEL_STATUS[row.status] ? MODEL_STATUS[row.status][1] : row.status}</b></span>
      <span class="chipbox">Latency<b>${fmtMs(metrics.avg_latency_ms)}</b></span>
      <span class="chipbox">p95<b>${fmtMs(metrics.p95_latency_ms)}</b></span>
      <span class="chipbox">Biaya<b>${fmtCost(metrics.cost_usd)}</b></span>
      ${regBadge(metrics)}
    </div><div id="detail-spark"></div>`;
  openModal(row.model, body);
  for (const check of ["basic", "streaming", "tools", "json"]) {
    if (checks[check]) $("#modal-body").insertAdjacentHTML("beforeend", checkDetailRow(check, checks[check]));
  }
  try {
    const params = new URLSearchParams({ model: row.model, kind: "functional" });
    if (row.provider_id != null) params.set("provider_id", row.provider_id);
    const hist = await api(`/api/models/history?${params}`);
    const svg = sparkline((hist.points || []).map((point) => point.value).reverse());
    if (svg) $("#detail-spark").innerHTML = `<div class="muted">Tren ${hist.metric}: ${svg}</div>`;
  } catch { /* ignore */ }
}

/* ---------- benchmark ---------- */
async function loadTasks() {
  try { state.tasks = (await api("/api/tasks")).tasks || []; } catch { state.tasks = []; }
  const sel = $("#bench-task");
  sel.innerHTML = state.tasks.map((task) => `<option value="${esc(task.id)}">${esc(task.name)}</option>`).join("");
  if (state.tasks[0]) $("#bench-max-turns").value = state.tasks[0].max_turns || 30;
}

async function startBench() {
  const taskId = $("#bench-task").value;
  if (!taskId) return toast("Pilih task", "err");
  const targets = selectedTargets("bench");
  if (!targets.length) return toast("Pilih minimal satu target untuk benchmark", "err");
  const body = {
    task_id: taskId,
    targets,
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
  for (const row of models) {
    const outcome = row.outcome || row.status;
    outcomes[outcome] = (outcomes[outcome] || 0) + 1;
  }
  let chips = "";
  for (const [key, value] of Object.entries(outcomes)) {
    chips += `<span class="chip ${OUTCOMES[key] ? OUTCOMES[key][0] : "muted"}">${value} ${OUTCOMES[key] ? OUTCOMES[key][1] : key}</span>`;
  }
  const avg = run.summary ? run.summary.avg_error_rate : null;
  const cost = models.reduce((sum, row) => sum + (((row.metrics || {}).cost_usd) || 0), 0);
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
  const ranks = rankMap(models, (row) => (row.metrics || {}).gen_tok_per_s, true);
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Provider</th><th>Model</th><th>Rank</th><th>Outcome</th><th>Tool calls</th><th>Error</th><th>Error rate</th><th>Recovery</th><th>TTFT</th><th>tok/s</th><th>Turn</th><th>Token</th><th>Biaya</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const metrics = row.metrics || {};
    const rate = metrics.error_rate;
    let bar = "";
    if (rate != null) {
      const pct = Math.min(100, rate * 100);
      const color = rate === 0 ? "var(--ok)" : (rate <= 0.1 ? "var(--warn)" : "var(--err)");
      bar = `<div class="ratebar"><span style="width:${pct}%;background:${color}"></span></div>`;
    }
    html += `<tr><td>${esc(providerLabel(row))}</td><td class="mono">${esc(row.model)} ${regBadge(metrics)}</td><td>${rankBadge(ranks.get(row.id))}</td>
      <td>${kvBadge(OUTCOMES, row.outcome || row.status)}</td>
      <td>${metrics.tool_calls ?? "—"}</td>
      <td>${metrics.tool_errors ?? "—"}</td>
      <td>${bar} ${fmtPct(rate)}</td>
      <td>${fmtPct(metrics.recovery_rate)}</td>
      <td>${fmtMs(metrics.avg_turn_ttft_ms)}</td>
      <td>${fmtNum(metrics.gen_tok_per_s)}</td>
      <td>${metrics.turns ?? "—"}</td>
      <td>${fmtTokens(metrics.tokens_in)}→${fmtTokens(metrics.tokens_out)}</td>
      <td>${fmtCost(metrics.cost_usd)}</td>`;
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
    const metrics = detail.model.metrics || {};
    let body = `<div class="chips">
      <span class="chipbox">Provider<b>${esc(providerLabel(detail.model))}</b></span>
      <span class="chipbox">Outcome<b>${OUTCOMES[detail.model.outcome] ? OUTCOMES[detail.model.outcome][1] : (detail.model.outcome || "—")}</b></span>
      <span class="chipbox">Tool calls<b>${metrics.tool_calls ?? "—"}</b></span>
      <span class="chipbox">Error<b>${metrics.tool_errors ?? "—"}</b></span>
      <span class="chipbox">Error rate<b>${fmtPct(metrics.error_rate)}</b></span>
      <span class="chipbox">Recovery<b>${fmtPct(metrics.recovery_rate)}</b></span>
      <span class="chipbox">TTFT/turn<b>${fmtMs(metrics.avg_turn_ttft_ms)}</b></span>
      <span class="chipbox">Latency/turn<b>${fmtMs(metrics.avg_turn_latency_ms)}</b></span>
      <span class="chipbox">tok/s<b>${fmtNum(metrics.gen_tok_per_s)}</b></span>
      <span class="chipbox">Model vs tool<b>${metrics.model_tool_ratio ?? "—"}</b></span>
      <span class="chipbox">Token<b>${fmtTokens(metrics.tokens_in)}→${fmtTokens(metrics.tokens_out)}</b></span>
      <span class="chipbox">Biaya<b>${fmtCost(metrics.cost_usd)}</b></span>
      <span class="chipbox">Durasi<b>${fmtDur(detail.model.started_at, detail.model.finished_at)}</b></span>
      ${metrics.stream_fallback ? '<span class="chipbox">Stream<b>fallback ' + metrics.stream_fallback + '×</b></span>' : ""}
      ${regBadge(metrics)}
    </div><div id="detail-spark"></div>`;

    const cats = metrics.by_category || {};
    if (Object.keys(cats).length) {
      body += `<h3>Kategori error</h3><div class="summary">`;
      for (const [key, value] of Object.entries(cats)) body += `<span class="chip err">${value}× ${CATEGORY_LABELS[key] || key}</span>`;
      body += `</div>`;
    }

    body += `<h3>Artefak (${files.files.length})</h3>`;
    if (files.files.length) {
      body += `<div class="filelist">`;
      for (const file of files.files) body += `<button class="btn tiny" data-file="${esc(file.path)}">${esc(file.path)} <span class="muted">(${file.size}b)</span></button>`;
      body += `</div><div id="preview-host"></div>`;
    } else {
      body += `<div class="muted">Tidak ada file yang dihasilkan.</div>`;
    }

    body += `<h3>Trace (${detail.steps.length} langkah)</h3><div class="timeline">`;
    let turn = null;
    for (const step of detail.steps) {
      if (step.turn !== turn) { turn = step.turn; body += `<div class="turnsep">Turn ${turn ?? "—"}</div>`; }
      if (step.type === "assistant") {
        const content = (step.result || "").trim();
        body += `<div class="step assistant"><div class="stephead">assistant
          <span class="muted">${fmtMs(step.latency_ms)}${step.ttft_ms != null ? " · ttft " + fmtMs(step.ttft_ms) : ""} · ${fmtTokens(step.tokens_in)}→${fmtTokens(step.tokens_out)} tok</span></div>
          ${content ? `<div class="code">${esc(content)}</div>` : '<div class="muted">→ memanggil tool</div>'}</div>`;
      } else if (step.type === "tool_call") {
        body += `<div class="step call"><div class="stephead">tool call: <span class="mono">${esc(step.tool || "?")}</span></div>
          <div class="code">${esc(step.args || "")}</div></div>`;
      } else if (step.type === "tool_result") {
        body += `<div class="step result ${step.ok ? "ok" : "err"}"><div class="stephead">hasil
          ${step.ok ? badge("ok", "ok") : badge("err", CATEGORY_LABELS[step.category] || step.category || "error")}
          ${step.flags ? badge("warn", step.flags) : ""}<span class="muted">${fmtMs(step.latency_ms)}</span></div>
          <div class="code">${esc(step.result || "")}</div></div>`;
      } else if (step.type === "error") {
        body += `<div class="step result err"><div class="stephead">error ${badge("err", CATEGORY_LABELS[step.category] || step.category || "error")}</div>
          <div class="code">${esc(step.result || "")}</div></div>`;
      }
    }
    body += `</div>`;
    $("#modal-body").innerHTML = body;
    $("#modal-title").textContent = row.model + " — trace benchmark";

    $("#modal-body").querySelectorAll("[data-file]").forEach((b) =>
      b.addEventListener("click", () => showPreview(b.dataset.file, rmId)));
    const htmlFile = files.files.find((file) => file.path.endsWith(".html"));
    if (htmlFile) showPreview(htmlFile.path, rmId);

    try {
      const params = new URLSearchParams({ model: row.model, kind: "bench" });
      if (row.provider_id != null) params.set("provider_id", row.provider_id);
      const hist = await api(`/api/models/history?${params}`);
      const svg = sparkline((hist.points || []).map((point) => point.value).reverse());
      if (svg) $("#detail-spark").innerHTML = `<div class="muted">Tren ${hist.metric}: ${svg}</div>`;
    } catch { /* ignore */ }
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
  const targets = selectedTargets("perf");
  if (!targets.length) return toast("Pilih minimal satu target", "err");
  const btn = $("#perf-start");
  btn.disabled = true;
  let path, body;
  if (mode === "load") {
    path = "/api/runs/load";
    body = { targets, levels: parseCsv($("#perf-levels").value), requests_per_level: parseInt($("#perf-rpl").value, 10) || 8 };
  } else {
    path = "/api/runs/context";
    body = { targets, sizes: parseCsv($("#perf-sizes").value) };
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
  const ranks = rankMap(models, (row) => (row.metrics || {}).max_throughput_rps, true);
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Provider</th><th>Model</th><th>Rank</th><th>Status</th><th>Total req</th><th>429</th><th>Max throughput</th><th>p95</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const metrics = row.metrics || {};
    const levels = metrics.levels || [];
    const maxTp = metrics.max_throughput_rps;
    const p95s = levels.map((level) => level.p95_ms).filter((value) => value != null);
    html += `<tr><td>${esc(providerLabel(row))}</td><td class="mono">${esc(row.model)} ${regBadge(metrics)}</td><td>${rankBadge(ranks.get(row.id))}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>
      <td>${metrics.total_requests ?? "—"}</td><td>${metrics.total_429 ?? "—"}</td>
      <td>${maxTp != null ? maxTp + " rps" : "—"}</td>
      <td>${p95s.length ? fmtMs(Math.max(...p95s)) : "—"}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-perf="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function contextTableHtml(models, actions) {
  const ranks = rankMap(models, (row) => (row.metrics || {}).max_prefill_tok_per_s, true);
  let html = `<div class="scroll"><table class="tbl"><thead><tr><th>Provider</th><th>Model</th><th>Rank</th><th>Status</th><th>Max TTFT</th><th>Max prefill tok/s</th>`;
  if (actions) html += `<th></th>`;
  html += `</tr></thead><tbody>`;
  for (const row of models) {
    const metrics = row.metrics || {};
    html += `<tr><td>${esc(providerLabel(row))}</td><td class="mono">${esc(row.model)} ${regBadge(metrics)}</td><td>${rankBadge(ranks.get(row.id))}</td><td>${kvBadge(MODEL_STATUS, row.status)}</td>
      <td>${fmtMs(metrics.max_ttft_ms)}</td><td>${fmtNum(metrics.max_prefill_tok_per_s)}</td>`;
    if (actions) html += `<td><button class="btn tiny" data-perf="${row.id}">Detail</button></td>`;
    html += `</tr>`;
  }
  return html + "</tbody></table></div>";
}

function openPerfDetail(rmId, kind) {
  const row = state.rmIndex[rmId];
  if (!row) return;
  const metrics = row.metrics || {};
  const levels = metrics.levels || [];
  let body = `<div class="chips"><span class="chipbox">Provider<b>${esc(providerLabel(row))}</b></span><span class="chipbox">Model<b class="mono">${esc(row.model)}</b></span>${regBadge(metrics)}</div>`;
  if (kind === "load") {
    body += `<div class="scroll"><table class="tbl"><thead><tr><th>Concurrency</th><th>Requests</th><th>Success</th><th>429</th><th>Error</th><th>p50</th><th>p95</th><th>Throughput</th><th>Wall</th></tr></thead><tbody>`;
    for (const level of levels) {
      body += `<tr><td>${level.concurrency}</td><td>${level.requests}</td><td>${level.success}</td><td>${level.http_429}</td><td>${level.errors}</td>
        <td>${fmtMs(level.p50_ms)}</td><td>${fmtMs(level.p95_ms)}</td><td>${level.throughput_rps ?? "—"} rps</td><td>${fmtMs(level.wall_ms)}</td></tr>`;
    }
    body += `</tbody></table></div>`;
  } else {
    body += `<div class="scroll"><table class="tbl"><thead><tr><th>Target tok</th><th>Actual tok</th><th>TTFT</th><th>Latency</th><th>Prefill tok/s</th><th>Status</th></tr></thead><tbody>`;
    for (const level of levels) {
      body += `<tr><td>${level.target_tokens}</td><td>${fmtTokens(level.tokens_in)}</td><td>${fmtMs(level.ttft_ms)}</td><td>${fmtMs(level.latency_ms)}</td>
        <td>${fmtNum(level.prefill_tok_per_s)}</td><td>${level.status === "pass" ? badge("ok", "ok") : level.status === "context_exceeded" ? badge("warn", "melebihi konteks") : badge("err", "error")}${level.error ? `<div class="muted">${esc(level.error)}</div>` : ""}</td></tr>`;
    }
    body += `</tbody></table></div>`;
  }
  openModal(row.model + (kind === "load" ? " — load probe" : " — context scaling"), body);
}

/* ---------- comparison ---------- */
const chartRegistry = {};
const COMPARISON_CHARTS = [
  { id: "chart-fn-tokps", key: "functional_tok_per_s", unit: "tok/s" },
  { id: "chart-ttft", key: "ttft_ms", unit: "ms" },
  { id: "chart-itl", key: "itl_ms", unit: "ms/token" },
  { id: "chart-latency", key: "avg_latency_ms", unit: "ms" },
  { id: "chart-pass", key: "pass_rate", unit: "0–1" },
  { id: "chart-cost", key: "cost_usd", unit: "USD" },
];
const CHART_COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#dc2626", "#4d7c0f", "#be185d"];

async function startComparison() {
  const targets = selectedTargets("comparison");
  const body = {
    targets: targets.length ? targets : null,
    repeat: parseInt($("#cmp-repeat").value, 10) || 3,
    include_blocked: $("#fn-include-blocked") ? $("#fn-include-blocked").checked : false,
  };
  const btn = $("#cmp-start");
  btn.disabled = true;
  try {
    const res = await api("/api/runs/comparison", { method: "POST", body: JSON.stringify(body) });
    state.cmpActiveRunId = res.run_id;
    ensurePoll();
    toast("Comparison run dimulai");
  } catch (e) { toast(e.message, "err"); btn.disabled = false; }
}

async function loadComparisonRuns() {
  const sel = $("#cmp-run");
  const previous = sel.value;
  try {
    const data = await api("/api/runs?kind=functional&limit=40");
    const runs = data.runs || [];
    let html = `<option value="">Histori agregat (window/mode)</option>`;
    for (const run of runs) {
      const repeat = (run.options || {}).repeat;
      const comparisonFlag = (run.options || {}).comparison ? " · comparison" : "";
      html += `<option value="${run.id}">#${run.id} · ${esc(run.started_at || "")} · ${run.status}${repeat ? " · " + repeat + "×" : ""}${comparisonFlag}</option>`;
    }
    sel.innerHTML = html;
    if (previous && [...sel.options].some((opt) => opt.value === previous)) sel.value = previous;
    else if (runs[0]) sel.value = String(runs[0].id);
  } catch (e) {
    sel.innerHTML = `<option value="">Histori agregat (window/mode)</option>`;
  }
}

async function loadComparison() {
  const host = $("#cmp-score");
  host.innerHTML = '<div class="muted">Memuat…</div>';
  const params = new URLSearchParams();
  const runId = $("#cmp-run").value;
  if (runId) params.set("run_id", runId);
  params.set("mode", $("#cmp-mode").value);
  params.set("window", $("#cmp-window").value || "5");
  params.set("top_n", $("#cmp-topn").value || "20");
  const selectedProviders = [...new Set(selectedTargets("comparison").map((t) => t.provider_id))];
  if (!runId && selectedProviders.length) params.set("provider_ids", selectedProviders.join(","));
  try {
    const data = await api(`/api/comparison?${params}`);
    renderComparison(data);
  } catch (e) {
    host.innerHTML = `<div class="muted">Gagal memuat comparison: ${esc(e.message)}</div>`;
  }
}

function metricValue(model, key) {
  return model.metrics && model.metrics[key] ? model.metrics[key].value : null;
}

function renderComparison(data) {
  const models = data.models || [];
  const weights = data.weights || {};
  const host = $("#cmp-score");
  if (!models.length) {
    host.innerHTML = '<div class="muted">Belum ada data fungsional. Jalankan comparison atau tes fungsional dulu.</div>';
    $("#cmp-weights").textContent = "";
    $("#cmp-providers").innerHTML = "";
    $("#cmp-table").innerHTML = "";
    drawComparisonCharts([]);
    return;
  }

  let html = `<div class="scroll"><table class="tbl"><thead><tr>
    <th>Rank</th><th>Provider</th><th>Model</th><th>Score</th>
    <th>Tok/s</th><th>TTFT</th><th>ITL</th><th>Latency</th><th>Pass rate</th><th>Biaya</th>
    <th>Chat</th><th>Stream</th><th>Tools</th><th>JSON</th></tr></thead><tbody>`;
  models.forEach((model, index) => {
    const score = model.score;
    const color = score == null ? "#6b7684" : (score >= 80 ? "var(--ok)" : score >= 55 ? "var(--warn)" : "var(--err)");
    const bar = score == null ? "" : `<div class="ratebar"><span style="width:${score}%;background:${color}"></span></div>`;
    const checks = model.checks || {};
    const checkCell = (name) => checks[name] ? kvBadge(CHECK_STATUS, checks[name]) : '<span class="badge muted">—</span>';
    html += `<tr><td>${rankBadge(score == null ? null : index + 1)}</td>
      <td>${esc(model.provider_name || "—")}</td><td class="mono">${esc(model.model)}</td>
      <td>${bar} ${score == null ? "—" : score.toFixed(1)}</td>
      <td>${fmtNum(metricValue(model, "functional_tok_per_s"))}</td>
      <td>${fmtMs(metricValue(model, "ttft_ms"))}</td>
      <td>${metricValue(model, "itl_ms") != null ? fmtNum(metricValue(model, "itl_ms")) + " ms" : "—"}</td>
      <td>${fmtMs(metricValue(model, "avg_latency_ms"))}</td>
      <td>${fmtPct(metricValue(model, "pass_rate"))}</td>
      <td>${fmtCost(metricValue(model, "cost_usd"))}</td>
      <td>${checkCell("basic")}</td><td>${checkCell("streaming")}</td><td>${checkCell("tools")}</td><td>${checkCell("json")}</td></tr>`;
  });
  html += "</tbody></table></div>";
  host.innerHTML = html;

  $("#cmp-weights").textContent = "Bobot skor: " + Object.entries(weights)
    .map(([key, value]) => `${Math.round(value * 100)}% ${WEIGHT_LABELS[key] || key}`).join(" · ");

  const providers = data.providers || [];
  let phtml = `<div class="scroll"><table class="tbl"><thead><tr>
    <th>Provider</th><th>Model</th><th>Score</th><th>Tok/s</th><th>TTFT</th><th>ITL</th><th>Latency</th><th>Pass rate</th><th>Biaya</th><th>Model terbaik</th></tr></thead><tbody>`;
  for (const provider of providers) {
    const median = provider.median || {};
    phtml += `<tr><td>${esc(provider.name)}</td><td>${provider.models}</td>
      <td>${provider.score == null ? "—" : provider.score.toFixed(1)}</td>
      <td>${fmtNum(median.functional_tok_per_s)}</td>
      <td>${fmtMs(median.ttft_ms)}</td>
      <td>${fmtNum(median.itl_ms)}</td>
      <td>${fmtMs(median.avg_latency_ms)}</td>
      <td>${fmtPct(median.pass_rate)}</td>
      <td>${fmtCost(median.cost_usd)}</td>
      <td class="mono">${esc(provider.best_model || "—")}</td></tr>`;
  }
  phtml += "</tbody></table></div>";
  $("#cmp-providers").innerHTML = phtml;

  $("#cmp-table").innerHTML = html;
  drawComparisonCharts(models);
}

function drawComparisonCharts(models) {
  if (typeof Chart === "undefined") {
    $("#cmp-table").insertAdjacentHTML("beforeend", '<div class="muted">Chart.js gagal dimuat (CDN offline). Chart tidak tersedia.</div>');
    return;
  }
  for (const def of COMPARISON_CHARTS) {
    const canvas = document.getElementById(def.id);
    if (!canvas) continue;
    if (chartRegistry[def.id]) chartRegistry[def.id].destroy();
    const holder = canvas.parentElement;
    holder.querySelectorAll(".nodata").forEach((node) => node.remove());
    const values = models.map((model) => metricValue(model, def.key));
    if (values.every((value) => value == null)) {
      holder.insertAdjacentHTML("beforeend", '<div class="nodata muted" style="padding:12px">Belum ada data untuk chart ini.</div>');
      continue;
    }
    const labels = models.map((model) => `${model.provider_name || "—"} / ${model.model}`);
    chartRegistry[def.id] = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: def.key,
          data: values,
          backgroundColor: values.map((_, index) => CHART_COLORS[index % CHART_COLORS.length] + "cc"),
          borderWidth: 0,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#6b7280", font: { size: 10 }, maxRotation: 40 } },
          y: { beginAtZero: true, title: { display: true, text: def.unit, color: "#6b7280", font: { size: 10 } }, ticks: { color: "#6b7280", font: { size: 10 } } },
        },
      },
    });
  }
}

/* ---------- providers ---------- */
function renderProviderList() {
  const host = $("#provider-list");
  if (!host) return;
  const providers = state.providers || [];
  if (!providers.length) {
    host.innerHTML = '<div class="muted">Belum ada provider. Tambahkan endpoint OpenAI-compatible pertama.</div>';
    return;
  }
  let html = `<div class="scroll"><table class="tbl"><thead><tr>
    <th>Nama</th><th>Endpoint</th><th>Kind</th><th>Model</th><th>Key</th><th>Status</th><th></th></tr></thead><tbody>`;
  for (const provider of providers) {
    const keys = [
      provider.has_api_key ? '<span class="badge ok">api ✓</span>' : '<span class="badge muted">api —</span>',
      provider.has_admin_key ? '<span class="badge ok">admin ✓</span>' : '<span class="badge muted">admin —</span>',
    ].join(" ");
    html += `<tr>
      <td><strong>${esc(provider.name)}</strong><div class="muted mono">#${provider.id}</div></td>
      <td class="mono">${esc(provider.base_url)}</td>
      <td>${esc(provider.kind)}</td>
      <td>${provider.model_count ?? (provider.models || []).length}</td>
      <td>${keys}</td>
      <td>${provider.enabled ? badge("ok", "aktif") : badge("muted", "disabled")}</td>
      <td class="nowrap">
        <button class="btn tiny" data-provider-action="edit" data-id="${provider.id}">Edit</button>
        <button class="btn tiny" data-provider-action="models" data-id="${provider.id}">Model</button>
        <button class="btn tiny" data-provider-action="test" data-id="${provider.id}">Test</button>
        <button class="btn tiny" data-provider-action="refresh" data-id="${provider.id}">Refresh</button>
        <button class="btn tiny" data-provider-action="toggle" data-id="${provider.id}">${provider.enabled ? "Disable" : "Enable"}</button>
      </td></tr>`;
  }
  host.innerHTML = html + "</tbody></table></div>";
}

$("#provider-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-provider-action]");
  if (!button) return;
  const id = Number(button.dataset.id);
  const action = button.dataset.providerAction;
  const provider = state.providers.find((item) => item.id === id);
  if (!provider) return;
  if (action === "edit") return openProviderForm(id);
  if (action === "models") return openModelsModal(id);
  if (action === "test") {
    button.disabled = true;
    try {
      const result = await api(`/api/providers/${id}/test`, { method: "POST" });
      if (result.ok) {
        let message = `Koneksi OK · ${result.model_count} model · ${fmtMs(result.latency_ms)}`;
        if (result.chat) {
          message += result.chat.ok
            ? ` · chat ${result.chat.model} ${fmtMs(result.chat.latency_ms)}`
            : ` · chat gagal: ${result.chat.error || result.chat.http_status}`;
        }
        toast(message, result.chat && !result.chat.ok ? "err" : undefined);
      } else toast(result.error || "Tes provider gagal", "err");
    } catch (e) { toast(e.message, "err"); }
    button.disabled = false;
    return;
  }
  if (action === "refresh") {
    button.disabled = true;
    try {
      const result = await api(`/api/providers/${id}/refresh-models`, { method: "POST" });
      toast(`Model disegarkan: ${result.total} total, ${result.added} baru`);
      await refreshCatalog();
    } catch (e) { toast(e.message, "err"); }
    button.disabled = false;
    return;
  }
  if (action === "toggle") {
    try {
      await api(`/api/providers/${id}`, { method: "PATCH", body: JSON.stringify({ enabled: provider.enabled ? 0 : 1 }) });
      await refreshCatalog();
      toast(provider.enabled ? "Provider di-disable" : "Provider di-enable");
    } catch (e) { toast(e.message, "err"); }
  }
});

function openProviderForm(providerId) {
  const provider = providerId ? state.providers.find((item) => item.id === providerId) : null;
  const value = (key, fallback = "") => provider && provider[key] != null ? provider[key] : fallback;
  const adminPaths = Array.isArray(provider && provider.admin_models_paths)
    ? provider.admin_models_paths.join(", ") : (value("admin_models_paths"));
  const extraHeaders = provider && provider.extra_headers
    ? (typeof provider.extra_headers === "string" ? provider.extra_headers : JSON.stringify(provider.extra_headers))
    : "";
  const body = `<form id="provider-form" onsubmit="return false">
    <div class="grid2">
      <label class="field">Nama<input id="pf-name" value="${esc(value("name"))}" required></label>
      <label class="field">Kind<select id="pf-kind"><option value="openai">openai-compatible</option></select></label>
    </div>
    <label class="field">Base URL<input id="pf-base" placeholder="http://localhost:4000" value="${esc(value("base_url"))}" required></label>
    <div class="grid2">
      <label class="field">Chat path<input id="pf-chat" value="${esc(value("chat_path", "/v1/chat/completions"))}"></label>
      <label class="field">Models path<input id="pf-models" value="${esc(value("models_path", "/v1/models"))}"></label>
    </div>
    <label class="field">Admin model list paths (pisahkan koma)<input id="pf-admin-paths" value="${esc(adminPaths)}"></label>
    <div class="grid2">
      <label class="field">Auth scheme
        <select id="pf-auth">
          <option value="bearer">Bearer</option>
          <option value="x-api-key">x-api-key</option>
          <option value="query">query api_key</option>
          <option value="none">none</option>
        </select>
      </label>
      <label class="field">Extra headers (JSON)<input id="pf-headers" placeholder='{"X-Org":"..."}' value="${esc(extraHeaders)}"></label>
    </div>
    <div class="grid2">
      <label class="field">Timeout (detik)<input type="number" id="pf-timeout" min="1" value="${esc(value("timeout_s", ""))}"></label>
      <label class="field">Max concurrency<input type="number" id="pf-concurrency" min="1" value="${esc(value("max_concurrency", ""))}"></label>
    </div>
    <div class="grid2">
      <label class="field">API key <input type="password" id="pf-api-key" placeholder="${provider && provider.has_api_key ? "••• tersimpan (kosongkan jika tidak mengubah)" : "opsional"}"></label>
      <label class="field">Admin key <input type="password" id="pf-admin-key" placeholder="${provider && provider.has_admin_key ? "••• tersimpan (kosongkan jika tidak mengubah)" : "opsional"}"></label>
    </div>
    <label class="switch"><input type="checkbox" id="pf-enabled" ${!provider || provider.enabled ? "checked" : ""}> Provider aktif</label>
    <p class="hint">Key disimpan di SQLite dan tidak pernah dikembalikan oleh API. Kosongkan key saat edit untuk mempertahankan nilai lama.</p>
    <div class="row"><button id="pf-save" class="btn primary">Simpan provider</button></div>
  </form>`;
  openModal(provider ? `Edit provider: ${provider.name}` : "Tambah provider", body);
  $("#pf-auth").value = value("auth_scheme", "bearer");
  $("#pf-kind").value = value("kind", "openai");
  $("#pf-save").addEventListener("click", () => saveProviderForm(providerId));
}

async function saveProviderForm(providerId) {
  const payload = {
    name: $("#pf-name").value.trim(),
    kind: $("#pf-kind").value,
    base_url: $("#pf-base").value.trim(),
    chat_path: $("#pf-chat").value.trim(),
    models_path: $("#pf-models").value.trim(),
    admin_models_paths: $("#pf-admin-paths").value.trim(),
    auth_scheme: $("#pf-auth").value,
    extra_headers: $("#pf-headers").value.trim(),
    timeout_s: $("#pf-timeout").value || null,
    max_concurrency: $("#pf-concurrency").value || null,
    enabled: $("#pf-enabled").checked ? 1 : 0,
  };
  const apiKey = $("#pf-api-key").value.trim();
  const adminKey = $("#pf-admin-key").value.trim();
  try {
    if (providerId) {
      await api(`/api/providers/${providerId}`, { method: "PATCH", body: JSON.stringify(payload) });
      const secrets = {};
      if (apiKey) secrets.api_key = apiKey;
      if (adminKey) secrets.admin_key = adminKey;
      if (Object.keys(secrets).length) {
        await api(`/api/providers/${providerId}/secrets`, { method: "PUT", body: JSON.stringify(secrets) });
      }
    } else {
      if (apiKey) payload.api_key = apiKey;
      if (adminKey) payload.admin_key = adminKey;
      await api("/api/providers", { method: "POST", body: JSON.stringify(payload) });
    }
    $("#modal").classList.add("hidden");
    await refreshCatalog();
    toast("Provider tersimpan");
  } catch (e) { toast(e.message, "err"); }
}

async function openModelsModal(providerId) {
  const provider = state.providers.find((item) => item.id === providerId);
  if (!provider) return;
  let models = [];
  try { models = (await api(`/api/providers/${providerId}/models`)).models || []; }
  catch (e) { return toast(e.message, "err"); }
  const costInput = (value) => value == null ? "" : String(Math.round(value * 1e6 * 1e6) / 1e6);
  let html = `<p class="hint">Harga dalam USD per 1 juta token. Centang tools/json jika provider mendukungnya. Perubahan harga dan nama tidak akan ditimpa saat refresh discovery.</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>Aktif</th><th>Model</th><th>Display</th><th>Mode</th><th>Tools</th><th>JSON</th><th>Blocked</th><th>Context</th><th>In $/1M</th><th>Out $/1M</th></tr></thead><tbody>`;
  for (const model of models) {
    html += `<tr data-model-row="${model.id}">
      <td><input type="checkbox" data-field="enabled" ${model.enabled ? "checked" : ""}></td>
      <td class="mono">${esc(model.model_name)}</td>
      <td><input data-field="display_name" value="${esc(model.display_name || "")}" style="width:110px"></td>
      <td><input data-field="mode" value="${esc(model.mode || "")}" style="width:70px"></td>
      <td><input type="checkbox" data-field="supports_tools" ${model.supports_tools ? "checked" : ""}></td>
      <td><input type="checkbox" data-field="supports_json" ${model.supports_json ? "checked" : ""}></td>
      <td><input type="checkbox" data-field="blocked" ${model.blocked ? "checked" : ""}></td>
      <td><input type="number" data-field="context_window" value="${model.context_window ?? ""}" style="width:92px"></td>
      <td><input type="number" step="0.000001" data-field="input_cost" value="${costInput(model.input_cost_per_token)}" style="width:100px"></td>
      <td><input type="number" step="0.000001" data-field="output_cost" value="${costInput(model.output_cost_per_token)}" style="width:100px"></td>
    </tr>`;
  }
  html += `</tbody></table></div><div class="row"><button id="models-save" class="btn primary">Simpan perubahan</button></div>`;
  openModal(`Model · ${provider.name}`, html);
  $("#models-save").addEventListener("click", () => saveModels());
}

async function saveModels() {
  const rows = document.querySelectorAll("[data-model-row]");
  try {
    for (const row of rows) {
      const id = Number(row.dataset.modelRow);
      const field = (name) => row.querySelector(`[data-field="${name}"]`);
      const cost = (name) => {
        const raw = field(name).value.trim();
        return raw === "" ? null : Number(raw) / 1e6;
      };
      const payload = {
        enabled: field("enabled").checked ? 1 : 0,
        display_name: field("display_name").value.trim() || null,
        mode: field("mode").value.trim() || null,
        supports_tools: field("supports_tools").checked ? 1 : 0,
        supports_json: field("supports_json").checked ? 1 : 0,
        blocked: field("blocked").checked ? 1 : 0,
        context_window: field("context_window").value.trim() || null,
        input_cost_per_token: cost("input_cost"),
        output_cost_per_token: cost("output_cost"),
      };
      await api(`/api/provider-models/${id}`, { method: "PATCH", body: JSON.stringify(payload) });
    }
    $("#modal").classList.add("hidden");
    await refreshCatalog();
    toast("Model diperbarui");
  } catch (e) { toast(e.message, "err"); }
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
    if (state.cmpActiveRunId) {
      const data = await api(`/api/runs/${state.cmpActiveRunId}`);
      if (data.run.status !== "running") {
        const finished = state.cmpActiveRunId;
        state.cmpActiveRunId = null;
        await loadComparisonRuns();
        $("#cmp-run").value = String(finished);
        await loadComparison();
        toast("Comparison selesai");
      }
    }
    if (!state.funcRunId && !state.benchRunId && !state.perfRunId && !state.cmpActiveRunId) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch { /* transient */ }
  state.polling = false;
}

/* ---------- history ---------- */
async function loadHistory() {
  try {
    const data = await api("/api/runs");
    const runs = data.runs || [];
    let html = `<div class="scroll"><table class="tbl"><thead><tr>
      <th>#</th><th>Jenis</th><th>Mulai</th><th>Status</th><th>Ringkasan</th><th></th></tr></thead><tbody>`;
    for (const run of runs) {
      let summary = "";
      const s = run.summary || {};
      if (run.kind === "functional" && s.counts) summary = Object.entries(s.counts).map(([key, value]) => `${value} ${key}`).join(" · ");
      else if (s.outcomes) summary = Object.entries(s.outcomes).map(([key, value]) => `${value} ${OUTCOMES[key] ? OUTCOMES[key][1] : key}`).join(" · ");
      else if (s.counts) summary = Object.entries(s.counts).map(([key, value]) => `${value} ${key}`).join(" · ");
      if (s.regressions) summary += ` · ${s.regressions} regresi`;
      html += `<tr><td>${run.id}</td><td>${KIND_LABELS[run.kind] || run.kind}</td><td class="mono">${esc(run.started_at || "")}</td>
        <td>${kvBadge(MODEL_STATUS, run.status)}</td><td>${esc(summary)}</td>
        <td><button class="btn tiny" data-hist="${run.id}">Lihat</button></td></tr>`;
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
    $("#set-func-conc").value = s.FUNCTIONAL_CONCURRENCY || "4";
    $("#set-bench-conc").value = s.BENCH_CONCURRENCY || "2";
    $("#set-timeout").value = s.DEFAULT_TIMEOUT_S || "60";
    $("#set-token-cap").value = s.BENCH_TOKEN_CAP || "150000";
    $("#set-regression").value = s.REGRESSION_THRESHOLD_PCT || "25";
    $("#set-cmp-repeat").value = s.COMPARISON_REPEAT || "3";
    $("#set-cmp-window").value = s.COMPARISON_WINDOW || "5";
    $("#set-host").value = s.HOST || "";
    $("#set-port").value = s.PORT || "";
    $("#set-load-levels").value = s.LOAD_LEVELS || "1,2,4,8";
    $("#set-load-rpl").value = s.LOAD_REQUESTS_PER_LEVEL || "8";
    $("#set-context-sizes").value = s.CONTEXT_SIZES || "1000,8000,32000";
    $("#set-w-tokps").value = s.SCORE_W_TOKPS || "0.30";
    $("#set-w-ttft").value = s.SCORE_W_TTFT || "0.20";
    $("#set-w-pass").value = s.SCORE_W_PASS || "0.20";
    $("#set-w-itl").value = s.SCORE_W_ITL || "0.10";
    $("#set-w-latency").value = s.SCORE_W_LATENCY || "0.10";
    $("#set-w-cost").value = s.SCORE_W_COST || "0.10";
  } catch { /* ignore */ }
}

async function saveSettings() {
  const body = {
    FUNCTIONAL_CONCURRENCY: $("#set-func-conc").value,
    BENCH_CONCURRENCY: $("#set-bench-conc").value,
    DEFAULT_TIMEOUT_S: $("#set-timeout").value,
    BENCH_TOKEN_CAP: $("#set-token-cap").value,
    REGRESSION_THRESHOLD_PCT: $("#set-regression").value,
    COMPARISON_REPEAT: $("#set-cmp-repeat").value,
    COMPARISON_WINDOW: $("#set-cmp-window").value,
    HOST: $("#set-host").value.trim(),
    PORT: $("#set-port").value.trim(),
    LOAD_LEVELS: $("#set-load-levels").value.trim(),
    LOAD_REQUESTS_PER_LEVEL: $("#set-load-rpl").value,
    CONTEXT_SIZES: $("#set-context-sizes").value.trim(),
    SCORE_W_TOKPS: $("#set-w-tokps").value,
    SCORE_W_TTFT: $("#set-w-ttft").value,
    SCORE_W_PASS: $("#set-w-pass").value,
    SCORE_W_ITL: $("#set-w-itl").value,
    SCORE_W_LATENCY: $("#set-w-latency").value,
    SCORE_W_COST: $("#set-w-cost").value,
  };
  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    toast("Settings tersimpan");
    await Promise.all([refreshStatus(), loadSettings()]);
  } catch (e) { toast(e.message, "err"); }
}

/* ---------- wiring ---------- */
$("#refresh-models").addEventListener("click", async () => {
  await refreshCatalog();
  toast("Katalog dimuat ulang");
});
$("#provider-add").addEventListener("click", () => openProviderForm(null));
$("#fn-start").addEventListener("click", startFunctional);
$("#bench-start").addEventListener("click", startBench);
$("#perf-start").addEventListener("click", startPerf);
$("#cmp-start").addEventListener("click", startComparison);
$("#cmp-refresh").addEventListener("click", loadComparison);
$("#cmp-run").addEventListener("change", loadComparison);
$("#cmp-mode").addEventListener("change", loadComparison);
$("#fn-include-blocked").addEventListener("change", () => {
  renderPickerList("functional");
  renderPickerList("comparison");
});
$("#set-save").addEventListener("click", saveSettings);
$("#bench-task").addEventListener("change", () => {
  const task = state.tasks.find((item) => item.id === $("#bench-task").value);
  if (task && task.max_turns) $("#bench-max-turns").value = task.max_turns;
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
  buildPicker("comparison");
  await Promise.allSettled([refreshStatus(), loadSettings(), loadTasks()]);
  await refreshCatalog();
  await loadComparisonRuns();
  await loadComparison();
})();