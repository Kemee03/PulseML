const $ = (selector) => document.querySelector(selector);
const num = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));

let latestPredictions = [];
let activePredictionFilter = "all";
let apiKeyPromptPromise = null;
const LAST_BATCH_KEY = "pulseml-last-batch";
const DASHBOARD_CACHE_KEY = "pulseml-dashboard-cache";

function showBatchRibbon(batch, live = false) {
  if (!batch?.batch_id) return;
  const count = batch.row_count ?? batch.count;
  const summary = `${count ? `${count} predictions · ` : ""}AstroLens v${batch.version ?? "?"} · Batch ID: ${batch.batch_id}`;
  $("#batch-ribbon").hidden = false;
  $("#batch-ribbon-title").textContent = batch.evaluated ? "Latest batch evaluated" : "Latest batch saved";
  $("#batch-ribbon-detail").textContent = summary;
  $("#batch-live-state").textContent = live ? "LIVE" : "LAST SAVED";
  $("#batch-ribbon").classList.toggle("is-stale", !live);
  localStorage.setItem(LAST_BATCH_KEY, JSON.stringify(batch));
}

function showCachedRegistry(models) {
  if (!Array.isArray(models) || !models.length) return;
  const markup = models.map((model) => {
    const production = model.deployed_version === model.version;
    return `<div class="model-card"><div class="model-left"><div class="model-icon">◈</div><div><div class="model-name">${escapeHtml(model.name)}</div><div class="model-meta">${escapeHtml(model.algorithm)} · v${escapeHtml(model.version)}</div></div></div><div class="model-right">${production ? '<span class="prod">● PRODUCTION</span>' : '<span class="staging">STAGING</span>'}</div></div>`;
  }).join("");
  $("#model-list").innerHTML = `<p class="cache-note">Last saved registry · reconnect to manage versions</p>${markup}`;
  $("#overview-model-list").innerHTML = `<p class="cache-note">Last saved registry · reconnect to manage versions</p>${markup}`;
  $("#model-count").textContent = $("#overview-model-count").textContent = String(models.length).padStart(2, "0");
}

try { const batch = JSON.parse(localStorage.getItem(LAST_BATCH_KEY) || "null"); if (batch) showBatchRibbon(batch); } catch {}
try { const cache = JSON.parse(localStorage.getItem(DASHBOARD_CACHE_KEY) || "null"); if (cache?.models) showCachedRegistry(cache.models); } catch {}

async function requestApiKey() {
  const savedKey = sessionStorage.getItem("pulseml-api-key");
  if (savedKey) return savedKey;
  if (!apiKeyPromptPromise) {
    apiKeyPromptPromise = Promise.resolve().then(() =>
      window.prompt("PulseML is protected. Enter the API key configured by the server:") || ""
    ).then((key) => {
      if (key) sessionStorage.setItem("pulseml-api-key", key);
      return key;
    }).finally(() => { apiKeyPromptPromise = null; });
  }
  return apiKeyPromptPromise;
}

async function api(url, options) {
  const send = (key) => {
    const headers = new Headers(options?.headers || {});
    if (key) headers.set("X-API-Key", key);
    return fetch(url, { ...options, headers });
  };
  let key = sessionStorage.getItem("pulseml-api-key") || "";
  let response = await send(key);
  if (response.status === 401 && url !== "/api/health") {
    if (key && sessionStorage.getItem("pulseml-api-key") === key) {
      sessionStorage.removeItem("pulseml-api-key");
    }
    key = await requestApiKey();
    if (!key) throw Error("API key required to use PulseML.");
    response = await send(key);
    if (response.status === 401 && sessionStorage.getItem("pulseml-api-key") === key) {
      sessionStorage.removeItem("pulseml-api-key");
    }
  }
  const data = await response.json();
  if (!response.ok) throw Error(data.detail || "Request failed");
  return data;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 3500);
}

function showEvaluation(result) {
  const matrix = result.confusion_matrix;
  const warningText = result.warning ? ` ⚠ ${result.warning}` : "";
  const element = $("#feedback-result");
  element.className = result.warning ? "panel-note metric-warning" : "panel-note";
  element.textContent = `AstroLens v${result.version} · ${result.count} labeled rows (${result.planet_rows} planet examples) · Accuracy ${(result.accuracy * 100).toFixed(1)}% · Precision ${(result.precision * 100).toFixed(1)}% · Recall ${(result.recall * 100).toFixed(1)}% · F1 ${(result.f1 * 100).toFixed(1)}%. Confusion matrix (TN/FP/FN/TP): ${matrix.true_negative}/${matrix.false_positive}/${matrix.false_negative}/${matrix.true_positive}. ${result.note}${warningText}`;
  const bars = [
    ["true-negative", "True no-planet", matrix.true_negative, "#9fc7ab"],
    ["false-positive", "False planet alarm", matrix.false_positive, "#e6b69d"],
    ["false-negative", "Missed planet", matrix.false_negative, "#d5c78d"],
    ["true-positive", "Found planet", matrix.true_positive, "#9990cf"],
  ];
  const graphic = $("#quality-visualization");
  graphic.hidden = false;
  graphic.innerHTML = `<div class="quality-topline"><b>What the labels say</b><span>${result.count} rows</span></div><div class="quality-bars">${bars.map(([type, label, count]) => count ? `<span class="${type}" style="width:${(count / result.count * 100).toFixed(2)}%" title="${label}: ${count}"></span>` : "").join("")}</div><div class="quality-key">${bars.map(([type, label, count, color]) => `<span><i style="background:${color}"></i>${label}: ${count}</span>`).join("")}</div>`;
}

function predictionRowHtml(prediction) {
  const failed = prediction.status !== "success";
  const result = failed ? "error" : prediction.prediction === 1 ? "planet" : "no_planet";
  const probability = failed ? "—" : `${(prediction.probability * 100).toFixed(1)}%`;
  return `<tr><td>${escapeHtml(prediction.request_id.slice(0, 8))}…</td><td><span class="tag${failed ? " error" : ""}">${result}</span></td><td>${probability}</td><td>v${escapeHtml(prediction.version)}</td><td>${num(prediction.latency_ms)} ms</td><td>${new Date(prediction.created_at).toLocaleTimeString()}</td></tr>`;
}

function renderPredictionRows() {
  const filtered = latestPredictions.filter((prediction) => {
    if (activePredictionFilter === "all") return true;
    if (activePredictionFilter === "error") return prediction.status !== "success";
    return prediction.status === "success" && (prediction.prediction === 1 ? "planet" : "no_planet") === activePredictionFilter;
  });
  $("#prediction-rows").innerHTML = filtered.map(predictionRowHtml).join("") || `<tr><td colspan="6" class="empty">${activePredictionFilter === "all" ? "No predictions yet. Try a sample prediction or upload a CSV." : "No recent predictions match this filter."}</td></tr>`;
  $("#overview-prediction-count").textContent = String(latestPredictions.length).padStart(2, "0");
  $("#overview-prediction-rows").innerHTML = latestPredictions.slice(0, 5).map(predictionRowHtml).join("") || '<tr><td colspan="6" class="empty">No predictions yet. Run a sample prediction to get started.</td></tr>';
}

function drawLatencyChart(predictions) {
  const chart = $("#chart");
  const rows = predictions.filter((prediction) => prediction.status === "success").slice().reverse();
  if (!rows.length) {
    chart.innerHTML = '<span class="empty-chart">Make a prediction and your first dots will appear here.</span>';
    return;
  }
  const width = 640;
  const height = 114;
  const values = rows.map((prediction) => Number(prediction.latency_ms));
  const ceiling = Math.max(10, Math.ceil(Math.max(...values) / 10) * 10);
  const points = values.map((value, index) => {
    const x = rows.length === 1 ? width / 2 : 8 + index * (width - 16) / (rows.length - 1);
    const y = height - 7 - Math.min(value, ceiling) / ceiling * (height - 18);
    return { x, y, value };
  });
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `8,${height} ${line} ${width - 8},${height}`;
  chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Recent inference latency line chart"><defs><linearGradient id="latency-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#b6acdf" stop-opacity=".36"/><stop offset="100%" stop-color="#b6acdf" stop-opacity=".01"/></linearGradient></defs><polygon points="${area}" fill="url(#latency-fill)"/><polyline points="${line}" fill="none" stroke="#8d82c1" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>${points.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="3.1" fill="#fffefa" stroke="#8176b7" stroke-width="2" vector-effect="non-scaling-stroke"><title>${point.value.toFixed(2)} ms</title></circle>`).join("")}</svg>`;
}

function showPage(pageName, updateLocation = true) {
  const page = document.getElementById(`page-${pageName}`);
  if (!page) pageName = "overview";
  document.querySelectorAll(".page").forEach((item) => {
    const active = item.id === `page-${pageName}`;
    item.hidden = !active;
    item.classList.toggle("active", active);
  });
  document.querySelectorAll(".nav-link").forEach((button) => {
    const active = button.dataset.page === pageName;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  const crumb = $("#page-crumb");
  if (crumb) crumb.textContent = document.querySelector(`#page-${pageName}`).dataset.title;
  if (updateLocation && location.hash !== `#${pageName}`) history.pushState({ pageName }, "", `#${pageName}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function restoreLatestFeedback() {
  try {
    const result = await api("/api/feedback/latest");
    if (!result.batch_id) return;
    showBatchRibbon({ ...result, row_count: result.count }, true);
    $("#feedback-batch-id").value = result.batch_id;
    if (result.evaluated) {
      showEvaluation(result);
    $("#batch-result").textContent = `Restored the saved ${result.count}-row prediction batch for AstroLens v${result.version}. Batch ID: ${result.batch_id}.`;
    } else {
      $("#feedback-result").textContent = `${result.message} Batch ID: ${result.batch_id}.`;
      $("#batch-result").textContent = `Latest batch: AstroLens v${result.version}. Batch ID: ${result.batch_id}.`;
    }
  } catch (error) { console.error(error); }
}

async function refresh() {
  try {
    const health = await api("/api/health");
    $("#api-health").textContent = health.model_loaded ? "API healthy" : "Model unavailable";
    $("#signal-api").textContent = health.model_loaded ? "Healthy" : "No model";
    $("#overview-live-api").textContent = health.model_loaded ? "Healthy" : "No model";
    const [metrics, monitor, registry, predictions, drift, alerts] = await Promise.all([
      api("/api/metrics"), api("/api/monitoring"), api("/api/models"),
      api("/api/predictions?limit=12"), api("/api/drift"), api("/api/alerts"),
    ]);
    localStorage.setItem(DASHBOARD_CACHE_KEY, JSON.stringify({ models: registry.models, predictions: predictions.predictions, saved_at: new Date().toISOString() }));
    const savedBatch = JSON.parse(localStorage.getItem(LAST_BATCH_KEY) || "null");
    if (savedBatch) showBatchRibbon(savedBatch, true);
    $("#requests").textContent = num(metrics.requests);
    $("#latency").textContent = num(metrics.avg_latency);
    $("#errors").textContent = num(metrics.error_rate);
    $("#drift").textContent = monitor.prediction_drift === "normal" ? "Normal" : monitor.prediction_drift === "insufficient_data" ? "Need data" : "Investigate";
    $("#drift").style.color = monitor.prediction_drift === "normal" ? "var(--green)" : monitor.prediction_drift === "insufficient_data" ? "#8a94a2" : "#c38a28";
    $("#positive-rate").textContent = `${(monitor.prediction_positive_rate * 100).toFixed(1)}% positive`;
    $("#overview-positive-rate").textContent = `${(monitor.prediction_positive_rate * 100).toFixed(1)}%`;
    $("#planet-share").textContent = `${(monitor.prediction_positive_rate * 100).toFixed(1)}%`;
    $("#planet-count").textContent = num(metrics.positive);
    $("#no-planet-count").textContent = num(metrics.negative);
    $("#distribution-ring").style.setProperty("--planet-share", `${monitor.prediction_positive_rate * 100}%`);
    $("#feature-drift-status").textContent = drift.status === "ready"
      ? drift.severity.toUpperCase()
      : drift.status === "insufficient_data" ? "Need data" : "Unavailable";
    $("#feature-drift-status").className = drift.severity === "high" ? "watch high" : drift.severity === "moderate" ? "watch moderate" : "watch";
    $("#overview-feature-drift").textContent = drift.status === "ready"
      ? drift.severity.toUpperCase()
      : drift.status === "insufficient_data" ? "Need data" : "Unavailable";
    $("#drift-sample-count").textContent = `${drift.sample_count} / ${drift.minimum_sample_count}`;
    $("#drift-message").textContent = drift.message;
    $("#drift-features").innerHTML = drift.top_features.length
      ? drift.top_features.slice(0, 5).map((feature) => `<div class="feature-row"><span>${escapeHtml(feature.feature)}</span><span class="psi-value">${Number(feature.psi).toFixed(4)}</span><span class="severity-pill ${escapeHtml(feature.severity)}">${escapeHtml(feature.severity)}</span></div>`).join("")
      : '<div class="loading">A report appears after at least 1,000 predictions on the current production version.</div>';
    $("#alert-count").textContent = String(alerts.alerts.length).padStart(2, "0");
    $("#nav-alert-count").textContent = String(alerts.alerts.length);
    $("#overview-open-alerts").textContent = String(alerts.alerts.length);
    $("#alert-list").innerHTML = alerts.alerts.length
      ? alerts.alerts.map((alert) => `<div class="alert-row"><div class="alert-mark">!</div><div class="alert-copy"><b>${escapeHtml(alert.title)}</b><small>${escapeHtml(alert.detail)}</small><small>v${escapeHtml(alert.version)} · ${new Date(alert.created_at).toLocaleString()}</small></div><button class="small-action" data-resolve="${alert.id}">Resolve</button></div>`).join("")
      : '<div class="loading">No open alerts.</div>';
    $("#model-count").textContent = String(registry.models.length).padStart(2, "0");
    $("#overview-model-count").textContent = String(registry.models.length).padStart(2, "0");
    const modelMarkup = registry.models.map((model) => {
      const production = model.deployed_version === model.version;
      return `<div class="model-card"><div class="model-left"><div class="model-icon">◈</div><div><div class="model-name">${escapeHtml(model.name)}</div><div class="model-meta">${escapeHtml(model.algorithm)} · v${escapeHtml(model.version)}</div></div></div><div class="model-right">${production ? '<span class="prod">● PRODUCTION</span>' : `<span class="staging">STAGING</span><button class="small-action" data-promote="${escapeHtml(model.model_id)}" data-version="${escapeHtml(model.version)}">Promote</button>`}</div></div>`;
    }).join("") || '<div class="loading">No models registered.</div>';
    $("#model-list").innerHTML = modelMarkup;
    $("#overview-model-list").innerHTML = modelMarkup;

    const deployedModel = registry.models.find((model) => model.deployed_version === model.version);
    $("#production-summary").innerHTML = deployedModel
      ? `<div class="production-version"><span class="production-glyph">◈</span><span><b>${escapeHtml(deployedModel.name)}</b><small>${escapeHtml(deployedModel.algorithm)} · v${escapeHtml(deployedModel.version)}</small></span><span class="production-badge">ON DUTY</span></div>`
      : '<div class="loading">No model is in production yet.</div>';

    const deployed = registry.models.find((model) => model.deployed_version === model.version);
    $("#rollback").disabled = !deployed || !deployed.has_previous;
    $("#rollback").title = deployed?.has_previous ? "Restore the previous production version" : "No previous production version yet";

    latestPredictions = predictions.predictions;
    $("#prediction-count").textContent = String(latestPredictions.length).padStart(2, "0");
    renderPredictionRows();
    drawLatencyChart(latestPredictions);
  } catch (error) {
    const connected = await fetch("/api/health").then((response) => response.ok).catch(() => false);
    if (!connected) {
      $("#api-health").textContent = "API offline";
      $("#signal-api").textContent = "Offline";
      $("#overview-live-api").textContent = "Offline";
    }
    const cache = JSON.parse(localStorage.getItem(DASHBOARD_CACHE_KEY) || "null");
    if (cache?.predictions && !latestPredictions.length) { latestPredictions = cache.predictions; renderPredictionRows(); }
    if (cache?.models) showCachedRegistry(cache.models);
    $("#overview-feature-drift").textContent = "Unavailable";
    console.error(error);
  }
}

document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => showPage(button.dataset.page)));
document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => showPage(button.dataset.go)));
document.querySelectorAll("[data-pred-filter]").forEach((button) => button.addEventListener("click", () => {
  activePredictionFilter = button.dataset.predFilter;
  document.querySelectorAll("[data-pred-filter]").forEach((item) => {
    const selected = item === button;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-pressed", String(selected));
  });
  renderPredictionRows();
}));
document.querySelectorAll("[data-pred-filter]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.predFilter === activePredictionFilter)));
window.addEventListener("popstate", () => showPage(location.hash.slice(1) || "overview", false));
window.addEventListener("hashchange", () => showPage(location.hash.slice(1) || "overview", false));
showPage(location.hash.slice(1) || "overview", false);

$("#refresh").addEventListener("click", refresh);
$("#predict").addEventListener("click", async () => {
  const button = $("#predict");
  button.disabled = true;
  button.firstChild.textContent = "Running demo… ";
  try {
    const result = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: Array(3197).fill(0) }),
    });
    toast(`Prediction saved · ${result.class_name} · ${result.latency_ms} ms`);
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.firstChild.textContent = "Run a demo prediction "; }
});

$("#show-upload").addEventListener("click", () => {
  const form = $("#upload-form");
  form.hidden = !form.hidden;
  if (!form.hidden) form.querySelector('[name="version"]').focus();
});

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = "Checking and registering…";
  try {
    const result = await api("/api/models/register", { method: "POST", body: new FormData(event.currentTarget) });
    toast(`${result.name} v${result.version} registered in staging`);
    event.currentTarget.reset();
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = "Upload as staging"; }
});

$("#batch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = "Predicting…";
  $("#batch-result").textContent = "Uploading the CSV and recording predictions…";
  try {
    const result = await api("/api/predict/batch", { method: "POST", body: new FormData(form) });
    $("#batch-result").textContent = `${result.count} predictions saved with AstroLens v${result.version}. Batch ID: ${result.batch_id}. Total model time: ${result.batch_latency_ms} ms. ${result.note}`;
    showBatchRibbon({ ...result, row_count: result.count }, true);
    $("#feedback-batch-id").value = result.batch_id;
    $("#feedback-result").className = "panel-note";
    $("#feedback-result").textContent = `Batch created. Submit the same CSV here to evaluate it. Batch ID: ${result.batch_id}.`;
    toast(`${result.count} batch predictions saved`);
    form.reset();
    await refresh();
  } catch (error) {
    $("#batch-result").textContent = error.message;
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run batch predictions";
  }
});

$("#feedback-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = "Calculating metrics…";
  try {
    const result = await api("/api/feedback/batch", { method: "POST", body: new FormData(form) });
    showEvaluation(result);
    showBatchRibbon({ ...result, row_count: result.count, evaluated: true }, true);
    toast("Labeled batch evaluated");
    await refresh();
  } catch (error) {
    $("#feedback-result").textContent = error.message;
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Evaluate labeled predictions";
  }
});

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-promote]");
  if (!button) return;
  button.disabled = true;
  try {
    const result = await api(`/api/models/${encodeURIComponent(button.dataset.promote)}/${encodeURIComponent(button.dataset.version)}/promote`, { method: "POST" });
    toast(`Production is now ${result.model} v${result.production_version}`);
    await refresh();
  } catch (error) { toast(error.message); button.disabled = false; }
});

$("#rollback").addEventListener("click", async () => {
  const deployed = await api("/api/models").then((data) => data.models.find((model) => model.deployed_version === model.version));
  if (!deployed) return;
  const button = $("#rollback");
  button.disabled = true;
  try {
    const result = await api(`/api/models/${encodeURIComponent(deployed.model_id)}/rollback`, { method: "POST" });
    toast(`Rolled back to ${result.model} v${result.production_version}`);
    await refresh();
  } catch (error) { toast(error.message); }
});

$("#alert-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-resolve]");
  if (!button) return;
  button.disabled = true;
  try {
    await api(`/api/alerts/${button.dataset.resolve}/resolve`, { method: "POST" });
    toast("Alert resolved");
    await refresh();
  } catch (error) { toast(error.message); button.disabled = false; }
});

refresh();
restoreLatestFeedback();
setInterval(refresh, 10000);

$("#copy-batch-id").addEventListener("click", async () => {
  const batch = JSON.parse(localStorage.getItem(LAST_BATCH_KEY) || "null");
  if (!batch?.batch_id) return;
  try { await navigator.clipboard.writeText(batch.batch_id); toast("Batch ID copied"); }
  catch { toast(batch.batch_id); }
});
