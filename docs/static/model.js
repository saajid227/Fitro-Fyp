function $(id) {
  return document.getElementById(id);
}

function show(el) {
  el.classList.remove("hidden");
}

function appendLog(text) {
  const box = $("logBox");
  box.textContent += text;
  box.scrollTop = box.scrollHeight;
}

function apiBase() {
  const base = String(window.FITARO_API_BASE || "").trim();
  return base ? base.replace(/\/+$/, "") : "";
}

async function fetchModelInfo() {
  const url = `${apiBase()}/api/model/info`;
  const res = await fetch(url);
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Failed to load model info");
  return data;
}

function renderPlots(plots) {
  const grid = $("plotsGrid");
  grid.innerHTML = (plots || [])
    .map(
      (p) => `
        <a class="plot-card" href="${apiBase()}${p.url}" target="_blank" rel="noreferrer">
          <div class="plot-name">${p.name}</div>
          <img class="plot-img" src="${apiBase()}${p.url}" alt="${p.name}" loading="lazy" />
        </a>
      `
    )
    .join("");
}

async function startTraining() {
  const url = `${apiBase()}/api/train/start`;
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Failed to start training");
  return data.job_id;
}

function connectStream(jobId, onDone) {
  const url = `${apiBase()}/api/train/stream/${jobId}`;
  const es = new EventSource(url);

  es.onmessage = (ev) => {
    appendLog(ev.data + "\n");
  };

  es.addEventListener("done", (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      onDone(payload);
    } finally {
      es.close();
    }
  });

  es.onerror = () => {
    appendLog("\n(Stream disconnected)\n");
  };
}

async function loadAndRender() {
  const data = await fetchModelInfo();

  const meta = data.meta || { available: false };
  $("trainedAt").textContent = meta.trained_at || "—";

  const reportText = (data.report && data.report.available && data.report.report_text) || "";
  $("reportBox").textContent = reportText || "No evaluation report found yet. Train the model first.";

  const lines = reportText.split("\n");
  const findLine = (prefix) => lines.find((l) => l.startsWith(prefix));
  $("accStat").textContent = (findLine("Accuracy") || "").split(":", 2)[1]?.trim() || "—";
  $("f1Stat").textContent = (findLine("Macro-F1") || "").split(":", 2)[1]?.trim() || "—";
  $("adjStat").textContent = (findLine("Adjacent-size error rate") || "").split(":", 2)[1]?.trim() || "—";

  renderPlots(data.plots || []);
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadAndRender();
  } catch (err) {
    $("reportBox").textContent =
      String(err?.message || err) +
      "\n\nConfigure docs/static/config.js with your backend URL (window.FITARO_API_BASE).";
  }

  const btn = $("trainAgainBtn");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Starting...";
    $("trainState").textContent = "";
    $("logBox").textContent = "";
    show($("logCard"));

    try {
      const jobId = await startTraining();
      btn.textContent = "Training...";
      $("trainState").textContent = `Job: ${jobId}`;
      connectStream(jobId, async (payload) => {
        if (!payload.ok) appendLog(`\nERROR: ${payload.error}\n`);
        btn.disabled = false;
        btn.textContent = "Train again";
        $("trainState").textContent = "Done";
        await loadAndRender();
      });
    } catch (err) {
      appendLog(String(err?.message || err) + "\n");
      btn.disabled = false;
      btn.textContent = "Train again";
    }
  });
});

