function $(id) {
  return document.getElementById(id);
}

function show(el) {
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : value;
}

function apiBase() {
  const base = String(window.FITARO_API_BASE || "").trim();
  return base ? base.replace(/\/+$/, "") : "";
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities || {});
  entries.sort((a, b) => b[1] - a[1]);
  return entries
    .map(([size, prob]) => {
      const pct = Math.round(prob * 1000) / 10;
      const w = Math.max(2, Math.min(100, pct));
      return `
        <div class="prob-row">
          <div class="prob-size">${size}</div>
          <div class="prob-bar"><div class="prob-fill" style="width:${w}%"></div></div>
          <div class="prob-pct">${pct}%</div>
        </div>
      `;
    })
    .join("");
}

async function predict(payload) {
  const url = `${apiBase()}/api/predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (!ct.includes("application/json")) {
    const text = await res.text();
    const snippet = String(text || "").trim().slice(0, 200);
    throw new Error(
      `Backend did not return JSON (HTTP ${res.status}). Check window.FITARO_API_BASE. Response starts with: ${JSON.stringify(snippet)}`
    );
  }

  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data?.error || `Prediction failed (HTTP ${res.status})`);
  return data.result;
}

window.addEventListener("DOMContentLoaded", () => {
  const form = $("predictForm");
  const predictBtn = $("predictBtn");
  const resultCard = $("resultCard");
  const errorCard = $("errorCard");
  const errorOut = $("errorOut");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hide(errorCard);
    hide(resultCard);
    predictBtn.disabled = true;
    predictBtn.textContent = "Recommending...";

    try {
      const fd = new FormData(form);
      const payload = {};
      for (const [k, v] of fd.entries()) payload[k] = toNumber(v);

      const result = await predict(payload);
      $("sizeOut").textContent = result.size;
      $("confidencePill").textContent = `Confidence: ${Math.round(result.confidence * 1000) / 10}%`;
      $("probOut").innerHTML = renderProbabilities(result.probabilities);
      $("justOut").textContent = result.justification || "";
      show(resultCard);
    } catch (err) {
      const base = apiBase();
      const hint = base
        ? ""
        : "\n\nConfigure docs/static/config.js with your backend URL (window.FITARO_API_BASE).";
      errorOut.textContent = String(err?.message || err) + hint;
      show(errorCard);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Recommend me";
    }
  });
});

