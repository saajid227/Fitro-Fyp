function $(id) {
  return document.getElementById(id);
}

function show(el) {
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

function appendLog(text) {
  const box = $("logBox");
  box.textContent += text;
  box.scrollTop = box.scrollHeight;
}

async function startTraining() {
  const res = await fetch("/api/train/start", { method: "POST" });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Failed to start training");
  return data.job_id;
}

function connectStream(jobId) {
  const es = new EventSource(`/api/train/stream/${jobId}`);

  es.addEventListener("meta", (ev) => {
    $("trainState").textContent = `Job: ${jobId}`;
  });

  es.onmessage = (ev) => {
    appendLog(ev.data + "\n");
  };

  es.addEventListener("done", (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      if (!payload.ok) appendLog(`\nERROR: ${payload.error}\n`);

      const stats = payload.stats || { available: false };
      if (stats.available) {
        $("adjStat").textContent = stats.adjacent_error_rate || "—";
        $("nonAdjStat").textContent = stats.non_adjacent_error_rate || "—";
        $("reportBox").textContent = stats.report_text || "";
        show($("statsCard"));
      } else {
        $("reportBox").textContent = "No evaluation report found yet.";
      }
    } catch (e) {
      appendLog(`\n(Unable to parse done payload)\n`);
    } finally {
      $("startTrainBtn").disabled = false;
      $("startTrainBtn").textContent = "Start training";
      $("trainState").textContent = "Done";
      es.close();
    }
  });

  es.onerror = () => {
    appendLog("\n(Stream disconnected)\n");
  };
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = $("startTrainBtn");
  btn.addEventListener("click", async () => {
    hide($("statsCard"));
    $("logBox").textContent = "";
    btn.disabled = true;
    btn.textContent = "Starting...";
    $("trainState").textContent = "";

    try {
      const jobId = await startTraining();
      btn.textContent = "Training...";
      connectStream(jobId);
    } catch (err) {
      appendLog(String(err?.message || err) + "\n");
      btn.disabled = false;
      btn.textContent = "Start training";
    }
  });
});

