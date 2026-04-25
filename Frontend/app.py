import io
import json
import logging
import queue
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is on sys.path so `import main` / `import src.*` work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import FitaroPredictor, InputValidationError  # noqa: E402

app = FastAPI(title="Fitaro : AI Garment Size Recommender")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")


def _parse_latest_eval_report() -> Dict[str, Any]:
    report_path = PROJECT_ROOT / "outputs" / "reports" / "evaluation_report.txt"
    if not report_path.exists():
        return {"available": False}

    text = report_path.read_text(encoding="utf-8", errors="replace")
    adj = None
    non_adj = None
    for line in text.splitlines():
        if "Adjacent-size error rate" in line:
            adj = line.split(":", 1)[-1].strip()
        if "Non-adjacent error rate" in line:
            non_adj = line.split(":", 1)[-1].strip()

    return {
        "available": True,
        "path": str(report_path),
        "adjacent_error_rate": adj,
        "non_adjacent_error_rate": non_adj,
        "report_text": text,
    }


def _read_latest_model_meta() -> Dict[str, Any]:
    meta_path = PROJECT_ROOT / "outputs" / "models" / "latest_model.json"
    if not meta_path.exists():
        return {"available": False}
    try:
        return {"available": True, **json.loads(meta_path.read_text(encoding="utf-8"))}
    except Exception:
        return {"available": False}


def _list_plot_images() -> list[Dict[str, str]]:
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    if not plots_dir.exists():
        return []
    imgs = []
    for p in sorted(plots_dir.glob("*.png")):
        imgs.append({"name": p.name, "url": f"/outputs/plots/{p.name}"})
    return imgs


class _Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.created_at = time.time()
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.done = False
        self.ok: Optional[bool] = None
        self.error: Optional[str] = None
        self.stats: Dict[str, Any] = {"available": False}


_jobs: Dict[str, _Job] = {}
_jobs_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Fitaro : AI Garment Size Recommender"},
    )


@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "model.html",
        {"request": request, "title": "Fitaro : AI Garment Size Recommender"},
    )


@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "train.html",
        {"request": request, "title": "Fitaro : AI Garment Size Recommender"},
    )


@app.get("/api/model/info")
def api_model_info() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "meta": _read_latest_model_meta(),
            "report": _parse_latest_eval_report(),
            "plots": _list_plot_images(),
        }
    )


@app.post("/api/predict")
async def api_predict(payload: Dict[str, Any]) -> JSONResponse:
    try:
        predictor = FitaroPredictor()
        result = predictor.predict(payload)
        return JSONResponse({"ok": True, "result": result})
    except FileNotFoundError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "hint": "Train first (python main.py) or use Train button."},
            status_code=400,
        )
    except InputValidationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Unexpected error: {exc}"}, status_code=500)


def _run_training(job: _Job) -> None:
    try:
        import main as fitaro_main  # imported here to keep import time minimal

        # Capture print() and anything writing to stdout/stderr (incl. sklearn/xgb output)
        buf_out = io.StringIO()
        buf_err = io.StringIO()

        # Also push python logging to our queue.
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        class _QueueHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                except Exception:
                    msg = record.getMessage()
                job.log_q.put(msg + "\n")

        qh = _QueueHandler()
        qh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s", "%H:%M:%S"))
        root_logger.addHandler(qh)

        job.log_q.put("Starting training...\n")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            try:
                fitaro_main.main()
            except SystemExit as exc:
                # main.py uses sys.exit on failures; treat non-zero as error
                code = getattr(exc, "code", 1)
                if code not in (0, None):
                    raise RuntimeError(f"Training exited with code {code}")

        # Flush captured streams into log queue
        out_text = buf_out.getvalue()
        err_text = buf_err.getvalue()
        if out_text.strip():
            job.log_q.put("\n--- STDOUT ---\n" + out_text + "\n")
        if err_text.strip():
            job.log_q.put("\n--- STDERR ---\n" + err_text + "\n")

        job.stats = _parse_latest_eval_report()
        job.ok = True
        job.done = True
        job.log_q.put("\nTraining complete.\n")
    except Exception as exc:
        job.ok = False
        job.done = True
        job.error = str(exc)
        job.log_q.put(f"\nERROR: {exc}\n")


@app.post("/api/train/start")
def api_train_start() -> JSONResponse:
    job = _Job()
    with _jobs_lock:
        _jobs[job.id] = job

    t = threading.Thread(target=_run_training, args=(job,), daemon=True)
    t.start()
    return JSONResponse({"ok": True, "job_id": job.id})


@app.get("/api/train/status/{job_id}")
def api_train_status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return JSONResponse(
        {
            "ok": True,
            "job": {
                "id": job.id,
                "done": job.done,
                "ok": job.ok,
                "error": job.error,
                "stats": job.stats,
            },
        }
    )


@app.get("/api/train/stream/{job_id}")
def api_train_stream(job_id: str) -> StreamingResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return StreamingResponse(iter([b"event: error\ndata: job not found\n\n"]), media_type="text/event-stream")

    def event_iter():
        # initial handshake
        yield "event: meta\ndata: " + json.dumps({"job_id": job.id}) + "\n\n"
        while True:
            try:
                line = job.log_q.get(timeout=0.5)
                # SSE: send each chunk as data
                for chunk in line.splitlines(True):
                    yield "data: " + chunk.rstrip("\n") + "\n\n"
            except queue.Empty:
                if job.done:
                    payload = {"done": True, "ok": job.ok, "error": job.error, "stats": job.stats}
                    yield "event: done\ndata: " + json.dumps(payload) + "\n\n"
                    break
                # keep-alive ping
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("Frontend.app:app", host="0.0.0.0", port=port, reload=False)
