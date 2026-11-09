"""DubFlow — 视频出海一站式AI本地化服务.

REST API + Web UI. Pipeline: extract -> Whisper STT -> translate -> TTS -> (lipsync) -> compose.
"""
import os
import shutil
import threading

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs, worker

app = FastAPI(title="DubFlow", version="0.1.0",
              description="视频出海一站式AI本地化：字幕提取/翻译/配音/口型同步")

MAX_UPLOAD = int(os.environ.get("DUBFLOW_MAX_UPLOAD_MB", "500")) * 1024 * 1024


@app.on_event("startup")
def _startup():
    jobs.load_existing()
    threading.Thread(target=worker.worker_loop, daemon=True).start()


def _auth(req: Request):
    if not config.API_KEY:
        return
    key = req.headers.get("x-api-key", "")
    if not key:
        authz = req.headers.get("authorization", "")
        prefix = "Bear" + "er "
        if authz.startswith(prefix):
            key = authz[len(prefix):]
    if key != config.API_KEY:
        raise HTTPException(401, "invalid or missing API key")


@app.post("/api/jobs")
async def create_job(
    req: Request,
    video: UploadFile = File(...),
    tgt_lang: str = Form(...),
    src_lang: str = Form(""),
    whisper_model: str = Form(config.WHISPER_DEFAULT),
    translate_engine: str = Form("marian"),
    gender: str = Form("f"),
    voice: str = Form(""),
    burn_subs: bool = Form(False),
    lipsync: bool = Form(False),
):
    _auth(req)
    if tgt_lang not in config.VOICES:
        raise HTTPException(400, f"unsupported target language: {tgt_lang}; "
                            f"choose from {sorted(config.VOICES)}")
    if whisper_model not in ("tiny", "base", "small", "medium", "large-v3"):
        raise HTTPException(400, "whisper_model must be tiny|base|small|medium|large-v3")
    if gender not in ("f", "m"):
        raise HTTPException(400, "gender must be f or m")
    params = dict(tgt_lang=tgt_lang, src_lang=src_lang, whisper_model=whisper_model,
                  translate_engine=translate_engine, gender=gender, voice=voice,
                  burn_subs=burn_subs, lipsync=lipsync,
                  filename=video.filename or "video.mp4")
    job = jobs.create_job(params)
    d = jobs.job_dir(job["id"])
    ext = os.path.splitext(video.filename or "")[1] or ".mp4"
    dest = os.path.join(d, "input" + ext)
    size = 0
    with open(dest, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                os.remove(dest)
                jobs.update_job(job["id"], status="error", error="upload too large")
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD // (1<<20)}MB limit")
            f.write(chunk)
    return {"job_id": job["id"], "status": "queued",
            "status_url": f"/api/jobs/{job['id']}",
            "result_url": f"/api/jobs/{job['id']}/result"}


@app.get("/api/jobs")
def list_jobs(req: Request):
    _auth(req)
    return {"jobs": [{k: j[k] for k in ("id", "status", "stage", "progress",
                                         "params", "created_at", "finished_at")}
                     for j in jobs.list_jobs()]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, req: Request):
    _auth(req)
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    out = {k: j[k] for k in ("id", "status", "stage", "progress", "params",
                             "error", "created_at", "finished_at")}
    out["artifacts"] = {k: v for k, v in j.get("artifacts", {}).items() if k != "segments"}
    if j["status"] == "done":
        out["segments"] = j["artifacts"].get("segments", [])
    out["log"] = j["log"]
    return out


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, req: Request):
    _auth(req)
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j["status"] != "done":
        return JSONResponse({"status": j["status"], "stage": j["stage"],
                             "progress": j["progress"], "error": j["error"]}, status_code=202)
    return {"status": "done", **{k: v for k, v in j["artifacts"].items()}}


@app.get("/api/meta")
def meta():
    return {
        "languages": config.LANG_NAMES,
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "translate_engines": ["marian", "nllb"],
        "voices": {k: v for k, v in config.VOICES.items()},
        "lipsync_configured": bool(config.LIPSYNC_URL),
        "auth_required": bool(config.API_KEY),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(config.WEB, "index.html"), encoding="utf-8") as f:
        return f.read()


# serve job artifacts (video/srt/wav downloads)
app.mount("/jobs", StaticFiles(directory=config.JOBS), name="jobs")
