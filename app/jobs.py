"""Job store + single background worker queue."""
import json
import os
import queue
import threading
import time
import uuid

from . import config

_jobs = {}          # id -> dict
_lock = threading.Lock()
_q = queue.Queue()


def job_dir(job_id: str) -> str:
    d = os.path.join(config.JOBS, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _persist(job: dict):
    try:
        with open(os.path.join(job_dir(job["id"]), "job.json"), "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def create_job(params: dict) -> dict:
    job_id = time.strftime("%Y%m%d-") + uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "status": "queued",        # queued|running|done|error
        "stage": "queued",
        "progress": 0.0,
        "params": params,
        "error": None,
        "artifacts": {},
        "created_at": time.time(),
        "finished_at": None,
        "log": [],
    }
    with _lock:
        _jobs[job_id] = job
    _persist(job)
    _q.put(job_id)
    return job


def update_job(job_id: str, **kw):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(kw)
        if "note" in kw:
            job["log"].append({"t": round(time.time(), 1), "msg": kw.pop("note")})
        _persist(job)


def log_job(job_id: str, msg: str):
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job["log"].append({"t": round(time.time(), 1), "msg": msg})
            _persist(job)


def get_job(job_id: str):
    return _jobs.get(job_id)


def list_jobs(limit=50):
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]


def next_job():
    return _q.get()


def load_existing():
    """Rehydrate job list on startup (mark interrupted as error)."""
    base = config.JOBS
    if not os.path.isdir(base):
        return
    for name in os.listdir(base):
        p = os.path.join(base, name, "job.json")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    job = json.load(f)
                if job.get("status") in ("queued", "running"):
                    job["status"] = "error"
                    job["error"] = "interrupted by server restart"
                _jobs[job["id"]] = job
            except Exception:
                pass
