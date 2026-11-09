"""DubFlow MCP server (stdio) — exposes the dubbing pipeline as MCP tools.

Run:  python -m mcp_server   (from dubflow root, with .venv activated)
Client config example (Claude Desktop / Hermes):
  {"mcpServers": {"dubflow": {"command": "/home/ec2-user/dubflow/.venv/bin/python",
                              "args": ["-m", "mcp_server"],
                              "cwd": "/home/ec2-user/dubflow"}}}
"""
import os
import sys
import time
import base64
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

from app import config, jobs, worker

mcp = _Server("dubflow")

_worker_started = False


def _ensure_worker():
    global _worker_started
    if not _worker_started:
        import threading
        jobs.load_existing()
        threading.Thread(target=worker.worker_loop, daemon=True).start()
        _worker_started = True


@mcp.tool()
def dubflow_capabilities() -> dict:
    """List supported target languages, Whisper models, and translation engines."""
    return {
        "languages": config.LANG_NAMES,
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "translate_engines": ["marian", "nllb"],
        "lipsync_configured": bool(config.LIPSYNC_URL),
    }


@mcp.tool()
def dubflow_submit(video_path: str, tgt_lang: str, src_lang: str = "",
                   whisper_model: str = "small", translate_engine: str = "marian",
                   gender: str = "f", voice: str = "", burn_subs: bool = False,
                   lipsync: bool = False) -> dict:
    """Submit a local video file for AI dubbing into tgt_lang.

    video_path: absolute path to an mp4/mov/mkv on this machine.
    Returns job_id; poll with dubflow_status.
    """
    _ensure_worker()
    if not os.path.isfile(video_path):
        raise ValueError(f"video not found: {video_path}")
    if tgt_lang not in config.VOICES:
        raise ValueError(f"unsupported tgt_lang; choose from {sorted(config.VOICES)}")
    params = dict(tgt_lang=tgt_lang, src_lang=src_lang, whisper_model=whisper_model,
                  translate_engine=translate_engine, gender=gender, voice=voice,
                  burn_subs=burn_subs, lipsync=lipsync,
                  filename=os.path.basename(video_path))
    job = jobs.create_job(params)
    d = jobs.job_dir(job["id"])
    ext = os.path.splitext(video_path)[1] or ".mp4"
    _copy(video_path, os.path.join(d, "input" + ext))
    return {"job_id": job["id"], "status": "queued"}


def _copy(src, dst, limit=500 * 1024 * 1024):
    size = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise ValueError("file exceeds 500MB")
            fo.write(chunk)


@mcp.tool()
def dubflow_status(job_id: str) -> dict:
    """Check progress of a dubbing job: status, stage, progress, log, error."""
    j = jobs.get_job(job_id)
    if not j:
        raise ValueError(f"job not found: {job_id}")
    out = {k: j[k] for k in ("id", "status", "stage", "progress", "error")}
    out["artifacts"] = {k: v for k, v in j.get("artifacts", {}).items() if k != "segments"}
    if j["status"] == "done":
        out["output_file"] = os.path.join(config.JOBS, job_id, "output.mp4")
        out["translated_srt"] = os.path.join(config.JOBS, job_id, "translated.srt")
    out["log_tail"] = j["log"][-5:]
    return out


@mcp.tool()
def dubflow_wait(job_id: str, timeout_s: int = 900) -> dict:
    """Block until a dubbing job finishes (done/error) or timeout. Returns final status."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = jobs.get_job(job_id)
        if j and j["status"] in ("done", "error"):
            return dubflow_status(job_id)
        time.sleep(3)
    return {"status": "timeout", "job_id": job_id}


@mcp.tool()
def dubflow_transcribe(video_path: str, whisper_model: str = "small",
                       language: str = "") -> dict:
    """STT only: extract audio from a video and transcribe with Whisper.
    Returns segments with timestamps + SRT content (no translation/dubbing)."""
    from app import pipeline
    if not os.path.isfile(video_path):
        raise ValueError(f"video not found: {video_path}")
    outdir = pathlib.Path(video_path).parent
    wav = str(outdir / (pathlib.Path(video_path).stem + "_16k.wav"))
    pipeline.extract_audio(video_path, wav)
    segs, lang = pipeline.transcribe(wav, whisper_model, language)
    srt_path = str(outdir / (pathlib.Path(video_path).stem + ".srt"))
    pipeline.write_srt(segs, srt_path)
    os.remove(wav)
    return {"language": lang, "segments": segs, "srt_file": srt_path}


@mcp.tool()
def dubflow_translate(texts: list, src_lang: str, tgt_lang: str,
                      engine: str = "marian") -> dict:
    """Translate a list of texts using MarianMT or NLLB-200. No video needed."""
    from app import pipeline
    if tgt_lang not in config.VOICES:
        raise ValueError(f"unsupported tgt_lang; choose from {sorted(config.VOICES)}")
    out = pipeline.translate_texts(list(texts), src_lang, tgt_lang, engine)
    return {"src_lang": src_lang, "tgt_lang": tgt_lang, "translations": out}


if __name__ == "__main__":
    _ensure_worker()
    mcp.run(transport="stdio")
