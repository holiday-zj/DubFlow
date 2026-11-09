"""Worker thread: runs the 5-stage pipeline for each job."""
import os
import traceback

from . import config, jobs, pipeline

STAGES = [
    ("extract", 0.05, "抽取音轨 (16kHz mono WAV)"),
    ("stt", 0.40, "语音识别 (Whisper)"),
    ("translate", 0.55, "翻译 (MarianMT/NLLB)"),
    ("tts", 0.85, "配音合成 (TTS)"),
    ("compose", 1.00, "合成输出 (FFmpeg)"),
]


def run_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        return
    d = jobs.job_dir(job_id)
    p = job["params"]
    try:
        jobs.update_job(job_id, status="running", stage="extract", progress=0.02,
                        note="开始处理")
        video = os.path.join(d, "input.mp4")
        if not os.path.isfile(video):
            # uploaded file saved under original ext
            for f in os.listdir(d):
                if f.startswith("input."):
                    video = os.path.join(d, f)
                    break

        # 1. extract audio
        wav = os.path.join(d, "audio_16k.wav")
        pipeline.extract_audio(video, wav)
        dur = pipeline.probe_duration(video)
        jobs.update_job(job_id, stage="stt", progress=0.08,
                        note=f"音轨抽取完成，时长 {dur:.1f}s，开始转录")

        # 2. STT
        segs, detected = pipeline.transcribe(wav, p.get("whisper_model", config.WHISPER_DEFAULT),
                                             p.get("src_lang", ""))
        src_lang = p.get("src_lang") or detected
        pipeline.write_srt(segs, os.path.join(d, "source.srt"))
        jobs.update_job(job_id, stage="translate", progress=0.42,
                        note=f"转录完成（{len(segs)} 段，源语言 {src_lang}），开始翻译")
        if not segs:
            raise RuntimeError("未识别到任何语音内容")

        # 3. translate
        tgt = p["tgt_lang"]
        texts = [s["text"] for s in segs]
        trans = pipeline.translate_texts(texts, src_lang, tgt,
                                         p.get("translate_engine", "marian"))
        for s, t in zip(segs, trans):
            s["text_trans"] = t
        pipeline.write_srt(segs, os.path.join(d, "translated.srt"), field="text_trans")
        jobs.update_job(job_id, stage="tts", progress=0.58, note="翻译完成，开始配音")

        # 4. TTS -> full dubbed timeline
        voice = p.get("voice")
        if not voice:
            gender = p.get("gender", "f")
            voice = config.VOICES.get(tgt, config.VOICES["en"])[gender]
        dubbed = os.path.join(d, "dubbed_audio.wav")
        pipeline.build_dub_audio(segs, voice, dur, dubbed,
                                 on_error=lambda m: jobs.log_job(job_id, m))
        jobs.update_job(job_id, stage="bgm", progress=0.80, note="配音完成")

        # 4a. keep BGM/SFX: separate vocals from original audio, remix
        final_audio = dubbed
        bgm_ok = False
        if p.get("keep_bgm"):
            try:
                no_vocals = pipeline.separate_bgm(video, d)
                final_audio = os.path.join(d, "final_audio.wav")
                pipeline.mix_bgm(dubbed, no_vocals, final_audio, dur,
                                 bgm_volume=float(p.get("bgm_volume") or 1.0))
                bgm_ok = True
                jobs.log_job(job_id, "背景音乐/音效已分离并混回配音")
            except Exception as e:
                final_audio = dubbed
                jobs.log_job(job_id, f"背景音乐分离失败，仅保留配音: {e}")
        jobs.update_job(job_id, stage="lipsync", progress=0.86, note="配音完成")

        # 4b. lip-sync (optional external service)
        lip_ok = False
        base_video = video
        if p.get("lipsync"):
            try:
                lipsynced = os.path.join(d, "lipsynced.mp4")
                lip_ok = pipeline.lipsync(video, final_audio, lipsynced)
                if lip_ok:
                    base_video = lipsynced
                    jobs.log_job(job_id, "口型同步完成（外部服务）")
                else:
                    jobs.log_job(job_id, "未配置口型同步服务，跳过（画面保持原样）")
            except Exception as e:
                jobs.log_job(job_id, f"口型同步失败，跳过: {e}")

        # 5. compose
        out = os.path.join(d, "output.mp4")
        lang3 = {"en": "eng", "zh": "chi", "ja": "jpn", "ko": "kor", "es": "spa",
                 "fr": "fre", "de": "ger", "ru": "rus", "pt": "por", "it": "ita",
                 "ar": "ara", "id": "ind", "vi": "vie", "th": "tha", "tr": "tur",
                 "hi": "hin"}.get(tgt, "eng")
        pipeline.compose(base_video, final_audio, os.path.join(d, "translated.srt"),
                         out, lang=lang3, burn_subs=bool(p.get("burn_subs")))
        jobs.update_job(
            job_id, status="done", stage="done", progress=1.0, finished_at=__import__("time").time(),
            note="合成完成",
            artifacts={
                "video": f"/jobs/{job_id}/output.mp4",
                "source_srt": f"/jobs/{job_id}/source.srt",
                "translated_srt": f"/jobs/{job_id}/translated.srt",
                "dubbed_audio": f"/jobs/{job_id}/dubbed_audio.wav",
                "final_audio": f"/jobs/{job_id}/final_audio.wav" if bgm_ok else f"/jobs/{job_id}/dubbed_audio.wav",
                "segments": segs,
                "src_lang": src_lang,
                "lipsync": lip_ok,
                "bgm_kept": bgm_ok,
            })
    except Exception as e:
        traceback.print_exc()
        jobs.update_job(job_id, status="error", error=f"{type(e).__name__}: {e}",
                        note=f"失败: {e}")


def worker_loop():
    while True:
        job_id = jobs.next_job()
        run_job(job_id)
