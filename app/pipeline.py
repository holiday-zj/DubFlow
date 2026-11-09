"""Pipeline stages: each takes job dir paths, returns artifacts. CPU-only safe."""
import os
import re
import subprocess
import json

from . import config

# ---------------- audio extraction ----------------

def extract_audio(video_path: str, out_wav: str):
    """Extract 16kHz mono WAV (standard input for speech models)."""
    cmd = [config.FFMPEG, "-y", "-i", video_path, "-vn",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def probe_duration(path: str) -> float:
    out = subprocess.run(
        [config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path], check=True, capture_output=True).stdout
    return float(json.loads(out)["format"]["duration"])


# ---------------- stage 1: STT (Whisper) ----------------

_whisper_model = None
_whisper_key = None

def transcribe(wav_path: str, model_size: str, language: str = ""):
    """faster-whisper transcription -> list of segments {start,end,text}."""
    global _whisper_model, _whisper_key
    if _whisper_key != model_size:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(model_size, device=config.WHISPER_DEVICE,
                                      compute_type=config.WHISPER_COMPUTE)
        _whisper_key = model_size
    segments, info = _whisper_model.transcribe(
        wav_path, language=language or None, vad_filter=True,
        beam_size=1)
    segs = [{"start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text.strip()} for s in segments]
    return segs, info.language


# ---------------- stage 2: translation (MarianMT) ----------------

# NLLB language tags verified against nllb-200-distilled-600M vocab
_MT_TAG = {"zh": "zho_Hans", "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
           "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
           "pt": "por_Latn", "it": "ita_Latn", "ar": "arb_Arab", "id": "ind_Latn",
           "vi": "vie_Latn", "th": "tha_Thai", "tr": "tur_Latn", "hi": "hin_Deva"}

_pipeline_cache = {}

def _load_marian(src: str, tgt: str):
    key = (src, tgt)
    if key not in _pipeline_cache:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # HF opus-mt naming aliases (ja is "jap", no en-ko model, etc.)
        alias = {"ja": "jap"}
        name = f"Helsinki-NLP/opus-mt-{alias.get(src, src)}-{alias.get(tgt, tgt)}"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.eval()
        _pipeline_cache[key] = (tok, model)
    return _pipeline_cache[key]

def _load_nllb200():
    if "nllb200" not in _pipeline_cache:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        _pipeline_cache["nllb200"] = (tok, model)
    return _pipeline_cache["nllb200"]

def _split_batch(texts, maxlen=400):
    """Split long segments into sentences to keep Marian quality."""
    out = []
    for t in texts:
        parts = re.split(r"(?<=[。！？.!?；;])\s*", t)
        buf = ""
        for p in parts:
            if len(buf) + len(p) <= maxlen:
                buf += p
            else:
                if buf:
                    out.append(buf)
                buf = p
        out.append(buf)
    return out

def _marian_translate(texts, src, tgt):
    """MarianMT translate; raises MarianPairMissing if no direct model."""
    import torch
    try:
        tok, model = _load_marian(src, tgt)
    except Exception:
        raise MarianPairMissing(f"{src}-{tgt}")
    chunks = _split_batch(texts)
    trans = []
    for i in range(0, len(chunks), 8):
        batch = chunks[i:i + 8]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=512, padding=True)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=256)
        trans += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return _merge_back(trans, texts)


class MarianPairMissing(Exception):
    pass


def _nllb_translate(texts, src, tgt):
    import torch
    tok, model = _load_nllb200()
    tgt_tag = _MT_TAG.get(tgt, tgt + "_Latn")
    tok.src_lang = _MT_TAG.get(src, src + "_Latn")
    results = []
    for chunk in _split_batch(texts):
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=256,
                                 repetition_penalty=1.2, num_beams=4,
                                 forced_bos_token_id=tok.convert_tokens_to_ids(tgt_tag))
        results += [_clean_gen(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return _merge_back(results, texts)


def _clean_gen(text: str) -> str:
    """Drop NLLB degenerate repeats (variation-selector / emoji spam)."""
    import re as _re
    text = _re.sub(r"[\ufe0e\ufe0f\u200d]", "", text)
    text = _re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
    # collapse any char repeated >6x (degenerate loop)
    text = _re.sub(r"(.)\1{5,}", r"\1", text)
    return text.strip()


def translate_texts(texts: list, src: str, tgt: str, engine: str = "marian") -> list:
    if src == tgt:
        return list(texts)
    ms_src = "zh" if src == "zh" else src
    if engine == "nllb":
        return _nllb_translate(texts, ms_src, tgt)
    # default: MarianMT (lightweight); fall back to NLLB if pair unavailable
    # (no en-pivot: pivoting degrades quality badly, NLLB covers all pairs)
    try:
        return _marian_translate(texts, ms_src, tgt)
    except MarianPairMissing:
        return _nllb_translate(texts, ms_src, tgt)

def _merge_back(translated_chunks: list, originals: list) -> list:
    """Reassemble sentence-level translations back into segment groups."""
    counts = []
    for t in originals:
        parts = re.split(r"(?<=[。！？.!?；;])\s*", t)
        buf, n = "", 0
        for p in parts:
            if len(buf) + len(p) <= 400:
                buf += p
            else:
                n += 1
                buf = p
        counts.append(n + 1)
    out, i = [], 0
    for c in counts:
        out.append(" ".join(translated_chunks[i:i + c]).strip())
        i += c
    return out


# ---------------- SRT writer ----------------

def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments, path, field="text"):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s[field]}\n\n")
    return path


# ---------------- stage 3: TTS ----------------

def tts_edge(text: str, voice: str, out_wav: str, rate: str = "+0%"):
    """edge-tts (free neural voices) -> 16k mono wav."""
    import asyncio, edge_tts
    mp3 = out_wav + ".mp3"
    async def go():
        t = edge_tts.Communicate(text, voice, rate=rate)
        await t.save(mp3)
    asyncio.run(go())
    subprocess.run([config.FFMPEG, "-y", "-i", mp3, "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", out_wav], check=True, capture_output=True)
    os.remove(mp3)
    return out_wav


def build_dub_audio(segments, voice: str, total_dur: float, out_wav: str,
                    on_error=None):
    """Synthesize per-segment audio, speed-fit each into its time slot,
    then assemble a full-length timeline wav aligned with the video.
    A failing segment is skipped (reported via on_error) instead of
    killing the whole job."""
    import wave
    tmpdir = os.path.dirname(out_wav)
    sr = 16000
    silence_total = int(total_dur * sr) + sr  # full timeline samples
    timeline = bytearray(silence_total * 2)

    for i, seg in enumerate(segments):
        text = _clean_gen(seg.get("text_trans") or seg.get("text", ""))
        if not text.strip():
            continue
        slot = max(0.3, seg["end"] - seg["start"])
        part = os.path.join(tmpdir, f"seg{i}.wav")
        ok = False
        for attempt in range(3):
            try:
                tts_edge(text, voice, part)
                ok = True
                break
            except Exception as e:
                if attempt == 2:
                    if on_error:
                        on_error(f"segment {i} TTS failed, skipped: {e}")
                else:
                    import time as _t
                    _t.sleep(2)
        if not ok:
            continue
        dur = probe_duration(part)
        # speed-fit: stretch/compress within 0.5x..2.0x using atempo chain
        ratio = dur / slot
        if ratio > 1.05 or ratio < 0.95:
            ratio = min(2.0, max(0.5, ratio))
            chain = []
            r = ratio
            while r > 2.0:
                chain.append("atempo=2.0"); r /= 2.0
            while r < 0.5:
                chain.append("atempo=0.5"); r *= 2.0
            chain.append(f"atempo={r:.4f}")
            fitted = part + ".fit.wav"
            subprocess.run([config.FFMPEG, "-y", "-i", part, "-filter:a",
                            ",".join(chain), fitted], check=True, capture_output=True)
            part = fitted
            dur = probe_duration(part)
        # read samples
        with wave.open(part, "rb") as w:
            n = w.getnframes()
            data = w.readframes(n)
        off = int(seg["start"] * sr) * 2
        end = min(off + len(data), len(timeline))
        timeline[off:end] = data[: end - off]
        base_part = os.path.join(tmpdir, f"seg{i}.wav")
        for f in (base_part, base_part + ".fit.wav"):
            if os.path.exists(f):
                os.remove(f)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(bytes(timeline))
    return out_wav


# ---------------- stage 4a: vocal/BGM separation (demucs) ----------------

def separate_bgm(video_path: str, workdir: str) -> str:
    """Split audio into vocals vs background (music/SFX) with demucs htdemucs.
    Returns path to no_vocals.wav (44.1kHz stereo). Raises on failure."""
    full = os.path.join(workdir, "audio_full.wav")
    subprocess.run([config.FFMPEG, "-y", "-i", video_path, "-vn", "-ac", "2",
                    "-ar", "44100", "-c:a", "pcm_s16le", full],
                   check=True, capture_output=True)
    import sys
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                        "-n", "htdemucs", "-o", workdir, full],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("demucs failed: " + r.stderr[-500:])
    base = os.path.splitext(os.path.basename(full))[0]
    no_vocals = os.path.join(workdir, "htdemucs", base, "no_vocals.wav")
    if not os.path.isfile(no_vocals):
        raise RuntimeError(f"demucs produced no no_vocals.wav under {workdir}")
    return no_vocals


def mix_bgm(dubbed_wav: str, bgm_wav: str, out_wav: str, duration: float,
            bgm_volume: float = 1.0, duck: bool = True):
    """Mix dubbed voice track over the extracted background (music/SFX).
    With duck=True, background is sidechain-compressed when speech is present."""
    duck_fc = (f"[1:a]volume={bgm_volume}[bg];"
               f"[0:a]aresample=44100,aformat=channel_layouts=stereo,"
               f"asplit=2[dub][dubsc];"
               f"[bg][dubsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:"
               f"release=400[bgd];"
               f"[dub][bgd]amix=inputs=2:duration=longest:normalize=0,"
               f"atrim=0:{duration:.3f}[mix]")
    plain_fc = (f"[1:a]volume={bgm_volume}[bg];"
                f"[0:a]aresample=44100,aformat=channel_layouts=stereo[dub];"
                f"[dub][bg]amix=inputs=2:duration=longest:normalize=0,"
                f"atrim=0:{duration:.3f}[mix]")
    for fc in ((duck_fc, plain_fc) if duck else (plain_fc,)):
        try:
            subprocess.run([config.FFMPEG, "-y", "-i", dubbed_wav, "-i", bgm_wav,
                            "-filter_complex", fc, "-map", "[mix]",
                            "-ac", "2", "-ar", "44100", out_wav],
                           check=True, capture_output=True)
            return out_wav
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("bgm mix failed")


# ---------------- stage 4: lip-sync (external service hook) ----------------

def lipsync(video_path: str, audio_path: str, out_path: str) -> bool:
    """Call external LatentSync/Wav2Lip HTTP service if configured.
    Returns True on success; False if not configured (video kept as-is)."""
    if not config.LIPSYNC_URL:
        return False
    import httpx
    with open(video_path, "rb") as fv, open(audio_path, "rb") as fa:
        r = httpx.post(config.LIPSYNC_URL, timeout=1800,
                       files={"video": ("v.mp4", fv, "video/mp4"),
                              "audio": ("a.wav", fa, "audio/wav")})
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return True


# ---------------- stage 5: compose ----------------

def compose(video_path: str, audio_path: str, srt_path: str, out_path: str,
            lang: str = "eng", burn_subs: bool = False):
    """Mux dubbed audio + optional embedded subtitle track."""
    if burn_subs:
        # hard-burn subs, re-encode video
        cmd = [config.FFMPEG, "-y", "-i", video_path, "-i", audio_path,
               "-vf", f"subtitles={srt_path}",
               "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-shortest", out_path]
    else:
        cmd = [config.FFMPEG, "-y", "-i", video_path, "-i", audio_path, "-i", srt_path,
               "-map", "0:v", "-map", "1:a", "-map", "2:s",
               "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
               "-metadata:s:s:0", f"language={lang}", "-shortest", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
