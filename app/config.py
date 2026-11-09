import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
DATA = os.path.join(ROOT, "data")
JOBS = os.path.join(DATA, "jobs")
WEB = os.path.join(ROOT, "web")

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models"))
os.makedirs(JOBS, exist_ok=True)

FFMPEG = os.path.join(BIN, "ffmpeg")
FFPROBE = os.path.join(BIN, "ffprobe")

WHISPER_DEFAULT = os.environ.get("DUBFLOW_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("DUBFLOW_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("DUBFLOW_COMPUTE", "int8")

API_KEY = os.environ.get("DUBFLOW_API_KEY", "")  # empty = open access (demo)

# External lip-sync service (LatentSync / Wav2Lip HTTP endpoint, e.g. psyb0t/flickies).
# POST multipart: video=<mp4>, audio=<wav> -> returns mp4 bytes.
LIPSYNC_URL = os.environ.get("DUBFLOW_LIPSYNC_URL", "")

# External / commercial translation (optional)
DEEPL_KEY = os.environ.get("DEEPL_API_KEY", "")

# External voice-clone TTS (GPT-SoVITS / CosyVoice HTTP service, VoiceCloneTTS-style)
# POST json {"text": str, "voice_id": str} -> wav bytes
CLONE_TTS_URL = os.environ.get("DUBFLOW_TTS_URL", "")

# edge-tts voices per target language (gender variant selectable)
VOICES = {
    "en": {"f": "en-US-AriaNeural", "m": "en-US-GuyNeural"},
    "zh": {"f": "zh-CN-XiaoxiaoNeural", "m": "zh-CN-YunxiNeural"},
    "ja": {"f": "ja-JP-NanamiNeural", "m": "ja-JP-KeitaNeural"},
    "ko": {"f": "ko-KR-SunHiNeural", "m": "ko-KR-InJoonNeural"},
    "es": {"f": "es-ES-ElviraNeural", "m": "es-ES-AlvaroNeural"},
    "fr": {"f": "fr-FR-DeniseNeural", "m": "fr-FR-HenriNeural"},
    "de": {"f": "de-DE-KatjaNeural", "m": "de-DE-ConradNeural"},
    "ru": {"f": "ru-RU-SvetlanaNeural", "m": "ru-RU-DmitryNeural"},
    "pt": {"f": "pt-BR-FranciscaNeural", "m": "pt-BR-AntonioNeural"},
    "it": {"f": "it-IT-ElsaNeural", "m": "it-IT-DiegoNeural"},
    "ar": {"f": "ar-SA-ZariyahNeural", "m": "ar-SA-HamedNeural"},
    "id": {"f": "id-ID-GadisNeural", "m": "id-ID-ArdiNeural"},
    "vi": {"f": "vi-VN-HoaiMyNeural", "m": "vi-VN-NamMinhNeural"},
    "th": {"f": "th-TH-PremwadeeNeural", "m": "th-TH-NiwatNeural"},
    "tr": {"f": "tr-TR-EmelNeural", "m": "tr-TR-AhmetNeural"},
    "hi": {"f": "hi-IN-SwaraNeural", "m": "hi-IN-MadhurNeural"},
}

LANG_NAMES = {
    "en": "英语", "zh": "中文", "ja": "日语", "ko": "韩语", "es": "西班牙语",
    "fr": "法语", "de": "德语", "ru": "俄语", "pt": "葡萄牙语", "it": "意大利语",
    "ar": "阿拉伯语", "id": "印尼语", "vi": "越南语", "th": "泰语", "tr": "土耳其语",
    "hi": "印地语",
}
