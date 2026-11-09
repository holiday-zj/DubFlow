"""Generate a Chinese test video (TTS speech over testsrc visuals) for pipeline testing."""
import asyncio
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
OUT = os.path.join(ROOT, "data", "test")
os.makedirs(OUT, exist_ok=True)

TEXT = ("大家好，欢迎来到我的频道。今天给大家介绍一款非常好用的产品。"
        "它可以帮你把视频翻译成多种语言，还能用你自己的声音配音。")


async def make_speech():
    import edge_tts
    t = edge_tts.Communicate(TEXT, "zh-CN-YunxiNeural")
    await t.save(os.path.join(OUT, "speech.mp3"))


asyncio.run(make_speech())
subprocess.run([
    os.path.join(BIN, "ffmpeg"), "-y",
    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=12",
    "-i", os.path.join(OUT, "speech.mp3"),
    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
    "-shortest", os.path.join(OUT, "sample.mp4")], check=True)
print("wrote", os.path.join(OUT, "sample.mp4"))
