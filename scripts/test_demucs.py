import os, subprocess, sys, time
sys.path.insert(0, "/home/ec2-user/dubflow")
os.environ.setdefault("HF_HOME", "/home/ec2-user/dubflow/models")
os.environ.setdefault("TORCH_HOME", "/home/ec2-user/dubflow/models/torch")

FF = "/home/ec2-user/dubflow/bin/ffmpeg"
d = "/home/ec2-user/dubflow/data/test/bgm"
os.makedirs(d, exist_ok=True)
src = "/home/ec2-user/dubflow/data/test/sample_bgm.mp4"

full = os.path.join(d, "audio_full.wav")
subprocess.run([FF, "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
                "-c:a", "pcm_s16le", full], check=True, capture_output=True)
print("full wav bytes:", os.path.getsize(full))

t0 = time.time()
r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                    "-n", "htdemucs", "-o", d, full],
                   capture_output=True, text=True)
print("demucs exit:", r.returncode, "elapsed: %.1fs" % (time.time() - t0))
if r.returncode != 0:
    print("STDERR tail:", r.stderr[-2500:]); sys.exit(1)

base = os.path.splitext(os.path.basename(full))[0]
sep = os.path.join(d, "htdemucs", base)
for name in ("no_vocals.wav", "vocals.wav"):
    p = os.path.join(sep, name)
    print(f"{name}: exists={os.path.isfile(p)} bytes={os.path.getsize(p) if os.path.isfile(p) else 0}")
print("SEPDIR", sep)
