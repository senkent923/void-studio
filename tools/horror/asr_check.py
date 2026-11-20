"""Проверка разборчивости озвучки: распознаёт речь Whisper и сравнивает с текстом."""
import difflib, os, re, subprocess, sys, threading
_lock = threading.Lock()
import numpy as np
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from faster_whisper import WhisperModel

_m = None
def model():
    global _m
    if _m is None:
        _m = WhisperModel(os.environ.get("ASR_MODEL", "small"), device="cpu", compute_type="int8")
    return _m

def load16k(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)

def norm(s):
    return re.sub(r"[^а-яё ]", "", s.lower().replace("ё", "е")).split()

def score(path, text):
    with _lock:
        segs, _ = model().transcribe(load16k(path), language="ru", beam_size=5)
        got = " ".join(s.text for s in segs).strip()
    r = difflib.SequenceMatcher(None, norm(text), norm(got)).ratio()
    return r, got

if __name__ == "__main__":
    for path, text in zip(sys.argv[1::2], sys.argv[2::2]):
        r, got = score(path, text)
        print(f"{r:.2f} | {got}")
