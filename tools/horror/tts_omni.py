#!/usr/bin/env python3
"""Озвучка истории моделью OmniVoice (движок VoiceStudio) с проверкой каждой фразы.

Строки, начинающиеся с «—», читает второй голос (по умолчанию — шёпот старухи).
Голоса стабильны: сначала создаётся эталон голоса, затем каждая фраза клонирует его.

    python3 tts_omni.py story.json voice.wav words.json --work DIR
Выход совместим с tts_words.py: words.json = [{"t","d","w"}], звук без длинных пауз.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
OV = Path(os.environ.get("OMNIVOICE_HOME", "/opt/omnivoice"))
BIN = Path(os.environ.get("OMNIVOICE_BIN", OV / "src/build/omnivoice-tts"))
MODEL = OV / "models/omnivoice-base-Q8_0.gguf"
CODEC = OV / "models/omnivoice-tokenizer-Q8_0.gguf"
SR = 24000

VOICES = {
    "narrator": {"instruct": "male, middle-aged, low pitch", "seed": 11, "cps": 13.5,
                 "ref": "Эту историю мне рассказал один охотник. Я до сих пор не знаю, верить ему или нет."},
    "granny": {"instruct": "female, elderly, whisper", "seed": 5, "cps": 12.0,
               "ref": "Не ходи туда, милок. Там по ночам никто не спит."},
}


def run_tts(text, out, *, instruct=None, ref=None, ref_text=None, seed=1, duration=None, steps=32):
    tf = Path(str(out) + ".txt")
    tf.write_text(text, encoding="utf-8")
    argv = [str(BIN), "--model", str(MODEL), "--codec", str(CODEC), "--lang", "Russian",
            "--seed", str(seed), "--steps", str(steps), "-o", str(out)]
    if instruct:
        argv += ["--instruct", instruct]
    if ref:
        rt = Path(str(out) + ".ref.txt")
        rt.write_text(ref_text, encoding="utf-8")
        argv += ["--ref-wav", str(ref), "--ref-text", str(rt)]
    if duration:
        argv += ["--duration", f"{duration:.2f}"]
    with open(tf, encoding="utf-8") as fin:
        r = subprocess.run(argv, stdin=fin, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not Path(out).exists():
        raise RuntimeError(r.stderr[-1500:])


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return x if x.ndim == 1 else x.mean(1)


def trim(x, thr_db=-42):
    """Обрезать тишину по краям, оставив 40 мс."""
    env = np.convolve(np.abs(x), np.ones(240) / 240, mode="same")
    idx = np.where(env > 10 ** (thr_db / 20))[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(0.04 * SR)); b = min(len(x), idx[-1] + int(0.04 * SR))
    return x[a:b]


def split_lines(text):
    """[(voice, sentence, paragraph_start)]"""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        voice = "granny" if line.startswith("—") else "narrator"
        line = line.lstrip("— ").strip()
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", line) if s]
        for i, s in enumerate(sents):
            out.append((voice, s, i == 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story"); ap.add_argument("out_wav"); ap.add_argument("out_json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    story = json.loads(Path(a.story).read_text(encoding="utf-8"))
    work = Path(a.work) / "omni"; work.mkdir(parents=True, exist_ok=True)
    voices = {k: dict(v, **story.get("omni_voices", {}).get(k, {})) for k, v in VOICES.items()}

    sys.path.insert(0, str(HERE))
    from asr_check import score

    # 1. эталоны голосов
    refs = {}
    for name, v in voices.items():
        p = work / f"ref_{name}.wav"
        if not p.exists():
            print(f"[omni] эталон голоса {name}", file=sys.stderr)
            run_tts(v["ref"], p, instruct=v["instruct"], seed=v["seed"],
                    duration=len(v["ref"]) / v["cps"] + 0.3)
        refs[name] = p

    # 2. фразы
    lines = split_lines(story["text"])

    def synth(i):
        voice, sent, _ = lines[i]
        v = voices[voice]
        out = work / f"s{i:03}.wav"
        if out.exists() and out.with_suffix(".ok").exists():
            return i, out
        best = (-1, None)
        for attempt in range(3):
            tmp = work / f"s{i:03}_{attempt}.wav"
            dur = len(sent) / v["cps"] + 0.35
            run_tts(sent, tmp, ref=refs[voice], ref_text=v["ref"], seed=v["seed"] + 100 * attempt + i,
                    duration=dur)
            r, got = score(str(tmp), sent)
            print(f"[omni] {i:02} попытка {attempt + 1}: {r:.2f} | {got}", file=sys.stderr)
            if r > best[0]:
                best = (r, tmp)
            if r >= 0.9:
                break
        best[1].replace(out); out.with_suffix(".ok").write_text(f"{best[0]:.2f}")
        return i, out

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        files = dict(ex.map(synth, range(len(lines))))

    # 3. склейка с паузами + тайминги слов (пропорционально длине слов во фразе)
    pauses = story.get("omni_pauses", {})
    audio, words, t = [np.zeros(int(0.05 * SR), np.float32)], [], 0.05
    for i, (voice, sent, para) in enumerate(lines):
        if i > 0:
            gap = float(pauses.get(str(i), 0.55 if para else 0.32))
            audio.append(np.zeros(int(gap * SR), np.float32)); t += gap
        x = trim(load(files[i]))
        if voice == "granny":
            x = x * 1.35
        dur = len(x) / SR
        toks = [w for w in sent.split() if re.search(r"\w", w)]
        weights = np.array([len(re.sub(r"\W", "", w)) + 1.5 for w in toks], float)
        starts = np.concatenate([[0], np.cumsum(weights)[:-1]]) / weights.sum() * dur * 0.94
        lens = weights / weights.sum() * dur * 0.94
        for w, s0, ln in zip(toks, starts, lens):
            words.append({"t": t + s0 + 0.02, "d": ln, "w": re.sub(r"[^\wё-]", "", w), "voice": voice})
        audio.append(x); t += dur
    y = np.concatenate(audio)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    wavfile.write(a.out_wav, SR, (y * 32767).astype(np.int16))
    Path(a.out_json).write_text(json.dumps(words, ensure_ascii=False, indent=0))
    print(f"[omni] готово: {len(y) / SR:.1f} c, {len(words)} слов", file=sys.stderr)


if __name__ == "__main__":
    main()
