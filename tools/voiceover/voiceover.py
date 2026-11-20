#!/usr/bin/env python3
"""voiceover — качественная русская озвучка текста и видео.

Движки (по приоритету в режиме auto):
  1. voicestudio — локальный VoiceStudio (http://localhost:3900), если запущен;
  2. edge        — нейронные голоса Microsoft (ru-RU-Svetlana/Dmitry), лучшее качество;
  3. piper       — офлайн-синтез (голоса irina/denis/dmitri/ruslan), запасной вариант.

Примеры:
  voiceover say "Привет, мир!" -o hello.mp3 --srt hello.srt
  voiceover say -f script.txt -o voice.mp3 --voice dmitry --rate +5%
  voiceover video clip.mp4 -f script.txt -o out.mp4 --music bg.mp3 --subs
  voiceover voices
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

CACHE = Path(os.environ.get("VOICEOVER_CACHE", Path.home() / ".cache" / "voiceover"))
VOICESTUDIO_URL = os.environ.get("VOICESTUDIO_URL", "http://localhost:3900").rstrip("/")

EDGE_VOICES = {
    "svetlana": "ru-RU-SvetlanaNeural",  # женский, по умолчанию
    "dmitry": "ru-RU-DmitryNeural",      # мужской
}
PIPER_VOICES = {
    "irina": "irina/medium/ru_RU-irina-medium",
    "denis": "denis/medium/ru_RU-denis-medium",
    "dmitri": "dmitri/medium/ru_RU-dmitri-medium",
    "ruslan": "ruslan/medium/ru_RU-ruslan-medium",
}
# Соответствие голосов между движками (по полу), чтобы fallback звучал похоже.
FEMALE = {"svetlana", "irina"}
DEFAULT_VOICE = os.environ.get("VOICEOVER_VOICE", "svetlana")


def log(msg: str) -> None:
    print(f"[voiceover] {msg}", file=sys.stderr)


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


# ---------------------------------------------------------------- text helpers

def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    parts: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        for s in re.split(r"(?<=[.!?…])\s+", para):
            s = s.strip()
            while len(s) > max_len:  # длинные предложения режем по запятым/пробелам
                cut = max(s.rfind(",", 0, max_len), s.rfind(";", 0, max_len), s.rfind(" ", 0, max_len))
                cut = cut if cut > 40 else max_len
                parts.append(s[: cut + 1].strip())
                s = s[cut + 1 :].strip()
            if s:
                parts.append(s)
    return parts


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    lines = []
    for i, (a, b, t) in enumerate(cues, 1):
        lines += [str(i), f"{fmt_ts(a)} --> {fmt_ts(b)}", t, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- engines

def edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def voicestudio_available() -> bool:
    try:
        with urllib.request.urlopen(f"{VOICESTUDIO_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def synth_edge(text: str, voice: str, rate: str, pitch: str, out_mp3: Path) -> list[tuple[float, float, str]]:
    import edge_tts

    name = EDGE_VOICES.get(voice, voice if "Neural" in voice else EDGE_VOICES["dmitry" if voice not in FEMALE else "svetlana"])
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None

    async def go():
        comm = edge_tts.Communicate(text, name, rate=rate, pitch=pitch, proxy=proxy, boundary="SentenceBoundary")
        cues = []
        with open(out_mp3, "wb") as f:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                    a = chunk["offset"] / 1e7
                    cues.append((a, a + chunk["duration"] / 1e7, chunk["text"]))
        return cues

    last = None
    for attempt in range(3):
        try:
            cues = asyncio.run(go())
            if out_mp3.stat().st_size > 0:
                return cues
        except Exception as e:  # сеть, лимиты — пробуем ещё раз
            last = e
            log(f"edge попытка {attempt + 1} не удалась: {e}")
    raise RuntimeError(f"edge-tts недоступен: {last}")


def piper_model(voice: str) -> Path:
    key = voice if voice in PIPER_VOICES else ("irina" if voice in FEMALE else "denis")
    rel = PIPER_VOICES[key]
    d = CACHE / "piper"
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / (Path(rel).name + ".onnx")
    for suffix in (".onnx", ".onnx.json"):
        p = d / (Path(rel).name + suffix)
        if not p.exists() or p.stat().st_size == 0:
            url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/{rel}{suffix}"
            log(f"скачиваю голос piper: {url}")
            tmp = p.with_suffix(p.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(p)
    return onnx


def synth_piper_sentence(model, text: str, out_wav: Path, speed: float) -> None:
    from piper.config import SynthesisConfig

    cfg = SynthesisConfig(length_scale=1.0 / speed)
    with wave.open(str(out_wav), "wb") as w:
        model.synthesize_wav(text, w, syn_config=cfg)


def synth_voicestudio_sentence(text: str, voice: str, speed: float, out_wav: Path) -> None:
    body = {
        "model": "tts-1-hd",
        "voice": os.environ.get("VOICESTUDIO_VOICE", voice if voice not in EDGE_VOICES and voice not in PIPER_VOICES else "alloy"),
        "input": text,
        "language": "ru",
        "speed": speed,
        "response_format": "wav",
    }
    req = urllib.request.Request(
        f"{VOICESTUDIO_URL}/v1/audio/speech", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {os.environ['VOICESTUDIO_API_KEY']}"} if os.environ.get("VOICESTUDIO_API_KEY") else {})},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out_wav.write_bytes(r.read())
    if duration(out_wav) <= 0:
        raise RuntimeError("VoiceStudio вернул пустой звук")


def synth_by_sentence(sentences: list[str], synth_one, tmp: Path, pause: float) -> tuple[Path, list]:
    """Синтез по предложениям + склейка с паузами; возвращает wav и тайминги."""
    files, cues, t = [], [], 0.0
    sil = tmp / "sil.wav"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(pause), str(sil)])
    for i, s in enumerate(sentences):
        raw = tmp / f"s{i:04}.wav"
        synth_one(s, raw)
        norm = tmp / f"n{i:04}.wav"
        run(["ffmpeg", "-y", "-i", str(raw), "-ar", "24000", "-ac", "1", str(norm)])
        d = duration(norm)
        cues.append((t, t + d, s))
        files += [norm, sil]
        t += d + pause
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files[:-1]))
    out = tmp / "joined.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return out, cues


def rate_to_speed(rate: str) -> float:
    m = re.fullmatch(r"([+-]?\d+)%", rate.strip())
    return max(0.5, min(2.0, 1 + int(m.group(1)) / 100)) if m else 1.0


def synthesize(text: str, out: Path, voice: str, rate: str, pitch: str, engine: str,
               srt: Path | None, pause: float = 0.35) -> str:
    text = clean_text(text)
    if not text:
        raise SystemExit("пустой текст")
    order = {"auto": ["voicestudio", "edge", "piper"]}.get(engine, [engine])
    errors = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for eng in order:
            try:
                if eng == "voicestudio":
                    if not voicestudio_available():
                        raise RuntimeError(f"VoiceStudio не запущен на {VOICESTUDIO_URL}")
                    sp = rate_to_speed(rate)
                    raw, cues = synth_by_sentence(split_sentences(text),
                                                  lambda s, p: synth_voicestudio_sentence(s, voice, sp, p), tmp, pause)
                elif eng == "edge":
                    if not edge_available():
                        raise RuntimeError("пакет edge-tts не установлен (pip install edge-tts)")
                    raw = tmp / "edge.mp3"
                    cues = synth_edge(text, voice, rate, pitch, raw)
                elif eng == "piper":
                    from piper import PiperVoice
                    model = PiperVoice.load(str(piper_model(voice)))
                    sp = rate_to_speed(rate)
                    raw, cues = synth_by_sentence(split_sentences(text),
                                                  lambda s, p: synth_piper_sentence(model, s, p, sp), tmp, pause)
                else:
                    raise SystemExit(f"неизвестный движок {eng}")
                finalize_audio(raw, out)
                if srt:
                    write_srt(cues or [(0, duration(out), text)], srt)
                log(f"готово: {out} ({duration(out):.1f} c), движок {eng}")
                return eng
            except SystemExit:
                raise
            except Exception as e:
                errors.append(f"{eng}: {e}")
                log(f"движок {eng} недоступен: {e}")
    raise SystemExit("не удалось озвучить:\n  " + "\n  ".join(errors))


def finalize_audio(src: Path, out: Path) -> None:
    """Громкость по стандарту (-16 LUFS), лёгкая очистка, нужный формат."""
    out.parent.mkdir(parents=True, exist_ok=True)
    af = "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11"
    codec = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".wav": ["-c:a", "pcm_s16le"],
             ".m4a": ["-c:a", "aac", "-b:a", "192k"], ".ogg": ["-c:a", "libopus", "-b:a", "96k"],
             ".opus": ["-c:a", "libopus", "-b:a", "96k"]}.get(out.suffix.lower(), ["-c:a", "libmp3lame", "-b:a", "192k"])
    run(["ffmpeg", "-y", "-i", str(src), "-af", af, "-ar", "48000", *codec, str(out)])


# ---------------------------------------------------------------- video

def make_video(video: Path, voice_audio: Path, out: Path, music: Path | None, srt: Path | None,
               orig_volume: float, music_volume: float, fit: str, delay: float) -> None:
    vdur, adur = duration(video), duration(voice_audio) + delay
    inputs = ["-i", str(video), "-i", str(voice_audio)]
    filters = [f"[1:a]adelay={int(delay * 1000)}:all=1,aresample=48000[vo]"]
    mix = ["[vo]"]
    if orig_volume > 0 and has_audio(video):
        filters.append(f"[0:a]volume={orig_volume},aresample=48000[orig]")
        mix.append("[orig]")
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        # приглушение музыки под голосом (ducking)
        filters.append(f"[2:a]volume={music_volume},aresample=48000[m]")
        filters.append("[vo]asplit=2[vo1][vosc]")
        filters.append("[m][vosc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[mduck]")
        mix = ["[vo1]"] + mix[1:] + ["[mduck]"]
    filters.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,"
                   f"alimiter=limit=0.95[aout]")

    vf = []
    target = vdur
    if fit == "extend" and adur > vdur:  # удлиняем видео стоп-кадром под длину озвучки
        vf.append(f"tpad=stop_mode=clone:stop_duration={adur - vdur + 0.5:.2f}")
        target = adur + 0.5
    elif fit == "voice":
        target = adur + 0.5
    if srt:
        style = "FontName=DejaVu Sans,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40"
        esc = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf.append(f"subtitles='{esc}':force_style='{style}'")
    # голос короче видео — дополняем тишиной до конца ролика
    filters[0] = filters[0].replace("[vo]", f",apad=whole_dur={target:.2f}[vo]")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]"]
    if vf:
        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{target:.2f}", "-movflags", "+faststart", str(out)]
    run(cmd)
    log(f"видео готово: {out} ({duration(out):.1f} c)")


# ---------------------------------------------------------------- CLI

def read_text(args) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if getattr(args, "text", None):
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("нужен текст: аргументом, -f файл или через stdin")


def main() -> None:
    p = argparse.ArgumentParser(prog="voiceover", description="Русская озвучка текста и видео")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-f", "--file", help="файл с текстом (UTF-8)")
        sp.add_argument("--voice", default=DEFAULT_VOICE,
                        help="svetlana (жен., по умолчанию) | dmitry (муж.) | irina/denis/dmitri/ruslan (piper) | id голоса VoiceStudio")
        sp.add_argument("--rate", default="+0%", help="скорость, напр. +10%% или -5%%")
        sp.add_argument("--pitch", default="+0Hz", help="высота (только edge), напр. -2Hz")
        sp.add_argument("--engine", default=os.environ.get("VOICEOVER_ENGINE", "auto"),
                        choices=["auto", "voicestudio", "edge", "piper"])

    s = sub.add_parser("say", help="озвучить текст в аудиофайл")
    s.add_argument("text", nargs="?")
    common(s)
    s.add_argument("-o", "--out", default="voice.mp3")
    s.add_argument("--srt", help="сохранить субтитры .srt")

    v = sub.add_parser("video", help="наложить озвучку на видео")
    v.add_argument("video")
    v.add_argument("text", nargs="?")
    common(v)
    v.add_argument("-o", "--out", default=None)
    v.add_argument("--audio", help="готовая озвучка вместо синтеза")
    v.add_argument("--music", help="фоновая музыка (автоматически приглушается под голосом)")
    v.add_argument("--music-volume", type=float, default=0.25)
    v.add_argument("--orig-volume", type=float, default=0.0, help="громкость исходного звука видео (0 = убрать)")
    v.add_argument("--subs", action="store_true", help="вшить субтитры")
    v.add_argument("--fit", choices=["extend", "video", "voice"], default="extend",
                   help="extend: удлинить видео стоп-кадром, если голос длиннее; video: обрезать по видео; voice: по длине голоса")
    v.add_argument("--delay", type=float, default=0.3, help="задержка начала голоса, сек")

    sub.add_parser("voices", help="список голосов")
    d = sub.add_parser("doctor", help="проверить, какие движки работают")
    d.add_argument("--quick", action="store_true")

    a = p.parse_args()
    if a.cmd == "voices":
        print("edge (нейронные, лучшее качество): svetlana (жен.), dmitry (муж.)")
        print("piper (офлайн): irina (жен.), denis, dmitri, ruslan (муж.)")
        print(f"voicestudio: сохранённые профили из {VOICESTUDIO_URL}/v1/audio/voices (если запущен)")
        return
    if a.cmd == "doctor":
        ok = {}
        ok["ffmpeg"] = bool(shutil.which("ffmpeg"))
        ok["voicestudio"] = voicestudio_available()
        with tempfile.TemporaryDirectory() as td:
            for eng in ("edge", "piper"):
                try:
                    synthesize("Проверка связи.", Path(td) / f"{eng}.mp3", DEFAULT_VOICE, "+0%", "+0Hz", eng, None)
                    ok[eng] = True
                except BaseException as e:  # noqa: BLE001
                    ok[eng] = False
                    log(f"{eng}: {e}")
        for k, val in ok.items():
            print(f"{'OK ' if val else '-- '} {k}")
        sys.exit(0 if ok["ffmpeg"] and (ok["edge"] or ok["piper"] or ok["voicestudio"]) else 1)

    text = None if (a.cmd == "video" and a.audio) else read_text(a)
    if a.cmd == "say":
        out = Path(a.out)
        synthesize(text, out, a.voice, a.rate, a.pitch, a.engine, Path(a.srt) if a.srt else None)
        return

    video = Path(a.video)
    out = Path(a.out) if a.out else video.with_name(video.stem + "_ru.mp4")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        srt = tmp / "subs.srt" if a.subs else None
        if a.audio:
            audio = Path(a.audio)
            if srt:
                log("--subs с --audio: субтитры без таймингов не построить, пропускаю")
                srt = None
        else:
            audio = tmp / "voice.wav"
            synthesize(text, audio, a.voice, a.rate, a.pitch, a.engine, srt)
            if srt and a.delay:
                # сдвигаем субтитры на задержку голоса
                shifted = tmp / "subs_shift.srt"
                run(["ffmpeg", "-y", "-itsoffset", str(a.delay), "-i", str(srt), str(shifted)])
                srt = shifted
            keep = out.with_suffix(".srt")
            if srt:
                shutil.copy(srt, keep)
        make_video(video, audio, out, Path(a.music) if a.music else None, srt,
                   a.orig_volume, a.music_volume, a.fit, a.delay)


if __name__ == "__main__":
    main()
