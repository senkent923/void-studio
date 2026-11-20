"""Синтез голоса edge-tts с таймингами каждого слова -> audio.mp3 + words.json"""
import asyncio, json, os, sys
import edge_tts
text, out_audio, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
voice = os.environ.get("VOICE", "ru-RU-DmitryNeural")
rate = os.environ.get("RATE", "-8%"); pitch = os.environ.get("PITCH", "-6Hz")
proxy = os.environ.get("HTTPS_PROXY") or None
async def go():
    c = edge_tts.Communicate(open(text, encoding="utf-8").read(), voice, rate=rate, pitch=pitch,
                             boundary="WordBoundary", proxy=proxy)
    words = []
    with open(out_audio, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio": f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
    json.dump(words, open(out_json, "w"), ensure_ascii=False, indent=0)
for i in range(4):
    try: asyncio.run(go()); break
    except Exception as e: print("retry", e, file=sys.stderr)
