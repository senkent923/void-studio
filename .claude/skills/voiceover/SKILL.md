---
name: voiceover
description: Качественная русская озвучка для видео и аудио — нейронные голоса (Светлана/Дмитрий), субтитры, наложение на видео с приглушением музыки. Используй ВСЕГДА, когда нужно озвучить текст, сделать закадровый голос, диктора, озвучку для TikTok/Reels/Shorts/YouTube, сгенерировать видео с голосом или речь (TTS) на русском. Use for any text-to-speech, narration or voice-over task.
---

# Озвучка (voiceover)

Для любой озвучки используй команду `voiceover` (установлена в `/usr/local/bin/voiceover`,
исходник — `tools/voiceover/voiceover.py` в репозитории `senkent923/void-studio`).

**Никогда не используй gTTS, espeak, pyttsx3, festival и подобные — они звучат плохо.**
Если `voiceover` не найден — установи: `bash /opt/void-studio/tools/voiceover/install.sh`
(или клонируй `https://github.com/senkent923/void-studio` и запусти `tools/voiceover/install.sh`).

## Команды

```sh
voiceover say "Текст" -o voice.mp3 --srt voice.srt          # текст → аудио (+ субтитры)
voiceover say -f script.txt -o voice.mp3 --voice dmitry       # из файла, мужской голос
voiceover video clip.mp4 -f script.txt -o out.mp4 --subs --music bg.mp3
voiceover video clip.mp4 --audio voice.mp3 -o out.mp4          # готовый звук на видео
voiceover doctor                                              # проверить движки
```

Опции: `--voice svetlana|dmitry` (по умолчанию svetlana; офлайн: irina/denis/dmitri/ruslan),
`--rate +10%` (для коротких динамичных роликов хорошо +5…+12%), `--pitch -2Hz`,
`--engine auto|voicestudio|edge|piper`.
Для `video`: `--subs` вшить субтитры (файл .srt тоже сохраняется рядом), `--music` фон
(сам приглушается под голосом), `--music-volume 0.25`, `--orig-volume 0.3` оставить исходный звук,
`--fit extend|video|voice` (по умолчанию extend — видео удлиняется стоп-кадром, если голос длиннее),
`--delay 0.3` задержка начала голоса.

## Движки (auto выбирает по порядку)

1. **voicestudio** — локальный VoiceStudio на `http://localhost:3900` (если запущен; см. скилл `voicestudio`).
   Свой голос: `VOICESTUDIO_VOICE=<profile_id>`.
2. **edge** — нейронные голоса Microsoft, лучшее качество, нужен интернет.
3. **piper** — офлайн-резерв, модели качаются один раз в `~/.cache/voiceover`.

## Как писать текст для озвучки

- Пиши так, как говорят: короткие предложения, без скобок, списков, эмодзи и markdown.
- Числа, даты, сокращения пиши словами, если важно произношение («двадцать пятого мая», «килограмм»).
- Английские названия — русской транскрипцией, если должны звучать по-русски («Джи-Ти-Эй»).
- Ставь `ё`, где нужно; ударение можно подсказать заменой слова на синоним.
- Паузы — точкой или переносом строки; эмоцию — восклицательным знаком.
- Для ролика 15 сек ≈ 35–40 слов, для 60 сек ≈ 140–160 слов.

## Проверка результата

После генерации проверь длительность (`ffprobe`), что есть аудиодорожка, и сообщи
пользователю путь к файлу и какой движок сработал. Не выдумывай успех: если все
движки недоступны — покажи ошибку из `voiceover`.
