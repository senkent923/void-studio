# voiceover — русская озвучка для видео

Нейронные голоса (Microsoft Svetlana / Dmitry), офлайн-резерв (Piper), поддержка
локального VoiceStudio, субтитры, наложение на видео с автоприглушением музыки,
нормализация громкости −16 LUFS.

```sh
bash tools/voiceover/install.sh             # установка (команда voiceover + скиллы Claude)
voiceover say "Привет!" -o hi.mp3 --srt hi.srt
voiceover video clip.mp4 -f script.txt -o out.mp4 --subs --music bg.mp3 --voice dmitry
voiceover doctor
```

Переменные окружения: `VOICEOVER_VOICE` (голос по умолчанию), `VOICEOVER_ENGINE`
(`auto|voicestudio|edge|piper`), `VOICESTUDIO_URL`, `VOICESTUDIO_VOICE`, `VOICESTUDIO_API_KEY`.

В облачных сессиях Claude Code устанавливается автоматически скриптом окружения
(клон репозитория в `/opt/void-studio` + `install.sh`).
