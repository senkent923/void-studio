#!/usr/bin/env bash
# Установка voiceover: зависимости, команда в PATH, скиллы Claude, офлайн-голоса.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"

python3 -m pip install -q --disable-pip-version-check -r "$DIR/requirements.txt" 2>&1 | grep -v -i "warning" || true
command -v ffmpeg >/dev/null || { (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1 || brew install ffmpeg || true; }

BIN=/usr/local/bin; [ -w "$BIN" ] || { BIN="$HOME/.local/bin"; mkdir -p "$BIN"; }
chmod +x "$DIR/voiceover.py"
ln -sf "$DIR/voiceover.py" "$BIN/voiceover"

# Скиллы Claude Code — на уровне пользователя, чтобы работали в любом проекте.
mkdir -p "$HOME/.claude/skills"
for s in voiceover voicestudio horror-video; do
  rm -rf "$HOME/.claude/skills/$s"
  cp -r "$REPO/.claude/skills/$s" "$HOME/.claude/skills/$s"
done
MARK="<!-- voiceover-rule -->"
if ! grep -qs "$MARK" "$HOME/.claude/CLAUDE.md"; then
  cat >> "$HOME/.claude/CLAUDE.md" <<RULE

$MARK
## Озвучка
Для любой озвучки, закадрового голоса и TTS (особенно в видео) используй команду \`voiceover\`
и скилл \`voiceover\` (русские нейронные голоса). Не используй gTTS/espeak/pyttsx3.
RULE
fi

# Заранее скачать офлайн-голоса (не критично, если нет сети).
python3 -c "
import sys; sys.path.insert(0, '$DIR'); import voiceover as v
for name in ('irina', 'denis'):
    try: v.piper_model(name)
    except Exception as e: print('piper', name, e)
" || true
echo "voiceover установлен: $(command -v voiceover || echo $BIN/voiceover)"
# зависимости генератора хоррор-роликов
[ -f "$REPO/tools/horror/requirements.txt" ] && python3 -m pip install -q --disable-pip-version-check -r "$REPO/tools/horror/requirements.txt" 2>&1 | grep -v -i warning || true
