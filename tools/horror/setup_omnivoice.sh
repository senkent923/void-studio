#!/usr/bin/env bash
# Установка OmniVoice (нейросеть VoiceStudio) для CPU: сборка omnivoice.cpp + модели Q8_0 (~950 МБ).
set -euo pipefail
OV="${OMNIVOICE_HOME:-/opt/omnivoice}"
mkdir -p "$OV/models"
if [ ! -x "$OV/src/build/omnivoice-tts" ]; then
  [ -d "$OV/src/.git" ] || git clone -q https://github.com/ServeurpersoCom/omnivoice.cpp "$OV/src"
  git -C "$OV/src" submodule update --init --recursive -q
  cmake -S "$OV/src" -B "$OV/src/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$OV/src/build" --config Release -j "$(nproc)" --target omnivoice-tts >/dev/null
fi
for f in omnivoice-base-Q8_0.gguf omnivoice-tokenizer-Q8_0.gguf; do
  [ -s "$OV/models/$f" ] || curl -fsSL -o "$OV/models/$f.part" \
    "https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/$f" && mv -f "$OV/models/$f.part" "$OV/models/$f" 2>/dev/null || true
done
python3 -m pip install -q --disable-pip-version-check faster-whisper 2>&1 | grep -vi warning || true
echo "OmniVoice готов: $OV"
