#!/bin/bash
# Saglitz Photo Studio — başlatır: yerel görsel motoru (FastAPI + mflux).
# Varsayılan model (SAGLITZ_BASE_MODEL, vars. "schnell") açılışta yüklenir;
# diğerleri (ör. "z-image-turbo") ilk istekte yüklenir. http://127.0.0.1:8765
#   ./start-engine.sh
#
# Z-Image Turbo'yu eklemek için ağırlıkları BİR KEZ indir (WiFi önerilir, ~5.9 GB):
#   HF_HUB_OFFLINE=0 ./engine-venv/bin/huggingface-cli download filipstrand/Z-Image-Turbo-mflux-4bit
# İndirdikten sonra MCP/API'de model="z-image-turbo" ile kullanılabilir.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${SAGLITZ_PORT:-8765}"
HOST="${SAGLITZ_HOST:-127.0.0.1}"

if [ ! -x engine-venv/bin/python ]; then
  echo "✗ engine-venv yok."; exit 1
fi

# Allow HuggingFace model downloads: downloading + running models IS the app's
# core function (same as its Draw Things / Civitai downloads). Set =1 on a dev
# box whose weights are already local to avoid per-load etag checks.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# 8-bit: float16 ile aynı kalite, ~yarı RAM. "4" = hızlı taslak, "" = float16.
export SAGLITZ_QUANTIZE="${SAGLITZ_QUANTIZE:-8}"

echo "▶ Saglitz motoru → http://$HOST:$PORT  (durdurmak için Ctrl+C)"
exec ./engine-venv/bin/python -m uvicorn server:app --app-dir engine --host "$HOST" --port "$PORT" --no-access-log
