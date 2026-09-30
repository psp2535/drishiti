#!/usr/bin/env bash
# One-shot demo: simulated AOI -> index (7 scenes) -> incremental add (8th) -> start the offline UI.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
[ -f data/raw/S2_20251022_SYN_AOI.tif ] || python scripts/make_synthetic_aoi.py
rm -rf data/store
python scripts/build_index.py --src data/raw --store data/store --exclude S2_20251022_SYN_AOI.tif
python scripts/build_index.py --add data/raw/S2_20251022_SYN_AOI.tif --store data/store
echo "Open http://127.0.0.1:${PORT:-8000}  (Ctrl+C to stop)"
DRISHTI_STORE=data/store exec python -m uvicorn drishti.api:app --host 127.0.0.1 --port "${PORT:-8000}"
