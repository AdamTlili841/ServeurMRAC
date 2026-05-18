#!/bin/sh
set -e
PORT="${PORT:-10000}"
echo "MRAC-FND: uvicorn on 0.0.0.0:${PORT} (preload=${PRELOAD_MODEL:-true})"
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 120
