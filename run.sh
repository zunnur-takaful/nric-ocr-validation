#!/usr/bin/env bash
# Single-command entry point: builds the image (cached after first run) and
# validates a file or folder, e.g. ./run.sh samples/sample.jpg
set -euo pipefail

INPUT="${1:?Usage: ./run.sh <image file or folder> [--side front|back]}"
shift
ABS_PATH="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

if [ -d "$ABS_PATH" ]; then
  HOST_DIR="$ABS_PATH"
  CONTAINER_PATH="/data"
else
  HOST_DIR="$(dirname "$ABS_PATH")"
  CONTAINER_PATH="/data/$(basename "$ABS_PATH")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -q -t nric-validate "$SCRIPT_DIR" > /dev/null
docker run --rm -v "$HOST_DIR":/data nric-validate "$CONTAINER_PATH" "$@"
