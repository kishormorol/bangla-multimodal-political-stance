#!/usr/bin/env bash
# Mirror the shared Drive folder into data/raw/.
#
# Google rate-limits bulk downloads from a shared folder, so a first run may
# stop partway with "Cannot retrieve the public link ... have had many
# accesses". The download resumes: re-run this after a few minutes and it picks
# up only the files that are still missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" -m bmpb.data.download "$@"

csvs=$(find "$ROOT/data/raw" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')
images=$(find "$ROOT/data/raw/images" -type f 2>/dev/null | wc -l | tr -d ' ')
processed=$(find "$ROOT/data/raw/processed_images" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "data/raw: ${csvs} csv, ${images} images, ${processed} processed_images"
echo "expected: 25 csv, ~200 images, 149 processed_images (375 Drive entries in total)"
