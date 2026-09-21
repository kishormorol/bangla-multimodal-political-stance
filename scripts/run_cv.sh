#!/usr/bin/env bash
# Cross-validate every model under the one shared protocol.
#
#   bash scripts/run_cv.sh                    # every config
#   bash scripts/run_cv.sh configs/text       # one subtree
#
# Each model is independent: one that fails (a checkpoint that will not
# download, an OOM) is reported at the end and does not stop the rest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TARGET="${1:-$ROOT/configs}"
FOLDS="${FOLDS:-5}"

mapfile -t CONFIGS < <(find "$TARGET" -name '*.yaml' ! -name 'data.yaml' | sort)
echo "Cross-validating ${#CONFIGS[@]} configs at $FOLDS folds"

FAILED=()
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  echo; echo "=== $name ==="
  if ! "$PYTHON" -m bmpb.cli cv --config "$config" --folds "$FOLDS"; then
    echo "!!! $name failed" >&2
    FAILED+=("$name")
  fi
done

echo
"$PYTHON" -m bmpb.cli leaderboard

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo; echo "${#FAILED[@]} failed: ${FAILED[*]}" >&2
  exit 1
fi
