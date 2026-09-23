#!/usr/bin/env bash
# Cross-validate every ablation config.
#
#   bash scripts/run_ablations.sh               # all ablations
#   bash scripts/run_ablations.sh clip_unfrozen # one ablation by name
#
# Each ablation is independent: one that fails does not stop the rest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ABLATION_DIR="$ROOT/configs/ablations"
FOLDS="${FOLDS:-5}"

if [[ $# -gt 0 ]]; then
  # Run a single named ablation.
  CONFIGS=("$ABLATION_DIR/$1.yaml")
  if [[ ! -f "${CONFIGS[0]}" ]]; then
    echo "Config not found: ${CONFIGS[0]}" >&2
    exit 1
  fi
else
  mapfile -t CONFIGS < <(find "$ABLATION_DIR" -name '*.yaml' | sort)
fi

echo "Running ${#CONFIGS[@]} ablation(s) at $FOLDS folds"

FAILED=()
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  echo; echo "=== ablation: $name ==="
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
