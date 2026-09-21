#!/usr/bin/env bash
# Train every config in configs/ and rebuild the results table.
#
#   bash scripts/run_all.sh                 # everything
#   bash scripts/run_all.sh configs/text    # one subtree
#
# Runs are independent: a model that fails (a checkpoint that will not download,
# an OOM) is reported at the end and does not stop the rest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TARGET="${1:-$ROOT/configs}"

if [[ ! -x "$PYTHON" ]]; then
  echo "no interpreter at $PYTHON — run 'make install' first" >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "$TARGET" -name '*.yaml' ! -name 'data.yaml' | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "no configs found under $TARGET" >&2
  exit 1
fi

echo "Training ${#CONFIGS[@]} configs from $TARGET"
FAILED=()
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  echo
  echo "=== $name ==="
  if ! "$PYTHON" -m bmpb.cli train --config "$config"; then
    echo "!!! $name failed" >&2
    FAILED+=("$name")
  fi
done

echo
"$PYTHON" -m bmpb.cli leaderboard

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "${#FAILED[@]} config(s) failed: ${FAILED[*]}" >&2
  exit 1
fi
