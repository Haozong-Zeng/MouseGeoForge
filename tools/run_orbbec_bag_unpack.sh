#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT.bag OUTPUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$1"
OUTPUT="$2"

mkdir -p "$OUTPUT"
rm -f "$OUTPUT/exit_code.txt"

"$ROOT/tools/bin/orbbec_bag_unpack" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  >"$OUTPUT/unpack.log" 2>&1
status=$?

printf '%s\n' "$status" >"$OUTPUT/exit_code.txt.tmp"
mv -f "$OUTPUT/exit_code.txt.tmp" "$OUTPUT/exit_code.txt"
exit "$status"
