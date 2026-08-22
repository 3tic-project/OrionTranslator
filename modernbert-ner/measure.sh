#!/usr/bin/env bash
# Repeat a config N times on the full book and report the runs plus the best chars/s.
# Requires the binary to be built first: cargo build -p modernbert-ner --release
# Usage: measure.sh <jobs> <batch> <maxtok> [repeats]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="target/${PROFILE:-release}/modernbert-ner"
MODEL="${MODEL:-models/modernbert_ja_30m_combined_ja}"
INPUT="${INPUT:-data/test/300e9d904cab12c2296245f0f402b9cd.txt}"
J="${1:-6}"; B="${2:-16}"; T="${3:-1024}"; N="${4:-3}"

vals=()
for _ in $(seq "$N"); do
  v=$("$BIN" --model "$MODEL" --input "$INPUT" --output /tmp/measure.jsonl \
      --backend cpu --jobs "$J" --batch-size "$B" --max-tokens "$T" 2>&1 \
      | grep -oE '[0-9]+ chars/s' | grep -oE '^[0-9]+')
  vals+=("$v")
done
printf "jobs=%-3s batch=%-4s maxtok=%-6s runs=[%s] best=%s\n" \
  "$J" "$B" "$T" "$(IFS=,; echo "${vals[*]}")" \
  "$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)"
