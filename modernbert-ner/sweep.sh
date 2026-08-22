#!/usr/bin/env bash
# Sweep batching / job-count parameters and print chars/s.
# Requires the binary to be built first: cargo build -p modernbert-ner --release
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="target/${PROFILE:-release}/modernbert-ner"
MODEL="${MODEL:-models/modernbert_ja_30m_combined_ja}"
INPUT="${INPUT:-/tmp/book400.txt}"

printf "%-8s %-8s %-6s %s\n" batch maxtok jobs chars/s
for pair in "8:512" "16:1024" "24:1536" "48:4096" "96:8192" "192:16384"; do
  b="${pair%%:*}"; t="${pair##*:}"
  for j in 4 6 8 12; do
    out=$("$BIN" --model "$MODEL" --input "$INPUT" --output /tmp/sweep.jsonl \
      --backend cpu --jobs "$j" --batch-size "$b" --max-tokens "$t" 2>&1 \
      | grep -oE '\([0-9]+ chars/s\)' | tr -d '()')
    printf "%-8s %-8s %-6s %s\n" "$b" "$t" "$j" "$out"
  done
done
