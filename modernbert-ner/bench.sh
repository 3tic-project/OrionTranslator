#!/usr/bin/env bash
# Throughput benchmark for modernbert-ner on a real Japanese novel.
# Usage: crates/modernbert-ner/bench.sh [tag] [jobs]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-run}"
JOBS="${2:-0}"
MODEL="${MODEL:-models/modernbert_ja_30m_combined_ja}"
INPUT="${INPUT:-data/test/300e9d904cab12c2296245f0f402b9cd.txt}"
BATCH="${BATCH:-24}"
MAXTOK="${MAXTOK:-1536}"
PROFILE="${PROFILE:-release}"
OUT="/tmp/mbner_${TAG}.jsonl"

cargo build -p modernbert-ner --profile "$PROFILE" >/dev/null
BIN="target/$PROFILE/modernbert-ner"

echo "== bench tag=$TAG jobs=$JOBS batch=$BATCH max_tokens=$MAXTOK profile=$PROFILE =="
"$BIN" \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$OUT" \
  --backend cpu \
  --jobs "$JOBS" \
  --batch-size "$BATCH" \
  --max-tokens "$MAXTOK" \
  --profile 2>&1 | grep -E "profile|done|packed"

echo "output: $OUT ($(wc -l < "$OUT") lines, sha=$(shasum -a 256 "$OUT" | cut -c1-16))"
