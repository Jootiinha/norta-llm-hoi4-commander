#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHUNKS_DIR="$ROOT_DIR/data/processed/chunks"
HISTORY_DIR="$CHUNKS_DIR/history"
CONFIG_PATH="$ROOT_DIR/configs/pipeline.yaml"

files=(
  "chunks.jsonl"
  "manifest.jsonl"
  "summary.json"
)

has_snapshot_source=0
for file_name in "${files[@]}"; do
  if [[ -f "$CHUNKS_DIR/$file_name" ]]; then
    has_snapshot_source=1
    break
  fi
done

if [[ "$has_snapshot_source" -eq 0 ]]; then
  exit 0
fi

timestamp="$(date +"%Y%m%d-%H%M%S")"
snapshot_dir="$HISTORY_DIR/$timestamp"
suffix=1
while [[ -e "$snapshot_dir" ]]; do
  snapshot_dir="$HISTORY_DIR/${timestamp}-$(printf "%02d" "$suffix")"
  suffix=$((suffix + 1))
done

mkdir -p "$snapshot_dir"

for file_name in "${files[@]}"; do
  if [[ -f "$CHUNKS_DIR/$file_name" ]]; then
    cp "$CHUNKS_DIR/$file_name" "$snapshot_dir/$file_name"
  fi
done

{
  printf "created_at=%s\n" "$(date -Iseconds)"
  printf "source_dir=%s\n" "$CHUNKS_DIR"
  printf "config_path=%s\n" "$CONFIG_PATH"
} > "$snapshot_dir/backup.env"

if [[ -f "$CONFIG_PATH" ]]; then
  cp "$CONFIG_PATH" "$snapshot_dir/pipeline.yaml"
fi

printf "Backup de chunks salvo em %s\n" "$snapshot_dir"
