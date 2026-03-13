#!/bin/bash
# Process ALL FLEURS data (no sample limit) for scaled training.
# Run AFTER first training validates with 200/language subset.
#
# FLEURS has ~2,200 train samples/language × 13 languages = ~28,600 samples.
# This is a 14× scaling over the initial 200/language run.
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup bash scripts/preprocess_fleurs_full.sh &

set -e

OUTPUT_DIR="./data/tokens_fleurs_full"
LANGS="hi,kn,ta,te,ml,bn,gu,mr,pa,ur,ne,or,as"
DEVICE="cuda:0"

echo "=== FLEURS Full Preprocessing ==="
echo "Output: $OUTPUT_DIR"
echo "Languages: $LANGS"
echo "Started: $(date)"

# Process all train split samples (no --max_samples limit)
PYTHONPATH=/apps/voxtral CUDA_VISIBLE_DEVICES=0 uv run python scripts/preprocess_hf.py \
    --dataset fleurs \
    --languages "$LANGS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "=== Done ==="
echo "Finished: $(date)"
TOTAL=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)
echo "Total samples: $TOTAL"
