#!/bin/bash
# Waits for FLEURS preprocessing to finish, then launches real training.
# Usage: nohup bash scripts/launch_training.sh &
# Logs: logs/launch_training.log

set -e

DATA_DIR="./data/tokens_fleurs"
LOG_FILE="./logs/launch_training.log"
mkdir -p ./logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== OmniVoxtral Training Launcher ==="
echo "Started: $(date)"
echo "Data directory: $DATA_DIR"

# Get the actual Python process PID (not shell wrappers)
PREPROCESS_PID=$(pgrep -f "python3 scripts/preprocess_hf.py" | head -1 || echo "")

if [ -n "$PREPROCESS_PID" ]; then
    echo "Preprocessing PID: $PREPROCESS_PID"
    echo "Waiting for preprocessing to complete..."

    while kill -0 "$PREPROCESS_PID" 2>/dev/null; do
        COUNT=$(find "$DATA_DIR" -name "*.npy" 2>/dev/null | wc -l)
        LANGS=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read d; do
            c=$(find "$d" -name "*.npy" 2>/dev/null | wc -l)
            [ "$c" -gt 0 ] && echo "$c"
        done | wc -l)
        echo "$(date '+%H:%M:%S') - $COUNT samples across $LANGS languages"
        sleep 120  # Check every 2 minutes
    done

    echo "Preprocessing finished at $(date)"
    # Wait a moment for GPU memory to fully release
    sleep 10
else
    echo "No preprocessing running."
fi

# Final count
COUNT=$(find "$DATA_DIR" -name "*.npy" 2>/dev/null | wc -l)
echo ""
echo "Final sample count: $COUNT"

if [ "$COUNT" -lt 100 ]; then
    echo "ERROR: Only $COUNT samples found. Need at least 100 to train."
    exit 1
fi

# Show per-language breakdown
echo ""
echo "=== Per-Language Counts ==="
for d in "$DATA_DIR"/*/; do
    lang=$(basename "$d")
    lcount=$(find "$d" -name "*.npy" 2>/dev/null | wc -l)
    [ "$lcount" -gt 0 ] && echo "  $lang: $lcount"
done

echo ""
echo "=== Launching Training ==="
echo "Time: $(date)"
echo "Command: PYTHONPATH=/apps/voxtral CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_omni_real.py"
echo ""

cd /apps/voxtral
PYTHONPATH=/apps/voxtral CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_omni_real.py
