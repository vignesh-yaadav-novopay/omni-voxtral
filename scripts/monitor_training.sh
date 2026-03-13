#!/bin/bash
# Quick monitor for OmniVoxtral training progress.
# Usage: bash scripts/monitor_training.sh

echo "=== OmniVoxtral Training Monitor ==="
echo ""

# Check preprocessing
PREP_PID=$(pgrep -f "python3 scripts/preprocess_hf.py" | head -1 || echo "")
if [ -n "$PREP_PID" ]; then
    COUNT=$(find ./data/tokens_fleurs -name "*.npy" 2>/dev/null | wc -l)
    echo "Preprocessing: RUNNING (PID $PREP_PID), $COUNT samples"
    for d in ./data/tokens_fleurs/*/; do
        lang=$(basename "$d")
        lcount=$(find "$d" -name "*.npy" 2>/dev/null | wc -l)
        [ "$lcount" -gt 0 ] && echo "  $lang: $lcount"
    done
else
    COUNT=$(find ./data/tokens_fleurs -name "*.npy" 2>/dev/null | wc -l)
    echo "Preprocessing: DONE ($COUNT samples)"
fi

echo ""

# Check launcher
LAUNCH_PID=$(pgrep -f "launch_training.sh" | head -1 || echo "")
if [ -n "$LAUNCH_PID" ]; then
    echo "Launcher: RUNNING (PID $LAUNCH_PID)"
    tail -3 ./logs/launch_training.log 2>/dev/null
else
    echo "Launcher: NOT RUNNING"
fi

echo ""

# Check training
TRAIN_PID=$(pgrep -f "python.*train_omni_real.py" | head -1 || echo "")
if [ -n "$TRAIN_PID" ]; then
    echo "Training: RUNNING (PID $TRAIN_PID)"
    # Show GPU usage
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0 2>/dev/null
else
    echo "Training: NOT RUNNING"
fi

echo ""

# GPU status
echo "=== GPU:0 Status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader -i 0 2>/dev/null

echo ""

# Latest W&B logs
LATEST_LOG=$(find ./logs -name "*.log" -newer ./logs/launch_training.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "=== Latest Training Log ==="
    tail -5 "$LATEST_LOG"
fi
