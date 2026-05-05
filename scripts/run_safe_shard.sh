#!/bin/bash
# =============================================================================
# run_safe_shard.sh — Per-GPU watchdog wrapper for sharded preprocessing.
# =============================================================================
#
# Variant of autoresearch/scripts/run_safe.sh that uses a PER-GPU lock so 4
# shards (one per GPU) can run simultaneously. The upstream watchdog uses a
# single shared lock which would force shards to run sequentially.
#
# Usage:
#   bash scripts/run_safe_shard.sh "CUDA_VISIBLE_DEVICES=0 uv run scripts/X --rank 0" 86400
#   bash scripts/run_safe_shard.sh "CUDA_VISIBLE_DEVICES=1 uv run scripts/X --rank 1" 86400
#
# Lock files are at /tmp/omnivoxtral_gpu_<gpu_id>.lock.
# Resource thresholds and exit codes match run_safe.sh.
# =============================================================================

set -uo pipefail

COMMAND="${1:?Usage: run_safe_shard.sh COMMAND [TIME_BUDGET] [MONITOR_INTERVAL] [GPU_THRESHOLD]}"
TIME_BUDGET="${2:-86400}"           # default 24h for preprocessing
MONITOR_INTERVAL="${3:-60}"
GPU_THRESHOLD="${4:-90}"

# Extract CUDA_VISIBLE_DEVICES from the command for per-GPU lock naming.
GPU_ID=$(echo "$COMMAND" | grep -oP "CUDA_VISIBLE_DEVICES=\K[0-9]+(,[0-9]+)*" | head -1)
GPU_ID="${GPU_ID:-all}"

LOCK_FILE="/tmp/omnivoxtral_gpu_${GPU_ID}.lock"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$(cd "$SCRIPT_DIR/../autoresearch/scripts" && pwd)/monitor.py"

echo "=== run_safe_shard.sh ==="
echo "  Command:    $COMMAND"
echo "  Budget:     ${TIME_BUDGET}s"
echo "  GPU(s):     ${GPU_ID}"
echo "  Lock:       ${LOCK_FILE}"
echo "========================="

# Per-GPU lock check.
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[lock] GPU ${GPU_ID} locked by PID=$LOCK_PID. Exiting."
        exit 4
    else
        echo "[lock] Stale lock from PID=$LOCK_PID. Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

# Pre-flight only if monitor script exists.
if [ -f "$MONITOR_SCRIPT" ]; then
    PREFLIGHT_EXIT=0
    PREFLIGHT=$(python3 "$MONITOR_SCRIPT" --gpu-threshold "$GPU_THRESHOLD" 2>&1) || PREFLIGHT_EXIT=$?
    if [ $PREFLIGHT_EXIT -ne 0 ]; then
        echo "[pre-flight] UNSAFE — aborting"
        echo "$PREFLIGHT"
        exit 2
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Launch in a new process group so the whole group can be killed.
echo "[launch] Starting"
setsid bash -c "$COMMAND" &
CHILD_PID=$!
echo "[launch] PID=$CHILD_PID"

START_TIME=$(date +%s)

cleanup() {
    if kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "[cleanup] SIGTERM -$CHILD_PID"
        kill -TERM -- -"$CHILD_PID" 2>/dev/null
        sleep 5
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            kill -9 -- -"$CHILD_PID" 2>/dev/null
        fi
    fi
}
trap cleanup EXIT

while kill -0 "$CHILD_PID" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then
        break
    fi
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    if [ "$ELAPSED" -ge "$TIME_BUDGET" ]; then
        echo "[timeout] Budget reached (${ELAPSED}s)"
        kill -TERM -- -"$CHILD_PID" 2>/dev/null
        sleep 5
        kill -9 -- -"$CHILD_PID" 2>/dev/null
        wait "$CHILD_PID" 2>/dev/null
        trap - EXIT
        rm -f "$LOCK_FILE"
        exit 3
    fi
    if [ -f "$MONITOR_SCRIPT" ]; then
        SAFETY_JSON=$(python3 "$MONITOR_SCRIPT" --gpu-threshold "$GPU_THRESHOLD" 2>/dev/null)
        IS_SAFE=$(echo "$SAFETY_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['safe'])" 2>/dev/null || echo "True")
        if [ "$IS_SAFE" = "False" ]; then
            VIOLATIONS=$(echo "$SAFETY_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['violations']))" 2>/dev/null || echo "?")
            echo "[SAFETY] Threshold breached: $VIOLATIONS — killing"
            kill -TERM -- -"$CHILD_PID" 2>/dev/null
            sleep 5
            kill -9 -- -"$CHILD_PID" 2>/dev/null
            wait "$CHILD_PID" 2>/dev/null
            trap - EXIT
            rm -f "$LOCK_FILE"
            exit 2
        fi
    fi
done

# Child finished naturally
wait "$CHILD_PID"
EXIT_CODE=$?
trap - EXIT
rm -f "$LOCK_FILE"
echo "[done] exit=$EXIT_CODE"
exit $EXIT_CODE
