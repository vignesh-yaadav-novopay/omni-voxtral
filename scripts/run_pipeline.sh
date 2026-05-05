#!/bin/bash
# =============================================================================
# run_pipeline.sh — Orchestrate Phase 1 → 2 → 3 → 1b → 4 in sequence.
# =============================================================================
#
# Designed to be launched once and left running. Polls for completion of each
# phase, launches the next. Each phase is multi-GPU sharded; we wait for ALL
# shards to exit before advancing.
#
# Phases:
#   1a-fleurs      — retokenize_v2 FLEURS (4 GPUs)
#   1a-iv          — retokenize_v2 IndicVoices (1 GPU streaming) — gated on 1a-fleurs done
#   2-vad          — vad_chunker on chunks_indic_yt (4 CPU shards) — independent of 1a
#   3-lid          — detect_language on chunks_v2/yt — gated on 2-vad done
#   1b-yt          — retokenize_v2 YouTube — gated on 3-lid done
#   4-diarize      — diarize_v2 on YouTube source — gated on 3-lid + GPUs free
#
# Each phase's status is tracked in logs/pipeline/<phase>.status with one of:
#   pending | running | done | failed
#
# Usage:
#   nohup bash scripts/run_pipeline.sh > logs/pipeline/orchestrator.log 2>&1 &
#
# Stop:
#   pkill -f run_pipeline.sh && pkill -f retokenize_v2 && pkill -f vad_chunker
# =============================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs/pipeline logs/phase1a/iv logs/phase1b/yt logs/phase3/lid logs/phase4/diarize

LOG_DIR="logs/pipeline"
STATUS() { echo "$(date +%H:%M:%S) $1" >> "$LOG_DIR/orchestrator.log"; echo "$(date +%H:%M:%S) $1"; }
SET_STATE() { echo "$2" > "$LOG_DIR/$1.status"; }
GET_STATE() { cat "$LOG_DIR/$1.status" 2>/dev/null || echo "pending"; }

# Detect already-running ranks for a phase by counting matching pids.
phase_running() {
    local pattern="$1"
    pgrep -f "$pattern" >/dev/null 2>&1
}

# ---------- 1a-fleurs already running externally ----------
if phase_running "retokenize_v2.py --dataset fleurs"; then
    STATUS "1a-fleurs already running externally"
    SET_STATE 1a-fleurs running
fi

# ---------- 2-vad already running externally ----------
if phase_running "vad_chunker.py"; then
    STATUS "2-vad already running externally"
    SET_STATE 2-vad running
fi

# Wait for an external phase to finish (no-process check).
wait_for_phase() {
    local phase="$1" pattern="$2"
    while phase_running "$pattern"; do
        sleep 60
    done
    STATUS "phase $phase: all workers exited"
    SET_STATE "$phase" done
}

launch_indicvoices() {
    STATUS "launching 1a-iv (IndicVoices, 4 shards × 1 GPU)"
    SET_STATE 1a-iv running
    rm -f /tmp/omnivoxtral_gpu_*.lock
    for r in 0 1 2 3; do
        nohup env CUDA_VISIBLE_DEVICES=$r HF_HOME=/apps/hf-home UV_CACHE_DIR=/apps/uv-cache \
            uv run python scripts/retokenize_v2.py \
                --dataset indicvoices --languages all --rank $r --world_size 4 \
                --output_dir data/tokens_v2 --device cuda:0 \
                > logs/phase1a/iv/rank${r}.log 2>&1 &
    done
}

launch_phase3_lid() {
    STATUS "launching 3-lid (4 shards × 1 GPU)"
    SET_STATE 3-lid running
    rm -f /tmp/omnivoxtral_gpu_*.lock
    for r in 0 1 2 3; do
        nohup env CUDA_VISIBLE_DEVICES=$r HF_HOME=/apps/hf-home UV_CACHE_DIR=/apps/uv-cache \
            uv run python scripts/detect_language.py \
                --chunks_dir data/chunks_v2/yt --rank $r --world_size 4 \
                --device cuda:0 --skip_existing \
                > logs/phase3/lid/rank${r}.log 2>&1 &
    done
}

launch_phase1b_yt() {
    STATUS "launching 1b-yt (YouTube retokenization)"
    SET_STATE 1b-yt running
    rm -f /tmp/omnivoxtral_gpu_*.lock
    for r in 0 1 2 3; do
        nohup env CUDA_VISIBLE_DEVICES=$r HF_HOME=/apps/hf-home UV_CACHE_DIR=/apps/uv-cache \
            uv run python scripts/retokenize_v2.py \
                --dataset youtube --input_path data/chunks_v2/yt \
                --rank $r --world_size 4 \
                --output_dir data/tokens_v2 --device cuda:0 \
                > logs/phase1b/yt/rank${r}.log 2>&1 &
    done
}

# ----- Sequence -----

# 1) FLEURS
if [ "$(GET_STATE 1a-fleurs)" = "running" ]; then
    STATUS "waiting on 1a-fleurs ..."
    wait_for_phase 1a-fleurs "retokenize_v2.py --dataset fleurs"
fi

# 2) IndicVoices (gated on 1a-fleurs)
if [ "$(GET_STATE 1a-iv)" != "done" ]; then
    launch_indicvoices
    wait_for_phase 1a-iv "retokenize_v2.py --dataset indicvoices"
fi

# 3) VAD (independent — should already be done by now, but wait if not)
if [ "$(GET_STATE 2-vad)" = "running" ]; then
    STATUS "waiting on 2-vad ..."
    wait_for_phase 2-vad "vad_chunker.py"
fi

# 4) LID (gated on 2-vad)
if [ "$(GET_STATE 3-lid)" != "done" ]; then
    launch_phase3_lid
    wait_for_phase 3-lid "detect_language.py"
fi

# 5) YouTube tokenization (gated on 3-lid)
if [ "$(GET_STATE 1b-yt)" != "done" ]; then
    launch_phase1b_yt
    wait_for_phase 1b-yt "retokenize_v2.py --dataset youtube"
fi

STATUS "pipeline COMPLETE — Phase 4 (diarize) is run separately as it needs"
STATUS "                   the full source YouTube audio + Demucs gating"
