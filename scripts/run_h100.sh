#!/bin/bash
# =============================================================================
# OmniVoxtral Training Launch Script
# =============================================================================
#
# Hardware-agnostic launcher. Auto-detects GPU count. All config via env vars.
#
# Usage:
#   # Auto-detect GPUs, default config:
#   bash scripts/run_h100.sh
#
#   # Training only (data already preprocessed):
#   SKIP_PREPROCESS=1 bash scripts/run_h100.sh
#
#   # Resume from checkpoint:
#   CKPT_PATH=logs/<run_id>/checkpoint_500.pt bash scripts/run_h100.sh
#
#   # Scale to 8×H100:
#   BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2 bash scripts/run_h100.sh
#
#   # Conservative for smaller GPUs:
#   BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8 bash scripts/run_h100.sh
#
# =============================================================================

set -euo pipefail
cd /apps/voxtral

echo "============================================="
echo " OmniVoxtral — Training Pipeline"
echo "============================================="
echo ""

# --- Step 0: Environment ---
source .env 2>/dev/null || true
export PYTHONPATH=/apps/voxtral
export TOKENIZERS_PARALLELISM=false

# Auto-detect available GPUs if not set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# Count GPUs from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "[0/3] Environment check..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)')
"
echo ""

# --- Step 1: Preprocess remaining FLEURS languages ---
if [ "${SKIP_PREPROCESS:-0}" != "1" ]; then
    echo "[1/3] Preprocessing remaining FLEURS languages..."

    # Use first GPU only for preprocessing
    FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
    CUDA_VISIBLE_DEVICES=$FIRST_GPU uv run scripts/preprocess_hf.py \
        --dataset fleurs \
        --languages ml,bn,gu,mr,pa,ur,ne,or,as,en \
        --output_dir "${DATA_PATH:-./data/tokens_fleurs}" \
        --max_samples "${MAX_SAMPLES_PER_LANG:-200}"

    TOTAL_FILES=$(find "${DATA_PATH:-./data/tokens_fleurs}" -name "*.npy" | wc -l)
    echo "  Total token files after preprocessing: $TOTAL_FILES"
    echo ""
else
    TOTAL_FILES=$(find "${DATA_PATH:-./data/tokens_fleurs}" -name "*.npy" | wc -l)
    echo "[1/3] Skipping preprocessing (SKIP_PREPROCESS=1)"
    echo "  Token files available: $TOTAL_FILES"
    echo ""
fi

# Free GPU memory from preprocessing
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# --- Step 2: Validate data ---
echo "[2/3] Validating training data..."
python3 -c "
import os, numpy as np
data_dir = '${DATA_PATH:-./data/tokens_fleurs}'
npy_files = []
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.npy'):
            npy_files.append(os.path.join(root, f))

print(f'  Total .npy files: {len(npy_files)}')

if npy_files:
    sample = np.load(npy_files[0])
    print(f'  Sample shape: {sample.shape}')
    print(f'  Token range: [{sample.min()}, {sample.max()}]')

langs = {}
for f in npy_files:
    parts = f.split('/')
    for i, p in enumerate(parts):
        if p in ('tokens_fleurs', 'tokens') and i+1 < len(parts):
            lang = parts[i+1]
            langs[lang] = langs.get(lang, 0) + 1
            break
for lang, count in sorted(langs.items()):
    print(f'  {lang}: {count} files')
"
echo ""

# --- Step 3: Launch FSDP training ---
ACCUM=${GRADIENT_ACCUMULATION_STEPS:-1}
BATCH=${BATCH_SIZE:-4}
EFFECTIVE=$((BATCH * NUM_GPUS * ACCUM))

echo "[3/3] Launching training with FSDP..."
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Batch: ${BATCH}/GPU × ${NUM_GPUS} GPUs × ${ACCUM} accum = ${EFFECTIVE} effective"
echo "  Save every: ${SAVE_EVERY:-500} steps"
echo "  Max steps: ${MAX_STEPS:-100000}"
echo "  Keep checkpoints: ${KEEP_CHECKPOINTS:-5}"
echo ""

# NCCL optimization
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29500}" \
    scripts/train_omni_full.py

echo ""
echo "============================================="
echo " Training complete!"
echo "============================================="
echo " Checkpoints: logs/<run_id>/checkpoint_*.pt"
echo " W&B: https://wandb.ai/vignesh-yaadav/omnivoxtral"
echo ""
echo " Next steps:"
echo "   1. Eval:   CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_omni.py"
echo "   2. Resume: CKPT_PATH=logs/<id>/checkpoint_N.pt bash scripts/run_h100.sh"
echo "============================================="
