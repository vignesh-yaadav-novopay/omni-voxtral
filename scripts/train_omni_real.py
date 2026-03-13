"""OmniVoxtral training on real Indic data.

Preconfigured for A10G (23GB VRAM) with practical settings:
- LoRA rank 64 on temporal transformer (fits in VRAM)
- Layer pruning (drop every 2nd layer → 3.5B params)
- Gradient checkpointing enabled
- Batch size 2 (A10G constraint)

Usage:
    # Train on FLEURS tokens:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_omni_real.py

    # Train with language adapters:
    CUDA_VISIBLE_DEVICES=0 LANGUAGE_ADAPTERS=true uv run scripts/train_omni_real.py

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=0 CKPT_PATH=logs/xxx/checkpoint_5000.pt uv run scripts/train_omni_real.py

    # Override any param via env:
    CUDA_VISIBLE_DEVICES=0 MAX_STEPS=50000 LR=5e-5 uv run scripts/train_omni_real.py
"""

import datetime

from voxtral.tokenizer.model import VoxtralTokenizerConfig

from scripts.train_omni import train_omni
from voxtral.trainer.config import VoxtralTrainConfig

config = VoxtralTrainConfig(
    name=f"omnivoxtral-indic-{datetime.datetime.now().strftime('%d%m%y')}",
    seed=42,
    # Model: Mistral 7B, pruned to ~3.5B, with LoRA
    mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
    mistral_kwargs={},
    voxtral_tokenizer_config=VoxtralTokenizerConfig(
        mimi_path="kyutai/mimi",
        whisper_path="openai/whisper-large-v3",
        text_hz=5,
        mimi_num_quantizers=8,
        text_vocab_size=65536,
        language="hi",  # Default language (overridden per-batch with adapters)
        sp_tokenizer_path="./data/tokenizer/omnivoxtral_sp.model",
    ),
    new_vocab_size=81920,
    loss_weights=[100, 10, 1],  # text=100x, semantic=10x, acoustic=1x
    lora_rank=64,       # LoRA for A10G memory constraint
    prune_layers=2,     # Drop every 2nd layer → ~3.5B params
    codec_hz=55,
    # EMA
    ema_gamma=16,
    ema_every=1024,
    # Dataset: preprocessed tokens from FLEURS/IndicVoices
    data_path="./data/tokens_fleurs",
    fake=False,
    overfit=None,
    batch_size=2,       # A10G constraint
    num_workers=4,
    test_size=4,
    # Speed
    compile=False,      # Disable for first run (enable after validating)
    gradient_checkpointing=True,  # Essential for A10G
    # Optimizer (conservative for fine-tuning)
    lr=1e-4,
    weight_decay=0.1,
    lr_eps=1e-9,
    lr_betas=(0.9, 0.95),
    grad_norm=1.0,
    warmup_steps=500,
    max_steps=10_000,
    # Logging
    test_every=1_000,
    generate_kwargs={},
    watch_every=None,
    ckpt_path=None,     # Set via env var to resume
    save_every=2_000,
    push_every=None,
    wandb_project_name="omnivoxtral",
)

train_omni(config)
