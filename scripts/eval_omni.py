"""Quick assessment of a trained OmniVoxtral checkpoint.

Loads a checkpoint, runs forward on real data, reports metrics.
Auto-detects checkpoint type and loads training config from checkpoint if available.

Usage:
    # Check latest checkpoint:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_omni.py --ckpt logs/<run_id>/checkpoint_2000.pt

    # Quick loss check:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_omni.py --ckpt logs/<run_id>/checkpoint_2000.pt --num_samples 20

    # Force full (non-LoRA) mode:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_omni.py --ckpt ... --full
"""

import argparse
import glob
import os

import numpy as np
import torch
import tqdm

from scripts.train_omni import init_omni_train_state
from voxtral.tokenizer.model import VoxtralTokenizerConfig
from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import get_npy_files
from voxtral.trainer.omni_trainer import compute_omni_loss


def find_latest_checkpoint(logs_dir: str = "./logs") -> str | None:
    """Find the most recent checkpoint file."""
    pattern = os.path.join(logs_dir, "**/checkpoint_*.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def detect_checkpoint_type(ckpt_path: str) -> tuple[str, dict | None]:
    """Detect checkpoint type and extract saved config if available.

    Returns (mode, saved_config_dict_or_None).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(checkpoint.get("model", {}).keys())
    saved_config = checkpoint.get("config", None)

    has_lora = any("lora_" in k for k in keys)
    layer_indices = set()
    for k in keys:
        if "temporal.model.layers." in k:
            parts = k.split("temporal.model.layers.")[1]
            idx = parts.split(".")[0]
            if idx.isdigit():
                layer_indices.add(int(idx))
    num_layers = len(layer_indices) if layer_indices else 0
    del checkpoint
    torch.cuda.empty_cache()

    if has_lora:
        return "lora", saved_config
    elif num_layers > 0 and num_layers < 30:
        return "pruned", saved_config
    else:
        return "full", saved_config


def make_config(mode: str, data_path: str, saved_config: dict | None = None) -> VoxtralTrainConfig:
    """Create config matching the training mode.

    If saved_config is available (from checkpoint), use it directly.
    Otherwise fall back to hardcoded defaults for backward compatibility.
    """
    if saved_config is not None:
        # Override data_path and eval-specific settings
        saved_config["data_path"] = data_path
        saved_config["batch_size"] = 1
        saved_config["num_workers"] = 0
        saved_config["fake"] = False
        # Reconstruct VoxtralTokenizerConfig from nested dict
        if isinstance(saved_config.get("voxtral_tokenizer_config"), dict):
            saved_config["voxtral_tokenizer_config"] = VoxtralTokenizerConfig(
                **saved_config["voxtral_tokenizer_config"]
            )
        return VoxtralTrainConfig(**saved_config)

    # Fallback for checkpoints without saved config
    base_kwargs = dict(
        mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
        voxtral_tokenizer_config=VoxtralTokenizerConfig(
            mimi_path="kyutai/mimi",
            whisper_path="openai/whisper-large-v3",
            text_hz=5,
            mimi_num_quantizers=8,
            text_vocab_size=65536,
            sp_tokenizer_path="./data/tokenizer/omnivoxtral_sp.model",
        ),
        new_vocab_size=81920,
        loss_weights=[100, 10, 1],
        fake=False,
        data_path=data_path,
        batch_size=1,
        num_workers=0,
    )

    if mode == "lora":
        return VoxtralTrainConfig(lora_rank=64, prune_layers=2, **base_kwargs)
    elif mode == "pruned":
        return VoxtralTrainConfig(lora_rank=None, prune_layers=2, **base_kwargs)
    else:  # full
        return VoxtralTrainConfig(lora_rank=None, prune_layers=None, **base_kwargs)


@torch.no_grad()
def run_assessment(ckpt_path: str, data_path: str, num_samples: int = 50, mode: str | None = None) -> dict:
    """Run forward pass on real token data and compute metrics."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Auto-detect or use forced mode
    print("Analyzing checkpoint...")
    detected_mode, saved_config = detect_checkpoint_type(ckpt_path)
    if mode is None:
        mode = detected_mode
    print(f"Mode: {mode}" + (" (from saved config)" if saved_config else " (detected from weights)"))

    config = make_config(mode, data_path, saved_config if mode == detected_mode else None)

    # Initialize model
    print("Loading model...")
    state = init_omni_train_state(config)

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    from voxtral.trainer.utils import unwrap_model
    model = unwrap_model(state.model)
    model.load_state_dict(checkpoint["model"])
    model.train(False)

    step = checkpoint.get("step", "?")
    print(f"Checkpoint step: {step}")
    del checkpoint
    torch.cuda.empty_cache()

    # Load data files
    files = get_npy_files(data_path)
    if len(files) > num_samples:
        import random
        random.seed(42)
        files = random.sample(files, num_samples)

    print(f"Running on {len(files)} samples...")

    metrics = {
        "temporal_loss": 0.0, "depth_loss": 0.0, "total_loss": 0.0,
        "text_acc": 0.0, "audio_acc": 0.0,
        "depth_acc": 0.0, "depth_q0_acc": 0.0, "depth_q1_7_acc": 0.0,
        "count": 0,
    }

    for fpath in tqdm.tqdm(files, desc="assessing"):
        tokens = torch.from_numpy(np.load(fpath)).squeeze().unsqueeze(0).to(device)
        losses = compute_omni_loss(state.model, tokens, config)
        for k in metrics:
            if k != "count":
                metrics[k] += losses[k].item()
        metrics["count"] += 1

    # Average
    count = metrics["count"]
    for k in metrics:
        if k != "count":
            metrics[k] /= count

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: latest)")
    parser.add_argument("--data", type=str, default="./data/tokens_fleurs", help="Data directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--full", action="store_true", help="Force full (non-LoRA, non-pruned) mode")
    parser.add_argument("--lora", action="store_true", help="Force LoRA mode")
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Train first.")
        return

    mode = None
    if args.full:
        mode = "full"
    elif args.lora:
        mode = "lora"

    metrics = run_assessment(ckpt_path, args.data, args.num_samples, mode=mode)

    print("\n=== Results ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Samples: {metrics['count']}")
    print(f"")
    print(f"Temporal loss:  {metrics['temporal_loss']:.4f}")
    print(f"Depth loss:     {metrics['depth_loss']:.4f}")
    print(f"Total loss:     {metrics['total_loss']:.4f}")
    print(f"")
    print(f"Text accuracy:     {metrics['text_acc']:.4%}")
    print(f"Audio accuracy:    {metrics['audio_acc']:.4%}")
    print(f"Depth accuracy:    {metrics['depth_acc']:.4%}")
    print(f"  q0 (semantic):   {metrics['depth_q0_acc']:.4%}")
    print(f"  q1-7 (acoustic): {metrics['depth_q1_7_acc']:.4%}")


if __name__ == "__main__":
    main()
