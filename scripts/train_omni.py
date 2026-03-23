"""OmniVoxtral training script — CLI entry point.

Trains the dual-transformer architecture (Temporal + Depth) with:
- Weighted loss (text=100, semantic=1, acoustic=1)
- Karras EMA schedule
- Linear warmup → cosine decay LR schedule
- W&B logging with separate temporal/depth loss tracking
- Gradient checkpointing + optional torch.compile
- DDP support for multi-GPU training

Usage:
    # Single-stream (default, backward compatible with existing data):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_omni.py

    # Dual-stream (requires diarized data):
    CUDA_VISIBLE_DEVICES=0 DUAL_STREAM=true uv run scripts/train_omni.py

    # Quick test with fake data:
    CUDA_VISIBLE_DEVICES=0 FAKE=true uv run scripts/train_omni.py

    # Override any config via env vars:
    CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=4 LR=1e-4 uv run scripts/train_omni.py
"""

import copy
import datetime
import math
import os
import sys
import time

import dotenv
import torch
import torch.distributed as dist
import tqdm
import transformers as tr
import wandb
from torch.nn.parallel import DistributedDataParallel

import voxtral.trainer.utils as utils
from voxtral.model.init_embeddings import initialize_extended_embeddings
from voxtral.model.omnivoxtral import OmniVoxtral, OmniVoxtralConfig
from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import VoxtralDataset
from voxtral.trainer.omni_trainer import (
    OmniTrainState,
    compute_omni_loss,
    omni_train_step,
)

dotenv.load_dotenv()


def init_omni_train_state(config: VoxtralTrainConfig) -> OmniTrainState:
    """Initialize OmniVoxtral model, optimizer, scheduler, and dataset."""
    device = utils.get_device()

    dual_stream = os.getenv("DUAL_STREAM", "false").lower() == "true"
    lang_adapters = os.getenv("LANGUAGE_ADAPTERS", "false").lower() == "true"
    adapter_rank = int(os.getenv("ADAPTER_RANK", "8"))

    tok_cfg = config.voxtral_tokenizer_config
    num_q = tok_cfg.mimi_num_quantizers
    codebook_size = (config.new_vocab_size - tok_cfg.text_vocab_size) // num_q

    omni_config = OmniVoxtralConfig(
        temporal_pretrained_path=config.mistral_pretrained_path,
        temporal_kwargs=config.mistral_kwargs,
        new_vocab_size=config.new_vocab_size,
        prune_layers=config.prune_layers,
        lora_rank=config.lora_rank,
        num_codebooks=num_q,
        codebook_size=codebook_size,
        text_vocab_size=tok_cfg.text_vocab_size,
        text_hz=tok_cfg.text_hz,
        dual_stream=dual_stream,
        language_adapters=lang_adapters,
        adapter_rank=adapter_rank,
        adapter_alpha=adapter_rank * 2,
        depth_num_layers=config.depth_num_layers,
        depth_dim=config.depth_dim,
        depth_num_heads=config.depth_num_heads,
        depth_dropout=config.depth_dropout,
    )

    utils.pprint(
        f"Initializing OmniVoxtral (dual_stream={dual_stream}, "
        f"lang_adapters={lang_adapters}, "
        f"stride={omni_config.stride}, vocab={omni_config.new_vocab_size})",
        color="bold cyan",
    )

    model = OmniVoxtral(omni_config)

    # Set depth mask rate for exposure bias mitigation (0 = disabled)
    if config.depth_mask_rate > 0:
        model.depth._mask_rate = config.depth_mask_rate
        utils.pprint(f"Depth token masking: rate={config.depth_mask_rate}", color="bold yellow")

    # Initialize extended embeddings with principled initialization
    embed_stats = initialize_extended_embeddings(
        model.temporal,
        original_vocab_size=32768,
        text_vocab_size=omni_config.text_vocab_size,
        total_vocab_size=omni_config.new_vocab_size,
        num_codebooks=omni_config.num_codebooks,
        codebook_size=omni_config.codebook_size,
    )
    utils.pprint(
        f"Embedding init: pretrained_std={embed_stats['pretrained_std']:.4f}, "
        f"indic_std={embed_stats['indic_std']:.4f}, audio_std={embed_stats['audio_std']:.4f}",
        color="bold green",
    )

    # Apply LoRA to temporal transformer if configured.
    # Skip if language adapters are already enabled (they provide per-family LoRA).
    if config.lora_rank is not None and not lang_adapters:
        import peft

        # Attention-only LoRA saves ~3GB VRAM vs all-linear on 7B (15.6 vs 18.8 GB)
        attn_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
        lora_target_modules = [
            name
            for name, module in model.temporal.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(a in name for a in attn_names)
        ]
        lora_config = peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            target_modules=lora_target_modules,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            use_dora=config.use_dora,
            # Keep embed_tokens fully trainable — LoRA would freeze it, but
            # we need new Indic text + audio codebook embeddings to learn.
            modules_to_save=["model.embed_tokens"],
        )
        model.temporal = peft.get_peft_model(model.temporal, lora_config)
        model.temporal.print_trainable_parameters()

    model = model.train()
    model = model.to(device, torch.bfloat16)

    # EMA (on CPU to save GPU memory). Skip when ema_every=0 to save ~7GB CPU RAM at 7B scale.
    if config.ema_every > 0:
        ema = copy.deepcopy(model)
        for param in ema.parameters():
            param.requires_grad = False
        ema = ema.to("cpu")
    else:
        ema = None

    # Optimizer: separate groups for temporal vs depth with optional LR multiplier.
    # Moshi uses 7-17x higher LR for depth (AF-001/H001).
    depth_lr = config.lr * config.depth_lr_multiplier
    temporal_2d = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1 and "depth" not in n]
    temporal_1d = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() == 1 and "depth" not in n]
    depth_2d = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1 and "depth" in n]
    depth_1d = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() == 1 and "depth" in n]

    optimizer_params = [
        {"params": temporal_2d, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": temporal_1d, "lr": config.lr, "weight_decay": 0.0},
        {"params": depth_2d, "lr": depth_lr, "weight_decay": config.weight_decay},
        {"params": depth_1d, "lr": depth_lr, "weight_decay": 0.0},
    ]

    if config.depth_lr_multiplier != 1.0:
        print(f"Separate LRs: temporal={config.lr}, depth={depth_lr} ({config.depth_lr_multiplier}x)")

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config.lr,
        betas=config.lr_betas,
        eps=config.lr_eps,
        fused=device.type == "cuda",
    )

    scheduler = tr.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    train_dataset = VoxtralDataset(config, split="train")
    val_dataset = VoxtralDataset(config, split="val")

    # Print parameter counts
    param_counts = model.param_count()
    n_trainable, gb = utils.trainable_params(model)
    utils.pprint(
        f"Parameters: temporal={param_counts['temporal'] / 1e6:.1f}M, "
        f"depth={param_counts['depth'] / 1e6:.1f}M, "
        f"total={param_counts['total'] / 1e6:.1f}M | "
        f"trainable={n_trainable / 1e6:.1f}M ({gb:.2f}GB)",
        color="bold cyan",
    )

    return OmniTrainState(
        step=0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def save_omni_state(state: OmniTrainState, config: VoxtralTrainConfig) -> None:
    """Save checkpoint. For LoRA models, saves only adapter + depth (compact).

    Full model: ~18GB (too large for disk). LoRA-only: ~2-4GB (fits).
    """
    model = utils.unwrap_model(state.model)
    try:
        import peft
        if isinstance(model.temporal, peft.PeftModel):
            # Compact: LoRA adapter (~1.5GB) + depth (~350MB) + optimizer (~2GB)
            temporal_state = peft.get_peft_model_state_dict(model.temporal)
            checkpoint = {
                "temporal_lora": temporal_state,
                "depth": model.depth.state_dict(),
                "optimizer": state.optimizer.state_dict(),
                "scheduler": state.scheduler.state_dict(),
                "step": state.step,
                "data_step": state.step * config.batch_size * config.world_size,
                "compact": True,  # flag for loader
            }
            utils._save_checkpoint(checkpoint, step=state.step, run_id=config.run_id)
            return
    except ImportError:
        pass
    # Fallback: full state_dict (for non-LoRA models like tiny)
    checkpoint = {
        "model": model.state_dict(),
        "ema": state.ema.state_dict() if state.ema is not None else {},
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "step": state.step,
        "data_step": state.step * config.batch_size * config.world_size,
    }
    utils._save_checkpoint(checkpoint, step=state.step, run_id=config.run_id)


def load_omni_state(state: OmniTrainState, config: VoxtralTrainConfig) -> OmniTrainState:
    """Load checkpoint. Handles both compact (LoRA-only) and full checkpoints."""
    assert config.ckpt_path is not None
    checkpoint = utils._load_checkpoint(config.ckpt_path)
    model = utils.unwrap_model(state.model)

    if checkpoint.get("compact"):
        # Compact checkpoint: LoRA adapter + depth only
        import peft
        peft.set_peft_model_state_dict(model.temporal, checkpoint["temporal_lora"])
        model.depth.load_state_dict(checkpoint["depth"])
        utils.pprint(f"Loaded compact (LoRA+depth) checkpoint", color="bold yellow")
    else:
        # Full checkpoint
        model.load_state_dict(checkpoint["model"])
        if state.ema is not None and checkpoint.get("ema"):
            state.ema.load_state_dict(checkpoint["ema"])

    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.step = checkpoint["step"]
    state.train_dataset.data_step = checkpoint["data_step"]
    utils.pprint(f"Loaded checkpoint from: {config.ckpt_path} (step {state.step})", color="bold yellow")
    return state


def log_omni_metrics(
    state: OmniTrainState, stats: dict[str, float], config: VoxtralTrainConfig
) -> dict[str, float]:
    """Log metrics to W&B and reset stats."""
    count = stats["count"]
    if count == 0:
        return stats

    metrics = {
        "temporal_loss": stats["temporal_loss"] / count,
        "depth_loss": stats["depth_loss"] / count,
        "total_loss": stats["total_loss"] / count,
        "grad_norm": stats["grad_norm"] / count,
        "current_lr": state.scheduler.get_last_lr()[0],
        "text_acc": stats["text_acc"] / count,
        "audio_acc": stats["audio_acc"] / count,
        "depth_acc": stats["depth_acc"] / count,
        "depth_q0_acc": stats["depth_q0_acc"] / count,
        "depth_q1_7_acc": stats["depth_q1_7_acc"] / count,
    }

    utils.rank_0_only(wandb.log)(metrics, step=state.step)

    # Also print to stdout for log file parsing
    print(
        f"[step {state.step}] "
        f"total={metrics['total_loss']:.3f} "
        f"temporal={metrics['temporal_loss']:.3f} "
        f"depth={metrics['depth_loss']:.3f} "
        f"text_acc={metrics['text_acc']:.3f} "
        f"depth_q0={metrics['depth_q0_acc']:.3f} "
        f"depth_q1_7={metrics['depth_q1_7_acc']:.3f} "
        f"grad_norm={metrics['grad_norm']:.3f} "
        f"lr={metrics['current_lr']:.2e}",
        flush=True,
    )

    return {
        "temporal_loss": 0.0, "depth_loss": 0.0, "total_loss": 0.0,
        "grad_norm": 0.0, "count": 0,
        "text_acc": 0.0, "audio_acc": 0.0,
        "depth_acc": 0.0, "depth_q0_acc": 0.0, "depth_q1_7_acc": 0.0,
    }


def create_loader(
    dataset: VoxtralDataset, config: VoxtralTrainConfig
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def train_omni(config: VoxtralTrainConfig) -> None:
    """Main OmniVoxtral training loop."""
    utils.pprint(config.model_dump(), json=True)

    # Distributed setup
    utils.distributed_only(dist.init_process_group)(
        "nccl",
        rank=config.rank,
        world_size=config.world_size,
        timeout=datetime.timedelta(seconds=3600),
    )
    utils.distributed_only(dist.barrier)()

    utils.set_seed(config.seed)
    state = init_omni_train_state(config)

    if config.ckpt_path:
        state = load_omni_state(state, config)

    # Gradient checkpointing on temporal transformer
    if config.gradient_checkpointing:
        model = utils.unwrap_model(state.model)
        # use_reentrant=False is required for DDP compatibility (avoids
        # "mark a variable ready only once" errors with reentrant backward)
        model.temporal.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # DDP
    if config.world_size > 1:
        state.model = DistributedDataParallel(
            state.model,
            device_ids=[config.local_rank],
            find_unused_parameters=True,  # Required: LoRA freezes base params
        )

    # Backend flags
    utils.backend_flags()

    # torch.compile
    if config.compile:
        torch._dynamo.config.optimize_ddp = False
        state.model = torch.compile(state.model, mode="reduce-overhead")

    # W&B
    omni = utils.unwrap_model(state.model)
    utils.rank_0_only(wandb.init)(
        config=config.model_dump() | {
            "model_type": "omnivoxtral",
            "dual_stream": omni.config.dual_stream,
            "temporal_params": omni.param_count()["temporal"],
            "depth_params": omni.param_count()["depth"],
        },
        id=config.run_id,
        resume="allow",
        dir=os.path.join(os.getcwd(), "logs"),
        project=config.wandb_project_name,
        name=config.name,
    )
    if config.watch_every is not None:
        utils.rank_0_only(wandb.watch)(state.model, log_freq=config.watch_every)

    # Data loaders (created once to avoid spawning new worker processes per eval)
    train_loader = iter(create_loader(state.train_dataset, config))
    val_loader = create_loader(state.val_dataset, config) if state.val_dataset and len(state.val_dataset) > 0 else None

    stats = {
        "temporal_loss": 0.0, "depth_loss": 0.0, "total_loss": 0.0,
        "grad_norm": 0.0, "count": 0,
        "text_acc": 0.0, "audio_acc": 0.0,
        "depth_acc": 0.0, "depth_q0_acc": 0.0, "depth_q1_7_acc": 0.0,
    }

    train_bar = tqdm.trange(
        config.max_steps - state.step,
        initial=state.step,
        total=config.max_steps,
        colour="green",
        disable=config.rank > 0,
    )

    for _ in train_bar:
        start_time = time.time()
        try:
            batch = next(train_loader)
        except StopIteration:
            continue

        state.step += 1
        state, stats = omni_train_step(state, batch=batch, stats=stats, config=config)

        # Fast-fail on NaN/exploding loss (Karpathy pattern: catch in seconds, not minutes)
        last_loss = stats["total_loss"] / stats["count"] if stats["count"] else 0
        if math.isnan(last_loss) or last_loss > 100:
            utils.pprint(f"FAIL: loss={last_loss} (NaN or >100). Aborting.", color="bold red")
            sys.exit(1)

        elapsed = time.time() - start_time
        count = stats["count"]
        tl = stats["temporal_loss"] / count if count else 0
        dl = stats["depth_loss"] / count if count else 0
        train_bar.set_description(
            f"T:{tl:.2f} D:{dl:.2f} {elapsed * 1000:.0f}ms/step"
        )

        # EMA update (skipped when ema=None, i.e. ema_every=0)
        if state.ema is not None:
            ema_info = utils.update_ema_karras_(
                state.ema,
                state.model,
                step=state.step,
                gamma=config.ema_gamma,
                ema_every=config.ema_every,
            )
            if ema_info:
                utils.rank_0_only(wandb.log)(ema_info, step=state.step)

        # Periodic logging
        if state.step % config.log_every == 0:
            stats = log_omni_metrics(state, stats, config)

        # Validation loss (uses persistent val_loader to avoid worker leaks)
        if (config.test_every and state.step % config.test_every == 0
                and val_loader is not None):
            state.model.eval()
            val_losses = []
            val_iter = iter(val_loader)
            n_val = min(config.test_size, len(state.val_dataset))
            with torch.no_grad():
                for _ in range(n_val):
                    try:
                        vb = next(val_iter)
                    except StopIteration:
                        break
                    vl = compute_omni_loss(state.model, vb["tokens"].to(utils.get_device()), config)
                    val_losses.append({k: v.item() for k, v in vl.items()})
            if val_losses:
                val_avg = {}
                for k in val_losses[0]:
                    val_avg[f"val_{k}"] = sum(d[k] for d in val_losses) / len(val_losses)
                utils.rank_0_only(wandb.log)(val_avg, step=state.step)
                # Print val metrics to stdout
                vt = val_avg.get("val_total_loss", 0)
                vd = val_avg.get("val_depth_loss", 0)
                vtmp = val_avg.get("val_temporal_loss", 0)
                print(
                    f"[val step {state.step}] "
                    f"val_total={vt:.3f} val_temporal={vtmp:.3f} val_depth={vd:.3f}",
                    flush=True,
                )
            state.model.train()

        # Checkpointing
        if config.save_every and state.step % config.save_every == 0:
            utils.rank_0_only(save_omni_state)(state, config=config)

        # Push to hub
        if config.push_every and state.step % config.push_every == 0 and state.ema is not None:
            utils.rank_0_only(state.ema.push_to_hub)(
                f"{config.name}",
                commit_message=f"step {state.step}, run_id {config.run_id}",
                private=False,
            )

        if state.step >= config.max_steps:
            utils.pprint("\nMax steps reached, exiting...", color="bold red")
            break

        utils.distributed_only(dist.barrier)()

    # Final metrics summary (grep-friendly for auto-research agent)
    if stats["count"] > 0:
        c = stats["count"]
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"\n=== FINAL METRICS (step {state.step}) ===\n"
            f"total_loss={stats['total_loss']/c:.4f}\n"
            f"temporal_loss={stats['temporal_loss']/c:.4f}\n"
            f"depth_loss={stats['depth_loss']/c:.4f}\n"
            f"text_acc={stats['text_acc']/c:.4f}\n"
            f"depth_q0_acc={stats['depth_q0_acc']/c:.4f}\n"
            f"depth_q1_7_acc={stats['depth_q1_7_acc']/c:.4f}\n"
            f"peak_vram_gb={peak_vram:.1f}\n"
            f"================================",
            flush=True,
        )

    # Cleanup
    utils.rank_0_only(wandb.finish)()
    utils.distributed_only(dist.destroy_process_group)()


if __name__ == "__main__":
    from voxtral.tokenizer.model import VoxtralTokenizerConfig

    config = VoxtralTrainConfig(
        name=f"omnivoxtral-{datetime.datetime.now().strftime('%d%m%y')}",
        seed=42,
        mistral_pretrained_path="nilq/mistral-1L-tiny",  # Use tiny for testing
        mistral_kwargs={},
        voxtral_tokenizer_config=VoxtralTokenizerConfig(),
        new_vocab_size=81920,
        loss_weights=[100, 1, 1],  # validated: [100,1,1] beats [100,10,1] (exp036)
        lora_rank=None,
        prune_layers=None,
        # EMA
        ema_gamma=16,
        ema_every=1024,
        # Dataset
        data_path="./data/tokens",
        fake=True,  # Fake data by default for safety
        overfit=None,
        batch_size=2,
        num_workers=4,
        test_size=50,  # AF-203: evaluate more val batches for reliable metrics
        # Speed
        compile=False,
        gradient_checkpointing=False,
        # Optimizer
        lr=3e-4,
        weight_decay=0.0,  # validated: WD=0 beats 0.1 (exp026)
        lr_eps=1e-9,
        lr_betas=(0.9, 0.95),
        grad_norm=1.0,
        warmup_steps=100,
        max_steps=500,
        # Logging
        test_every=500,  # validate every 500 steps (was None — broke all validation)
        generate_kwargs={},
        watch_every=None,
        ckpt_path=None,
        save_every=None,
        push_every=None,
        wandb_project_name="omnivoxtral",
    )

    train_omni(config)
