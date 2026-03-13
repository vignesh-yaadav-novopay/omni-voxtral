"""OmniVoxtral FULL fine-tuning with FSDP across multiple GPUs.

No LoRA, no pruning — full 7B parameter training with FSDP sharding.
Supports checkpoint resume to survive crashes.
All values configurable via env vars for easy hardware transport.

Usage:
    # 2×H100:
    torchrun --nproc_per_node=2 scripts/train_omni_full.py

    # Resume from checkpoint:
    CKPT_PATH=logs/<run_id>/checkpoint_500.pt torchrun --nproc_per_node=2 scripts/train_omni_full.py

    # Override config via env:
    MAX_STEPS=100000 LR=3e-5 BATCH_SIZE=4 torchrun --nproc_per_node=2 scripts/train_omni_full.py

    # Scale to 8×H100:
    BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2 torchrun --nproc_per_node=8 scripts/train_omni_full.py
"""

import datetime
import functools
import glob
import os
import re
import time

import dotenv
import torch
import torch.distributed as dist
import tqdm
import transformers as tr
import wandb
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import voxtral.trainer.utils as utils
from voxtral.model.depth_transformer import DepthTransformerLayer
from voxtral.model.init_embeddings import initialize_extended_embeddings
from voxtral.model.omnivoxtral import OmniVoxtral, OmniVoxtralConfig
from voxtral.tokenizer.model import VoxtralTokenizerConfig
from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import VoxtralDataset
from voxtral.trainer.omni_trainer import (
    OmniTrainState,
    compute_omni_loss,
    omni_train_step,
)

dotenv.load_dotenv()


def init_fsdp_model(config: VoxtralTrainConfig) -> OmniVoxtral:
    """Build OmniVoxtral model from config. No hardcoded architecture values."""
    tok_cfg = config.voxtral_tokenizer_config
    num_q = tok_cfg.mimi_num_quantizers
    text_vocab = tok_cfg.text_vocab_size
    # Derive codebook_size from vocab layout instead of hardcoding
    codebook_size = (config.new_vocab_size - text_vocab) // num_q

    omni_config = OmniVoxtralConfig(
        temporal_pretrained_path=config.mistral_pretrained_path,
        temporal_kwargs=config.mistral_kwargs,
        new_vocab_size=config.new_vocab_size,
        prune_layers=config.prune_layers,
        lora_rank=config.lora_rank,
        num_codebooks=num_q,
        codebook_size=codebook_size,
        text_vocab_size=text_vocab,
        text_hz=tok_cfg.text_hz,
        dual_stream=config.dual_stream,
        language_adapters=config.language_adapters,
    )

    if config.rank == 0:
        utils.pprint(
            f"Initializing OmniVoxtral (prune={config.prune_layers}, lora={config.lora_rank}, "
            f"stride={omni_config.stride}, vocab={omni_config.new_vocab_size}, "
            f"codebook_size={codebook_size}, dual={config.dual_stream})",
            color="bold cyan",
        )

    model = OmniVoxtral(omni_config)

    # Initialize extended embeddings — derive original vocab from loaded model
    original_vocab_size = model.temporal.config.vocab_size
    # After resize, vocab_size changes — but we need the PRETRAINED size.
    # Mistral 7B v0.3 = 32768. Read from model config before resize.
    # Since resize already happened in OmniVoxtral.__init__, use the known constant.
    # This is a property of the pretrained model, not a tunable parameter.
    if original_vocab_size == config.new_vocab_size:
        # Already resized — use Mistral's known pretrained vocab
        original_vocab_size = 32768

    embed_stats = initialize_extended_embeddings(
        model.temporal,
        original_vocab_size=original_vocab_size,
        text_vocab_size=text_vocab,
        total_vocab_size=config.new_vocab_size,
        num_codebooks=num_q,
        codebook_size=codebook_size,
    )
    if config.rank == 0:
        utils.pprint(
            f"Embedding init: pretrained_std={embed_stats['pretrained_std']:.4f}, "
            f"indic_std={embed_stats['indic_std']:.4f}, audio_std={embed_stats['audio_std']:.4f}",
            color="bold green",
        )

    if config.gradient_checkpointing:
        model.temporal.gradient_checkpointing_enable()

    model = model.train()
    model = model.to(torch.bfloat16)

    return model


def wrap_fsdp(model: OmniVoxtral) -> FSDP:
    """Wrap OmniVoxtral with FSDP for multi-GPU sharding."""
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            tr.models.mistral.modeling_mistral.MistralDecoderLayer,
            DepthTransformerLayer,
        },
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )

    return model


def save_fsdp_checkpoint(model: FSDP, optimizer, scheduler, step, config):
    """Save full state dict from FSDP model (only on rank 0)."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    if config.rank == 0:
        log_dir = os.path.join(os.getcwd(), "logs", config.run_id)
        os.makedirs(log_dir, exist_ok=True)
        ckpt_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
        torch.save({
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config.model_dump(),
        }, ckpt_path)
        utils.pprint(f"Saved checkpoint: {ckpt_path}", color="bold yellow")

        # Rotate old checkpoints
        ckpt_files = glob.glob(os.path.join(log_dir, "checkpoint_*.pt"))
        ckpt_files.sort(
            key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
            reverse=True,
        )
        for old in ckpt_files[config.keep_checkpoints:]:
            os.remove(old)
            utils.pprint(f"Removed old checkpoint: {old}", color="dim")

    dist.barrier()


def load_fsdp_checkpoint(
    fsdp_model: FSDP, optimizer, scheduler, ckpt_path: str, rank: int
) -> int:
    """Load checkpoint into FSDP model. Returns the step number."""
    map_location = {"cuda:0": f"cuda:{rank}"}
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        fsdp_model.load_state_dict(checkpoint["model"])
        optim_state = FSDP.optim_state_dict_to_load(
            fsdp_model, optimizer, checkpoint.get("optimizer", {})
        )
        optimizer.load_state_dict(optim_state)

    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    step = checkpoint.get("step", 0)
    if rank == 0:
        utils.pprint(f"Resumed from checkpoint: {ckpt_path} (step {step})", color="bold yellow")

    # Free checkpoint memory
    del checkpoint
    torch.cuda.empty_cache()

    return step


def _log_gpu_memory(step: int):
    """Log GPU memory stats to wandb. Helps detect leaks over long runs."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    wandb.log({
        "gpu/allocated_gb": allocated,
        "gpu/reserved_gb": reserved,
        "gpu/peak_allocated_gb": max_allocated,
        "gpu/fragmentation_pct": (1 - allocated / reserved) * 100 if reserved > 0 else 0,
    }, step=step)


def _make_stats() -> dict[str, float]:
    return {
        "temporal_loss": 0.0, "depth_loss": 0.0, "total_loss": 0.0,
        "grad_norm": 0.0, "count": 0,
        "text_acc": 0.0, "audio_acc": 0.0,
        "depth_acc": 0.0, "depth_q0_acc": 0.0, "depth_q1_7_acc": 0.0,
    }


def train_full(config: VoxtralTrainConfig) -> None:
    """Full fine-tuning with FSDP across multiple GPUs."""
    # Distributed setup — generous timeout for model loading on slow networks
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    config.rank = int(os.environ.get("RANK", 0))
    config.local_rank = local_rank
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if config.rank == 0:
        utils.pprint(f"World size: {config.world_size} GPUs", color="bold cyan")
        effective_batch = config.batch_size * config.world_size * config.gradient_accumulation_steps
        utils.pprint(
            f"Effective batch: {config.batch_size} × {config.world_size} GPUs "
            f"× {config.gradient_accumulation_steps} accum = {effective_batch}",
            color="bold cyan",
        )
        utils.pprint(config.model_dump(), json=True)

    utils.set_seed(config.seed + config.rank)

    # Build model and wrap with FSDP
    model = init_fsdp_model(config)

    if config.rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        utils.pprint(
            f"Parameters: {total/1e9:.2f}B total, {trainable/1e9:.2f}B trainable",
            color="bold cyan",
        )

    fsdp_model = wrap_fsdp(model)

    # Optimizer — all params trainable, separate weight decay groups
    wd_params = [p for p in fsdp_model.parameters() if p.requires_grad and p.dim() > 1]
    no_wd_params = [p for p in fsdp_model.parameters() if p.requires_grad and p.dim() <= 1]

    optimizer = torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": config.weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.lr_betas,
        eps=config.lr_eps,
    )

    scheduler = tr.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if config.ckpt_path:
        start_step = load_fsdp_checkpoint(
            fsdp_model, optimizer, scheduler, config.ckpt_path, config.rank
        )

    # Dataset
    train_dataset = VoxtralDataset(config)
    train_loader = iter(torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    ))

    # W&B (rank 0 only)
    if config.rank == 0:
        wandb.init(
            config=config.model_dump() | {
                "model_type": "omnivoxtral-full",
                "fsdp": True,
                "num_gpus": config.world_size,
                "effective_batch_size": config.batch_size * config.world_size * config.gradient_accumulation_steps,
            },
            id=config.run_id,
            resume="allow",
            dir=os.path.join(os.getcwd(), "logs"),
            project=config.wandb_project_name,
            name=config.name,
        )

    # Backend flags
    utils.backend_flags()
    torch.cuda.empty_cache()

    accum_steps = config.gradient_accumulation_steps
    stats = _make_stats()
    device = torch.device(f"cuda:{local_rank}")

    remaining_steps = config.max_steps - start_step
    train_bar = tqdm.trange(
        remaining_steps,
        initial=start_step,
        total=config.max_steps,
        colour="green",
        disable=config.rank > 0,
    )

    step = start_step
    optimizer.zero_grad()

    for _ in train_bar:
        start_time = time.time()

        # Gradient accumulation: accumulate over multiple micro-batches
        for micro_step in range(accum_steps):
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = iter(torch.utils.data.DataLoader(
                    train_dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers, pin_memory=True,
                ))
                batch = next(train_loader)

            x = batch["tokens"].to(device)
            losses = compute_omni_loss(fsdp_model, x=x, config=config)

            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = losses["total_loss"] / accum_steps
            scaled_loss.backward()

        # Step optimizer after all micro-batches
        grad_norm = fsdp_model.clip_grad_norm_(config.grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1

        # Stats (use unscaled loss values for logging)
        elapsed = time.time() - start_time
        stats["count"] += 1
        for k in ["temporal_loss", "depth_loss", "total_loss"]:
            stats[k] += losses[k].item()
        stats["grad_norm"] += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        for k in ["text_acc", "audio_acc", "depth_acc", "depth_q0_acc", "depth_q1_7_acc"]:
            stats[k] += losses[k].item()

        count = stats["count"]
        tl = stats["temporal_loss"] / count
        dl = stats["depth_loss"] / count
        train_bar.set_description(f"T:{tl:.2f} D:{dl:.2f} {elapsed*1000:.0f}ms/step")

        # Log metrics
        if step % config.log_every == 0 and config.rank == 0:
            metrics = {k: stats[k] / count for k in stats if k != "count"}
            metrics["current_lr"] = scheduler.get_last_lr()[0]
            metrics["step_time_ms"] = elapsed * 1000
            metrics["samples_per_sec"] = (config.batch_size * config.world_size * accum_steps) / elapsed
            wandb.log(metrics, step=step)
            _log_gpu_memory(step)
            stats = _make_stats()

        # Periodic cache clearing to reduce fragmentation over long runs
        if step % (config.log_every * 10) == 0:
            torch.cuda.empty_cache()

        # Save checkpoint
        if config.save_every and step % config.save_every == 0:
            save_fsdp_checkpoint(fsdp_model, optimizer, scheduler, step, config)

        if step >= config.max_steps:
            if config.rank == 0:
                utils.pprint("\nMax steps reached.", color="bold red")
            break

        dist.barrier()

    # Final save
    save_fsdp_checkpoint(fsdp_model, optimizer, scheduler, step, config)

    if config.rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Default config for 2×H100.
    # EVERY value here is overridable via env var (pydantic-settings priority).
    # To scale to different hardware, just set env vars — no code changes needed.
    #
    # Examples:
    #   BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2  → 8×8×2 = 128 effective batch (8×H100)
    #   BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8  → 1×1×8 = 8 effective batch (1×A10G)
    #   MAX_STEPS=500000 SAVE_EVERY=2000            → long training, less frequent saves
    #   DUAL_STREAM=true LANGUAGE_ADAPTERS=true      → enable advanced features
    config = VoxtralTrainConfig(
        name=f"omnivoxtral-full-{datetime.datetime.now().strftime('%d%m%y')}",
        seed=42,
        mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
        mistral_kwargs={},
        voxtral_tokenizer_config=VoxtralTokenizerConfig(
            mimi_path="kyutai/mimi",
            whisper_path="openai/whisper-large-v3",
            text_hz=5,
            mimi_num_quantizers=8,
            text_vocab_size=65536,
            language="hi",
            sp_tokenizer_path="./data/tokenizer/omnivoxtral_sp.model",
        ),
        new_vocab_size=81920,
        loss_weights=[100, 10, 1],
        lora_rank=None,
        prune_layers=None,
        codec_hz=55,
        # EMA disabled for FSDP (complex to maintain across shards)
        ema_gamma=16,
        ema_every=999999,
        # Dataset
        data_path="./data/tokens_fleurs",
        fake=False,
        overfit=None,
        # Batch: per GPU. Effective = batch_size × world_size × gradient_accumulation_steps.
        batch_size=4,
        num_workers=4,
        test_size=4,
        # Speed
        compile=False,
        gradient_checkpointing=True,
        # Optimizer
        lr=2e-5,
        weight_decay=0.1,
        lr_eps=1e-9,
        lr_betas=(0.9, 0.95),
        grad_norm=1.0,
        warmup_steps=300,
        max_steps=100_000,
        gradient_accumulation_steps=1,
        # Logging & safety
        log_every=50,
        test_every=None,
        generate_kwargs={},
        watch_every=None,
        ckpt_path=None,
        save_every=500,
        keep_checkpoints=5,
        push_every=None,
        wandb_project_name="omnivoxtral",
        # Architecture (override via env: DUAL_STREAM=true LANGUAGE_ADAPTERS=true)
        dual_stream=False,
        language_adapters=False,
    )

    train_full(config)
