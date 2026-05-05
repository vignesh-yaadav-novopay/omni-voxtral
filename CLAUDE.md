# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**OmniVoxtral** (formerly Voxtral) — an end-to-end multilingual speech language model built by extending Mistral with a Mimi-codec audio vocabulary plus a Depth Transformer for per-codebook prediction. Tier-1 target: 22 Indic languages + English. The repo originated as a hackathon SpeechLM (single-stream, single-GPU) and is mid-evolution into a dual-stream duplex architecture.

Companion docs (read these for deeper context):
- `memories/CLAUDE.md` — vision, mission, values, full architecture rationale
- `memories/ARCHITECTURE.md`, `memories/TRAINING_STRATEGY.md`, `memories/CAPACITY.md` — design decisions
- `memories/CODEBASE_AUDIT.md` — known issues; `memories/PROGRESS.md` — phase status
- `memories/plan.md` and `memories/memory.md` — strategy + rolling working memory

## Build & Run

Package manager is `uv` (not pip, not conda). Entrypoints expect `uv run`.

```sh
# Full pipeline (A100+ class GPU):
uv run ./scripts/everything.py

# Stages:
uv run ./scripts/index.py        # YouTube search → URL list
uv run ./scripts/scrape.py       # yt-dlp + ffmpeg segment (no transcode)
uv run ./scripts/preprocess.py   # GPU tokenize → .npy under data/tokens/
uv run ./scripts/train.py        # original Voxtral trainer (Mistral + Mimi)
uv run ./scripts/serve.py        # Gradio voice UI

# OmniVoxtral (current architecture — temporal Mistral + depth transformer):
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_omni.py
DUAL_STREAM=true uv run scripts/train_omni.py     # requires diarized data
LANGUAGE_ADAPTERS=true ADAPTER_RANK=8 uv run scripts/train_omni.py
FAKE=true uv run scripts/train_omni.py            # smoke test, fake tokens

# Multi-GPU (DDP):
torchrun --nproc_per_node=4 scripts/train_omni.py

# Indic data tooling:
uv run scripts/load_indic_datasets.py             # FLEURS + IndicVoices loader
uv run scripts/preprocess_hf.py                   # HF dataset → token .npy
uv run scripts/diarize_audio.py                   # pyannote → dual-stream
uv run scripts/filter_audio.py                    # SNR/clipping/duration filter
uv run scripts/train_tokenizer.py                 # SentencePiece Unigram 65K
uv run scripts/benchmark_codec.py                 # Mimi PESQ/F0 per language
uv run scripts/eval_omni.py --ckpt_path <path>    # Whisper WER eval
uv run scripts/generate.py --ckpt_path <path>     # Audio generation
```

`.env` must define `HF_TOKEN` and `WANDB_API_KEY`. Secrets are read by `python-dotenv`; never commit `.env`.

## Configuration

All runtime configuration uses `pydantic-settings` with **environment-variable priority** (see `BaseConfig.settings_customise_sources` in `src/voxtral/trainer/config.py`). Override anything by exporting env vars — no CLI flags needed:

```sh
BATCH_SIZE=4 LR=1e-4 MAX_STEPS=20000 DEPTH_MASK_RATE=0.15 \
  DEPTH_FOCAL_GAMMA=2.0 DEPTH_Q_DROPOUT=0.3 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_omni.py
```

Key knobs in `VoxtralTrainConfig` (`src/voxtral/trainer/config.py`):
- `mistral_pretrained_path` — `nilq/mistral-1L-tiny` (default test) or `mistralai/Mistral-7B-v0.3`
- `prune_layers` — drop every Nth layer (cheap "distillation" — `2` halves the depth)
- `lora_rank`, `use_dora` — PEFT options on the temporal transformer
- `loss_weights = [text, semantic, acoustic]` — validated as `[100, 1, 1]` (Moshi §4.4)
- `depth_focal_gamma`, `depth_mask_rate`, `depth_q_dropout` — depth-loss regularizers
- `dual_stream`, `language_adapters`, `adapter_rank` — architecture toggles

## Architecture

### Token layout

The vocabulary is extended to 81,920 tokens: `text_vocab_size (65,536) + num_codebooks (8) × codebook_size (2,048)`. Mimi codec tokens are offset per quantizer to avoid collision with text. Sequences interleave text and audio at a 5 Hz / 50 Hz schedule using a stride of 21 (single-stream) or 42 (dual-stream). Text leads audio by a 2-window delay so that imperfect Whisper word alignment never violates causality.

### Dual-transformer (OmniVoxtral)

```
Temporal Transformer (Mistral, optionally pruned + LoRA / language adapters)
    │  hidden state at each audio position
    ▼
Depth Transformer (6L / 16H / 1024dim, ≈113.7M params)
    │  predicts q1 → q2 → … → q8 autoregressively at that timestep
    ▼
Combined loss: temporal CE (text/semantic) + depth CE (acoustic, optionally focal)
```

Implementation pointers:
- `src/voxtral/model/omnivoxtral.py` — `OmniVoxtral`, `OmniVoxtralConfig`
- `src/voxtral/model/depth_transformer.py` — small AR transformer over codebooks
- `src/voxtral/model/init_embeddings.py` — principled init for the 81,920-token table (text init from Mistral, audio init by k-means over Mimi codebook)
- `src/voxtral/model/language_adapters.py` — 5 per-family LoRA adapters (Devanagari / Dravidian / Indo-Aryan-other / Tibeto-Burman / Latin) covering 23 language tags
- `src/voxtral/trainer/omni_trainer.py` — `compute_omni_loss`, `omni_train_step` (gradient accumulation, focal loss, RVQ dropout, depth masking)
- `src/voxtral/tokenizer/dual_stream.py` — user/model-stream interleaving for duplex training
- `src/voxtral/tokenizer/mimi/` — vendored Kyutai Mimi neural codec
- `src/voxtral/tokenizer/word_level_whisper.py` — `TimedWhisperTokenizer`, word-level timestamps via Whisper cross-attention

### Trainer & data

- Custom PyTorch loop (not HF `Trainer`): DDP, EMA (Karras schedule), `torch.compile`, gradient checkpointing, W&B.
- `VoxtralDataset` (`src/voxtral/trainer/data.py`) is an `IterableDataset` over hashed `.npy` token files under `data/tokens/<2-char-prefix>/<uuid>.npy`.
- Audio scraping deliberately avoids re-encoding: `ffmpeg segment` copies streams directly (~8000× realtime).

### What lives where

```
src/voxtral/
  data/        indexing.py, scraping.py, preprocessing.py
  tokenizer/   model.py (VoxtralTokenizer), dual_stream.py, word_level_whisper.py, mimi/
  model/       omnivoxtral.py, depth_transformer.py, init_embeddings.py, language_adapters.py
  trainer/     config.py, trainer.py (legacy), omni_trainer.py, data.py, test.py, utils.py
  server.py    Gradio UI

scripts/       Entry points — both legacy Voxtral and current OmniVoxtral pipelines
data/          tokens/, tokenizer/, codec_benchmark_results/, indic_urls.txt, …
logs/          per-run W&B run dirs with checkpoint_<step>.pt
memories/      Strategic docs, audits, design records (read-only context)
autoresearch/  Autonomous experimentation harness — see next section
```

## Autoresearch (`autoresearch/`)

Two things share this directory:

1. **Karpathy's nanochat-style autoresearch harness** (vendored upstream): a single-GPU GPT pretraining loop with a fixed 5-minute time budget per experiment and `val_bpb` as the metric.
2. **The OmniVoxtral autonomous research workspace**: file-based memory and a safety watchdog that the `/auto-research` skill uses to run 5-minute training experiments on the actual OmniVoxtral codebase.

### Upstream harness — do not modify the contract

- `prepare.py` — fixed constants (`MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS=40·524288`, vocab 8192), data download from `karpathy/climbmix-400b-shuffle`, `rustbpe` tokenizer, dataloader, and **`evaluate_bpb` — the ground-truth metric, do NOT change**.
- `train.py` — single file the agent edits in upstream mode. Holds the GPT model, MuonAdamW optimizer, and training loop. Architecture / hyperparameters / batch size are all fair game.
- `program.md` — baseline instructions for the upstream loop. Describes the experiment protocol, `results.tsv` schema, and the "NEVER STOP" rule.
- `pyproject.toml` — its own deps (`torch==2.9.1`, `kernels`, `tiktoken`, `rustbpe`, `pyarrow`). Run `uv sync` inside `autoresearch/`. Don't add deps.
- Cache lives at `~/.cache/autoresearch/`. First-time setup: `uv run prepare.py`.

To run a single upstream experiment: `uv run train.py > run.log 2>&1` (always redirect — never tee). Then `grep "^val_bpb:\|^peak_vram_mb:" run.log`. If `grep` is empty, the run crashed; `tail -n 50 run.log` for the trace.

### OmniVoxtral autonomous research

Driven by the `/auto-research` skill (two parallel `/loop` instances — one literature, one experiments — sharing this filesystem). Files:

- `memory/results.tsv` — 400+ logged experiments. Schema: `commit  hypothesis_id  total_loss  temporal_loss  depth_loss  text_acc  depth_acc  memory_gb  status  description`. Tab-separated; commas break parsing. Status is `keep` / `discard` / `crash`.
- `memory/research_log.md` — narrative milestones (e.g. "7B graduation", "focal-loss breakthrough"). Append-only journal in reverse chronological order.
- `memory/research_questions.md` — open questions. Cross out / mark `→ ANSWERED` as cycles resolve them; cite the paper note that closed it.
- `memory/experiment_queue.md` — prioritized backlog. Entries are tagged (e.g. `MPD-001`, `GEN-FIX-1`, `EMBNORM-2`) and reference paper notes.
- `memory/audit_reviews.md` — "audit council" verdicts that gate phase transitions. Some verdicts are blocking (e.g. AF-201..AF-210: "stop tiny-model experiments").
- `memory/theory_findings.md`, `memory/training_insights.md` — distilled lessons (e.g. "WD=0 beats 0.1", "focal γ=2.0 is the only depth regularizer needed").
- `memory/7b_graduation_checklist.md` — checklist used to graduate from the tiny model to the pruned 7B.
- `memory/papers/` and `papers/` — paper notes (40+ files: Moshi, MaskGCT, DiSTAR, ResGen, MMS, DualCodec, ERVQ, OWLS, …). Use **`templates/paper-notes.md`** as the schema (Source / Authors / Read date / Key Techniques / Results / Applicability to OmniVoxtral / Generated Hypotheses / Citations to Follow).
- `RESEARCH_JOURNAL.md` — public-facing narrative of the project journey (separate from `memory/research_log.md`).

Naming convention: paper notes are `<topic>-<YYYY-MM-DD>.md` or `<paper-tag>-<year>.md`. New summaries / action plans usually carry a date suffix and are referenced from `experiment_queue.md` and `research_questions.md`.

### Safety watchdog

OmniVoxtral training will lock the host out via SSH if RAM/GPU pegs. Always launch through the watchdog:

```sh
bash autoresearch/scripts/run_safe.sh "CUDA_VISIBLE_DEVICES=0 uv run scripts/train_omni.py" 300 15 80
#                                       └ command (quoted)                                  │  │   └ GPU % threshold (default 80)
#                                                                                            │  └ monitor interval seconds (default 15)
#                                                                                            └ time budget seconds (default 300)
```

- Acquires `/tmp/omnivoxtral_gpu.lock` (PID-based; stale locks auto-removed). Concurrent runs exit with code 4.
- Runs `monitor.py` pre-flight (nvidia-smi + /proc/meminfo, no psutil) — aborts if GPU/CPU/disk thresholds are breached.
- Launches the child via `setsid` so the whole process group (incl. `torchrun` workers) gets SIGTERM/SIGKILL on cleanup.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid long-run fragmentation OOMs.
- Exit codes: `0` ok, `1` crash, `2` resource breach, `3` time budget, `4` lock conflict.

Other utilities: `scripts/monitor.py` (one-shot JSON safety check) and `scripts/diagnose_embeddings.py` (modality-gap diagnostic on a checkpoint — measures cosine similarity / norm distribution between text and speech embeddings; root-cause tool for the "speech embeddings 4× text norm" issue documented in `memory/papers/embedding-diagnostic-results-2026-03-25.md`).

### Working in autoresearch — rules of thumb

- `prepare.py` is read-only (upstream). `evaluate_bpb` is the contract.
- Always launch training under `run_safe.sh` — never bare. SSH lockout has happened.
- Redirect logs to a file (`> run.log 2>&1`); do not tee or stream into the agent context.
- After each experiment: append to `memory/results.tsv`, then either advance the branch (`keep`) or `git reset` (`discard`). Do not commit `results.tsv`.
- New paper or analysis → drop a dated note under `memory/papers/` or `papers/` (use `templates/paper-notes.md`) and link it from `experiment_queue.md` / `research_questions.md`.
- Audit-council verdicts in `memory/audit_reviews.md` are blocking. Read them before queueing experiments.

## Memory & sessions

- This project uses Claude Code's auto-memory at `~/.claude/projects/-apps-voxtral/memory/` — `MEMORY.md` is the index loaded into every conversation.
- Strategic, durable context lives in `memories/` (PRD, HLD, architecture, audits). Update `memories/memory.md` at session end with new insights.
- Long sessions (`session-*.md` at the repo root) are conversation transcripts kept for reference. They are not part of the build.

## Hardware notes

- Dev: 1× A10G (24 GB) — pruned 7B + attn-only LoRA rank 128 fits with `low_cpu_mem_usage=True`, bf16, GPU ≈78 % (18 GB).
- Multi-GPU: 4× A10G DDP works for the current 4.6 B-pruned config; per-rank val variance is significant (track average, not best rank).
- Target: 2× H100. Scripts have no hard dependency on either; control via `CUDA_VISIBLE_DEVICES` and `torchrun --nproc_per_node`.

## License

MIT (see `LICENSE`).
