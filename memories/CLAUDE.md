# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision — North Star

**Democratize SpeechLMs.** Make end-to-end speech language models as open, simple, and natural as text LLMs — no information bottleneck, no synthetic-sounding output, no closed-source gatekeeping.

## Mission — What We Do

Convert Mistral into an end-to-end SpeechLM (audio-in, audio-out) that preserves prosody, rhythm, tone, and learns natural conversational behaviors (interruptions, silence) directly from data. Unlike GPT-4o (closed) or Moshi (over-engineered), Voxtral is open and simple.

## Values — How We Work

- **First principles over convention**: Question every architectural decision. We chose discrete tokenization over continuous embeddings because it enables bidirectional modality (input AND output) without architectural hacks.
- **Real data over synthetic data**: Podcast data over audiobooks/TTS. Synthetic data makes SpeechLMs sound robotic — defeating their entire purpose.
- **Simplicity over complexity**: A single flat interleaved token sequence. No multi-stream decoders, no separate encoders/decoders. Just extend the vocabulary and finetune.
- **Modest compute, maximum impact**: Everything runs on a single GPU. Layer pruning (drop every Nth layer) over expensive distillation. Practical over theoretical.

## Build & Run

Package manager: `uv` (not pip, not conda)

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run full pipeline (needs A100+ GPU)
uv run ./scripts/everything.py

# Run individual stages
uv run ./scripts/index.py       # 1. Index YouTube podcast URLs
uv run ./scripts/scrape.py      # 2. Download & chunk audio (yt-dlp + ffmpeg)
uv run ./scripts/preprocess.py  # 3. Tokenize audio on GPU → .npy files
uv run ./scripts/train.py       # 4. Finetune Mistral on multimodal tokens
uv run ./scripts/serve.py       # 5. Gradio voice UI

# Training variants
uv run ./scripts/train_overfit.py   # Overfit on few samples (proof of concept)
uv run ./scripts/train_test.py      # Quick test with fake data
```

## Environment

Required in `.env`:
- `HF_TOKEN` — Hugging Face access token (for model downloads)
- `WANDB_API_KEY` — Weights & Biases logging

## Architecture

### The Big Idea: Discrete Multimodal Tokenization

Instead of continuous embeddings, Voxtral extends Mistral's vocabulary to 2^16 tokens and interleaves three token types in a single flat sequence at 55Hz:

```
[text_tok, audio_q1, audio_q2, audio_q3, audio_q4, text_tok, audio_q1, ...]
```

- **Text tokens** (5Hz): Mistral's original vocab (0–32767), produced by a modified Whisper that uses cross-attention weights to timestamp individual words
- **Audio tokens** (50Hz, 4 quantizers): Mimi neural audio codec tokens, offset per quantizer to avoid collisions (token = mimi_token + quantizer_idx * mimi_vocab_size + text_vocab_size)
- **2-window delay**: Text tokens lead audio by 2 time windows to ensure correct causal ordering despite imperfect Whisper alignment

### Source Structure

```
src/voxtral/
├── tokenizer/
│   ├── model.py              # VoxtralTokenizer — the core multimodal tokenizer
│   │                           (interleave/uninterleave, encode/decode)
│   ├── word_level_whisper.py  # TimedWhisperTokenizer — word-level timestamps via
│   │                           Whisper cross-attention, bucketed at configurable Hz
│   └── mimi/                  # Kyutai's Mimi neural audio codec (vendored)
│       ├── models/            # compression.py (MimiModel), lm.py, loaders.py
│       ├── modules/           # conv, seanet, transformer, rope, streaming
│       └── quantization/      # RVQ (residual vector quantization)
├── data/
│   ├── indexing.py            # YouTube search → URL list (multithreaded)
│   ├── scraping.py            # yt-dlp download + ffmpeg segment (no transcode)
│   └── preprocessing.py       # Batch GPU tokenization → .npy files
├── trainer/
│   ├── config.py              # VoxtralTrainConfig (pydantic-settings, env priority)
│   ├── trainer.py             # Custom training loop (not HF Trainer)
│   ├── data.py                # VoxtralDataset (IterableDataset over .npy files)
│   ├── test.py                # Generation test — decode samples, log audio to W&B
│   └── utils.py               # DDP helpers, EMA (Karras schedule), checkpointing
└── server.py                  # Gradio UI for voice interaction
```

### Key Design Decisions

1. **Config via pydantic-settings**: All configs use `BaseSettings` with env vars taking priority over init args. Override any parameter via environment variable.
2. **Custom trainer, not HF Trainer**: Direct PyTorch training loop with DDP support, EMA (Karras schedule), gradient checkpointing, torch.compile, and W&B integration.
3. **Layer pruning**: Drop every Nth layer (`prune_layers=2` drops half) as a simple alternative to LoRA for fitting 7B on single GPU. Both options available.
4. **No transcoding in scraping**: ffmpeg `segment` mode copies audio without re-encoding, achieving ~8000x realtime download rate.
5. **Hashed filenames**: Audio chunks and token files use MD5-based UUIDs with 2-char subdirectories for filesystem-friendly storage.

### Data Flow

```
YouTube searches → urls.txt → yt-dlp chunks (data/chunks/) → GPU tokenization → .npy tokens (data/tokens/) → training
```

## Memory Management

- See `memories/memory.md` for session-persistent context and working memory
- See `memories/plan.md` for strategic planning documents (PRD, HLD, LLD)
- Update memory.md at session end to preserve insights for future sessions

## Skills & Hooks

Available superpowers skills for this project:
- `superpowers:brainstorming` — Before any creative/architectural work
- `superpowers:writing-plans` — For multi-step implementation planning
- `superpowers:executing-plans` — For plan execution with review checkpoints
- `superpowers:test-driven-development` — Before implementing features
- `superpowers:systematic-debugging` — For any bug/failure investigation
- `superpowers:verification-before-completion` — Before claiming work is done
- `superpowers:requesting-code-review` — After completing major features
- `feature-dev:feature-dev` — Guided feature development
