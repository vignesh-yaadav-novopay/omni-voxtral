# Voxtral Working Memory

This file serves as persistent working memory across Claude Code sessions. Update at session end.

## Project Identity
- **Name**: Voxtral
- **Origin**: A16z × Cerebral Valley × Mistral London Hackathon submission
- **Author**: Vignesh Yadav
- **License**: MIT
- **Stage**: Hackathon MVP / proof of concept (overfitted model demo)

## Key Technical Facts
- Python ≥3.11, managed with `uv` (not pip)
- PyTorch-based, uses HuggingFace transformers for Mistral model definition
- Custom training loop (not HF Trainer — author prefers low-level control)
- Mimi codec from Kyutai (vendored in `src/voxtral/tokenizer/mimi/`)
- Whisper-tiny.en used for timed text tokenization
- Mistral-7B-v0.3 as base model (nilq/mistral-1L-tiny for testing)
- Token vocab: 2^16 = 65536 (original 32768 text + audio tokens)
- Audio: 24kHz sample rate for Mimi, 16kHz for Whisper
- Training: bfloat16, AdamW, linear warmup schedule, gradient checkpointing
- EMA: Karras-style (not simple exponential), updated every N steps
- Checkpoints: last 5 kept, supports HuggingFace URL loading

## Known Bugs / TODOs
- `loss_weights` config exists but is NOT wired into `compute_loss` — critical gap
- `server.py` hardcodes `output.wav` path — will break with concurrent users
- No unit tests exist
- No data validation for .npy token files
- Sample rates (24000, 16000) hardcoded in multiple places

## Config Pattern
All configs inherit from `pydantic_settings.BaseSettings` with env vars taking priority.
Override any parameter by setting its uppercase name as an env var:
```
BATCH_SIZE=16 uv run ./scripts/train.py
```

## Session Log
- **2026-03-12**: Initial memory setup. Created CLAUDE.md, plan.md, memory.md. Full codebase audit completed. No llm-council directory accessible at referenced path.
