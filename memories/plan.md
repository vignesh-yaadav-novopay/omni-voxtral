# Voxtral Strategic Plan

## Strategy — How We Get There

### Current State (Hackathon MVP)
- Full pipeline working end-to-end: index → scrape → preprocess → train → serve
- Proof-of-concept model overfitted on few samples
- Single-GPU training only
- 4 Mimi quantizers (trades audio quality for training speed)
- No loss weighting across token types (text vs semantic vs acoustic)

### Phase 1: Foundation (Immediate)
- [ ] Implement weighted loss across token types (text=100x, semantic=10x, acoustic=1x) — config exists but not wired
- [ ] Scale training data: run full indexing + scraping pipeline on a good machine
- [ ] Evaluate audio quality vs number of quantizers tradeoff (4 vs 8)
- [ ] Add proper evaluation metrics for SpeechLM quality

### Phase 2: Quality (Near-term)
- [ ] Multi-GPU training (DDP infrastructure exists, needs testing at scale)
- [ ] Streaming conversation mode (Mistral's sliding window attention + rolling KV cache)
- [ ] Better data filtering: remove low-quality audio, music, non-conversational content
- [ ] Implement proper audio quality metrics (PESQ, STOI, or MOS prediction)

### Phase 3: Scale (Medium-term)
- [ ] Train on full 8k+ hours of podcast data
- [ ] Explore full finetuning vs LoRA vs pruning tradeoffs at scale
- [ ] Community model releases on HuggingFace
- [ ] Real-time bidirectional streaming conversation

---

## PRD — Product Requirements Document

### Problem Statement
Current voice AI systems either:
1. Use ASR→LLM→TTS pipeline: loses prosody, can't handle interruptions, feels robotic
2. GPT-4o: closed-source, over-safety-tuned, inaccessible
3. Moshi: over-engineered architecture, still sounds robotic from synthetic training data

### Target Users
- ML researchers building conversational AI
- Developers wanting natural voice interfaces
- Open-source community building on Mistral ecosystem

### Core Requirements
1. Audio-in, audio-out with no intermediate text bottleneck
2. Preserve prosody, rhythm, tone from input
3. Learn natural conversational dynamics (turn-taking, interruptions) from data
4. Run on modest hardware (single A100 minimum)
5. Fully open-source, buildable from scratch

### Success Metrics
- Audio output naturalness (MOS score > Moshi)
- Training convergence on 8k hours within 48h on 8xA100
- Community adoption (HuggingFace downloads, forks)

---

## HLD — High-Level Design

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VOXTRAL SYSTEM                       │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ Indexer   │──▶│ Scraper  │──▶│  Preprocessor    │   │
│  │ (YouTube) │   │ (yt-dlp) │   │  (GPU Tokenizer) │   │
│  └──────────┘   └──────────┘   └────────┬─────────┘   │
│                                          │              │
│                                    .npy tokens          │
│                                          │              │
│                                 ┌────────▼─────────┐   │
│                                 │     Trainer       │   │
│                                 │  (Mistral + DDP)  │   │
│                                 └────────┬─────────┘   │
│                                          │              │
│                                   model weights         │
│                                          │              │
│                                 ┌────────▼─────────┐   │
│                                 │   Gradio Server   │   │
│                                 │   (Voice I/O)     │   │
│                                 └──────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          MULTIMODAL TOKENIZER                    │   │
│  │  ┌────────────┐  ┌───────────────────────────┐  │   │
│  │  │TimedWhisper│  │    Mimi Audio Codec       │  │   │
│  │  │  (5Hz text)│  │(50Hz × 4 quantizers audio)│  │   │
│  │  └─────┬──────┘  └──────────┬────────────────┘  │   │
│  │        │                     │                   │   │
│  │        └──────┬──────────────┘                   │   │
│  │               │ interleave                       │   │
│  │        ┌──────▼──────┐                           │   │
│  │        │ 55Hz flat   │                           │   │
│  │        │ token stream│                           │   │
│  │        └─────────────┘                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Token Vocabulary Layout
```
[0, 32767]       → Mistral text tokens (from TimedWhisper)
[32768, 65535]   → Audio tokens (4 quantizers × mimi_vocab_size, offset to avoid collision)
```

### Interleaving Pattern (per time step at ~1/5s granularity)
```
[1 text token] [4 audio tokens (q1,q2,q3,q4) × 10 frames] = 41 tokens per window
```

---

## LLD — Low-Level Design

### VoxtralTokenizer (src/voxtral/tokenizer/model.py)
- `encode(waveform, sample_rate)`:
  1. Resample to 16kHz for Whisper, keep 24kHz for Mimi
  2. Run TimedWhisperTokenizer → text tokens at 5Hz
  3. Run Mimi encoder → 4 quantizer streams at 50Hz
  4. Offset each Mimi quantizer's tokens to unique range
  5. Interleave audio quantizers: [q1_t1, q2_t1, q3_t1, q4_t1, q1_t2, ...]
  6. Apply 2-window delay: trim first 2 windows of audio, last 2 of text
  7. Interleave text + audio: [text, audio×factor, text, audio×factor, ...]
- `decode(tokens)`:
  1. Uninterleave text and audio
  2. Discard text tokens
  3. Undo quantizer offsets
  4. Run Mimi decoder → waveform

### Training Loop (src/voxtral/trainer/trainer.py)
- Standard autoregressive cross-entropy on the interleaved token sequence
- EMA model updated via Karras schedule (not simple exponential)
- Generation test: take first half of sequence as prompt, generate second half, decode to audio, log to W&B
- Checkpoint: saves last 5, supports resume from HuggingFace URLs
- DDP: NCCL backend with 1-hour timeout

### Config Priority
Environment variables → init args → dotenv → file secrets (pydantic-settings customized order)

---

## Project Audit — Council Review

### Strengths
1. **Architectural clarity**: The discrete tokenization approach is elegant and well-motivated
2. **Pipeline completeness**: End-to-end from data acquisition to serving
3. **Practical engineering**: No-transcode scraping, GPU preprocessing, checkpoint management
4. **Clean abstractions**: Each stage is independently runnable with its own config

### Issues & Risks
1. **Loss weighting not implemented**: `loss_weights=[100, 10, 1]` exists in config but `compute_loss` uses uniform cross-entropy. This is the single most impactful TODO.
2. **No test suite**: Zero unit/integration tests. Fragile to refactoring.
3. **Whisper alignment quality**: 5Hz bucketing with max_length truncation may lose information for fast speech. No fallback for Whisper failures.
4. **Memory safety**: `preprocessing.py` loads entire audio files into memory — could OOM on long recordings (mitigated by chunking, but not guaranteed).
5. **Hardcoded sample rates**: 24kHz for Mimi and 16kHz for Whisper are scattered across files rather than centralized.
6. **No data validation**: No checks for corrupt .npy files, no manifest/index for the token dataset.
7. **Server hardcodes output path**: `server.py` writes to `output.wav` — breaks with concurrent requests.

### Recommendations (Priority Order)
1. Wire up `loss_weights` in `compute_loss` — differentiating text/semantic/acoustic loss is critical for quality
2. Add basic tests for tokenizer encode/decode roundtrip
3. Centralize sample rate constants
4. Add .npy file validation in dataset loading
5. Fix server concurrent write issue
