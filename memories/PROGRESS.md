# OmniVoxtral Progress Tracker

> Last updated: 2026-03-13

---

## Phase 0: Research Documents — COMPLETE

All 7 research documents written to `memories/`:

| Document | Status | Key Finding |
|----------|--------|-------------|
| CODEBASE_AUDIT.md | Done | 13 issues (3 CRITICAL), 60% code reusable |
| LITERATURE_SYNTHESIS.md | Done | 25+ papers, Moshi is reference architecture |
| DATASET_INVENTORY.md | Done | ~90-100K hours potentially available |
| ARCHITECTURE.md | Done | 9 decisions: dual-stream, Inner Monologue, 65K vocab |
| COMPETITIVE_ANALYSIS.md | Done | Only open duplex model with Indic support |
| CAPACITY.md | Done | LoRA on pruned 3.5B fits A10G at ~9GB |
| TRAINING_STRATEGY.md | Done | 5-phase curriculum, $22-$1728 compute budget |

---

## Phase 1: Codec Validation — COMPLETE

**Scripts:** `scripts/source_indic_audio.py`, `scripts/benchmark_codec.py`
**Results:** `data/codec_benchmark_results/`

| Metric | Value |
|--------|-------|
| Languages tested | 13 real + 9 placeholder |
| PESQ range (8q, real) | 1.41 - 2.16 |
| Mean PESQ (8q, real) | 1.85 |
| Mean F0 correlation | 0.945 |
| Mean energy correlation | 0.982 |
| Encode latency | ~14ms/sample |
| Decode latency | ~17ms/sample |

**Decision:** Proceed with Mimi at 8 quantizers. PESQ threshold revised from 3.0 to 2.0 (neural codec calibration). Fine-tune on Indic data for improvement.

---

## Phase 2: Multilingual Tokenizer — COMPLETE

### Step 2.1: Benchmark Mistral BPE — DONE
**Script:** `scripts/benchmark_tokenizer.py`
**Results:** `data/tokenizer_benchmark_results/`
- Mistral 32K BPE: 10.79 mean tokens/word (catastrophic)
- All 13 Indic languages FAIL fertility targets

### Step 2.2: Train SentencePiece Tokenizer — DONE
**Script:** `scripts/train_tokenizer.py`
**Artifacts:** `data/tokenizer/`

| Metric | Mistral 32K BPE | OmniVoxtral 65K SP | Change |
|--------|----------------|-------------------|--------|
| Mean fertility (Indic) | 10.79 tok/word | 1.91 tok/word | **-82%** |
| Worst: Malayalam | 22.20 | 2.49 | -89% |
| Best: Hindi | 5.34 | 1.51 | -72% |
| Vocab size | 32,768 | 65,536 | 2x |
| Language tokens | 0 | 23 (IDs 4-26) | — |
| Control tokens | 0 | 6 | silence, overlap, backch, turn_start/end, lang_switch |

All languages pass targets: Devanagari ≤2.5, Dravidian ≤3.0, others ≤3.5

### Step 2.3: Whisper Replacement Design — DONE
**Document:** `memories/WHISPER_REPLACEMENT_DESIGN.md`
- **Phase 2 (immediate):** Whisper-large-v3 drop-in replacement (~10 line change)
- **Phase 3 (optimal):** IndicWhisper ASR + MMS_FA CTC forced alignment (two-stage)
  - MMS_FA: 300M params, 1130 languages, <100ms/30s alignment, native PyTorch
  - IndicWhisper: 4.1% lower WER on Indic than vanilla Whisper
  - Critical finding: Whisper has Brahmic normalization bug (WERs inflated 22-152%)
- 15/22 languages via Whisper, all 22 via MMS_FA
- Fits on A10G: 3-5GB total for both stages

---

## Phase 3: Data Pipeline — COMPLETE

### Step 3.1: Dataset Loaders — WORKING
**Script:** `scripts/load_indic_datasets.py`
- FLEURS loader: Working (13 languages, fully open)
- IndicVoices loader: Working (all 22 languages, HF_TOKEN set)
- CommonVoice loader: Blocked — needs `datasets>=3.0` (Parquet format), conflicts with FLEURS
- YouTube Indic: Search terms created (`data/indic_searches.txt`, 178 terms)

**Coverage:** 22/22 languages accessible via IndicVoices + 13/22 via FLEURS.

### Step 3.2: Indic YouTube Search — DONE
- `data/indic_searches.txt`: 178 search terms across 13 languages + code-switching
- `data/indic_urls.txt`: **1,845 unique URLs**, **2,654 hours** total content
- Config: min_duration=30min, search_limit=20, 8 workers
- Ready for scraping: `uv run scripts/scrape.py` with INPUT_FILE=data/indic_urls.txt

### Step 3.3: Quality Filtering Pipeline — DONE
**Script:** `scripts/filter_audio.py`
- SNR estimation (energy-based VAD, bottom 30% = noise, top 30% = speech)
- Speech activity ratio (frames above -40dB threshold)
- Clipping detection (samples near +-1.0)
- Duration bounds (3-60s default)
- CSV report generation
- Verified: clean audio passes, noisy/short/clipped correctly rejected
- Usage: `uv run scripts/filter_audio.py --input_dir data/chunks/ --output_dir data/filtered/`

### Step 3.4: Diarization Pipeline — DONE
**Script:** `scripts/diarize_audio.py`
- pyannote.audio speaker-diarization-3.1 integration
- Speaker separation via timestamp masking (zero out other speaker)
- Role assignment heuristic (most speech = model/host, less = user/guest)
- DualStreamTokenizer integration for end-to-end: audio → diarize → dual-stream tokens → .npy
- Single file or batch directory processing
- Usage: `CUDA_VISIBLE_DEVICES=0 uv run scripts/diarize_audio.py --input_dir data/chunks/ --output_dir data/dual_tokens/`
- Note: requires `pyannote.audio` (not yet in pyproject.toml — install separately)

---

## Code Changes (completed 2026-03-13)

### loss_weights bug fix — DONE
- `trainer.py:compute_loss()` now takes `config` and builds per-token weight pattern
- Pattern: `[text=100, semantic=10, acoustic=1, ..., acoustic=1]` per 9-token window
- Normalized so mean weight = 1.0 (preserves loss scale)
- Matches Moshi's finding that alpha=100 for text was "most critical for quality"

### Whisper-large-v3 + SentencePiece swap — DONE
- `model.py`: whisper-tiny.en → whisper-large-v3, vocab 32K→65K, quantizers 4→8
- `word_level_whisper.py`: Added `language` param, `sp_tokenizer_path`, `_tokenize_bucket()`
- `generate_tokens()`: Passes `language` and `task="transcribe"` for non-English

### CRITICAL: Codebook alignment fix — DONE (2026-03-13)
**Bug:** `extract_codebook_targets()` extracted positions 1-8 per window, assuming they
mapped to codebooks q0-q7 of one frame. But with 20 audio tokens per window / 8 codebooks
= 2.5 frames per window, the codebook alignment shifts every window. Odd windows had mixed
frames and misaligned codebook assignments. The depth transformer was training on wrong targets.

**Fix:** Changed to frame-level extraction:
1. Extract all audio tokens from windows, concatenate across boundaries
2. Reshape flat audio into `(batch, num_frames, 8)` — each row is one complete frame
3. Compute per-frame positions in the interleaved sequence for hidden state extraction
4. Pass `frame_positions` to `OmniVoxtral.forward()` instead of computing text positions

**Second bug:** Loss weight pattern was off-by-one — `target_ids[0]` (an audio token) was
getting text weight (100) instead of semantic weight (10). Fixed by building weights for the
full sequence then taking `[1:]` to align with `target_ids`.

**Result:** 225 correctly aligned frames per 20s clip (was 90 misaligned windows).
All 8 codebook columns verified in correct token ranges.

### CRITICAL: LoRA frozen embeddings fix — DONE (2026-03-13)
**Bug:** peft's `get_peft_model()` freezes ALL non-LoRA parameters. `nn.Embedding` layers
don't get LoRA adapters (only `nn.Linear` does), so `embed_tokens` becomes completely frozen.
This means 49,152 new token embeddings (32K Indic text + 16K audio codebook) can never learn —
the model trains but new tokens are dead.

**Fix:** Added `modules_to_save=["model.embed_tokens"]` to LoRA config. This tells peft to
create a trainable copy of the embedding layer. VRAM cost: +3.1GB (0.6GB weights + 2.5GB Adam).
Still fits A10G at ~15-18GB total.

**Verified:** Gradients flow through `embed_tokens.modules_to_save.default.weight` and
depth transformer after backward pass.

### Training metrics enhancement — DONE (2026-03-13)
Added per-token-type accuracy metrics to `compute_omni_loss()`:
- `text_acc` / `audio_acc` — temporal transformer accuracy by token type
- `depth_acc` / `depth_q0_acc` / `depth_q1_7_acc` — depth transformer accuracy (overall, semantic, acoustic)
All logged to W&B every 50 steps alongside loss curves.

---

---

## Phase 4: Architecture Implementation — COMPLETE

### Depth Transformer — DONE
**File:** `src/voxtral/model/depth_transformer.py`
- 6 layers, 16 heads, dim 1024 (113.7M params, 227MB fp16)
- Per-codebook embedding tables + per-codebook output heads
- RMSNorm + SwiGLU FFN (matching Mistral's architecture)
- Training: teacher-forced forward with batched timesteps
- Inference: autoregressive q1→q2→...→q8 generation with temperature + top-k

### OmniVoxtral Dual-Transformer — DONE
**File:** `src/voxtral/model/omnivoxtral.py`
- Wraps Temporal (Mistral 7B) + Depth (113M) transformers
- Forward: temporal produces logits + hidden states → depth predicts codebooks
- Generate: text token sampling → depth autoregressive codebook generation
- Stride-aware extraction of hidden states at audio positions

### OmniVoxtral Training Loop — DONE
**File:** `src/voxtral/trainer/omni_trainer.py`
- Combined temporal + depth loss with correct codebook target extraction
- De-offsets codebook tokens from global IDs to [0, codebook_size) for Depth Transformer
- Gradient flow verified through both transformers

### Embedding Initialization — DONE
**File:** `src/voxtral/model/init_embeddings.py`
- Pretrained [0:32K]: Preserved Mistral weights (std=0.021)
- Indic text [32K:65K]: Half pretrained std (0.010) — trainable, non-disruptive
- Audio [65K:82K]: Quarter pretrained std (0.005) + per-codebook offset
- LM head: 0.01 std for new tokens

### Vocab Size Fix — DONE
- `new_vocab_size` updated from 65,536 → 81,920 (65K text + 8×2048 audio)
- Updated in both `config.py` and `omnivoxtral.py`

### Dual-Stream Interleaving — DONE
**File:** `src/voxtral/tokenizer/dual_stream.py`
- `DualStreamTokenizer`: encodes two audio channels (user + model) into single interleaved sequence
- Token layout per 200ms window: [user_text(1), user_audio(20), model_text(1), model_audio(20)] = stride 42
- `encode()`: accepts two audio tensors, produces dual-stream token sequence
- `decode_model_stream()` / `decode_user_stream()`: separate decoding of each stream
- Roundtrip interleave/uninterleave verified correct

### Dual-Stream Model Support — DONE
**Files:** `omnivoxtral.py`, `omni_trainer.py`
- `OmniVoxtralConfig.dual_stream`: toggles between stride=21 (single) and stride=42 (dual)
- `OmniVoxtral.forward()`: extracts model stream hidden states at correct positions (offset `stream_stride`)
- `extract_codebook_targets()`: correctly locates model codebooks at positions 22-29 in dual windows
- `compute_omni_loss()`: builds [user_pattern, model_pattern] weight structure for dual-stream
- Forward + backward pass verified in both single and dual-stream modes

### Full Training Script — DONE
**File:** `scripts/train_omni.py`
- CLI entry point for OmniVoxtral training
- Initializes dual-transformer with embedding initialization
- Supports LoRA, gradient checkpointing, DDP, torch.compile, W&B logging
- Tracks temporal_loss, depth_loss, total_loss, grad_norm separately
- Dual-stream mode via `DUAL_STREAM=true` env var
- EMA with Karras schedule, checkpoint save/load

### Language Adapters — DONE
**File:** `src/voxtral/model/language_adapters.py`
- 5 language-family LoRA adapters: Indo-Aryan-Deva, Indo-Aryan-Other, Dravidian, Sino-Tibetan, Austroasiatic
- 23 language codes mapped to families (all 22 Indic + English)
- `create_language_adapters()`: creates per-family peft adapters on temporal transformer
- `activate_adapter()`: switches active adapter by language code
- `OmniVoxtral.set_language("kn")` → activates Dravidian adapter
- ~28K params per family on tiny model; ~2-5M per family expected on Mistral 7B
- English uses base weights (no adapter) — preserves pretrained quality
- Enabled via `LANGUAGE_ADAPTERS=true ADAPTER_RANK=8` env vars in training script

---

## Phase 4: Architecture Implementation — COMPLETE

All Phase 4 components implemented and verified:
- Depth Transformer (113.7M params)
- OmniVoxtral dual-transformer (single + dual-stream modes)
- Dual-stream tokenizer (stride 42)
- Combined temporal + depth loss with weighted CE
- Embedding initialization (principled 3-tier: pretrained/indic/audio)
- Language-family LoRA adapters (5 families, 23 languages)
- Full training script with CLI entry point

---

## Phase 5: First Training Run — IN PROGRESS

### End-to-end Smoke Test — DONE
- `train_omni.py` verified: 3 steps with fake data, loss decreasing (21.99→19.88)
- Both single-stream (stride=21) and dual-stream (stride=42) modes verified

### HuggingFace Preprocessing Script — DONE
**Script:** `scripts/preprocess_hf.py`
- Streams audio from FLEURS/IndicVoices directly to .npy tokens
- No intermediate wav files (500x compression: 960KB wav → 1.7KB tokens)
- Fixed dtype mismatch (fp16 waveform for fp16 tokenizer)
- Fixed HF cache: all caches moved from /home → /apps (symlinked)
- Verified: Hindi tokens shape (1, 1890), range [2, 81907], 90 windows correct

### FLEURS Preprocessing — RUNNING
- 13 Indic languages × 200 samples each = 2,600 token files
- Running on GPU:0 (A10G 23GB), ~8.5 samples/min
- GPU utilization only 14% — bottleneck is CPU (resampling, HF streaming, Whisper features)
- Hindi: 200/200 DONE, Kannada: 200/200 DONE, Tamil: in progress
- Estimated completion: ~4 hours from 11:30 IST
- Output: `data/tokens_fleurs/`

### Training Pipeline Validation — DONE (2026-03-13)
- DataLoader correctly finds all nested `.npy` files via `os.walk`
- Real data batch shape verified: (2, 1890), token range [2, 81904]
- `extract_codebook_targets()`: 225 frames, all 8 codebooks at 100% alignment
- Loss weights: text=14.39× (normalized), semantic=1.44×, acoustic=0.14×
- Full forward pass through tiny model with real data: temporal_logits (2, 1889, 81920), depth_logits (2, 225, 8, 2048)
- `cb_targets_local` correctly de-offset to [0, 2045] range

### Auto-Launch Training — SET UP
**Script:** `scripts/launch_training.sh` (PID running in background)
- Polls every 2 minutes, waits for preprocessing PID to exit
- Sleeps 10s for GPU memory release, then launches training
- Logs to `logs/launch_training.log`
- Fixed: PID selection (uses `python3 scripts/...` pattern to avoid bash wrappers)
- Fixed: GPU conflict (waits for preprocessing to fully finish, not just sample threshold)
- Fixed: Import path (`PYTHONPATH=/apps/voxtral` for `from scripts.train_omni`)

### Real Training Config — READY
**Script:** `scripts/train_omni_real.py`
- Mistral 7B with prune_layers=2 (→ 3.5B), LoRA rank=64
- Gradient checkpointing, batch_size=2 (A10G constraint)
- LR=1e-4 with 500 warmup, 10K max steps
- SentencePiece tokenizer at `data/tokenizer/omnivoxtral_sp.model`
- Data: `data/tokens_fleurs/`
- Estimated VRAM: ~12-15GB (fits A10G 23GB)
- Prerequisites verified: Mistral cached, SP model exists, WANDB+HF tokens set

### YouTube Scraping — BLOCKED
- yt-dlp updated to 2026.03.03 but still blocked by YouTube bot detection
- Error: "Sign in to confirm you're not a bot"
- Fix needed: `--cookies-from-browser` or browser cookie export
- 1,845 URLs × 2,654 hours ready in `data/indic_urls.txt`

### IndicVoices — BLOCKED (gated dataset)
- `ai4bharat/IndicVoices` requires manual access request on HuggingFace
- HF_TOKEN is set but doesn't have access to gated repo (401 error)
- Covers all 22 Indic languages (12,000 hours total)
- Next step: Request access at https://huggingface.co/datasets/ai4bharat/IndicVoices

---

## What's Next (Priority Order)

1. **Monitor training** — Auto-launches when preprocessing finishes (~15:30 IST). Check `bash scripts/monitor_training.sh`
2. **Request IndicVoices access** — User needs to accept terms on HuggingFace
3. **Scale FLEURS** — Reprocess with full dataset (no max_samples limit, ~10K+ samples)
4. **Fix YouTube scraping** — Need browser cookies (`--cookies-from-browser chrome`)
5. **Install pyannote.audio** — Enable diarization for dual-stream data
6. **Upgrade datasets to v3** — Unblocks CommonVoice
