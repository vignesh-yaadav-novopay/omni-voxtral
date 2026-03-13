# Training Strategy for OmniVoxtral

> Curriculum, data mixing, loss design, and multi-GPU strategy.
> Date: 2026-03-13 | Based on Moshi's 5-phase approach, adapted for multilingual + constrained compute.

---

## Training Curriculum Overview

Following Moshi's proven multi-phase approach, adapted for multilingual and constrained compute:

```
Phase 1: Text Pre-training (INHERITED — Mistral 7B weights)
    ↓
Phase 2: Multilingual Audio Pre-training (LONGEST — 100K+ steps)
    ↓
Phase 3: Diarized Dual-Stream Post-training (CRITICAL — enables duplex)
    ↓
Phase 4: Duplex Fine-tuning (Fisher + Indic conversations)
    ↓
Phase 5: Instruction Fine-tuning (SHORTEST — alignment)
```

---

## Phase 1: Text Pre-training (Inherited)

**Action:** Start from Mistral 7B v0.3 pretrained weights. No additional text training.

**What we get for free:**
- 32K English vocabulary with strong language modeling
- Instruction following capability (from instruct variant)
- World knowledge encoded in 7B parameters

**What we change:**
- Extend vocabulary from 32K → 65K (resize_token_embeddings)
- Add new embedding weights for Indic text tokens + audio tokens
- Initialize new embeddings randomly (will be trained in Phase 2)
- Apply layer pruning if using A10G (`prune_layers=2`)

**Embedding initialization for new tokens:**
```python
# After resize_token_embeddings(65536):
# Tokens [0, 32767] → inherited Mistral weights
# Tokens [32768, 65535] → random init with same std as existing
existing_std = model.model.embed_tokens.weight[:32768].std()
with torch.no_grad():
    model.model.embed_tokens.weight[32768:].normal_(0, existing_std)
```

---

## Phase 2: Multilingual Audio Pre-training

**Objective:** Teach the model to predict next audio token autoregressively, across all target languages.

**Duration:** 100K-500K steps (dependent on data availability and convergence)

**Data mix:**
| Source | Hours | Percentage | Purpose |
|--------|-------|-----------|---------|
| IndicVoices (all 22 langs) | 12,000 | 35% | Multilingual coverage |
| BhasaAnuvaad | 10,000 (sampled) | 30% | Scale + translation alignment |
| YouTube English (existing) | 5,000 | 15% | Maintain English quality |
| YouTube Indic (new scrape) | 3,000 | 10% | Conversational Indic |
| Common Voice + OpenSLR | 2,000 | 5% | Clean per-language data |
| Code-switching corpora | 500 | 5% | Code-switch exposure |

**Architecture:** Single-stream (current Voxtral architecture, extended for multilingual).
- Reason: Phase 2 is about learning audio representations, not duplex.
- Dual-stream introduced in Phase 3 after audio representations are learned.

**Token format (per sample):**
```
<|lang:kn|> [text_t1, audio_q1_t1, ..., audio_qN_t1, text_t2, audio_q1_t2, ...]
```

**Training config:**
```python
# Phase 2 config overrides
batch_size = 2          # A10G constraint
max_steps = 100_000     # Initial target
lr = 1e-4               # Lower than default (fine-tuning, not training from scratch)
warmup_steps = 5_000
lora_rank = 64          # LoRA for A10G
prune_layers = 2        # Layer pruning for A10G
gradient_checkpointing = True
```

**Checkpointing:** Save every 5K steps. Evaluate on held-out FLEURS samples every 1K steps.

---

## Phase 3: Diarized Dual-Stream Post-training

**Objective:** Transition from single-stream to dual-stream. Model learns to track two speakers simultaneously.

**Prerequisites:**
- Phase 2 converged (audio prediction quality verified)
- Dual-stream tokenizer implemented (rewrite of `tokenizer/model.py`)
- Speaker-diarized data prepared

**Data:**
| Source | Hours | Type |
|--------|-------|------|
| IndicVoices conversational (diarized) | 2,000 | Multi-speaker, Indic |
| YouTube Indic interviews (diarized) | 1,000 | Multi-speaker, Indic |
| YouTube English podcasts (diarized) | 2,000 | Multi-speaker, English |
| Fisher corpus (pre-diarized) | 2,000 | Duplex, English |

**Diarization pipeline:**
1. Apply speaker diarization (pyannote.audio or WhisperX) to multi-speaker audio
2. Separate into user channel + model channel
3. Encode each channel independently with Mimi
4. Interleave into dual-stream token format

**Token format (dual-stream):**
```
Per timestep: [user_text, user_q1...q8, model_text, model_q1...q8]
```

**Training approach:**
- Initialize from Phase 2 checkpoint
- Freeze first 8 layers of Temporal Transformer (preserve learned representations)
- Train remaining layers + new Depth Transformer from scratch
- Gradually unfreeze layers over first 10K steps

**Duration:** 50K-100K steps

---

## Phase 4: Duplex Fine-tuning

**Objective:** Teach natural turn-taking, interruptions, backchannels.

**Data (all must be genuine conversations with overlaps):**
| Source | Hours | Key Feature |
|--------|-------|-------------|
| Fisher corpus | 2,000 | Natural English turn-taking, overlaps |
| IndicVoices conversational | 500 | Indic turn-taking patterns |
| Vaani (if available) | 1,000 | Indian conversational norms |
| YouTube Indian interviews | 500 | Naturalistic Indic duplex |

**What this phase teaches:**
1. When to yield the floor (backchannel signals)
2. When to take the floor (response timing)
3. How to handle interruptions (truncate gracefully)
4. Silence management (comfortable pauses vs. awkward gaps)
5. Backchannel generation ("uh-huh", "hmm", "accha", "sari")

**Training config:**
- Lower learning rate: 5e-5 (fine-tuning, not learning)
- Shorter sequences: focus on turn boundaries
- Higher weight on turn-boundary tokens

**Duration:** 10K-30K steps

---

## Phase 5: Instruction Fine-tuning

**Objective:** Align model behavior — helpfulness, safety, task following.

**Data (can be partially synthetic):**
| Source | Hours | Type |
|--------|-------|------|
| Synthetic instructional (TTS + scripted) | 5,000 | Task-following in 22 languages |
| ShareGPT-style voice conversations | 1,000 | Natural instruction following |
| Safety examples | 500 | Refusal patterns, content boundaries |

**Generation of synthetic data:**
1. Use GPT-4 / Claude to generate text conversations in Indic languages
2. Use IndicTTS or Coqui to synthesize audio
3. Format as duplex conversations (with synthetic backchannels)

**Training config:**
- Lowest learning rate: 2e-5
- Focus on model stream quality (user stream is always ground truth)
- RLHF/DPO possible but not in initial version

**Duration:** 5K-10K steps

---

## Loss Design (fixing the critical bug)

### Current (broken)
```python
# trainer.py:166-175
loss = cross_entropy(logits.view(-1, V), targets.view(-1))
# All token types weighted equally. loss_weights=[100,10,1] IGNORED.
```

### Target: Weighted per-token-type loss

```python
def compute_loss(model, x, config):
    input_ids = x[:, :-1].contiguous()
    target_ids = x[:, 1:].contiguous()

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (B, T, V)

    # Create per-token weight mask based on token type
    text_mask = (target_ids < config.text_vocab_size).float()
    semantic_mask = (
        (target_ids >= config.text_vocab_size) &
        (target_ids < config.text_vocab_size + config.codebook_size)  # First codebook
    ).float()
    acoustic_mask = (
        target_ids >= config.text_vocab_size + config.codebook_size  # Remaining codebooks
    ).float()

    # Weighted loss: text=100, semantic=10, acoustic=1
    weights = (
        text_mask * config.loss_weights[0] +
        semantic_mask * config.loss_weights[1] +
        acoustic_mask * config.loss_weights[2]
    )

    # Per-token cross entropy
    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        target_ids.view(-1),
        reduction='none'
    ).view_as(target_ids)

    # Weighted mean
    loss = (per_token_loss * weights).sum() / weights.sum()
    return loss
```

### Why these weights matter (from Moshi paper)

Moshi uses `alpha=100` for semantic/text tokens and `alpha=1` for acoustic tokens. Their ablation showed:
- Without text token weighting: WebQ accuracy 9%
- With text token weighting (100x): WebQ accuracy 26.6%
- **3x improvement from loss weighting alone**

The intuition: text tokens carry semantic meaning (what to say), acoustic tokens carry signal detail (how it sounds). The model should prioritize understanding what to say over reproducing audio artifacts.

---

## Data Processing Pipeline

### Step 1: Download and organize
```bash
# IndicVoices from HuggingFace
python -c "from datasets import load_dataset; load_dataset('ai4bharat/IndicVoices')"

# Common Voice
python -c "from datasets import load_dataset; load_dataset('mozilla-foundation/common_voice_16_1', 'hi')"

# YouTube Indic (extended pipeline)
CUDA_VISIBLE_DEVICES=0 uv run scripts/index.py  # with Indic search terms
uv run scripts/scrape.py
```

### Step 2: Language detection and labeling
```python
# For YouTube-scraped data without language labels:
# Use MMS language ID model to classify each audio chunk
from transformers import Wav2Vec2ForSequenceClassification
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-4017")
```

### Step 3: Speaker diarization (for Phase 3+)
```python
# Using pyannote.audio for diarization
from pyannote.audio import Pipeline
diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

### Step 4: GPU tokenization
```bash
# Extended preprocessing with language labels
CUDA_VISIBLE_DEVICES=0 uv run scripts/preprocess.py
```

### Step 5: Quality filtering
- Discard samples with SNR < 10 dB
- Discard samples where language ID confidence < 0.8
- Discard samples shorter than 3 seconds (insufficient context)
- Discard samples where Whisper produces no text (silence/noise)

---

## Evaluation Strategy

### Per-Phase Evaluation

**Phase 2 (Audio Pre-training):**
- Audio reconstruction quality: PESQ > 3.0 on held-out FLEURS
- Text prediction accuracy: token-level accuracy > 60%
- Cross-lingual: verify all 22 languages show improvement over random baseline

**Phase 3 (Dual-Stream):**
- Speaker separation quality: can model track which speaker is talking?
- Turn-taking accuracy: does model yield/take floor at correct times?
- Both measured on held-out Fisher + IndicVoices conversational data

**Phase 4 (Duplex):**
- Turn gap distribution: should match Fisher corpus statistics (200-250ms median)
- Backchannel rate: 5-15% of conversation time
- Interruption handling: model truncates within 200ms of user interruption
- MOS (Mean Opinion Score): human evaluation, target > 3.5

**Phase 5 (Instruction):**
- Task completion rate on held-out instructions
- Safety: refusal rate on harmful requests > 95%
- Naturalness MOS > 3.5

### Automated Benchmarks

| Metric | Tool | Target | Phase |
|--------|------|--------|-------|
| PESQ | pesq library | > 3.0 | All |
| WER (ASR quality) | Whisper-large-v3 | < 15% per language | 2+ |
| F0 correlation | librosa.pyin | > 0.9 | All |
| Turn gap (ms) | Custom (diarization) | 200-250ms median | 4 |
| Token accuracy | Custom | > 60% text, > 40% audio | 2+ |
| Latency (e2e) | time.perf_counter | ≤ 200ms | 5 |

---

## Compute Budget

### Minimum Viable (A10G, single GPU)

| Phase | Steps | Time | VRAM | Config |
|-------|-------|------|------|--------|
| Phase 2 | 50K | ~16 hours | 9 GB | LoRA r=64, pruned 3.5B |
| Phase 3 | 30K | ~10 hours | 10 GB | LoRA r=64, pruned 3.5B + Depth |
| Phase 4 | 10K | ~3 hours | 10 GB | LoRA r=64 |
| Phase 5 | 5K | ~2 hours | 10 GB | LoRA r=64 |
| **Total** | **95K** | **~31 hours** | | |

**Cost at $0.70/hr (A10G spot): ~$22**

### Recommended (H100, single GPU)

| Phase | Steps | Time | VRAM | Config |
|-------|-------|------|------|--------|
| Phase 2 | 200K | ~24 hours | 40 GB | Full fine-tune, 7B |
| Phase 3 | 100K | ~12 hours | 45 GB | Full fine-tune + Depth |
| Phase 4 | 30K | ~4 hours | 45 GB | Full fine-tune |
| Phase 5 | 10K | ~1 hour | 45 GB | Full fine-tune |
| **Total** | **340K** | **~41 hours** | | |

**Cost at $4/hr (H100 on-demand): ~$164**

### Full Scale (8× H100, FSDP)

| Phase | Steps | Time | Config |
|-------|-------|------|--------|
| Phase 2 | 500K | ~32 hours | Full fine-tune, FSDP, batch=128 |
| Phase 3 | 200K | ~16 hours | Full fine-tune, FSDP |
| Phase 4 | 50K | ~4 hours | Full fine-tune, FSDP |
| Phase 5 | 20K | ~2 hours | Full fine-tune, FSDP |
| **Total** | **770K** | **~54 hours** | |

**Cost at $32/hr (8× H100): ~$1,728**

---

## Risk Mitigation

### Risk: Mimi codec fails on Indic languages
**Probability:** Medium (30%)
**Impact:** High — blocks all training
**Mitigation:**
1. Run codec benchmark FIRST (Phase 1, no model training needed)
2. If PESQ < 3.0 on >50% languages: fine-tune Mimi on IndicVoices audio
3. Mimi fine-tuning: ~10K steps, ~2 hours on A10G

### Risk: Insufficient conversational Indic data
**Probability:** High (60%)
**Impact:** Medium — degrades duplex quality for Indic
**Mitigation:**
1. YouTube Indic scraping (scale existing pipeline)
2. Transfer duplex behavior from English Fisher to Indic via multilingual backbone
3. Synthetic conversational data (TTS + scripted dialogues)

### Risk: A10G too small for dual-stream training
**Probability:** Low (10% — math shows it fits with LoRA)
**Impact:** Medium — need cloud GPU
**Mitigation:**
1. Reduce sequence length (2048 instead of 4096)
2. Reduce batch size to 1
3. Use CPU offloading (DeepSpeed ZeRO-3) as last resort
4. Rent single H100 (~$4/hr) for training phases

### Risk: Code-switching quality is unacceptable
**Probability:** High (50%)
**Impact:** Low for MVP — code-switching is P1, not P0
**Mitigation:**
1. Gather more code-switching data from YouTube
2. Data augmentation: splice monolingual segments
3. Explicit `<|lang_switch|>` token training
4. Accept degraded code-switching in v1, improve in v2
