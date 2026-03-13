# OmniVoxtral Architecture

> Design decisions with first-principles justification and citations.
> Date: 2026-03-13 | All decisions traceable to research or measured constraints.

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          OmniVoxtral (7B params)        │
                    │                                         │
  User Audio ──►    │  ┌──────────┐     ┌──────────────────┐ │    ──► Model Audio
  (Mimi encode)     │  │ Temporal  │     │ Depth Transformer│ │    (Mimi decode)
                    │  │ Transform │────►│ (6L, per-step)   │─┼──►
  User Text  ──►    │  │ (32L)    │     └──────────────────┘ │    ──► Model Text
  (Inner Mon.)      │  └──────────┘                          │    (Inner Monologue)
                    └─────────────────────────────────────────┘
```

**Two transformers, four streams:**
- **Temporal Transformer** (32 layers, 4096 dim): Models time dependencies across the full context. Initialized from Mistral 7B.
- **Depth Transformer** (6 layers, 1024 dim): Models inter-codebook dependencies at each timestep. Trained from scratch.
- **User stream**: Always-on encoding of incoming user audio (silence = silence tokens, not absence of tokens)
- **Model stream**: Always-on generation of model audio (silence when listening, speech when responding)

---

## Decision 1: Codec → Mimi (validate first, fine-tune if needed)

**Decision:** Use Kyutai's Mimi codec as-is, validate on 22 Indic languages, fine-tune only if needed.

**Rationale:**
1. Mimi's first codebook is distilled from WavLM — captures semantic content, enables Inner Monologue (Moshi paper §3)
2. 80ms latency is duplex-compatible (matches human turn-gap of 200-250ms with room for model inference)
3. Already vendored and working in Voxtral codebase (`tokenizer/mimi/`)
4. No other codec has semantic distillation + low latency + streaming support

**Risk:** Untested on Indic languages. English PESQ is 2.45 (below our 3.0 pass threshold).

**Mitigation:** Phase 1 codec benchmark on all 22 languages before committing. Decision matrix:
| Outcome | Action |
|---------|--------|
| >80% languages PESQ ≥ 3.0 at 8q | Mimi is viable → proceed |
| 50-80% pass | Fine-tune Mimi on IndicVoices audio |
| <50% pass | Evaluate DM-Codec, potentially train new codec |

**Quantizer decision:** Benchmark 4q vs 8q. If quality difference < 0.5 PESQ, keep 4q (2x faster training). Moshi uses 8q.

**Citation:** Moshi paper §3, AudioLM semantic-acoustic hierarchy, SpeechTokenizer semantic distillation

---

## Decision 2: Architecture → Temporal + Depth Transformer

**Decision:** Follow Moshi's dual-transformer architecture.

**Temporal Transformer (main model):**
- 32 layers, 32 heads, dim 4096
- Initialized from Mistral 7B v0.3 pretrained weights
- Processes flattened time-interleaved tokens from both streams
- Context window: 4096 tokens (~37 seconds at 55Hz × 2 streams)

**Depth Transformer (new, trained from scratch):**
- 6 layers, 16 heads, dim 1024
- At each timestep, predicts codebook tokens sequentially (q1→q2→...→q8)
- Input: Temporal Transformer's hidden state at current timestep
- Separate from Temporal to keep per-timestep computation bounded

**Why not monolithic?**
1. Monolithic transformer wastes capacity modeling inter-codebook dependencies globally
2. Depth Transformer constrains codebook prediction to local context (correct inductive bias)
3. Moshi ablations show Temporal+Depth outperforms monolithic on both quality and latency

**Why not Voxtral's current single-stream approach?**
1. Single-stream cannot model overlapping speech (both speakers talking simultaneously)
2. No concept of "my turn" vs "their turn" — fundamental limitation for duplex
3. Cannot generate while listening (no dual-stream)

**Citation:** Moshi paper §4, dGSLM dual-channel architecture

---

## Decision 3: Dual-Stream → Yes (user audio + model audio)

**Decision:** Always-on dual streams. No turn segmentation. No VAD-based switching.

**Implementation:**
- User stream: Mimi-encoded user audio tokens, always flowing (silence = silence tokens)
- Model stream: Generated model audio tokens, always flowing (silence when listening)
- Both streams interleaved into Temporal Transformer input
- At each timestep: `[user_text, user_q1...q8, model_text, model_q1...q8]`

**Why always-on?**
1. Eliminates need for turn detection (VAD is error-prone, adds 100ms+ latency)
2. Model learns turn-taking from data (Fisher corpus), not from rules
3. Backchannels ("uh-huh"), overlaps, and interruptions emerge naturally
4. Moshi demonstrated this works with 200ms practical latency

**What changes from Voxtral:**
- `tokenizer/model.py` must be rewritten for dual-stream interleaving
- Token layout doubles (user + model at each timestep)
- Training data must include speaker diarization labels

**Citation:** Moshi paper §4.3, dGSLM dual-channel, Fisher corpus turn-taking statistics

---

## Decision 4: Inner Monologue → Yes (text tokens as intermediate)

**Decision:** Generate text tokens as prefix to audio tokens at each timestep.

**Evidence:**
- Moshi: WebQ accuracy 9% → 26.6% with Inner Monologue (3x improvement, §4.4)
- Spirit LM: Interleaved text+speech on Llama backbone works for cross-modal tasks
- Text provides semantic grounding that pure audio tokens lack

**Implementation:**
- At each model timestep: first predict text token(s), then predict audio codebook tokens
- Text tokens use multilingual tokenizer (not Mistral's English-only BPE)
- Text can be discarded at inference for pure speech-to-speech (optional)
- For multilingual: text tokens in target language script

**For Indic languages:**
- Text tokens must handle Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia, Gurmukhi, Meitei, Ol Chiki scripts
- Requires multilingual tokenizer (see Decision 5)

**Citation:** Moshi paper §4.4, Spirit LM interleaved speech+text

---

## Decision 5: Multilingual Tokenizer → SentencePiece Unigram 65K vocab

**Decision:** Train new SentencePiece Unigram tokenizer with 65,536 vocab.

**Why Unigram over BPE:**
1. Dravidian languages (Tamil, Telugu, Kannada, Malayalam) are agglutinative — words can be very long
2. Unigram's probabilistic segmentation handles agglutination better than BPE's greedy merges
3. MMS and SeamlessM4T both use character/subword models that handle morphological richness

**Vocab allocation (65,536 = 2^16):**
- ~32,000: Inherit Mistral's existing English tokens (maintain English quality)
- ~20,000: Indic language tokens across 10 scripts (Devanagari, Tamil, etc.)
- ~5,000: Shared multilingual tokens (numerals, punctuation, common borrowings)
- ~8,536: Audio codebook special tokens + language tokens + control tokens

**Token ID layout:**
```
[0, 32767]     → Text tokens (Mistral inherited + Indic extension)
[32768, 32768 + num_codebooks * codebook_size - 1]  → Audio tokens (per-quantizer offset)
[remaining]    → Special tokens (<|lang:kn|>, <|silence|>, <|overlap|>, etc.)
```

**Initialization:** Copy Mistral's embedding weights for shared tokens. Random init for new tokens.

**Target fertility (tokens/word):**
- Latin (English): ≤ 1.5
- Devanagari (Hindi): ≤ 2.5
- Dravidian (Kannada/Tamil/Telugu): ≤ 3.0
- Low-resource (Bodo/Santali): ≤ 4.0

**Citation:** MMS language-specific adapters, Whisper multitask format, SentencePiece Unigram paper

---

## Decision 6: Language Conditioning → Hybrid (token + adapter)

**Decision:** Three-layer language conditioning:
1. **Language token**: `<|lang:kn|>` prepended to each turn (like Whisper)
2. **Language adapters**: Per-language-family LoRA (~100K params each, like MMS)
3. **Implicit detection**: Model learns to detect language from audio context

**Language families for adapter grouping:**
- Indo-Aryan-Deva: Hindi, Marathi, Nepali, Konkani, Dogri, Maithili, Sanskrit, Bodo (Devanagari script)
- Indo-Aryan-Other: Bengali, Assamese (Bengali script), Gujarati, Punjabi, Odia, Sindhi, Kashmiri, Urdu
- Dravidian: Tamil, Telugu, Kannada, Malayalam
- Sino-Tibetan: Manipuri (Meitei)
- Austroasiatic: Santali

**For code-switching:** Inject `<|lang_switch:en|>` token when language changes mid-utterance. Model learns switch points from code-switching training data.

**Why hybrid over token-only?**
1. Tokens alone can't capture phonological differences between language families
2. Adapters allow fine-grained per-family tuning without interference
3. MMS showed adapters work for 1,100+ languages with minimal parameter overhead
4. Implicit detection provides fallback when no language token is provided

**Citation:** MMS §3.2 (language adapters), Whisper §2.1 (language tokens), IndicLID

---

## Decision 7: Data Strategy → Scale Conversational Data

**Decision:** Follow Whisper's thesis (scale > cleverness) with emphasis on conversational speech.

**Data priority:**
1. **Conversational Indic speech** (IndicVoices conversational + YouTube Indic podcasts + Vaani)
2. **English duplex data** (Fisher corpus — 2K hours of real conversations)
3. **Read Indic speech** (IndicVoices read + CommonVoice + OpenSLR)
4. **Code-switching data** (HiACC + Hindi-Marathi + YouTube Hinglish)

**Minimum per language:**
- 50 hours supervised conversational (for usable generation quality)
- 500 hours total (supervised + unsupervised, for SSL pre-training)
- MMS showed 32 hours supervised is enough with SSL, but we target higher quality

**Data mixing strategy (training batches):**
- 40% Indic conversational
- 20% English conversational (Fisher + existing YouTube)
- 20% Indic read speech
- 10% Code-switching
- 10% Instruction-following (synthetic)

**Citation:** Whisper scale thesis, MMS (32h/language sufficient with SSL), Fisher corpus for duplex

---

## Decision 8: Serving → Custom Async WebSocket Server

**Decision:** Replace Gradio with custom async server. No half measures.

**Architecture:**
```
Browser/App ──WebRTC──► aiortc adapter ──► Audio Buffer (80ms frames)
                                                    │
                                                    ▼
                                            Mimi Encode (GPU)
                                                    │
                                                    ▼
                                         Temporal Transformer
                                          (streaming inference)
                                                    │
                                                    ▼
                                            Mimi Decode (GPU)
                                                    │
                                                    ▼
Browser/App ◄──WebRTC──  Audio Buffer  ◄───── Opus encode
```

**Transport options:**
- **WebSocket**: Binary Opus frames, 80ms chunks, asyncio + uvicorn
- **WebRTC**: aiortc for browser/mobile (preferred for quality — handles jitter, packet loss)

**Session management:**
- Per-session KV cache (isolated, no cross-contamination)
- Session timeout: 5 minutes idle → evict KV cache
- Dynamic batching: 5ms collection window for concurrent sessions

**Latency budget (target: ≤200ms end-to-end):**
| Component | Budget |
|-----------|--------|
| Network (WebRTC) | 20ms |
| Mimi encode | 10ms |
| Temporal Transformer (1 step) | 50ms |
| Depth Transformer (1 step) | 30ms |
| Mimi decode | 10ms |
| Network (return) | 20ms |
| Buffer overhead | 60ms |
| **Total** | **200ms** |

**Why not Gradio?**
1. Gradio is request-response — no streaming, no duplex
2. No WebSocket/WebRTC support built-in
3. No session management or KV cache persistence
4. `server.py` currently hardcodes `output.wav` — fundamentally broken for concurrent users

**Citation:** Moshi 200ms practical latency, WebRTC standard for real-time audio

---

## Decision 9: Training Infrastructure → FSDP + Gradient Checkpointing

**Decision:** Use PyTorch FSDP (Fully Sharded Data Parallel) for multi-GPU, gradient checkpointing for memory.

**Why FSDP over DDP:**
1. DDP replicates full model on each GPU — 7B × 2 bytes × N GPUs
2. FSDP shards model across GPUs — 7B × 2 bytes / N GPUs per shard
3. On 4× A10G (23GB each): DDP can't fit 7B (14GB model + overhead > 23GB). FSDP can.
4. On 8× H100: FSDP enables training with larger batch sizes

**Gradient checkpointing:** Must fix the inactive bug (`trainer.py:283-284`). Use `model.gradient_checkpointing_enable()` instead of bare attribute. Reduces memory by ~40% at cost of ~30% slower training.

**For immediate work (GPU:0 only, A10G 23GB):**
- Single GPU: No FSDP needed
- Layer pruning (`prune_layers=2`) reduces 7B → ~3.5B → ~7GB in bf16
- With gradient checkpointing: fits comfortably on A10G

**Citation:** PyTorch FSDP documentation, existing Voxtral DDP support in `utils.py`

---

## Token Layout (detailed)

### Current Voxtral (single-stream, 55Hz)
```
Per window (0.2s):
[text_tok, q1, q2, q3, q4, q1, q2, q3, q4, ..., q1, q2, q3, q4]
  1 text  +  40 audio tokens (10 timesteps × 4 quantizers)  = 41 tokens/window
Total: 41 × 5Hz = 205 tokens/sec
```

### OmniVoxtral (dual-stream, target)
```
Per timestep (80ms = 1 Mimi frame):
[user_text, user_q1...q8, model_text, model_q1...q8]
  1 + 8 + 1 + 8 = 18 tokens per timestep
At 12.5Hz: 18 × 12.5 = 225 tokens/sec

With text at 5Hz (1 text per 2.5 audio frames):
Effective: ~(8+8) × 12.5 + (1+1) × 5 = 210 tokens/sec
```

### Vocabulary Address Space (2^16 = 65,536)
```
[0 — 32,767]         Text tokens (Mistral + Indic extension)
[32,768 — 49,151]    Audio codebook tokens (8 quantizers × 2048 codes each)
[49,152 — 65,535]    Special tokens:
                       <|lang:xx|> × 30 languages
                       <|silence|>
                       <|overlap|>
                       <|backch|> (backchannel)
                       <|lang_switch:xx|> × 30
                       <|turn_start|>
                       <|turn_end|>
                       <|pad|>
                       Reserved for future
```

---

## Architecture Comparison

| Feature | Voxtral (current) | OmniVoxtral (target) | Moshi (reference) |
|---------|-------------------|---------------------|-------------------|
| Streams | 1 (single) | 2 (dual) | 2 (dual) |
| Transformer | 1 (Mistral) | 2 (Temporal + Depth) | 2 (Temporal + Depth) |
| Codebooks | 4 | 8 (or 4 if benchmark passes) | 8 |
| Languages | English | 22 Indic + English | English |
| Inner Monologue | No (text in sequence) | Yes (text prefix per step) | Yes |
| Duplex | No | Yes | Yes |
| Loss weighting | Defined but unused | 100:10:1 (text:semantic:acoustic) | 100:1 (semantic:acoustic) |
| Tokenizer | Mistral BPE 32K | SentencePiece Unigram 65K | Helium BPE |
| Serving | Gradio (request-response) | WebSocket/WebRTC (streaming) | WebSocket |
| Training | DDP | FSDP + gradient checkpointing | 1016 H100s |

---

## Open Questions (to resolve during implementation)

1. **Depth Transformer initialization:** Train from scratch (Moshi approach) or distill from Temporal layers?
2. **Cross-stream attention:** Allow model stream to attend to user stream (Moshi: yes)? Or keep separate (simpler)?
3. **Streaming Mimi:** Current vendored Mimi supports streaming. Need to verify it works with 8 quantizers.
4. **Language adapter granularity:** Per-language or per-language-family? MMS uses per-language but has 1B params.
5. **Code-switching detection:** Explicit `<|lang_switch|>` tokens or implicit (model figures it out)?
6. **Silence representation:** Dedicated `<|silence|>` token or actual silence codec tokens?
