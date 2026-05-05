# Research Findings — OmniVoxtral Pipeline Rebuild

Two parallel research streams: codebase audit (existing pipeline) and external best-practice survey (rebuild components).

---

## PART A — CODEBASE AUDIT

### A.1 Data flow diagram (current state)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ YouTube URLs (data/urls.txt) → yt-dlp → ffmpeg segment (no transcode)       │
│ scraping.py:download_and_chunk_video, chunk_duration=20s                    │
│                              ↓                                              │
│                data/chunks_indic_yt/*.m4a (20s chunks, 57,668 files)        │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPTIONAL FILTER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ filter_audio.py — SNR>10dB, speech_ratio>0.3, clip_ratio<0.01, 3s≤dur≤60s   │
│                              ↓                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              SINGLE-STREAM TOKENIZATION (default — used today)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ preprocessing.py / preprocess_hf.py:                                        │
│   AudioChunkDataset → (batch, 1, 480000) @ 24kHz, padded with zeros         │
│                              ↓                                              │
│ VoxtralTokenizer.encode(x, 24000)  [model.py:73]                            │
│   ┌─ AUDIO BRANCH (Mimi 24kHz):                                             │
│   │    mimi.encode(x) → (batch, 8, ~250) — 8 codebooks @ 12.5 Hz            │
│   │    + token_offset[q] + text_vocab_size = 65536-81920 range              │
│   │    interleave 8 codebooks → (batch, ~2000)                              │
│   │                                                                          │
│   └─ TEXT BRANCH (Whisper 16kHz):                                           │
│        resample(24000→16000)                                                │
│        TimedWhisperTokenizer.forward(x, 16000)                              │
│          ├ generate_tokens(language="en") ← HARDCODED at init               │
│          ├ tokens_to_words → WordTiming[]                                   │
│          ├ separate_into_buckets(bucket_size=1.0s)                          │
│          └ _tokenize_bucket → Mistral BPE (default, sp_path=None)           │
│             pad/truncate to text_hz=5 tokens/sec                            │
│        → (batch, 100) text tokens                                           │
│                                                                              │
│ DELAY: text leads audio by 2 windows (0.4s)                                 │
│   audio = audio[..., 8*12.5*2:]                                             │
│   text  = text[..., :-5*2]                                                  │
│                                                                              │
│ FINAL INTERLEAVE: stride=21 per window [text(1), audio(20)]                 │
│ Output: (batch, ~505) flattened tokens                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              DUAL-STREAM TOKENIZATION (built but NEVER USED on YT)          │
├─────────────────────────────────────────────────────────────────────────────┤
│ diarize_audio.py:                                                            │
│   pyannote-3.1 → segments [{speaker, start, end}]                           │
│   separate_speakers (zero-mask other speakers) → tracks dict                │
│   assign_roles (more speech → "model", less → "user")                       │
│                              ↓                                              │
│ DualStreamTokenizer.encode(user_audio, model_audio, 24000)                  │
│   per-stream: text(1) + audio(20)                                           │
│   dual interleave: stride=42 per window                                     │
│   [user_text, user_audio×20, model_text, model_audio×20]                    │
│ Output: (batch, ~1010) flattened dual-stream tokens                         │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STORAGE                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ _save_tokens → data/tokens/{aa-ff}/{filename}.npy                           │
│ NO METADATA SIDECAR. NO LANGUAGE TAG. NO TRANSCRIPT. NO CONFIDENCE.         │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TRAINING DATALOADER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ VoxtralDataset (IterableDataset, 90/10 split, seed=42)                      │
│  - get_npy_files(data_path) recursive walk                                  │
│  - corrupted file → fake item (random tokens)                               │
│  - distributed: stride by world_size × num_workers                          │
│  - max_seq_len truncation                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOSS COMPUTATION                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ omni_trainer.py:                                                             │
│  extract_codebook_targets(x, stride, num_codebooks, dual_stream)            │
│    → (batch, num_frames, 8) targets, (num_frames,) frame_positions          │
│  OmniVoxtral.forward(input_ids=x[:-1], targets=x[1:])                       │
│    temporal_loss: weighted CE [text=100, semantic=1, acoustic=1]            │
│    depth_loss: per-codebook CE, q0 weight=100, focal γ=2.0 default          │
│    total_loss = temporal_loss + depth_loss                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### A.2 Hardcoded numerics on the data path

| Constant | Where | Why | Breaking change? |
|---|---|---|---|
| `24000` (Mimi SR) | `loaders.py:15`, `preprocessing.py:96`, `dual_stream.py:126` | SEANet trained at this SR | Mimi retrain required |
| `12.5` (Mimi frame rate) | `loaders.py:16` | 24000 ÷ 1920 hop_length | Mimi retrain |
| `16000` (Whisper SR) | `word_level_whisper.py:243`, `model.py:77` | Whisper fixed input | Hard assertion |
| `20 × 24000 = 480000` (chunk frames) | `preprocessing.py:27`, `scraping.py:18` | 20s training window | **Soft — can change in v2** |
| `5` (text_hz) | `model.py:17`, `config.py:43` | Text tokens per second | Tied to bucket size 1.0s |
| `8` (num_quantizers) | `model.py:18`, `config.py:49` | Codebooks used (Mimi has 32 total) | Soft, but vocab math depends on it |
| `2048` (codebook_size) | `loaders.py:46`, `config.py:50` | Mimi quantizer cardinality | Hard |
| `65536` (text vocab) | `model.py:19`, `config.py:51` | Mistral / SP tokenizer vocab | Tokenizer retrain to change |
| `81920` (full vocab) | `config.py:38` | 65536 + 8×2048 | Must match `resize_token_embeddings` |
| `2` (delay windows) | `model.py:107`, `dual_stream.py:137` | Text leads audio 0.4s for causal alignment | Soft |
| `21` (single-stream stride) | `omnivoxtral.py:83` | 1 text + 20 audio per 200ms window | Stride math everywhere |
| `42` (dual-stream stride) | `dual_stream.py:66` | 2 × stream_stride | Dual-stream loss extraction |
| `1.0` (bucket size, sec) | `word_level_whisper.py:256` | Whisper sentence granularity | Soft |
| `10 dB` (SNR threshold) | `filter_audio.py:102` | Quality gate | Tunable |
| `0.3` (speech ratio) | `filter_audio.py:59,72` | Music/silence rejection | Tunable |
| `[100, 1, 1]` (loss weights) | `config.py:39`, `omni_trainer.py:162` | Moshi §4.4: text=100x audio | Soft |
| `1.0s` (Whisper bucket) | `word_level_whisper.py:256` | Bucketing window | Tied to text_hz |

### A.3 Public interfaces (signatures, shapes, contracts)

**`VoxtralTokenizer.encode(x, sample_rate)`** — `model.py:73`
- Input: `x: (batch, 1, num_samples)` float32 @ 24kHz, mono only (asserted).
- Output: `(batch, seq_len)` token IDs in `[0, 81920)`.
- For 20s @ 24kHz: input (1,1,480000) → output (1,~505).
- Assumes `sample_rate=24000` (asserted), `language` set at init time.
- Failure: silent audio → empty Whisper output → mostly audio tokens. Wrong language → garbled transcript with shifted boundaries.

**`DualStreamTokenizer.encode(user_audio, model_audio, sample_rate)`** — `dual_stream.py:107`
- Input: two `(batch, 1, num_samples)` tensors @ 24kHz.
- Output: `(batch, seq_len_dual)`. Layout per window: `[user_text(1), user_audio(20), model_text(1), model_audio(20)]`. Stride = 42.
- Trims both inputs to min length internally.
- Assumes diarization preprocessing has already separated speakers (non-overlapping).

**`TimedWhisperTokenizer.forward(audio, sample_rate)`** — `word_level_whisper.py:242`
- Input: `(batch, num_samples)` @ 16kHz.
- Output: `(batch, num_buckets)` where `num_buckets = ⌈duration⌉` and each bucket is exactly `hertz=5` tokens (padded/truncated).
- **Language fixed at construction.** No per-call override.
- **Pad token = `eos_token`** for Mistral BPE (line 214). Polluting "end of sentence" semantics into silence.

**`extract_codebook_targets(tokens, stride, num_codebooks, dual_stream)`** — `omni_trainer.py:35`
- Input: `(batch, seq_len)` interleaved tokens.
- Output: `((batch, num_frames, 8), (num_frames,))` — codebook targets and frame positions.
- Single-stream: extracts positions `[1:21]` per stride-21 window (audio portion).
- Dual-stream: extracts positions `[22:42]` (model_audio only).
- Truncates partial windows silently.

**`VoxtralDataset.__iter__`** — `trainer/data.py:45`
- Yields `{"tokens": torch.Tensor(seq_len,)}` infinitely.
- Deterministic 90/10 train/val split, seed=42.
- Per-worker offset `rank × num_workers + worker_id` for DDP.
- **No metadata** flows alongside tokens.

### A.4 Implicit assumptions (no validation in code)

1. **Language homogeneity** per tokenizer instance — language fixed at init, no detection.
2. **Whisper transcripts are accurate** — token timestamps trusted unconditionally; no confidence filter.
3. **Mono audio** — channel averaging hardcoded; stereo info destroyed.
4. **Frame alignment** — text bucket t aligns with audio frame [t, t+1) by construction; never verified per-file.
5. **No language fallback** — if Whisper fails or is wrong language, no MMS LID consulted.
6. **All `.m4a`/`.mp3`/`.wav` are valid** — corrupted files return zero-padded silence (`preprocessing.py:64-65`).
7. **Vocab partition is constant** — text in `[0, 65535]`, audio in `[65536, 81919]`. No dynamic vocab.
8. **Diarization quality** — pyannote output trusted; no post-validation.

### A.5 Failure modes

| Scenario | Current behavior | Impact |
|---|---|---|
| Silent / music-only audio | Whisper returns empty text → tokens are mostly audio | text_acc=0 on these files; `filter_audio.py` partially mitigates |
| Wrong-language Whisper config | Whisper transcribes in wrong language; word boundaries misaligned | Text tokens encode garbage; this is the **gibberish root cause** |
| Multi-speaker audio (single-stream) | Whisper merges all speakers into one transcript | Tokens semantically inconsistent; model learns no turn-taking |
| Diarization fails (single speaker, music, noise) | Returns empty segments → file skipped, OR single speaker → both streams duplicated | Silent loss of conversational data |
| Corrupted `.npy` at training time | `get_item` exception → returns `get_fake_item()` (random tokens, shape 220) | Silent corruption of training; no logging |
| Sequence not divisible by stride | Partial window discarded silently | Up to (num_codebooks-1) tokens lost per file |
| `max_seq_len` too small | Truncation, no warning | Long-utterance bias loss |

### A.6 Cross-component dependencies

```
preprocess_hf.py ─┬─ VoxtralTokenizer (model.py)
                  ├─ datasets (FLEURS, IndicVoices)
                  └─ writes data/tokens/{lang}/

diarize_audio.py ─┬─ pyannote.audio
                  ├─ DualStreamTokenizer (dual_stream.py)
                  ├─ VoxtralTokenizerConfig (model.py)
                  └─ writes data/dual_tokens/  [NEVER RUN ON YT CORPUS]

VoxtralTokenizer ─┬─ Mimi (loaders.py → compression.py)
                  └─ TimedWhisperTokenizer (word_level_whisper.py)

DualStreamTokenizer ─┬─ Mimi (shared instance)
                     ├─ TimedWhisperTokenizer (shared instance)
                     └─ interleave/uninterleave (model.py)

TimedWhisperTokenizer ─┬─ transformers.WhisperProcessor + WhisperForConditionalGeneration
                       ├─ transformers.AutoTokenizer (Mistral) — DEFAULT
                       └─ sentencepiece (optional, sp_tokenizer_path)

trainer/data.py ─┬─ VoxtralTrainConfig
                 └─ reads data/tokens/  [single-stream by default]

omni_trainer.py ─┬─ OmniVoxtral (model/omnivoxtral.py)
                 └─ extract_codebook_targets (local)
```

### A.7 Existing testing setup

- `scripts/train_test.py` — 2-line smoke test (calls `train()`).
- `src/voxtral/trainer/test.py` — evaluation loop, not unit tests.
- **NO** `conftest.py`, `pytest.ini`, or `tests/` directory.
- `__main__` smoke tests exist for: `preprocess.py`, `preprocess_hf.py`, `diarize_audio.py`, `filter_audio.py`, `scraping.py`.
- **Zero unit tests** for: `VoxtralTokenizer.encode/decode`, `DualStreamTokenizer`, `extract_codebook_targets`, `compute_omni_loss`, frame-alignment, codebook offset correctness.

### A.8 What's missing (rebuild gap analysis)

1. **Per-file language detection** — no LID, no confidence, no fallback.
2. **VAD** — silero/webrtc not wired in; all 20s chunks tokenized as-is.
3. **Metadata sidecar writer** — no JSON alongside `.npy`.
4. **Stereo / dual-channel handling** — hardcoded mono averaging.
5. **MMS LID + ASR fallback** — for the 7 Indic languages Whisper doesn't support.
6. **Streaming Whisper** — no path for short clips (<4s) where standard LID is unreliable.
7. **Schema validation** — no token-range or shape checks on `.npy` files.
8. **Per-codebook loss schedule** — depth weights are constant; no warmup curriculum.
9. **End-to-end integration tests** — synthetic audio → `.npy` → batch → loss roundtrip.
10. **Stereo source detection** — no auto-detect of L/R channel-as-speaker.

---

## PART B — EXTERNAL RESEARCH

### B.1 Mistral Voxtral models (deep dive — user-requested)

**Voxtral Transcribe 2** ([mistral.ai/news/voxtral-transcribe-2](https://mistral.ai/news/voxtral-transcribe-2)):
- Two variants: **Voxtral Mini Transcribe V2** (batch, $0.003/min, ~4% WER on FLEURS) and **Voxtral Realtime** (streaming, $0.006/min, also open-weights as `Voxtral-Mini-4B-Realtime-2602`, sub-200ms configurable, ~1-2% WER above offline at 480ms).
- Product layer adds: speaker diarization with timestamps, **context biasing up to 100 words/phrases** (domain vocab injection), word-level timestamps, audio up to 3 hours/request.
- 13 languages: English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, German, Japanese, Korean, Italian, Dutch.
- **Diarization caveat:** "with overlapping speech, the model typically transcribes one speaker."

**`mistralai/voxtral` collection** ([huggingface.co/collections/mistralai/voxtral](https://huggingface.co/collections/mistralai/voxtral)):

| Model | Params | Type | Notes |
|---|---|---|---|
| Voxtral-Small-24B-2507 | 24B | Audio-Text→Text | Server-side. 8 langs incl. Hindi. 32K ctx. ~55GB bf16. |
| Voxtral-Mini-3B-2507 | 3B | Audio-Text→Text | Edge. 8 langs incl. Hindi. **~9.5GB bf16, fits A10G.** Auto-LID + transcription + translation + Q&A. |
| Voxtral-Mini-4B-Realtime-2602 | 4B | Streaming ASR | 13 langs. <500ms. NOT Mimi-based. |
| Voxtral-4B-TTS-2603 | 4B | TTS | Speech synthesis (Mar 2026). |

**Voxtral-Mini-4B-Realtime-2602** architecture:
- ~970M causal audio encoder + ~3.4B LM decoder. **NOT Whisper-style despite the HF code naming** — it's a **causal/streaming** transformer with sliding window 750, 32 layers, 1280 dim, 32 heads, 128 mel bins, hop_length=160 at 16kHz.
- **80ms per text token, 12.5 Hz frame rate** (matches Mimi exactly!). MLP adapter 4× downsamples encoder output.
- Streaming: `num_samples_first_audio_chunk` + `num_samples_per_audio_chunk`. **480ms = sweet spot** matching offline quality. `default_num_delay_tokens=6`.
- **No explicit LID head** — language is emergent from text generation. Short-clip LID below ~480ms is unreliable.
- **No word timestamps** documented for the realtime model.
- WER at 480ms: English 8.47%, FLEURS 13-lang avg **8.72%**.

**`antirez/voxtral.c`** ([github.com/antirez/voxtral.c](https://github.com/antirez/voxtral.c)):
- Single-file C reference. Reveals streaming pattern.
- API: `vox_stream_feed(samples) → vox_stream_get(text) → vox_stream_finish()`.
- `vox_stream_flush(s)` for manual silence boundary.
- **Default processing interval 2.0s** — every 2s of new audio triggers an encoder pass.
- Decoder KV cache auto-compacts at 8192 positions (sliding window).
- **VAD/silence built-in:** "silence is automatically detected and stripped to reduce encoder/decoder work."

**Bottom line:** Use **Voxtral-Mini-3B-2507 as the upstream LID + transcription oracle for Hindi** in our preprocessing (~9.5 GB bf16, fits A10G alongside Mimi/pyannote). For the other 21 Indic languages, fall back to Whisper-large-v3 + MMS. Do NOT copy Voxtral-Realtime architecture — it's single-stream encoder→LM, not Moshi-style. **Portable design choices:** 80ms / 12.5 Hz frame rate (already matches Mimi), causal sliding-window encoder pattern, `num_delay_tokens` for streaming alignment.

### B.2 whisper-diarization joint pipeline

[github.com/MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) — 5.5k stars, actively maintained.

**Pipeline order (load-bearing):**
1. **Demucs** vocal extraction (strip music/noise).
2. **faster-whisper** transcription on vocal stem.
3. **ctc-forced-aligner** correct word timestamps.
4. **NeMo MarbleNet** VAD + segmentation.
5. **NeMo TitaNet** speaker embeddings.
6. Speaker labels merged onto word-level timestamps.

**API:** `python diarize.py -a AUDIO_FILE` plus `--whisper-model`, `--language`, `--no-stem`, `--suppress_numerals`. Output: per-speaker timed segments with text.

**VRAM:** ≥10 GB for `diarize_parallel.py` (Whisper + NeMo concurrent). Sequential `diarize.py` fits more comfortably on A10G 24GB.

**Indic caveat:** No published Indic benchmarks. Pyannote/NeMo speaker models have known issues on Indian languages (open issue [pyannote-audio Discussion #1856](https://github.com/pyannote/pyannote-audio/discussions/1856) — "the pipeline fails to segregate speakers accurately for Indian languages"). Whisper-large-v3 supports 15 Indic languages; **NeMo TitaNet trained on mostly English/Mandarin** → expect quality drop on Indic clusters.

**Gotchas:** Long audio (4+ hr) crashes via RAM exhaustion. Overlapping speech unsupported. Demucs strips music aggressively — desirable for podcasts with intro music, undesirable for sung-language datasets (use `--no-stem`).

**Bottom line:** Adopt the **whisper-diarization stage ordering** (Demucs → faster-whisper → ctc-forced-aligner → MarbleNet → embeddings) as our blueprint, but **swap TitaNet for pyannote-3.1** since we've already validated pyannote in `diarize_audio.py`, and **skip Demucs for clean studio podcasts** to preserve native acoustics for Mimi tokenization. Run sequentially to stay <20GB on A10G.

### B.3 Streaming Whisper for short clips

- **Whisper hard constraint:** 30-second receptive field. `detect_language()` returns full softmax → use top-1 prob as confidence.
- **faster-whisper:** exposes `segment.avg_logprob` (segment-level confidence), no per-language confidence in streaming mode without manually calling `model.detect_language()`.
- **whisper-timestamped:** DTW alignment, per-word confidence.
- **whisper-streaming** (UFAL): self-adaptive LocalAgreement-2, ~1s polling.
- **WhisperLive** (Collabora): websocket, ~3GB VRAM fp16.
- **WhisperRT** ([arxiv 2508.12301](https://arxiv.org/html/2508.12301v2)): causal streaming Whisper with token-level confidence over time.
- **Voxtral-Realtime:** 1 text token / 80ms, LID emergent from token output.

**VAD in faster-whisper:** Built-in Silero VAD, `vad_filter=True`, default `min_silence_duration_ms=2000` is **too long** for podcast preprocessing → reduce to **500ms**.

**Pause thresholds (typical production):**
- 200ms — streaming partial-result emission.
- 500ms — sentence boundary cut for batch chunking.
- 1000-2000ms — paragraph / turn boundary.

**Bottom line:** **faster-whisper-large-v3** with `vad_filter=True, vad_parameters={min_silence_duration_ms: 500}`, post-validate language via `model.detect_language()` per chunk. **Discard chunks where top-1 lang prob < 0.7 OR avg_logprob < -1.0**. For chunks <4s, **always concatenate ±2s of neighbor context before LID** — short-clip LID is unreliable.

### B.4 VAD-aware variable-length chunking

**Silero VAD** ([github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)) — dominant choice over WebRTC.
- Multilingual ("thousands of languages" trained).
- v6 ROC-AUC 0.97 vs WebRTC 0.73 on multi-domain validation.
- 8kHz / 16kHz only.
- <1ms per 30ms chunk on a single CPU thread.

**Recommended Silero defaults for OmniVoxtral:**

| Param | Default | Suggested |
|---|---|---|
| `threshold` | 0.5 | 0.5 |
| `min_speech_duration_ms` | 250 | **1000** (prevent sub-1s fragments) |
| `min_silence_duration_ms` | 100 | **500** (sentence boundary) |
| `speech_pad_ms` | 30 | **150** (preserve onsets/offsets — critical for Mimi) |
| `max_speech_duration_s` | inf | **20** (Mimi memory cap) |

**Stereo handling:** Silero operates **per-channel** (must call on each channel array). Same for WebRTC. **pyannote-3.1 auto-downmixes stereo → mono** (averaging), losing channel-as-speaker info. **For podcast stereo where each speaker is on a separate channel, run VAD per channel and merge → free diarization without pyannote.**

**Recommended chunk distribution for speech-LM training:**
- Min: 1.0s (12 Mimi frames) — anything shorter has unreliable LID.
- Max: 20s (250 Mimi frames) — A10G DDP memory.
- Mean target: 5-10s (Moshi paper).

**Bottom line:** **Silero VAD v6 per-channel** with `min_speech_duration_ms=1000, min_silence_duration_ms=500, speech_pad_ms=150, max_speech_duration_s=20`. WebRTC is dead — Silero wins by 24 ROC-AUC points.

### B.5 MMS LID + ASR for low-resource Indic

**`facebook/mms-lid-256`** ([huggingface.co/facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256)):
- 256 languages, Wav2Vec2ForSequenceClassification, 1B params, ~4 GB fp32 (~2GB fp16).
- Indic in lid-256: hin, ben, tam, tel, kan, mal, mar, pan, urd, asm, guj, ory, npi, mai, mag, bho.
- **Missing from lid-256:** Bodo, Dogri, Konkani, Manipuri, Santali, Sindhi.
- Use **`facebook/mms-lid-1024`** or **`facebook/mms-lid-2048`** for full Indic-22 coverage (sat, snd, mni, gom).
- License: **CC-BY-NC 4.0** (non-commercial — train your own classifier if commercial).

**`facebook/mms-1b-all`** for ASR fallback:
- 1162 languages via per-language adapters (~10MB each).
- API: `processor.tokenizer.set_target_lang("ben"); model.load_adapter("ben")` — switch language without reloading 1B base.
- Quality: WER on Librispeech-clean ~12.6% (worse than Whisper) but on truly low-resource Indic where Whisper has zero training data, **MMS is the only open option**.

**Indic-22 routing strategy:**

| Language | Primary | Fallback |
|---|---|---|
| Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Urdu, Nepali, Sinhalese, Assamese, Odia, Maithili (15) | Whisper-large-v3 | MMS-1B-all |
| Bodo, Dogri, Konkani, Manipuri, Santali, Sindhi (6) | MMS-1B-all | IndicWhisper |
| Kashmiri | MMS (perso-arabic) | — |

**Bottom line:** **mms-lid-2048 as primary LID** (covers all 22 Indic), **Whisper-large-v3 LID as confidence tie-breaker** on the 15 it supports. ASR routing: Whisper for the 15, MMS-1B-all for the 7 unsupported. Mind CC-BY-NC if commercial.

### B.6 SentencePiece + dual-script handling

**Vocabulary expansion:**
- SentencePiece **does not natively support adding tokens** to a trained Unigram model — must retrain or use `--vocab` restriction trick.
- Clean path: **train new SP unigram with `--user_defined_symbols=<lang_hi>,...,<turn>,<bos_audio>` BEFORE corpus training** so symbols are reserved.
- Our `omnivoxtral_sp.model` already has language tokens at IDs 4-26 — **keep this design, don't append.**

**State of the art — IndicSuperTokenizer** ([arxiv 2511.03237, Nov 2025](https://arxiv.org/html/2511.03237v1)):
- 2-stage BPE (subword + superword), 200K vocab, 39.5% fertility improvement over LLaMA-4.
- Hindi 1.23, Tamil 2.12, Odia 1.65 (vs LLaMA-4's 10.51).
- **Our 1.91 fertility @ 65K is competitive** — don't switch unless scaling to 200K vocab.

**Moshi tokenizer + inner-monologue pattern** ([arxiv 2410.00037](https://arxiv.org/html/2410.00037v2)):
- 32K SentencePiece Unigram, byte-backoff for unseen chars, all numbers split to single digits.
- **Inner Monologue:** time-aligned text tokens predicted as **prefix** to acoustic tokens **per 80ms frame**.
- Text token at frame N = the word that *starts* in frame N (or PAD if mid-word).

**Language-tag injection (literature patterns):**
1. **Prefix token** (Whisper, mBART): `<lang_hi>` at start of text stream. Simple, mono-lingual chunks.
2. **Per-frame language stream**: parallel token stream with language ID per frame. Better for code-switching, doubles depth-transformer compute.

**IndicTrans2** unifies 5 Indic scripts via SentencePiece preprocessing. **IndicWhisper** (AI4Bharat Vistaar) Whisper-medium fine-tuned on 10.7K hours, lowest WER on 39/59 Vistaar benchmarks.

**Bottom line:** Keep our **65K SP Unigram with `<lang_xx>` at fixed IDs** — fertility 1.91 is excellent. Adopt **Moshi's inner-monologue per-frame prefix pattern**: `<lang_hi>` emitted in FIRST text-frame of each utterance, then text aligned per 80ms. For mid-utterance code-switching, emit fresh `<lang_xx>` token at boundary.

### B.7 Dual-channel Mimi encoding

**Mimi stereo support** ([huggingface.co/docs/transformers/model_doc/mimi](https://huggingface.co/docs/transformers/model_doc/mimi)):

```
audio_channels (int, optional, defaults to 1) — Number of channels in the audio data.
                                                  Either 1 for mono or 2 for stereo.
```

**However:** the released `kyutai/mimi` checkpoint is trained at `audio_channels=1` (mono). Architecture supports `audio_channels=2` as config flag, but **no public stereo Mimi checkpoint exists**.

**Mimi specs (kyutai/mimi mono):**
- 24 kHz sampling (NOT 16kHz).
- 12.5 Hz frame rate (= 80ms per frame, matches Voxtral exactly).
- 1.1 kbps.
- 32 quantizers total, **1 semantic** (WavLM-distilled), **31 acoustic RVQ**.
- `codebook_size=2048`, `codebook_dim=256`. Causal conv, sliding window 250.

**Moshi's dual-stream pattern** ([arxiv 2410.00037](https://arxiv.org/html/2410.00037v2)):

> "Moshi models two streams of audio: one corresponds to Moshi speaking, and the other one to the user speaking"

Moshi explicitly **does NOT use Mimi's stereo mode**. Each channel is a separate Mimi-mono stream. Interleaved at depth-transformer level. **Our stride-21 / 42 design already mirrors this**.

**Pre-Mimi step for podcast stereo:**
1. **If channels = speakers** (interview podcast, separate channels): encode each channel as separate Mimi-mono stream → "free" dual-stream training data.
2. **If channels = spatial/ambient** (most YouTube): diarize → mask non-target-speaker to silence per stream → encode both.

**Bottom line:** Treat Mimi as **mono-only in practice**. Detect channel separation: correlate channels — if low correlation → speaker-per-channel → encode each. If high correlation → diarize + mask. Both feed our existing dual-stream depth transformer without architectural change. **Resample to 24kHz before Mimi**, not 16kHz.

### B.8 A10G resource budget (preprocessing)

| Stage | Model | VRAM | Time per 1hr audio |
|---|---|---|---|
| VAD | Silero v6 | <100 MB CPU | ~1 min |
| Source sep | Demucs htdemucs | ~2 GB | ~3-5 min |
| LID primary | mms-lid-2048 | ~4 GB fp16 | ~2 min |
| ASR primary | faster-whisper-large-v3 | ~3 GB fp16 | ~2-4 min |
| ASR fallback | mms-1b-all | ~4 GB fp16 + adapters | ~5 min |
| Diarization | pyannote-3.1 | ~2 GB | ~2 min |
| Forced align | ctc-forced-aligner | ~1 GB | ~1 min |
| Mimi encode | kyutai/mimi | ~1 GB | ~2 min |

**Total peak ~15-18 GB if stages serialized.** Do NOT run Demucs + Whisper + MMS concurrently on a 24GB A10G — load/unload between stages, or run VAD/Demucs on CPU to save VRAM headroom.

---

## PART C — KEY DECISIONS THE PLAN MUST RESOLVE

1. **Option A vs B** — source-language inner monologue (recommended) vs English-driven cross-lingual TTS.
2. **Whisper vs Voxtral-Mini-3B vs faster-whisper** for primary ASR. Voxtral-Mini-3B is 9.5GB and offers built-in translation, but Whisper-large-v3 has better Indic coverage. **Recommendation: faster-whisper-large-v3 + MMS fallback** (covers all 22 Indic).
3. **Demucs in or out** — strips music aggressively, helpful for podcasts with intro music, harmful for sung-language datasets. **Recommendation: gate behind a flag, default off for v1, opt-in for noisy YouTube content.**
4. **Stereo strategy** — auto-detect speaker-per-channel via correlation, fall back to pyannote diarize+mask. Both produce dual-stream Mimi encoding.
5. **Tokenizer schema lock** — when do we re-train SentencePiece? Recommend: never in this rebuild; use existing `omnivoxtral_sp.model`. Vocab expansion is a Phase 6 architectural change.
6. **Metadata sidecar schema** — JSON vs JSONL vs Parquet. **Recommendation: JSON-per-file** for human-readability and partial-write safety; aggregate to Parquet manifest at training time.
7. **Re-tokenize existing 57,668 YT files vs from scratch** — the existing tokens were built with `language="en"` default and Mistral BPE. **Recommendation: discard, do not migrate**. Re-tokenization budget ~24-36 GPU-hr diarization + ~200 GPU-hr Whisper+Mimi.
