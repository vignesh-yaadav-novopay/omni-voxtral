# OmniVoxtral Pipeline Rebuild — High-Level Design

## 1. Purpose

This is the architectural overview of the rebuilt training data pipeline. It defines the components, their responsibilities, and the data flow between them. For implementation specifics (function signatures, file edits, concrete schemas) see `LLD.md`. For the prose narrative and rationale see `plan.md`.

## 2. System diagram (target end-state, post-Phase 5)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              SOURCE DATA LAYER                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   FLEURS (HF)        IndicVoices (HF)         YouTube                          │
│   13 langs           22 langs                 1,845 URLs / 2,654 hr            │
│   ~12h/lang          conversational           podcast / monologue              │
│   studio-clean       multi-speaker            mixed quality                    │
│       │                  │                        │                            │
│       │                  │                        ▼                            │
│       │                  │            scraping.py (yt-dlp + ffmpeg segment)    │
│       │                  │                        │                            │
│       │                  │                        ▼                            │
│       │                  │            data/chunks_indic_yt/*.m4a               │
│       │                  │            (legacy 20s chunks — input only)         │
│       │                  │                                                     │
│       └──────────────────┴────────────┬───────────┘                            │
│                                       ▼                                        │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                          PHASE 0 — PRE-FLIGHT VALIDATION                       │
├────────────────────────────────────────────────────────────────────────────────┤
│   scripts/phase0_preflight.py                                                  │
│     1. SP <lang_xx> roundtrip                                                  │
│     2. Mimi -45 dB vs bit-zero codebook differentiation                        │
│     3. (dropped: resume gradient sanity — fresh-init only)                     │
│     4. 200-chunk LID rejection pilot                                           │
│   Output: planning/phase0_results.md                              │
│   Gates: Phase 1 SP injection strategy, Phase 4 silence strategy, Phase 3      │
│          quarantine threshold                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                       PHASE 1 — TOKENIZER FIXES (1a + 1b)                      │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Phase 1a (FLEURS + IV, ~4 days)            Phase 1b (YouTube, ~10 days)      │
│      │                                          (gated on Phase 3 LID)         │
│      ▼                                                                         │
│   scripts/retokenize_v2.py                                                     │
│     ┌────────────────────────────────────┐                                     │
│     │ VoxtralTokenizer.encode(            │                                    │
│     │   x, sample_rate,                   │  per-call language + task          │
│     │   language=lang,                    │                                    │
│     │   task="transcribe",                │                                    │
│     │   source_metadata=...               │                                    │
│     │ )                                   │                                    │
│     │   ├─ Mimi.encode (24 kHz, 8q)       │                                    │
│     │   └─ TimedWhisperTokenizer.forward  │                                    │
│     │       ├─ Whisper transcribe         │                                    │
│     │       └─ SentencePiece bucketing    │  (default sp_tokenizer_path)       │
│     └────────────────────────────────────┘                                     │
│                       │                                                        │
│                       ▼                                                        │
│   _save_tokens() + write_metadata_sidecar()  (atomic — temp+rename)            │
│                       │                                                        │
│                       ▼                                                        │
│   data/tokens_v2/{source}/{lang}/{shard}/file.npy + file.meta.json            │
│                                                                                │
│   Phase 1.5: valid_token_mask in sidecar → trainer ignores zero-pad regions    │
│                                                                                │
│   Lazy translation (10% sample initially, fills in background):                │
│     VoxtralTokenizer.translate() → sidecar.translation_en                      │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                  PHASE 2 — VAD-AWARE VARIABLE-LENGTH CHUNKING                  │
├────────────────────────────────────────────────────────────────────────────────┤
│   scripts/vad_chunker.py                                                       │
│     ├─ Stereo channel correlation detector                                     │
│     │     correlation < 0.5 → speaker-per-channel                              │
│     │     correlation ≥ 0.5 → mixed audio                                      │
│     ├─ Silero VAD v6 (per-channel)                                             │
│     │     min_speech_duration_ms=300                                           │
│     │     min_silence_duration_ms=500                                          │
│     │     speech_pad_ms=150                                                    │
│     │     max_speech_duration_s=20                                             │
│     └─ Variable-length chunks (no zero-padding ever)                           │
│                                       │                                        │
│                                       ▼                                        │
│       data/chunks_v2/{source}/{lang_or_unknown}/{shard}/{uuid}.wav             │
│                                              + chunk.json                      │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                PHASE 3 — PER-CHUNK LID + STREAMING WHISPER FALLBACK            │
├────────────────────────────────────────────────────────────────────────────────┤
│   scripts/detect_language.py    (batched-pass: LID sweep, then ASR sweep)      │
│     ├─ Primary: facebook/mms-lid-2048 (covers all 22 Indic)                    │
│     ├─ Secondary: Whisper-large-v3 detect_language() (15 Indic supported)      │
│     ├─ Agreement check + confidence threshold 0.7                              │
│     └─ Streaming context for chunks <4s (concat ±2s neighbors)                 │
│                                       │                                        │
│                                       ▼                                        │
│       <chunk>.lang.json {language, confidence, method, agreement}              │
│                                       │                                        │
│   Routing per language:                                                        │
│     ├─ 15 Whisper-supported Indic → faster-whisper-large-v3                    │
│     └─ 7 Whisper-unsupported (brx, doi, kok, mni, sat, sd, ks)                 │
│         → MMSASR (mms-1b-all + per-language adapters)                          │
│                                       │                                        │
│                                       ▼                                        │
│       Quarantine: low-confidence or LID-disagree chunks                        │
│       data/chunks_v2_quarantine/                                               │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│              PHASE 4 — DIARIZATION + DUAL-STREAM TOKENIZATION                  │
├────────────────────────────────────────────────────────────────────────────────┤
│   scripts/diarize_v2.py                                                        │
│     (operates on raw source audio, NOT on chunks_v2)                           │
│                                                                                │
│   detect_music.py → music_likely flag → gates Demucs                           │
│                                                                                │
│   ┌───────────────────────────────────────────────────────┐                    │
│   │ Stage 1: (optional) Demucs vocal separation           │                    │
│   │ Stage 2: faster-whisper transcription                 │                    │
│   │ Stage 3: ctc-forced-aligner (correct word timestamps) │                    │
│   │ Stage 4: Silero v6 VAD/segmentation                   │ ← consistent with  │
│   │ Stage 5: pyannote-3.1 speaker embeddings              │   Phase 2          │
│   │ Stage 6: speaker labels merged onto words             │                    │
│   └───────────────────────────────────────────────────────┘                    │
│                                       │                                        │
│   Stereo speaker-per-channel detected upstream → skip pyannote, use channels   │
│                                       │                                        │
│                                       ▼                                        │
│       data/dual_chunks_v2/{source}/{lang}/{shard}/{uuid}.json                  │
│           + S0.wav (other speaker masked to room-tone or -45dB noise)          │
│           + S1.wav                                                             │
│                                       │                                        │
│                                       ▼                                        │
│   DualStreamTokenizer.encode(user_audio, model_audio, 24000)                   │
│       Layout per 200ms window: [user_text(1), user_audio(20),                  │
│                                  model_text(1), model_audio(20)]               │
│       Stride=42                                                                │
│                                       │                                        │
│                                       ▼                                        │
│       data/tokens_v2_dual/{source}/{lang}/{shard}/file.npy + meta.json         │
│           stream_layout="dual", stride=42, speaker_segments populated          │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                      PHASE 5 — TRAINING WITH V2 DATA                           │
├────────────────────────────────────────────────────────────────────────────────┤
│   VoxtralDataset                                                               │
│     ├─ Reads .meta.json sidecar alongside .npy                                 │
│     ├─ Pinned val split via data/val_split_v2.json                             │
│     ├─ random.Random(seed) (no global state)                                   │
│     ├─ Stride homogeneity by stream_layout (sampler groups)                    │
│     ├─ Temperature sampling τ=3.3 across languages                             │
│     ├─ Fail-fast at >5% missing sidecars                                       │
│     └─ Yields {tokens, language, duration_s, source, valid_token_mask}         │
│                                       │                                        │
│                                       ▼                                        │
│   omni_trainer.py                                                              │
│     ├─ extract_codebook_targets (stride from sidecar, not hardcoded)           │
│     ├─ compute_omni_loss (zero loss on valid_token_mask=False positions)       │
│     ├─ Mean-across-ranks val reporting (AF-406 fix)                            │
│     └─ Combined temporal + depth CE loss                                       │
│                                       │                                        │
│                                       ▼                                        │
│   FRESH-INIT TRAINING (no checkpoint resume, user decision)                    │
│   4×A10G DDP. Target: WER ≤50% on 13 FLEURS langs within 5,000 steps.          │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────────────┐
│                              EVALUATION HARNESS                                │
├────────────────────────────────────────────────────────────────────────────────┤
│   scripts/eval_wer.py    (5 samples × 13 langs, runs every N checkpoints)      │
│   tests/test_no_gibberish.py    (Whisper LID + autocorrelation + F0 std)       │
│   tests/test_interruption_emission.py    (Goal 4: silence emission ≥70%)       │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Component responsibilities (one-line summaries)

| Component | Responsibility |
|---|---|
| `scripts/phase0_preflight.py` | Run 4 validation experiments before Phase 1 starts. Outputs gate downstream phases. |
| `VoxtralTokenizerConfig` | Configuration object. Default `sp_tokenizer_path` flips to SentencePiece. |
| `VoxtralTokenizer` | Single-stream encode. Per-call language + task. Forwards to Mimi + Whisper. |
| `TimedWhisperTokenizer` | Whisper transcription with per-call language + task. Bucketed text token output. |
| `MMSASR` (new) | ASR fallback wrapper for the 7 Indic languages Whisper-large-v3 doesn't support. |
| `DualStreamTokenizer` | Two-channel encode with stride 42 layout. Silence regions encoded as non-degenerate Mimi tokens. |
| `scripts/retokenize_v2.py` | Phase 1 driver. Walks source datasets, calls VoxtralTokenizer.encode, writes tokens + sidecars. |
| `scripts/vad_chunker.py` | Phase 2 driver. Silero VAD v6 per-channel. Variable-length chunks. Stereo correlation detector. |
| `scripts/detect_language.py` | Phase 3 driver. Batched LID + agreement check + streaming context for short clips. |
| `scripts/detect_music.py` | Phase 4 helper. Spectral flux + harmonicity + tempo. Outputs `music_likely` per source. |
| `scripts/diarize_v2.py` | Phase 4 driver. Demucs (gated) → Whisper → ctc-forced-aligner → Silero VAD → pyannote. |
| `VoxtralDataset` | Phase 5 data loader. Sidecar reader, pinned val split, temperature sampling, stride homogeneity. |
| `compute_omni_loss` | Loss function. Honors `valid_token_mask`. Mean-across-ranks aggregation. |
| `scripts/eval_wer.py` | WER + UTMOS evaluation. Runs on every checkpoint per language. |
| `tests/test_no_gibberish.py` | Regression test. Whisper LID + autocorrelation + F0 std. |
| `tests/test_interruption_emission.py` | Goal 4 eval harness. Synthetic dual-stream input, asserts silence emission ≥ 70%. |

## 4. Data flow contracts

### Source → token contract

| Source | Per-file inputs | Per-file outputs |
|---|---|---|
| FLEURS HF dataset | `audio.array`, `audio.sampling_rate`, `language` (from config), `transcription` | `tokens.npy`, `tokens.meta.json` |
| IndicVoices HF dataset | `audio_filepath.array`, `language` (from config) | same |
| YouTube `.m4a` | raw bytes + `<chunk>.lang.json` (from Phase 3) | same |

All three converge into the unified `data/tokens_v2/{source}/{lang}/{shard}/file.npy` layout. Training is source-agnostic — the `source` field in the sidecar is metadata only.

### Token → batch contract (Phase 5)

| Field | Type | Origin |
|---|---|---|
| `tokens` | `Tensor[(seq_len,)]` int64 | `.npy` file |
| `language` | `str` | `.meta.json[language]` |
| `duration_s` | `float` | `.meta.json[duration_s]` |
| `source` | `str` | `.meta.json[source]` |
| `valid_token_mask` | `Tensor[(seq_len,)]` bool | `.meta.json[valid_token_mask]` (default: all-True if missing) |
| `stream_layout` | `str` | `.meta.json[stream_layout]` ("single" or "dual") |

Sampler groups by `stream_layout` so a single batch never mixes stride-21 and stride-42 samples.

### Batch → loss contract

`compute_omni_loss(model, batch, config)` returns:

| Field | Type | Notes |
|---|---|---|
| `temporal_loss` | scalar | Mean across ranks. |
| `depth_loss` | scalar | Mean across ranks. |
| `total_loss` | scalar | `temporal_loss + depth_loss`. |
| `text_acc`, `audio_acc`, `depth_acc`, `depth_q0_acc`, `depth_q1_7_acc` | scalar | Per-rank computed, mean across ranks. |
| `text_acc_std`, ... | scalar | Std across ranks (audit AF-406). |

Loss contributions from positions where `valid_token_mask=False` are zeroed before reduction.

## 5. Storage layout

```
data/
├── chunks_indic_yt/          # legacy 20s YT chunks (input to Phase 1b)
├── chunks_v2/                # VAD-chunked variable-length (Phase 2 output)
│   └── yt/<lang>/<shard>/
│       ├── *.wav             # raw audio per speech region
│       ├── *.lang.json       # Phase 3 output: language + confidence
│       └── chunk.json        # source, channels, stream_role
├── chunks_v2_quarantine/     # rejected chunks (low LID confidence, disagree)
├── dual_chunks_v2/           # diarized speaker-masked pairs (Phase 4 output)
│   └── yt/<lang>/<shard>/
│       ├── *.json            # speaker segments + paths
│       ├── S0.wav
│       └── S1.wav
├── tokens_v2/                # Phase 1 tokens on existing 20s chunks
│   ├── fleurs/<lang>/<shard>/*.npy + .meta.json
│   ├── iv/<lang>/<shard>/*.npy + .meta.json
│   └── yt/<lang>/<shard>/*.npy + .meta.json
├── tokens_v2_chunked/        # Phase 2 tokens on VAD-chunked audio
├── tokens_v2_dual/           # Phase 4 dual-stream tokens
├── tokens_legacy/            # archived data/tokens/ (corrupted v1 output)
├── tokenizer/
│   └── omnivoxtral_sp.model  # 65K SentencePiece Unigram (existing, unchanged)
└── val_split_v2.json         # pinned validation file list (Phase 5 input)

logs/
└── archive/
    └── checkpoint_40800.pt.legacy   # archived corrupted-data checkpoint
```

Total disk budget: 200-500 GB during the rebuild. `tokens_legacy/` removed only after Phase 5 success criteria met.

## 6. Hardware allocation

| Resource | Allocation |
|---|---|
| 1×A10G | Preprocessing (serialized): Silero VAD → Demucs (opt-in) → mms-lid-2048 → faster-whisper → MMS-1B-all (fallback) → pyannote → Mimi. Peak ~15-18 GB VRAM. |
| 3×A10G | Training (DDP), continues during preprocessing. Switches to v2 data path at Phase 1a cutover. |
| 4×A10G | After preprocessing completes, full DDP for sustained training. |
| Disk | 200-500 GB on the same machine. Watchdog checks `df` before each preprocessing batch. |

## 7. Phase ordering and gates

```
Phase 0 ──► Phase 1a ──► (training resumes on FLEURS+IV)
   │              │
   │              ├──► Phase 1.5 (valid_token_mask in trainer)
   │              │
   │              └──► Phase 2 (re-chunk with VAD)
   │                      │
   │                      └──► Phase 3 (per-chunk LID + MMS routing)
   │                              │
   │                              └──► Phase 1b (re-tokenize YT with proper LID)
   │                                       │
   │                                       └──► Phase 4 (diarize YT + dual-stream)
   │                                                │
   │                                                └──► Phase 5 (full v2 training)
   │
   └─► Phase 0 outputs gate Phase 1 (SP injection), Phase 4 (silence strategy),
       Phase 3 (quarantine threshold)
```

Training is **never paused** for the entire rebuild. Phase 1a unblocks training on cleaner FLEURS+IV data within ~4 days. Phases 2-4 enrich the corpus while training runs in the background. Phase 5 represents full convergence.

## 8. Failure modes and circuit breakers

| Failure | Detection | Action |
|---|---|---|
| Phase 0 SP roundtrip test fails | Unit test in `test_phase0_preflight.py` | Switch language-tag injection to global prepend mode. |
| Phase 0 Mimi silence differentiation fails | Unit test | Phase 4 silence strategy switches to room-tone splice. |
| Phase 3 quarantine rate > 50% on pilot | `phase0_results.md` review | Loosen confidence threshold to 0.5, mandate streaming context for all chunks. |
| Sidecar missing on > 5% of files at training time | `VoxtralDataset` raises | Halt training. Audit preprocessing run. |
| Stride heterogeneity in batch | Shape mismatch in `extract_codebook_targets` | Sampler grouping by `stream_layout` prevents this; if it triggers, treat as a bug. |
| Disk fills during preprocessing | `run_safe.sh` watchdog | Pause preprocessing, alert. |
| Loss explodes during fresh-init training | Standard NaN check | Halt, dump samples, audit recent v2 sidecars for corruption. |
| WER regression > 80% on `eval_wer.py` | `test_no_gibberish.py` | Block checkpoint promotion. |

## 9. Reference implementations

- **whisper-diarization** (https://github.com/MahmoudAshraf97/whisper-diarization) — Phase 4 stage ordering. Substitute Silero for MarbleNet to match Phase 2.
- **antirez/voxtral.c** (https://github.com/antirez/voxtral.c) — Streaming pattern. `vox_stream_flush` reference for short-clip context accumulation in Phase 3.
- **Voxtral-Mini-4B-Realtime-2602** — 80ms / 12.5 Hz token grid (matches Mimi). `default_num_delay_tokens=6` reference for streaming alignment.
- **Silero VAD v6** — Phases 2 and 4 VAD.
- **Moshi paper §4** — Inner monologue + dual-stream layout.

## 10. Out-of-scope (Phase 6 follow-ups)

Architectural model changes deferred:
- MPD (Masked Parallel Depth) — eliminates AR error chain.
- Embedding-norm rescaling — addresses 3.92× speech-vs-text norm mismatch.
- Semantic distillation from Whisper-tiny on q0 (DualCodec recipe).
- FSQ codec swap — eliminates depth transformer.
- IndicSuperTokenizer 200K vocab expansion.
- 22-language coverage (full IndicVoices-R 1,704 hours).

These are tracked in `experiment_queue.md` and remain Phase 6 only.
