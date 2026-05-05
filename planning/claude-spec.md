# OmniVoxtral Training Pipeline Rebuild — Synthesized Specification

This is the consolidated spec combining: (a) the original problem statement and 5 defects, (b) external research findings on rebuild components, and (c) the interview answers locking in architectural decisions.

---

## 1. Problem statement

After 528 logged experiments, OmniVoxtral reaches val_loss=1.489 (text_acc=97%, depth_acc=66%) on teacher-forced one-step prediction, but **autoregressive generation produces repetitive Hindi-shaped gibberish** ("आपाशाशा…"). Whisper auto-detects the model's own output as Swedish. The architecture is sound; the training data labels are corrupt.

A first-principles audit found five compounding defects in the data pipeline that silently destroyed the text stream and erased the prosody/interruption signal Mimi was supposed to preserve. This spec defines a ground-up rebuild with a phased delivery so training can resume on cleaner data within days, not weeks.

## 2. Goals (with concrete metrics)

- **Eliminate gibberish.** Whisper-large-v3 WER on generated Hindi audio drops below 50% (currently >80%).
- **Preserve prosody.** F0 correlation between generated and reference > 0.85; UTMOS > 2.5.
- **Learn interruptions.** Model emits silence tokens during user-stream speech in dual-stream eval >70% of the time.
- **Make the pipeline auditable.** Every token file has a paired metadata sidecar.
- **Phase delivery so training can resume on the cleanest available data within ~1 week, full corpus within ~1 month.**

## 3. Non-goals

- Architectural changes to the model itself (no MPD, no FSQ codec swap, no language adapters in this rewrite — gated behind Phase 6 follow-up).
- Inference-side fixes (repetition penalty, RAS sampling, embedding norm rescaling) — tracked separately in `experiment_queue.md` as Phase NOW.
- 22-language coverage in v2.0 — start with the 13 languages we already have FLEURS data for; expand as IndicVoices-R is processed.

## 4. Architectural decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| **Inner monologue language** | Option A — source-language text drives audio. English translation stored as sidecar metadata only. | Preserves Moshi's causal text-leads-audio alignment. Enables Indic-Indic duplex. Matches Voxtral 80ms tokenization. |
| **Existing 57,668 corrupted YT tokens** | Discard, re-tokenize from `data/chunks_indic_yt/`. | Cleanest training signal. Migration cost ~230 GPU-hr is acceptable. |
| **License posture** | Research / non-commercial. MMS is fine. | Unlocks all 22 Indic languages via MMS-1B-all + mms-lid-2048. |
| **First phase to ship** | Phase 1: tokenizer fixes only (defects 2, 3, 5). | Re-tokenize 20s chunks WITHOUT re-chunking. Training resumes in days. |
| **Primary ASR** | faster-whisper-large-v3. | Battle-tested for Indic; built-in Silero VAD; ~3GB fp16; use OpenAI Whisper for translation pass. |
| **Demucs default** | Off, opt-in flag. | Adds 3-5 min/hr preprocessing; over-strips music in some cases. Per-source toggle. |
| **Storage layout** | `data/tokens_v2/{source}/{lang}/{shard}/file.npy` + `.meta.json` sidecar. | Source-aware, language-aware. Old `data/tokens/` untouched until cutover. |
| **Test framework** | pytest + integration tests on every new component. | `tests/` directory, fixtures for synthetic audio. Each new module ships with unit + 1 integration test. |
| **Mimi handling** | Treat as mono. Stereo audio → two separate Mimi-mono streams (Moshi pattern). | Released `kyutai/mimi` checkpoint is mono-trained. Architecture supports stereo but no public weights. |
| **VAD** | Silero v6 per-channel. `min_speech_duration_ms=1000, min_silence_duration_ms=500, speech_pad_ms=150, max_speech_duration_s=20`. | ROC-AUC 0.97 vs WebRTC's 0.73. Multilingual-trained. CPU-only inference. |
| **LID** | mms-lid-2048 primary; Whisper-large-v3 LID as confidence tie-breaker. | MMS covers all 22 Indic; Whisper validates the 15 it supports. Reject if both disagree or top-1 prob < 0.7. |
| **ASR fallback for 7 Whisper-unsupported Indic** | MMS-1B-all with per-language adapters. | Bodo, Dogri, Konkani, Manipuri, Santali, Sindhi, Kashmiri. |
| **Text tokenizer** | Existing `data/tokenizer/omnivoxtral_sp.model` (65K SP Unigram, fertility 1.91 for Indic). No retraining. | Already trained and benchmarked. Vocab expansion is a Phase 6 architectural change. |
| **Language tag injection** | Moshi-style per-frame prefix: `<lang_xx>` emitted in first text-frame of each utterance. | Simpler than parallel language stream. Code-switching handled by emitting fresh tag at boundary. |

## 5. The 5 defects and the Phase that fixes each

| # | Defect | Fix | Phase |
|---|---|---|---|
| 1 | Whisper language hardcoded at construction time, no per-file detection | Per-call language API + LID stage (mms-lid-2048 + Whisper LID tie-breaker) | Phase 3 |
| 2 | Whisper transcripts re-tokenized with Mistral BPE for Indic | Default `sp_tokenizer_path` = `data/tokenizer/omnivoxtral_sp.model` | Phase 1 |
| 3 | `task="translate"` never an option | `task` becomes per-call argument; translate runs as second pass into sidecar | Phase 1 |
| 4 | 20s hard chunking + zero padding + stereo→mono mean + single-stream tokenization on conversational audio | VAD-aware variable-length chunking (Phase 2), dual-channel + diarization-driven dual-stream tokenization (Phase 4) | Phases 2, 4 |
| 5 | No metadata sidecars | JSON sidecar per `.npy` file with language, confidence, transcript, translation, num_speakers, duration, source, etc. | Phase 1 |

## 6. Phase plan

### Phase 1 — Tokenizer fixes only (Defects 2, 3, 5)

**Scope:** Make the existing 20-second chunked tokenization correct. No re-chunking.

**Deliverables:**
- `VoxtralTokenizerConfig.sp_tokenizer_path` defaults to `data/tokenizer/omnivoxtral_sp.model`. Mistral BPE remains opt-in.
- `TimedWhisperTokenizer.forward(audio, sample_rate, language=None, task="transcribe")` — per-call language and task.
- `TimedWhisperTokenizer.translate(audio, ...)` — second-pass English translation for sidecar.
- `_save_tokens()` writes paired `<file>.meta.json` with the schema in §7.
- `VoxtralTokenizer.encode(x, sample_rate, language=None, task="transcribe", source_metadata=None)` — forwards everything to TimedWhisperTokenizer.
- `tests/test_tokenizer_v2.py` — pytest unit + integration coverage.

**Re-tokenization run:** Re-tokenize FLEURS + IndicVoices + the existing YouTube chunks (corrupted text discarded). Output to `data/tokens_v2/{fleurs,iv,yt}/{lang}/{shard}/`.

**Training cutover:** Update `VoxtralTrainConfig.data_path` to `./data/tokens_v2`. DDP training resumes within ~1 week.

**Estimated cost:** ~200 GPU-hr (single A10G, batched).

### Phase 2 — VAD-aware variable-length chunking (Defect 4 partial)

**Scope:** Replace 20s hard cuts with VAD-driven variable-length chunks. Eliminate zero padding.

**Deliverables:**
- `scripts/vad_chunker.py` — Silero VAD per-channel, produces variable-length chunks at `data/chunks_v2/{source}/{lang}/`.
- `AudioChunkDataset` updated to load variable-length audio (no padding; data loader bucketed by length).
- `VoxtralDataset` already supports packing; verify behavior.
- Pre-roll/post-roll 150ms preserved at chunk boundaries.
- Stereo source preserved through to tokenization (correlation-based detection of speaker-per-channel).
- `tests/test_vad_chunker.py` — synthetic audio with known silence boundaries verifies output count + duration distribution.

**Re-chunking run:** Process the 1,845 source URLs through the v2 chunker. New chunks under `data/chunks_v2/yt/`.

**Re-tokenization run:** Phase 1 tokenizer applied to v2 chunks. Output to `data/tokens_v2_chunked/`.

**Estimated cost:** ~50 GPU-hr (single A10G).

### Phase 3 — Per-chunk LID + streaming Whisper fallback (Defect 1)

**Scope:** Detect language per chunk; reject low-confidence; streaming-Whisper for short clips.

**Deliverables:**
- `scripts/detect_language.py` — primary mms-lid-2048, secondary Whisper LID, agreement-based confidence; writes `<chunk>.lang.json`.
- `TimedWhisperTokenizer` integration: read `<chunk>.lang.json` if present; else run inline LID.
- Streaming Whisper wrapper: for chunks <4s, accumulate ±2s neighbor context before LID + transcription.
- `MMS-1B-all` ASR adapter loader for the 7 Whisper-unsupported Indic languages.
- `tests/test_language_detection.py` — verifies per-language routing, confidence rejection, streaming fallback.

**Estimated cost:** ~30 GPU-hr (single A10G).

### Phase 4 — Diarization + dual-stream tokenization on YouTube corpus (Defect 4 full)

**Scope:** Wire conversational audio through the dual-stream path so the model can learn interruptions.

**Deliverables:**
- `scripts/diarize_v2.py` — orchestrates Demucs (opt-in) → faster-whisper transcription → ctc-forced-aligner → MarbleNet VAD → pyannote-3.1 speaker embeddings.
- Stereo channel correlation detector — when channels are speaker-separated, skip pyannote and use channel-as-speaker.
- Updated `DualStreamTokenizer` consuming diarized output with proper turn-taking semantics; silence between turns encoded as a learnable "listening" signal (not zero-padded).
- `tests/test_dual_stream_pipeline.py` — synthetic two-speaker audio with known turn boundaries; verifies dual-stream token layout and silence encoding.

**Tokenization run:** All YouTube chunks. Output to `data/tokens_v2_dual/yt/{lang}/{shard}/`.

**Estimated cost:** ~36 GPU-hr diarization + part of Phase 1's Whisper budget.

### Phase 5 — Wire everything into `train_omni.py`

**Scope:** Training loop reads v2 tokens with sidecars; per-language temperature sampling; dual-stream toggle; training resumes on clean data.

**Deliverables:**
- `VoxtralDataset` reads `<file>.meta.json` alongside `.npy`; emits language tag in batch metadata.
- Temperature sampling τ=3.3 for language balancing across batches.
- Per-rank val reporting fixed (mean across ranks, not best — see audit AF-406).
- Training resumes from `logs/68g46p5g/checkpoint_40800.pt` (best focal-loss checkpoint) on v2 data.
- WER eval pipeline (`scripts/eval_wer.py`) generates 20 samples per language and reports Whisper WER + UTMOS.

**Estimated cost:** Engineering only; training cost is ongoing.

### Phase 6 — Architectural follow-ups (DEFERRED)

Out of scope for this rebuild. Tracked in `experiment_queue.md`:
- MPD (masked parallel depth) — eliminates AR error chain.
- Embedding norm rescaling fix — addresses 3.92× speech vs text norm.
- Semantic distillation from Whisper-tiny on q0.
- FSQ codec exploration — replace Mimi with TAAE-style FSQ.
- Vocab expansion to 200K (IndicSuperTokenizer recipe).
- 22-language coverage (full IndicVoices-R 1704 hours).

## 7. Metadata sidecar schema

For every `<file>.npy`, write `<file>.meta.json`:

```json
{
  "schema_version": 2,
  "preprocessing_version": "v2.0",
  "preprocessing_run_id": "<uuid>",
  "preprocessing_timestamp": "2026-05-05T14:32:00Z",

  "source": "youtube",
  "source_id": "<youtube-video-id>",
  "source_url": "https://www.youtube.com/watch?v=...",
  "chunk_index": 12,

  "language": "hi",
  "language_confidence": 0.94,
  "language_method": "mms_lid_2048",
  "language_secondary": {"language": "hi", "confidence": 0.91, "method": "whisper_lid"},

  "transcript": "आज मौसम बहुत अच्छा है",
  "translation_en": "The weather is very good today",
  "transcript_method": "faster_whisper_large_v3",
  "transcript_avg_logprob": -0.32,

  "duration_s": 12.4,
  "sample_rate": 24000,
  "num_channels": 1,
  "num_speakers": 2,
  "speaker_segments": [
    {"speaker": "S0", "start": 0.0, "end": 3.2},
    {"speaker": "S1", "start": 3.2, "end": 7.8}
  ],

  "snr_db": 18.5,
  "speech_ratio": 0.78,
  "clip_ratio": 0.001,

  "stream_layout": "single",
  "tokenizer_config": {
    "sp_model": "omnivoxtral_sp.model",
    "text_hz": 5,
    "mimi_num_quantizers": 8,
    "stride": 21
  },
  "token_count": 505,
  "token_range": [0, 81920]
}
```

For dual-stream files, `stream_layout = "dual"`, `stride = 42`, and `speaker_segments` informs which speaker is on which stream.

## 8. Hardware budget

**Preprocessing (single A10G, sequential):**

| Stage | VRAM | Time per 1hr audio |
|---|---|---|
| Silero VAD | <100 MB CPU | ~1 min |
| Demucs (opt-in) | ~2 GB | 3-5 min |
| mms-lid-2048 | ~4 GB fp16 | ~2 min |
| faster-whisper-large-v3 | ~3 GB fp16 | 2-4 min |
| MMS-1B-all (fallback) | ~4 GB fp16 + adapters | ~5 min |
| pyannote-3.1 | ~2 GB | ~2 min |
| ctc-forced-aligner | ~1 GB | ~1 min |
| Mimi encode | ~1 GB | ~2 min |

Stages serialized → peak ~15-18 GB. Run preprocessing on a single A10G, training continues on the other 3 in DDP.

**Training:** existing 4×A10G DDP setup. No change.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Re-tokenization budget (~200 GPU-hr) blocks training | Phase 1 ships first on existing chunks → training resumes in ~1 week. Phase 2 (re-chunking) runs in background. |
| MMS LID disagrees with Whisper LID frequently | Reject (or quarantine) chunks where confidence < 0.7 OR LID disagree → flag for manual review. Track rejection rate in dashboards. |
| Pyannote-3.1 underperforms on Indic | Stereo correlation fallback (channel-as-speaker). Quarantine chunks with single-speaker-detected audio for monolingual single-stream training. |
| Demucs over-strips music in sung-language data | Default off. Per-source flag. |
| Variable-length packing breaks DDP gradient sync | Existing `make_dataloader` packs sequences; verify with synthetic test before scaling up. |
| Sidecar schema drift between phases | `schema_version` field; loader rejects unknown schemas with explicit error. |
| Disk space (~57K chunks × 2 paths × 50KB) | Old `data/tokens_yt/` quarantined to `data/tokens_yt_corrupted/`, deleted after v2 cutover and audit. |

## 10. Reference repositories and prior art

- `https://github.com/MahmoudAshraf97/whisper-diarization` — joint timed-Whisper + diarization (stage ordering blueprint).
- `https://github.com/antirez/voxtral.c` — minimal C streaming Voxtral (streaming pattern reference).
- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602` — streaming Whisper alternative (80ms / 12.5 Hz token grid match Mimi).
- `https://huggingface.co/mistralai/Voxtral-Mini-3B-2507` — Mistral audio LLM (built-in translation; potential Phase 6 swap).
- `https://github.com/snakers4/silero-vad` — multilingual VAD.
- `https://huggingface.co/facebook/mms-lid-2048` — primary LID for 22 Indic.
- `https://huggingface.co/facebook/mms-1b-all` — ASR fallback for 7 Indic Whisper doesn't support.
- `https://github.com/SYSTRAN/faster-whisper` — primary ASR.
- Moshi paper — `https://arxiv.org/html/2410.00037v2` — Inner Monologue + dual-stream training.
- IndicSuperTokenizer — `https://arxiv.org/html/2511.03237v1` — Phase 6 vocab expansion reference.

## 11. Output deliverables

Three files in `planning/`:

- `plan.md` — narrative plan (renamed from `claude-plan.md` at workflow end).
- `HLD.md` — high-level architectural design (data flow diagrams, component boundaries, phase boundaries).
- `LLD.md` — low-level design (file-by-file changes, function signatures, data schemas, dependencies, runtime contracts).

Workflow scaffolding (`claude-research.md`, `claude-interview.md`, `claude-spec.md`, `claude-plan-tdd.md`, `sections/`) remains in the planning directory for traceability.
