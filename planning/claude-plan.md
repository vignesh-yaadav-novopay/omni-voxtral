# OmniVoxtral Training Pipeline Rebuild — Implementation Plan

## 0. What this plan is for

This document defines the ground-up rebuild of the OmniVoxtral training data pipeline. The current pipeline produces tokens that look fine in aggregate (val_loss=1.489, text_acc=97%, depth_acc=66%) but cause autoregressive generation to fail catastrophically — the model emits repetitive Hindi-shaped gibberish. A first-principles audit traced the failure to five compounding defects in how audio is converted into training tokens. This plan describes how to fix all five, phased so training can resume on cleaner data within ~1 week and the full corpus is rebuilt within ~1 month.

The audience for this plan is an engineer (or `deep-implement` agent) with no prior context on the project. Read it top to bottom; you should not need to consult the spec or research to implement.

## 1. Background — the bug, in one paragraph

OmniVoxtral is a Moshi-style dual-transformer speech LM. The temporal transformer is a layer-pruned, LoRA-finetuned Mistral-7B; the depth transformer is a small autoregressive predictor over Mimi codec codebooks. Audio is encoded with Mimi (24 kHz, 8 quantizers, 12.5 Hz frame rate) and text comes from Whisper-large-v3 in a parallel stream interleaved at stride 21 (1 text token + 20 audio tokens per 200 ms window). The model trains on this interleaved sequence with weighted cross-entropy. Training looked successful but audio generation produced repetitive gibberish because the text labels in training were silently corrupted: Whisper was hardcoded to language="en" and ran on Indic audio with no per-file detection; the resulting transcripts were re-tokenized with Mistral BPE (fertility ~10.8 for Devanagari) and then truncated to 5 tokens per second; conversational audio was tokenized as if it were single-speaker monologue; 20-second hard chunking severed prosody contours; zero-padding on short clips taught the model to emit silent loops; and no metadata persisted so none of this was auditable. Mimi itself was the only correctly-functioning component.

## 2. Goals (concrete, measurable)

| Goal | Metric | Target |
|---|---|---|
| Eliminate gibberish | Whisper-large-v3 WER on generated Hindi audio | < 50% (currently > 80%) |
| Preserve prosody | F0 correlation generated vs reference | > 0.85 |
| Speech naturalness | UTMOS | > 2.5 |
| Learn interruptions | Silence-token emission during user-stream speech (dual-stream eval) | > 70% |
| Pipeline auditability | Token files with paired metadata sidecars | 100% |
| Phased delivery — clean text on FLEURS+IV | Training resumes on Phase 1a output | ≤ 4 days |
| Phased delivery — YouTube re-tokenized end-to-end | Phase 3 LID + Phase 1b retokenization complete | ≤ 3 weeks |
| Full corpus rebuilt with dual-stream | Phase 4 dual-stream tokens for YouTube available | ≤ 5 weeks |

(Original Goal 6 of "≤ 1 week" was unrealistic — the translation pass plus YouTube LID gap make Phase 1 a 12-16 day operation. Honest timeline above.)

## 3. Non-goals (deferred to Phase 6)

- Architectural changes to the model: MPD (masked parallel depth), FSQ codec swap, semantic distillation, embedding-norm rescaling, language adapters.
- Inference-side fixes: repetition-aware sampling, temperature stratification, KV caching for depth.
- 22-language coverage. Start with the 13 FLEURS languages we already have; expand as IndicVoices-R is processed.
- Tokenizer retraining. The existing `data/tokenizer/omnivoxtral_sp.model` (65K SentencePiece Unigram, fertility 1.91 for Indic) is canonical for this rebuild. Vocab expansion is Phase 6.

## 4. Architectural decisions (locked)

These are the decisions every phase depends on. Do not relitigate during implementation.

| Decision | Choice |
|---|---|
| Inner monologue language | **Option A** — text tokens carry source-language transcript. English translation is sidecar metadata only. |
| Existing 57,668 corrupted YT tokens | Discard, re-tokenize from `data/chunks_indic_yt/`. |
| License posture | Research / non-commercial. MMS is fine. |
| First phase to ship | Phase 1 (tokenizer fixes), no re-chunking. Training resumes on existing 20s chunks with corrected text. |
| Primary ASR | `faster-whisper-large-v3` (built-in Silero VAD, ~3 GB fp16). |
| Translation pass | OpenAI `whisper-large-v3` with `task="translate"` in a separate call. Sidecar only. |
| Demucs source separation | Off by default. Per-source opt-in flag. |
| Storage layout | `data/tokens_v2/{source}/{lang}/{shard}/file.npy` + paired `.meta.json`. |
| Test framework | pytest with synthetic-audio fixtures. New `tests/` directory. |
| Mimi handling | Treat as mono. Stereo → two separate Mimi-mono streams (Moshi pattern). |
| VAD | Silero v6 per-channel. `min_speech_duration_ms=300, min_silence_duration_ms=500, speech_pad_ms=150, max_speech_duration_s=20`. (Lowered from 1000 ms after review — 1 s floor erased Indic backchannels critical for Goal 4.) |
| LID | `mms-lid-2048` primary, Whisper-large-v3 LID as confidence tie-breaker. Reject if disagree or top-1 < 0.7. |
| ASR fallback for 7 unsupported Indic | `MMS-1B-all` with per-language adapters (Bodo, Dogri, Konkani, Manipuri, Santali, Sindhi, Kashmiri). |
| Text tokenizer | Existing `omnivoxtral_sp.model`. Default `sp_tokenizer_path` flips from `None` to that path. |
| Language tag injection | Moshi-style: `<lang_xx>` emitted in first text-frame of each utterance. Code-switching handled with fresh tag at boundary. |

## 5. The five defects mapped to phases

| # | Defect | Fixed in |
|---|---|---|
| 1 | Whisper language hardcoded at construction | Phase 3 (LID + streaming Whisper for short clips) |
| 2 | Mistral BPE re-tokenization (fertility 10.8 for Indic) | Phase 1 (default to SentencePiece) |
| 3 | `task="translate"` not exposed | Phase 1 (per-call task argument + translation sidecar) |
| 4 | 20s hard chunking + zero pad + stereo→mono mean + single-stream on conversational audio | Phase 2 (VAD chunking) + Phase 4 (dual-stream + diarization) |
| 5 | No metadata sidecars | Phase 1 (sidecar writer) |

## 6. Phase-by-phase plan

### Phase 0 — Pre-flight validation (1 day)

**Why first.** Three of the locked architectural decisions rest on assumptions that have not been empirically tested. Phase 0 runs four small experiments before any Phase 1 code is written. Each Phase 1+ task is gated on a Phase 0 result.

**Experiments.**

1. **SentencePiece `<lang_xx>` roundtrip.** Encode `"<lang_hi>आज मौसम बहुत अच्छा है"` through `data/tokenizer/omnivoxtral_sp.model`. Verify `<lang_hi>` becomes a single user-defined-symbol token ID (in [4, 26]) and not a byte-fallback sequence. **Gate:** if byte-fallback, language-tag injection switches from per-utterance prefix to global prepend (one slot total at sequence start).

2. **Mimi silence differentiation.** Generate 5 s of (a) bit-zero silence, (b) -45 dB white noise, (c) -45 dB pink noise, (d) real room-tone sample from a quiet FLEURS region. Encode all four through Mimi at 8 quantizers. Verify q1 (semantic) codebook IDs differ across the four conditions. **Gate:** if indistinguishable from bit-zero, Phase 4 silence strategy switches from "low-energy noise floor" to "real room-tone splice."

3. ~~Resume gradient sanity~~ **Dropped** — user locked fresh-init. No checkpoint resume planned.

4. **200-chunk LID rejection pilot.** Random sample 200 chunks from `data/chunks_indic_yt/`. Run mms-lid-2048 + Whisper LID through the proposed Phase 3 pipeline. Measure: agreement rate, low-confidence rate (top-1 < 0.7), total quarantine rate. **Gate:** if quarantine rate > 50%, Phase 3 thresholds are loosened OR a second-pass LID with neighbor-context is mandated for all chunks (not just <4 s).

**Output.** `planning/phase0_results.md` with all four experiment outcomes. Phase 1 cannot start until Phase 0 outputs exist and gates are evaluated.

---

### Phase 1 — Tokenizer fixes (Defects 2, 3, 5)

**Why second.** Phase 1 fixes the biggest single source of training-label corruption (Indic transcripts being clipped to 5 byte-level Mistral pieces) without re-chunking audio. The other defects compound the problem but Phase 1 alone is expected to move depth_acc and gibberish behavior measurably.

**Phase 1 splits into 1a and 1b.**

- **Phase 1a (FLEURS + IndicVoices, ~4 days, ~80 GPU-hr).** Both datasets carry per-sample language labels in their HF metadata, so `encode(language=lang)` works without LID. Ships first; training resumes on clean text within ~4 days.
- **Phase 1b (YouTube, ~10 days, ~200 GPU-hr).** Gated on Phase 3 LID landing — YouTube has no per-file language source. Phase 1b retokenizes YT chunks using the language tags Phase 3 produces.

**Components changed.**

`src/voxtral/tokenizer/model.py`
- `VoxtralTokenizerConfig`: change default `sp_tokenizer_path` from `None` to `"data/tokenizer/omnivoxtral_sp.model"`. Mistral BPE remains opt-in via explicit `sp_tokenizer_path=None`.
- `VoxtralTokenizer.encode(x, sample_rate, language=None, task="transcribe", source_metadata=None)`: add three new arguments. `language` is forwarded to Whisper. `task` selects transcribe vs translate. `source_metadata` is an optional dict that flows through to the sidecar writer (URL, source_id, chunk_index).
- New method `VoxtralTokenizer.translate(x, sample_rate, language=None)`: convenience wrapper that calls `encode(... task="translate")` and returns only the text token sequence (no Mimi pass).

`src/voxtral/tokenizer/word_level_whisper.py`
- `TimedWhisperTokenizer.forward(audio, sample_rate, language=None, task="transcribe")`: language and task become per-call arguments. The `__init__` still accepts `language` as a default, but per-call wins.
- `generate_tokens(processor, model, audio, language=None, task="transcribe")`: always forward both `language` and `task` to `model.generate()`. Drop the conditional `if language != "en"` block — it was the source of Defect 1's silent fallback.
- New method `TimedWhisperTokenizer.translate(audio, sample_rate, language=None)`: a second-pass call with `task="translate"` returning a plain Python string (not bucketed, not tokenized — destined for sidecar metadata).

`src/voxtral/data/preprocessing.py` and `scripts/preprocess_hf.py`
- Tokenizer instantiation no longer hardcodes `language="en"`. `VoxtralTokenizerConfig()` is built without a language; per-file language is passed at `encode()` time.
- New helper `write_metadata_sidecar(npy_path, metadata)` writes a paired `<file>.meta.json` to the schema in §7. Called immediately after `_save_tokens()`.
- `preprocess_dataset()` in `preprocess_hf.py` already knows which language each sample comes from (via the FLEURS / IndicVoices `lang` argument). Pipe that into the new `encode(..., language=lang)` call. Run a second `tokenizer.translate()` pass to populate `translation_en` in the sidecar.
- Output path changes from `data/tokens/{lang}/...` to `data/tokens_v2/{source}/{lang}/{shard}/...`. `source` ∈ `{fleurs, iv, yt, ...}`.

`src/voxtral/trainer/data.py`
- `VoxtralDataset` extended to read `<file>.meta.json` alongside `<file>.npy`. The sidecar populates a `language` field on each yielded sample. Missing sidecar → tag as `lang="unknown"`, log a warning.
- The yielded dict becomes `{"tokens": Tensor, "language": str, "duration_s": float, "source": str}`. Existing code that expects only `tokens` keeps working.

`tests/test_tokenizer_v2.py` (new)
- Synthetic 1-second 24 kHz Hindi audio fixture (use existing FLEURS sample via `pytest.fixture`).
- Unit: `VoxtralTokenizer.encode(..., language="hi", task="transcribe")` produces tokens whose text portion decodes to non-empty Devanagari.
- Unit: `VoxtralTokenizer.encode(..., language="hi", task="translate")` produces tokens whose text portion decodes to English Latin script.
- Unit: missing language argument falls back to `__init__` default.
- Integration: full pipeline FLEURS file → encode → save → reload → matches expected shape.
- Integration: sidecar `.meta.json` is written and contains `language`, `transcript`, `translation_en`, `source`, `duration_s`.

**Re-tokenization run.**

A new `scripts/retokenize_v2.py` walks `data/chunks_indic_yt/`, `data/tokens_indicvoices/` source files, and FLEURS samples, and re-tokenizes through the new pipeline into `data/tokens_v2/{source}/{lang}/`. The old `data/tokens/` directory is renamed to `data/tokens_legacy/` and removed only after the v2 cutover is verified.

**Translation pass strategy (revised from review).** The English-translation sidecar field is a hidden ~200 GPU-hr cost if run on every chunk. Make translation **lazy**:
- `tokenizer.translate()` runs on a sampled 10% of Phase 1a chunks initially (sidecar `translation_en` allowed to be `null`).
- Background job fills the remaining 90% over weeks, opportunistically when training GPU is idle.
- Phase 5 evaluation does not block on full translation coverage.

**Cost.** Phase 1a (FLEURS + IV transcribe + 10% translate sample) ≈ 80 GPU-hr on a single A10G. Phase 1b (YouTube transcribe + 10% translate, post-Phase 3) ≈ 200 GPU-hr.

**Training cutover.**

`VoxtralTrainConfig.data_path` flips from `./data/tokens` to `./data/tokens_v2`. A small adapter in `VoxtralDataset` handles the new layout (recursive glob over `{source}/{lang}/{shard}/*.npy` works without changes).

**Fresh-init only — no checkpoint resume (user decision).** All prior training (528 experiments, checkpoint_40800.pt) was on corrupted text labels. The model has memorized a wrong-language → wrong-text → autoregressive-loop manifold. Resuming from that checkpoint risks anchoring the new run to the corrupted basin. **Decision locked: train from scratch on v2 data.** The 40,800 prior steps are sunk cost; the corrupted checkpoint is archived to `logs/archive/checkpoint_40800.pt.legacy` and not used as a starting point.

This means:
- Phase 1a cutover starts a fresh-init training run on v2 data with the same architecture (Mistral-7B pruned + LoRA + Depth Transformer 4L).
- Initial training will take longer to reach val_loss < 1.5 than a resume would have — budget ~5,000-10,000 fresh-init steps to match prior corrupted-data benchmarks.
- The Phase 0 "resume gradient sanity" pre-flight (experiment 3) is dropped since we no longer resume.
- Loss explosion is no longer a risk on cutover; the model starts at random init regardless.

### Phase 1.5 — Valid-token mask in sidecar

**Why this exists.** Phase 1 produces tokens on the existing 20-second hard chunks, which include zero-padded silence at the end of files shorter than 20 s. Training on these still teaches the silent-loop failure mode. Re-chunking is Phase 2 (~10 days). The interim fix:

- `_save_tokens()` emits a `valid_token_mask` field in the sidecar — a bit array marking which token positions correspond to real audio vs zero-pad.
- `compute_omni_loss` ignores positions where `valid_token_mask=False` (loss is zeroed for those positions, gradient masked).
- Cost: ~50 lines in `omni_trainer.py`. Implemented during Phase 1a so it's live at cutover.

This decouples the silent-loop fix from the re-chunking timeline.

### Phase 2 — VAD-aware variable-length chunking (Defect 4 partial)

**Why second.** With clean text from Phase 1, the next biggest training-signal corruption is the 20-second hard chunker plus zero-padding. These together teach the model that "emit silence forever" is a normal continuation. VAD-aware variable-length chunking removes both, and it's a self-contained change that doesn't require touching the model.

**Components added/changed.**

`scripts/vad_chunker.py` (new)
- Loads source audio from `data/chunks_indic_yt/` (or any source directory).
- Detects whether input is stereo by reading channel count without auto-mixing.
- For each channel, runs Silero VAD v6 with the parameters in §4 (`min_speech_duration_ms=300` to capture Indic backchannels). For mono input, runs once.
- Emits one chunk per detected speech region, with 150 ms speech pad on either side.
- Writes chunks to `data/chunks_v2/{source}/{lang_or_unknown}/{shard}/{uuid}.wav` at 24 kHz mono per channel.
- For stereo input, computes inter-channel correlation. If correlation < 0.5, it's likely speaker-per-channel — emit two chunks per detected region (one per channel) and tag both with `stream_role` ∈ `{user, model}` in a small `chunk.json` next to each `.wav`. Otherwise emit a single mixed chunk.
- Skips chunks shorter than `min_speech_duration_ms=1000`.

`src/voxtral/data/preprocessing.py`
- `AudioChunkDataset` updated to handle variable-length audio. Padding to fixed length is removed entirely. Chunks shorter than the minimum are skipped.
- `VoxtralDataset` (already an `IterableDataset`) uses bucketed batching by length to keep per-batch padding minimal.
- Make sure `make_dataloader` in `prepare.py` (in `autoresearch/`) is left alone — it's the upstream Karpathy loader and is read-only by contract.

`tests/test_vad_chunker.py` (new)
- Synthetic audio: 30 seconds with three known speech regions (5 s, 8 s, 4 s) separated by 2-second silences.
- Verify: VAD produces three chunks with durations ≈ matching, with 150 ms padding on each side.
- Verify: stereo with speaker-per-channel produces two chunks per region (one per channel) tagged with stream_role.
- Verify: stereo with speaker-mixed produces one chunk per region.

**Re-chunking run.** Reprocess all 1,845 source URLs through `vad_chunker.py`. Output to `data/chunks_v2/yt/`. Existing `data/chunks_indic_yt/` is preserved as fallback.

**Re-tokenization run.** Phase 1 tokenizer is then run on the v2 chunks to produce `data/tokens_v2_chunked/`. Final training data path becomes `data/tokens_v2_chunked` once Phase 2 completes.

**Cost.** ~50 GPU-hours (CPU-bound Silero on top of the GPU re-tokenization).

### Phase 3 — Per-chunk LID + streaming Whisper fallback (Defect 1)

**Why third.** With clean text and prosody-preserving chunks, the remaining text-side corruption is wrong-language transcription. Phase 3 wires in a real LID and a fallback path for clips too short for confident detection.

**Components added/changed.**

`scripts/detect_language.py` (new)
- Walks `data/chunks_v2/{source}/{lang_or_unknown}/{shard}/*.wav`.
- For each file, runs `mms-lid-2048` (primary) and `whisper-large-v3.detect_language()` (secondary).
- For chunks shorter than 4 seconds, builds a streaming context window: concatenates ±2 seconds of neighboring audio from the same source video before running LID.
- Writes `<file>.lang.json` with `{language, confidence, method, secondary, agreement}` to the chunk directory.
- Skips chunks where confidence < 0.7 OR primary and secondary disagree → moves them to `data/chunks_v2_quarantine/`.

`src/voxtral/tokenizer/word_level_whisper.py`
- `TimedWhisperTokenizer` reads `<file>.lang.json` if present; otherwise calls inline LID. Removes the dependency on per-file language being passed by the caller (Phase 1 made it possible; Phase 3 makes it automatic).
- New helper `transcribe_with_streaming_context(audio, neighbor_audio_before, neighbor_audio_after, language)` for short clips — accumulates context for transcription as well.

`src/voxtral/tokenizer/mms_asr.py` (new)
- Lightweight wrapper around `facebook/mms-1b-all` for ASR fallback on the 7 Whisper-unsupported Indic languages.
- Loads the base 1B model once, switches per-language adapters at call time using `processor.tokenizer.set_target_lang()` and `model.load_adapter()`.
- Exposes `MMSASR.transcribe(audio, sample_rate, language) -> str`.

`src/voxtral/tokenizer/model.py`
- `VoxtralTokenizer` adds optional `mms_asr` parameter. If language ∈ `{brx, doi, kok, mni, sat, sd, ks}`, route ASR through `MMSASR` instead of Whisper.

`tests/test_language_detection.py` (new)
- Verify Hindi 5-second clip: primary mms-lid-2048 returns `hi`, agreement with Whisper-large-v3.
- Verify 1-second clip without context: rejected (confidence < 0.7).
- Verify same 1-second clip with ±2 s context: passes.
- Verify Bodo clip: routed through MMS-1B-all, not Whisper.
- Verify quarantine path: synthetic audio with disagreeing LIDs gets moved to quarantine directory.

**Cost.** ~30 GPU-hours.

### Phase 4 — Diarization + dual-stream tokenization on YouTube corpus (Defect 4 full)

**Why fourth.** With clean text, prosody-aware chunks, and reliable language tags, the model can now learn **interruptions** — the "knows when to speak vs listen" goal. This phase wires conversational audio through the dual-stream path properly.

**Components added/changed.**

`scripts/diarize_v2.py` (new — replaces `scripts/diarize_audio.py` which becomes legacy)
- **Operates on raw source audio**, not on `chunks_v2/`. Diarized output replaces `chunks_v2/` for conversational sources. (This was ambiguous in v1 of the plan; locked here.)
- Pipeline: optional Demucs vocal separation → faster-whisper transcription → ctc-forced-aligner → **Silero v6 VAD** (consistent with Phase 2; MarbleNet dropped per review) → pyannote-3.1 speaker embeddings → per-speaker timed segments with words.
- Output: `data/dual_chunks_v2/{source}/{lang}/{shard}/{uuid}.json` describing per-speaker segments + paths to two `.wav` files (one per speaker, with non-target-speaker masked to silence).
- For stereo input where Phase 2 detected speaker-per-channel, skip pyannote and use the channel-as-speaker assignment directly. Faster, more reliable than re-diarizing.

**Silence encoding (revised after Phase 0 result).**
- If Phase 0 experiment 2 confirmed -45 dB noise produces different Mimi codes than bit-zero → use noise floor.
- If Phase 0 experiment 2 showed Mimi maps both to the same codes → switch to **room-tone splice**: extract 200 ms of low-energy region from the same source video, loop it during silence regions. Maintains acoustic continuity with the speaker's environment.

`src/voxtral/tokenizer/dual_stream.py` (existing, modified)
- `DualStreamTokenizer.encode(user_audio, model_audio, sample_rate)` is unchanged in interface but the input contract becomes "per-speaker masked audio at full chunk length" rather than "two arbitrary audio streams." Add a docstring assertion.
- Critically: silence between turns must be encoded as Mimi tokens, not zero-padded. The fix is upstream — `diarize_v2.py` masks the non-active speaker to a low-energy noise floor (-45 dB) instead of bit-zero. This avoids the silent-loop failure mode.
- New method `DualStreamTokenizer.encode_with_metadata(user_audio, model_audio, sample_rate, segments_metadata)` writes a `stream_layout="dual"` sidecar with speaker_segments populated.

`tests/test_dual_stream_pipeline.py` (new)
- Synthetic two-speaker audio: 10 seconds with known turn-taking pattern (S0 0-3s, S1 3-6s, overlap 5-7s, S0 7-10s).
- Verify: diarize_v2 emits per-speaker segments with the right boundaries.
- Verify: `DualStreamTokenizer.encode` produces tokens with the dual-stride layout `[user_text, user_audio×20, model_text, model_audio×20]`.
- Verify: silence regions in the user stream produce non-degenerate Mimi tokens (not the same repeated codebook ID).

**Tokenization run.** All YouTube chunks plus IndicVoices conversational subset go through diarize_v2 → DualStreamTokenizer. Output to `data/tokens_v2_dual/yt/{lang}/{shard}/`.

**Cost.** ~36 GPU-hours diarization + part of Phase 1's Whisper budget.

### Phase 5 — Wire everything into `train_omni.py`

**Why fifth.** Training has been running on Phase 1 outputs throughout Phases 2-4. Phase 5 finalizes the data loader changes that exploit metadata, dual-stream tokens, and per-language sampling.

**Components changed.**

`src/voxtral/trainer/data.py`
- `VoxtralDataset` reads metadata sidecar; emits language tag in batch metadata.
- Implements **temperature sampling** with τ=3.3 across languages: file selection is weighted by `count(lang)^(1/τ)` so high-resource (Hindi) and low-resource (Bodo) languages contribute on similar order of magnitude.
- Adds a `dual_stream` flag that, when True, reads from `data/tokens_v2_dual/`. Stride 42 is detected from sidecar `tokenizer_config.stride` rather than hardcoded.
- **Stride homogeneity in batches** (review-driven fix): all samples in a batch share `stride`. Sampler groups by `stream_layout` field before batching. Without this, mixed batches cause shape mismatches in `extract_codebook_targets`.
- **Random determinism fix**: replace `random.seed(config.seed)` (process-global state) with `random.Random(config.seed)` local instance.
- **Pinned val split**: read `data/val_split_v2.json` (a fixed list of file paths captured at Phase 1a cutover) instead of using a deterministic seed-shuffle over the file list. This makes val_loss comparisons valid across phases even as the directory tree changes.
- **Fail-fast on missing sidecars**: if more than 5% of consumed files are missing sidecars, raise. Prevents silent corruption from `get_fake_item()` fallback.

`src/voxtral/trainer/omni_trainer.py`
- `extract_codebook_targets` already supports `dual_stream=True`. Verify it works on real v2 dual-stream tokens (tests, not new code).
- Per-rank validation reporting fixed: replace "best rank" with "mean across ranks ± std." This is audit AF-406, never landed.

`scripts/train_omni.py`
- New env var `DATA_PATH` defaults to `./data/tokens_v2_chunked` (Phase 2 single-stream output) or `./data/tokens_v2_dual` (Phase 4 dual-stream). Both work; the `dual_stream` flag drives the choice.
- Resume-from-checkpoint path validated with the new dataset layout.

`scripts/eval_wer.py` (new)
- Generates **5 samples per language across all 13 FLEURS languages** = 65 samples total (revised from review — single-language Hindi smoke is insufficient).
- Runs Whisper-large-v3 transcription on each sample.
- Computes WER vs the prompt's reference transcript when prompt mode, vs LID-only check in unprompted mode.
- Optionally runs UTMOS for naturalness. Fails the run if WER > 80% (regression guard).

`tests/test_no_gibberish.py` (new — review-driven addition)
- Generates 30 s of audio from the current checkpoint per target language.
- Asserts:
  - Whisper LID == target language with confidence ≥ 0.9.
  - Output token autocorrelation at lag 50-200 < 0.5 (no repetitive loops).
  - F0 std > 30 Hz (not a monotone drone).
- Runs on every checkpoint, not just at "done" time. Catches regression to the original gibberish failure mode.

`tests/test_interruption_emission.py` (new — Goal 4 eval harness)
- Synthetic dual-stream audio: 30 s, two known speakers with turn-taking pattern + 3 backchannels per speaker.
- Asserts the model emits silence tokens during user-stream speech ≥ 70% of the time. This is the operational definition of Goal 4.

**Tests.** Integration test that runs `train_omni.py` for 5 steps on synthetic v2 data and asserts no shape mismatches and decreasing loss.

### Phase 6 — Architectural follow-ups (DEFERRED, listed for completeness)

Out of scope for this rebuild. Tracked in `experiment_queue.md`:
- MPD (masked parallel depth) — eliminates AR error chain in depth transformer.
- Embedding-norm rescaling fix — addresses the 3.92× speech-vs-text norm mismatch found by `diagnose_embeddings.py`.
- Semantic distillation from Whisper-tiny on q0 (DualCodec recipe).
- FSQ codec exploration — replace Mimi with TAAE-style FSQ and eliminate depth transformer entirely.
- Vocab expansion to 200K (IndicSuperTokenizer 2-stage BPE recipe).
- 22-language coverage with full IndicVoices-R 1,704 hours.

## 7. Sidecar metadata schema

Every `.npy` token file ships with a paired `.meta.json` matching this shape. Write the file atomically (temp + rename) so a partial preprocessing run never produces an unreadable sidecar.

```python
@dataclass
class ChunkMetadata:
    schema_version: int                       # 2
    preprocessing_version: str                # "v2.0"
    preprocessing_run_id: str                 # uuid for the batch run
    preprocessing_timestamp: str              # ISO8601

    source: str                               # "youtube" | "fleurs" | "indicvoices" | ...
    source_id: str                            # video_id or sample_id
    source_url: str | None                    # http URL if applicable
    chunk_index: int                          # which chunk within the source

    language: str                             # ISO code, e.g. "hi"
    language_confidence: float                # 0.0-1.0
    language_method: str                      # "mms_lid_2048" | "whisper_lid" | "tie_break"
    language_secondary: dict | None           # {language, confidence, method}

    transcript: str                           # source-language transcript
    translation_en: str | None                # English translation, sidecar only
    transcript_method: str                    # "faster_whisper_large_v3" | "mms_1b_all"
    transcript_avg_logprob: float             # for filtering low-confidence

    duration_s: float
    sample_rate: int                          # 24000
    num_channels: int                         # 1 or 2
    num_speakers: int                         # 1 for FLEURS, ≥1 for IV/YT
    speaker_segments: list[dict]              # [{speaker, start, end}]

    snr_db: float | None
    speech_ratio: float | None
    clip_ratio: float | None

    stream_layout: str                        # "single" | "dual"
    tokenizer_config: dict                    # {sp_model, text_hz, mimi_num_quantizers, stride}
    token_count: int
    token_range: tuple[int, int]              # min/max token ID, for sanity
```

The data loader treats unknown `schema_version` values as a hard failure (raise, do not silently fall back).

**Additional fields added after review:**

```python
@dataclass
class ChunkMetadataAddenda:
    valid_token_mask: list[bool] | None       # Phase 1.5 — per-token mask for zero-pad regions
    data_license_class: str                   # "cc-by-nc" | "apache" | "mit" — for license filtering
    lid_inherits_from_neighbor: bool          # True for sub-1s chunks; LID borrowed from adjacent chunk
    music_likely: bool | None                 # from scripts/detect_music.py — gates Demucs
    quarantine_reason: str | None             # if file is in quarantine, why
```

**Atomic write contract.** All sidecars and tokens must be written atomically:
```python
tmp_path = f"{final_path}.tmp"
np.save(tmp_path, data)
os.replace(tmp_path, final_path)
```
Same for `.meta.json` (write to `.meta.json.tmp`, then `os.replace`). Prevents partially-written sidecars from being read mid-write.

## 8. Directory structure after Phase 5

```
data/
  chunks_indic_yt/         # original 20s YT chunks (Phase 1 input)
  chunks_v2/               # VAD-chunked variable-length (Phase 2 output)
    yt/
      <lang>/
        <shard>/*.wav      # raw audio per speech region
                *.json     # chunk-level metadata (lang, source, channels)
  dual_chunks_v2/          # diarized speaker-masked pairs (Phase 4 output)
    yt/
      <lang>/
        <shard>/*.json     # speaker segments + paths to S0.wav, S1.wav
  tokens_v2/               # Phase 1 tokens on existing 20s chunks
    fleurs/<lang>/<shard>/*.npy + .meta.json
    iv/<lang>/<shard>/*.npy + .meta.json
    yt/<lang>/<shard>/*.npy + .meta.json
  tokens_v2_chunked/       # Phase 2 tokens on VAD-chunked audio
    yt/<lang>/<shard>/*.npy + .meta.json
  tokens_v2_dual/          # Phase 4 dual-stream tokens
    yt/<lang>/<shard>/*.npy + .meta.json
  tokens_v2_quarantine/    # rejected chunks (low LID confidence, disagree, etc.)
  tokens_legacy/           # archived data/tokens/ (old corrupted output)

src/voxtral/
  tokenizer/
    model.py                # VoxtralTokenizer (modified, per-call language/task)
    word_level_whisper.py   # TimedWhisperTokenizer (modified, per-call args, sidecar reader)
    dual_stream.py          # DualStreamTokenizer (modified, silence-as-token)
    mms_asr.py              # NEW: MMS-1B-all wrapper
    mimi/                   # unchanged, vendored Kyutai Mimi
  data/
    preprocessing.py        # AudioChunkDataset (modified, no padding)
  trainer/
    data.py                 # VoxtralDataset (modified, sidecar reader, temperature sampling)
    omni_trainer.py         # mean-across-ranks val (modified, AF-406 fix)

scripts/
  phase0_preflight.py       # NEW: Phase 0 — runs the 4 pre-flight experiments
  retokenize_v2.py          # NEW: Phase 1 driver
  vad_chunker.py            # NEW: Phase 2 driver
  detect_language.py        # NEW: Phase 3 driver
  detect_music.py           # NEW: spectral flux + harmonicity + tempo detection
  diarize_v2.py             # NEW: Phase 4 driver
  diarize_audio.py          # legacy, retained for reference
  eval_wer.py               # NEW: Phase 5 WER eval

tests/                      # NEW directory
  conftest.py               # synthetic-audio fixtures (TTS-generated, offline-cached)
  test_phase0_preflight.py  # Phase 0
  test_tokenizer_v2.py      # Phase 1
  test_vad_chunker.py       # Phase 2
  test_language_detection.py# Phase 3
  test_dual_stream_pipeline.py # Phase 4
  test_train_omni_smoke.py  # Phase 5 (5-step training smoke test)
  test_no_gibberish.py      # Regression test for gibberish failure mode
  test_interruption_emission.py # Goal 4 eval harness
```

**Disk space estimate.** With chunk audio + tokens_v2 + tokens_legacy + tokens_v2_chunked + tokens_v2_dual + dual_chunks_v2, total disk needed is **200-500 GB**. `tokens_legacy/` is removed only after Phase 5 success criteria are met. Monitor `df` in `run_safe.sh`; pre-allocate the storage budget.

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Re-tokenization budget (~200 GPU-hr) blocks training | High | Low | Phase 1 ships first on existing chunks → training resumes in ~1 week. Phase 2 (re-chunking) runs in background. |
| MMS LID disagrees with Whisper LID frequently | Medium | Medium | Quarantine on disagree. Dashboard rejection rate. Tune confidence thresholds per-language if rejection > 30%. |
| Pyannote-3.1 underperforms on Indic speakers | Medium | Medium | Stereo-correlation fallback (channel-as-speaker). Quarantine single-speaker-detected files for monolingual single-stream training. |
| Demucs over-strips music in sung-language data | Low | Low | Default off. Per-source flag. Sung-language inputs flagged manually. |
| Variable-length packing breaks DDP gradient sync | Low | High | Existing `make_dataloader` packs sequences; verify with synthetic test (`tests/test_train_omni_smoke.py`) before scaling. |
| Sidecar schema drift between phases | Medium | Medium | `schema_version` field; loader rejects unknown schemas with explicit error. Monthly schema review. |
| Disk space (57K chunks × ~50 KB tokens × multiple paths) | Low | Low | `data/tokens_legacy/` deleted after v2 cutover and audit. Monitor `df` in `run_safe.sh`. |
| MMS license (CC-BY-NC) becomes a problem if commercial intent emerges | Low | High | Decision locked at "research only" for now. If commercial path opens, swap to in-house LID classifier (Phase 6 follow-up). |
| Streaming Whisper context-window approach degrades long-clip quality | Low | Medium | Streaming path only triggers for clips < 4 s. Long clips use plain Whisper. Dual-mode behavior is in `tests/test_language_detection.py`. |
| Training resumes from `checkpoint_40800.pt` but new data distribution causes loss spike | Medium | Low | Expected and desirable — old data was corrupted; loss going up briefly is the model relearning correct alignment. Monitor for 5,000 steps before deciding to roll back. |

## 10. Success criteria for declaring "done"

- All five defects + Phase 0 pre-flight experiments have passing unit tests in `tests/`.
- `data/tokens_v2_dual/yt/{hi,ta,bn,...}/` contains **≥ 30,000 token files** with paired sidecars (revised from 50,000 — accounts for ~30% Phase 3 quarantine + Phase 4 single-speaker rejection).
- **Single fresh-init training run** (no resume) trains past 5,000 post-cutover steps and achieves val_loss within 0.3 of the prior corrupted-data best of 1.489 — but with intelligible audio.
- `scripts/eval_wer.py` reports **Hindi WER ≤ 50% on 5 samples × 13 FLEURS languages** (not just Hindi) within 5,000 post-cutover training steps.
- `tests/test_no_gibberish.py` passes for all 13 target languages: Whisper LID detects each generated sample as the target language with confidence ≥ 0.9, autocorrelation lag 50-200 < 0.5, F0 std > 30 Hz.
- A 30-second dual-stream eval shows the model emitting silence tokens during ≥ 70% of user-stream speech (`tests/test_interruption_emission.py`).
- `data/val_split_v2.json` exists and is referenced by `VoxtralDataset` — val comparisons across phases are valid.
- This plan, the HLD, and the LLD live in `planning/` for posterity.

**Rollback plan.** Fresh-init run replaces the resume hedge. If Phase 1 v2 data turns out to be unexpectedly corrupted (a bug in the rebuild, not the spec):
- `data/tokens/` is preserved as `data/tokens_legacy/`. We do NOT roll back the *training* (fresh-init from corrupted data is no better), but we do roll back the *data path* and re-debug the v2 pipeline.
- Phase 1.5 valid_token_mask code stays in place; harmless on legacy data (mask defaults to all-True).
- If a Phase 0 pre-flight gate fails, the affected Phase 1+ task is paused until the fallback strategy is implemented (e.g., room-tone splice for Phase 4 silence if -45 dB noise didn't differentiate from bit-zero).

## 11. Out-of-scope reminder

Inference-side fixes (repetition penalty, RAS sampling, embedding-norm rescaling, modality alignment warmup) belong to Phase NOW in `experiment_queue.md`. They run in parallel with this rebuild but are not part of it. Architectural model changes (MPD, FSQ, MoE, semantic distillation) are Phase 6 follow-ups and not in this plan.

## 12. References

- `https://github.com/MahmoudAshraf97/whisper-diarization` — joint timed-Whisper + diarization (Phase 4 stage-ordering blueprint).
- `https://github.com/antirez/voxtral.c` — minimal C streaming Voxtral (Phase 3 streaming pattern).
- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602` — streaming reference (80 ms / 12.5 Hz token grid match Mimi).
- `https://huggingface.co/mistralai/Voxtral-Mini-3B-2507` — Mistral audio LLM (Phase 6 candidate replacement for faster-whisper).
- `https://github.com/snakers4/silero-vad` — multilingual VAD (Phase 2).
- `https://huggingface.co/facebook/mms-lid-2048` — primary LID for 22 Indic.
- `https://huggingface.co/facebook/mms-1b-all` — ASR fallback for Bodo, Dogri, Konkani, Manipuri, Santali, Sindhi, Kashmiri.
- `https://github.com/SYSTRAN/faster-whisper` — primary ASR.
- `https://arxiv.org/html/2410.00037v2` — Moshi paper (Inner Monologue + dual-stream training).
- `https://arxiv.org/html/2511.03237v1` — IndicSuperTokenizer (Phase 6 vocab expansion reference).
