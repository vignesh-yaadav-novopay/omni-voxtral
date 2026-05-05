# TDD Plan — OmniVoxtral Pipeline Rebuild

Mirrors the structure of `claude-plan.md`. For each implementation section, this defines test stubs to write **before** implementation. Tests are stubs (prose / signatures), not full implementations — the implementer writes the actual test code.

Project uses pytest with synthetic-audio fixtures from `tests/conftest.py`. Run with `uv run pytest tests/` per existing repo convention.

---

## Phase 0 — Pre-flight validation

`tests/test_phase0_preflight.py`:

- Test: `<lang_hi>` token IDs round-trip through `omnivoxtral_sp.model` as a single user-defined-symbol ID.
- Test: `<lang_hi>` is decoded back as the literal string `<lang_hi>`, not as byte-fallback fragments.
- Test: All 23 language tags `<lang_xx>` map to IDs in the inclusive range [4, 26].
- Test: Mimi encode of 5 s bit-zero produces a different q1 codebook ID histogram than 5 s of -45 dB white noise.
- Test: Mimi encode of 5 s real room-tone differs in q1 from 5 s bit-zero.
- Test: 200-chunk LID rejection pilot returns a JSON report with `agreement_rate`, `low_confidence_rate`, `total_quarantine_rate`.

---

## Phase 1 — Tokenizer fixes (Defects 2, 3, 5)

`tests/test_tokenizer_v2.py`:

### `VoxtralTokenizerConfig`
- Test: default `sp_tokenizer_path` resolves to `data/tokenizer/omnivoxtral_sp.model` and the file exists.
- Test: explicit `sp_tokenizer_path=None` falls back to Mistral BPE (backward compatibility).

### `TimedWhisperTokenizer.forward(audio, sample_rate, language=None, task="transcribe")`
- Test: synthetic 1 s Hindi audio with `language="hi", task="transcribe"` returns tokens whose decoded text contains Devanagari characters.
- Test: synthetic 1 s Hindi audio with `language="hi", task="translate"` returns tokens whose decoded text is ASCII Latin.
- Test: omitting `language` falls back to `__init__` default; if `__init__` default is also None, raises a clear ValueError.
- Test: Whisper's `task` and `language` are forwarded to `model.generate()` (use a mock to verify the kwargs).

### `VoxtralTokenizer.encode(x, sample_rate, language=None, task="transcribe", source_metadata=None)`
- Test: full pipeline on 20 s FLEURS Hindi sample produces token tensor of expected shape (`(1, ~505)`).
- Test: tokens are in valid range `[0, 81920)`.
- Test: `source_metadata` flows through to the sidecar writer.

### `VoxtralTokenizer.translate(audio, sample_rate, language=None)`
- Test: returns a Python `str` containing English text.
- Test: warns and truncates if returned string > 2000 characters (defensive cap).

### Sidecar writer `write_metadata_sidecar(npy_path, metadata)`
- Test: writes `<file>.meta.json` with `schema_version=2`.
- Test: written atomically — temp file is `<file>.meta.json.tmp`, renamed via `os.replace`. Verify by injecting a failure between write and rename and asserting no partial file is visible.
- Test: `valid_token_mask` field is populated with a bit array matching token count.
- Test: `data_license_class` field is set per the source (FLEURS=apache, IV=cc-by, YT=user-set).

### Phase 1.5 valid-token mask in trainer
- Test: `compute_omni_loss` zeros loss contribution from positions where `valid_token_mask=False`.
- Test: gradient w.r.t. masked positions is zero.
- Test: backward compatibility — if `valid_token_mask=None` (legacy data), all positions contribute (mask defaults to all-True).

### `VoxtralDataset` sidecar reader
- Test: yielded dict contains `language`, `duration_s`, `source` fields when sidecar exists.
- Test: missing sidecar produces a logged warning and tags the sample as `language="unknown"`.
- Test: if more than 5% of consumed samples are missing sidecars, the iterator raises a clear error.

---

## Phase 2 — VAD-aware variable-length chunking

`tests/test_vad_chunker.py`:

### Synthetic audio fixtures
- Fixture: 30 s mono audio with three known speech regions (5 s, 8 s, 4 s) separated by 2 s silences. Generated via TTS or pre-recorded.
- Fixture: 10 s stereo audio with speaker A on left channel only, speaker B on right channel only.
- Fixture: 10 s stereo audio with both speakers on both channels (mixed).

### `vad_chunker.py`
- Test: mono fixture produces three chunks. Chunk durations within ±50 ms of expected (5 s, 8 s, 4 s) after 150 ms speech_pad.
- Test: chunks shorter than `min_speech_duration_ms=300` are skipped.
- Test: chunks longer than `max_speech_duration_s=20` are split at silence boundaries.
- Test: each chunk has 150 ms pre-roll and post-roll preserved (check first/last sample energy).

### Stereo handling
- Test: speaker-per-channel stereo fixture produces two parallel chunks per region (one per channel) tagged with `stream_role` ∈ `{user, model}` in `chunk.json`.
- Test: mixed stereo fixture (correlation > 0.5) produces single mixed chunks.
- Test: `chunk.json` correctly records `num_channels` and `stream_role`.

### Backchannel preservation (Goal 4 critical)
- Test: synthetic 30 s audio with a 400 ms "haan" backchannel produces a chunk for the backchannel (would be filtered at `min_speech_duration_ms=1000`).
- Test: the backchannel chunk is tagged with `lid_inherits_from_neighbor=True` in its sidecar.

---

## Phase 3 — Per-chunk LID + streaming Whisper fallback

`tests/test_language_detection.py`:

### `detect_language.py`
- Test: clean 5 s Hindi clip → primary `mms-lid-2048` returns `hi` with confidence ≥ 0.9.
- Test: clean 5 s Hindi clip → secondary Whisper LID also returns `hi`. `agreement=True` recorded.
- Test: 1 s clip without context → top-1 confidence < 0.7 → quarantined.
- Test: 1 s clip with ±2 s neighbor context concatenated → confidence ≥ 0.7 → accepted with `lid_inherits_from_neighbor=False` (got its own confidence).
- Test: synthetic adversarial clip with disagreeing LIDs (mms says `bn`, whisper says `hi`) → quarantined to `data/chunks_v2_quarantine/`, `quarantine_reason="lid_disagreement"`.

### MMS routing for Whisper-unsupported Indic
- Test: for `language ∈ {brx, doi, kok, mni, sat, sd, ks}`, ASR call routes to `MMSASR.transcribe`, not Whisper.
- Test: MMS adapter switches correctly without reloading the base model.

### `MMSASR` wrapper
- Test: switching from `language="ben"` to `language="hin"` reuses the loaded 1B base; only the adapter changes.
- Test: returns transcript as `str`.

---

## Phase 4 — Diarization + dual-stream tokenization

`tests/test_dual_stream_pipeline.py`:

### `diarize_v2.py`
- Test: synthetic two-speaker audio (S0: 0-3s, S1: 3-6s, S0: 6-10s) produces three segments with correct boundaries.
- Test: diarize_v2 operates on raw source audio path, not on `chunks_v2/` paths (verify via path inspection in test).
- Test: stereo input where Phase 2 detected `speaker-per-channel=True` skips pyannote and uses channel-as-speaker assignment.
- Test: Demucs runs only when source's `music_likely=True` flag is set.

### `DualStreamTokenizer.encode_with_metadata(user_audio, model_audio, sample_rate, segments_metadata)`
- Test: dual-stream output has stride 42, layout `[user_text(1), user_audio(20), model_text(1), model_audio(20)]`.
- Test: silence regions in the user stream produce non-degenerate Mimi tokens (not the same repeated codebook ID — assert >5 unique q1 IDs in any 200 ms silence window).
- Test: `stream_layout="dual"` is recorded in sidecar.
- Test: speaker_segments populated from `segments_metadata`.

### Silence encoding strategy (Phase 0 gated)
- Test: if Phase 0 experiment 2 result is `room_tone_required=True`, diarize_v2 splices room-tone instead of -45 dB noise. Verify by checking output q1 codebook variance.

---

## Phase 5 — Wire into `train_omni.py`

`tests/test_train_omni_smoke.py`:

### `VoxtralDataset` upgrades
- Test: temperature sampling τ=3.3 — when 100 samples are drawn from a corpus with 90% Hindi + 10% Bodo, drawn distribution is approximately 70% Hindi / 30% Bodo (within ±5%).
- Test: `random.Random(seed)` local instance produces deterministic shuffle even when other code calls `random.seed()` afterwards.
- Test: pinned val split — `data/val_split_v2.json` is loaded; iteration produces exactly the listed files in val mode regardless of directory contents.
- Test: stride homogeneity — sampler groups samples by `stream_layout` and a single batch never mixes stride-21 and stride-42 samples.

### Training smoke test
- Test: 5-step training run on synthetic v2 data: forward pass, loss computation, backward pass, optimizer step. Assert no shape mismatches, no NaN, decreasing average loss.

### Per-rank val reporting (audit AF-406)
- Test: `compute_omni_loss` aggregated metrics use mean across DDP ranks, not best rank.
- Test: std across ranks is reported alongside mean.

### `eval_wer.py`
- Test: end-to-end pipeline: load checkpoint → generate 5 samples × 13 langs → Whisper transcribe → compute WER.
- Test: regression guard — fails the run if any language WER > 80%.

---

## Regression tests (run on every checkpoint)

`tests/test_no_gibberish.py`:

- Test: for each of 13 target languages, generate 30 s of audio. Whisper-large-v3 LID detects the audio as the target language with confidence ≥ 0.9.
- Test: token autocorrelation at lag 50-200 < 0.5 (no repetitive loops).
- Test: F0 std > 30 Hz (output is not a monotone drone).
- Test: failure produces a clear diff: prints which test failed, what the actual values were, what the targets were.

`tests/test_interruption_emission.py`:

- Test: generate 30 s of dual-stream output from a synthetic two-speaker prompt. Compute the fraction of model-stream output positions that are silence tokens during user-stream speech regions. Assert ≥ 70%.
- Test: verify the protocol — that the synthetic input has user-stream speech in known regions, and the metric is computed over those regions only.

---

## Test fixtures (`tests/conftest.py`)

- Fixture: `synthetic_hindi_audio` — generated via TTS-from-text, cached locally (offline-safe). 5 s, 24 kHz, mono.
- Fixture: `synthetic_two_speaker_audio` — concatenated TTS samples from two speakers, 30 s with known turn boundaries.
- Fixture: `silence_5s` — bit-zero 5 s.
- Fixture: `noise_5s_minus_45db` — Gaussian noise scaled to -45 dB RMS, 5 s.
- Fixture: `room_tone_5s` — extracted from a quiet region of an existing FLEURS sample, 5 s.
- Fixture: `tokenizer_v2` — initialized `VoxtralTokenizer` with v2 config; module-scoped to amortize Whisper + Mimi load time.

---

## Phase 0 sequencing constraint

Tests for Phase 1 cannot pass until Phase 0 fixtures and gates are evaluated. `tests/test_phase0_preflight.py` must run first; its outputs (`phase0_results.md`) become inputs to subsequent test parametrization (e.g., the silence-encoding strategy parameter).
