# Opus Review

**Model:** claude-opus-4
**Generated:** 2026-05-05T00:00:00Z

---

# Adversarial Review: OmniVoxtral Pipeline Rebuild Plan

## Executive Assessment

The plan is well-structured and the defect mapping is rigorous, but it contains several **load-bearing assumptions that are likely wrong** and **at least one foundational arithmetic problem** that will sink Phase 1 if not caught. The biggest concerns are: (1) the Phase 1 / Phase 2 transitional state actively contradicts itself on what data trains the model, (2) the language-tag injection scheme is incompatible with the existing `text_hz=5` rigid bucketing, (3) silence-as-Mimi-tokens may not actually solve the silent-loop failure mode, and (4) Phase 5's "resume from checkpoint_40800.pt" is taking on far more risk than the plan acknowledges.

## 1. Critical Defects in the Plan

### 1.1 Phase 1 is NOT shippable in days

200 GPU-hr / 24 hr/day = 8.3 days minimum. Plus a second `task="translate"` Whisper pass (~doubles cost). Plus YouTube has no language source for Phase 1's `encode(language=lang)` call — so Phase 1 either excludes YT or merges with Phase 3 LID.

**Recommendation:** Phase 1 ships for FLEURS + IndicVoices only (language known). YT joins in Phase 3 after LID lands.

### 1.2 Phase 1 / Phase 2 transition is internally contradictory

During Phases 2-4, training runs on Phase 1 outputs (`data/tokens_v2/`) at 20-second hard chunks **with zero-padding still in the data** (Defect 4 not yet fixed). The model learns the corrupted "emit silent loop forever" pattern from Phase 1 data while Phase 2 is being built.

**Recommendation:** Phase 1.5 — add a `valid_token_mask` field to the sidecar and have the trainer ignore zero-padded regions. Removes the silent-loop signal without re-chunking.

### 1.3 Language-tag injection incompatible with text_hz=5 truncation

`_tokenize_bucket` truncates to exactly 5 tokens per second. Inserting `<lang_xx>` consumes 1 of 5 → 20% reduction in text bandwidth at the most info-dense first second.

**Recommendation:** Verify SP roundtrip behavior of `<lang_xx>` first (Phase 0 pre-flight test). Consider global prepend (one slot total) instead of per-utterance.

### 1.4 Silence-as-Mimi-tokens may not differ from bit-zero in Mimi codes

Plan assumes -45 dB noise floor produces different Mimi codes than bit-zero silence. **Untested.** If Mimi maps both to the same RVQ codes, the silent-loop failure mode persists.

**Recommendation:** Phase 0 pre-flight: encode 5s bit-zero + 5s -45 dB noise through Mimi, compare codebook IDs. 30-minute experiment that gates Phase 4.

### 1.5 Resume from checkpoint_40800.pt understates the risk

Existing checkpoint trained 40,800 steps on Mistral BPE byte-fragments + English Whisper. After SentencePiece cutover, text embedding rows for SP IDs 4-26 + new range are at near-init noise.

**Recommendation:** Train fresh in parallel to checkpoint resume. Freeze non-text-embedding layers for first 2,000 post-cutover steps. Define explicit checkpoint sunset criterion.

## 2. Sequencing and Integration Risks

### 2.1 GPU memory contention — MMS LID + Whisper concurrent

Plan implies running mms-lid-2048 (~4 GB) then Whisper (~3 GB) per-chunk. Loading/unloading 57K chunks is not viable.

**Recommendation:** Specify batched-pass architecture. One pass for LID over all chunks, then unload, then ASR pass.

### 2.2 MMS LID license enforcement (CC-BY-NC vs Apache)

Plan never specifies enforcement. If team later releases checkpoints, MMS-derived labels could infect derived models.

**Recommendation:** Add `data_license_class: "cc-by-nc" | "apache" | "mit"` to sidecar. Loader can filter by license.

### 2.3 VAD min_speech_duration_ms=1000 erases backchannels

Hindi "हाँ" (~400ms), Tamil "ஆமா" (~500ms), Kannada "ಹೂಂ" (~400ms) — backchannels are exactly the turn-taking signal Goal 4 needs. Filtering at 1000ms removes them.

**Recommendation:** Lower to 300 ms. Tag chunks <1s with `lid_inherits_from_neighbor=True`.

### 2.4 Demucs auto-detection of music

Manual flagging at 1,845 URLs is laborious. Spectral flux + harmonicity + tempo detection is straightforward.

**Recommendation:** `scripts/detect_music.py` emits `music_likely: bool` per source URL. Demucs gating becomes data-driven.

## 3. Missing Considerations

### 3.1 Validation set drift after re-tokenization

`VoxtralDataset` does deterministic 90/10 split with `seed=42` over `get_npy_files()`. After cutover, file paths change → val files differ. The val_loss=1.489 baseline is on a different val set than the new pipeline.

**Recommendation:** Pin `data/val_split_v2.json` (fixed file list or hash-based assignment) that survives directory restructuring.

### 3.2 Translation pass is hidden 200 GPU-hr budget

Second Whisper pass over every chunk doubles preprocessing time.

**Recommendation:** Make translation lazy / opt-in. Sidecar `translation_en` allowed to be `null`. Run on sampled 10% initially.

### 3.3 Stride heterogeneity in batches

If batch contains both stride-21 (single-stream) and stride-42 (dual-stream), `extract_codebook_targets` cannot unify them.

**Recommendation:** Add explicit decision: "All samples in a batch share `stride`. Sampler groups by `stream_layout` field."

### 3.4 Quarantine semantics are vague

If rejection rate is 30% on 57,668, that's 17,000 chunks deleted. The §10 success criterion "50,000 token files" is impossible if rejection AND VAD AND diarization stack.

**Recommendation:** 200-chunk pilot before locking thresholds. Realistic post-quarantine count is probably 25,000-35,000.

### 3.5 No regression test for gibberish failure mode

WER on gibberish output may still be 100%. The actual failure has structural signatures: token autocorrelation, F0 monotony, Whisper LID misdetection.

**Recommendation:** `tests/test_no_gibberish.py` regression test:
- Whisper LID == target language ≥ 0.9 confidence.
- Output token autocorrelation at lag 50-200 < 0.5.
- F0 std > 30 Hz.

### 3.6 random.seed vs random.Random in dataset shuffle

`VoxtralDataset` uses `random.seed(config.seed)` (process-level state). If any other code calls `random` first, shuffle order shifts.

**Recommendation:** Use `random.Random(config.seed)` (local instance).

## 4. Over- and Under-Specified Items

### 4.1 Over-specified: MarbleNet VAD vs Silero VAD inconsistency

Phase 2 uses Silero, Phase 4 uses MarbleNet (copied from whisper-diarization). Pick one — Silero.

### 4.2 Under-specified: chunks_v2 → dual_chunks_v2 transformation

Does Phase 4 re-VAD? Use Phase 2 boundaries? If misaligned, chunk reconciliation problem.

**Recommendation:** Specify: "diarize_v2.py operates on raw source audio, not on chunks_v2/. Outputs replace chunks_v2/ for conversational audio."

### 4.3 Under-specified: validation strategy for Phase 1

"val_loss ≤ 1.4 within 5,000 steps" is meaningless given val-set drift.

**Recommendation:** Replace with "WER on `eval_wer.py` 10-sample held-out Hindi prompts ≤ 50% within 5,000 post-cutover training steps."

## 5. Concrete Bugs / Footguns

- `tokenizer.translate(audio, ...)` returns unbounded string. Filter for length, log warning if truncation.
- `data/tokens_legacy/` removal — 5x source storage; quantify (200-500 GB).
- `_save_tokens` atomic write requires `np.save(tmp); os.replace(tmp, final)` in code, not just prose.
- Test fixture for synthetic Hindi audio: FLEURS not bundled — add offline-cached fixture or TTS-generated synthetic.
- `get_fake_item()` 220-token random tensor on load failure — silent corruption. Add fail-fast at >5% warning rate.

## 6. Things to Add to the Plan

1. **Phase 0 (pre-flight)** — 1 day of validation:
   - SP `<lang_xx>` round-trip.
   - Mimi -45 dB vs bit-zero codebook differentiation.
   - Text-embedding gradient sanity at resume.
   - 200-chunk LID rejection-rate pilot.

2. **Explicit checkpoint sunset criterion** — step count after which checkpoint_40800 is abandoned.

3. **Interruption-learning evaluation harness** — define eval protocol for Goal 4.

4. **Concrete rollback plan** — if Phase 1 cutover causes loss explosion, what's the recovery?

5. **CI/test execution plan** — are tests blocking Phase ship?

6. **Per-language smoke evaluation** — Phase 1 done = 5 samples per lang × 13 FLEURS langs, not just Hindi.

## 7. Summary Risk Ranking

| # | Risk | Severity | Likelihood |
|---|---|---|---|
| 1 | Phase 1 takes 16 days, not 7 | High | High |
| 2 | Phase 2-4 transitional state trains on uncorrected zero-pad | High | Certain |
| 3 | Resume from corrupted-data checkpoint destabilizes text embedding | High | Medium |
| 4 | Silence-as-Mimi-tokens may not differ from bit-zero | High | Unknown |
| 5 | min_speech_duration_ms=1000 erases backchannels | Medium | High |
| 6 | Val set drift invalidates phase comparisons | Medium | Certain |
| 7 | Stride heterogeneity causes shape mismatch | Medium | High |
| 8 | Quarantine rate ≥30% prevents §10 success criterion | Medium | Medium |
| 9 | Language-tag injection consumes 20% of text bandwidth | Medium | Certain |
| 10 | `<lang_xx>` SP tokens may not round-trip cleanly | Medium | Unknown |
| 11 | Translation pass GPU budget (~200 hr) not in plan | Medium | Certain |
| 12 | `random` global state non-determinism | Low | Medium |

## Closing

The plan is well-organized but optimistic. It will not ship Phase 1 in days as written, the resume-from-checkpoint hedge is more dangerous than the plan admits, and at least three of the locked decisions (VAD floor, language-tag injection, silence encoding) need empirical validation before they're locked. Add a Phase 0 of one day of pre-flight experiments. Add the val-split persistence fix. Acknowledge that "training resumes in days" is "training resumes in 2-3 weeks" honestly.
