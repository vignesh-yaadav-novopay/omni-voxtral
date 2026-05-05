# Integration Notes — Opus Review (iteration 1)

## Integrating

| # | Opus finding | Integration |
|---|---|---|
| 1.1 | Phase 1 takes 8-16 days, not 7 | Plan §6.Phase 1: split into Phase 1a (FLEURS+IV, ~4 days) and Phase 1b (YouTube, gated on Phase 3 LID, ~10 days). Goal 6 timeline rewritten honestly. |
| 1.2 | Phase 2-4 transitional state trains on zero-pad | New Phase 1.5: emit `valid_token_mask` field in sidecar; trainer ignores zero-padded regions on loss. Removes silent-loop signal without re-chunking. |
| 1.3 | `<lang_xx>` injection breaks 5-token bucket | Adopt **conditional path**: Phase 0 pre-flight tests SP roundtrip. If clean → per-utterance prefix. If byte-fallback → global prepend (one slot total). Decision recorded in Phase 0 output. |
| 1.4 | Mimi -45 dB vs bit-zero may produce same codes | Phase 0 pre-flight: encode 5s of each through Mimi, diff codebook IDs. If indistinguishable, Phase 4 silence strategy switches to room-tone samples (sourced from quiet regions of original audio). |
| 1.5 | Resume from checkpoint_40800 risk understated | **User decision (post-review): fresh-init only, no resume.** The corrupted-data checkpoint is archived to `logs/archive/checkpoint_40800.pt.legacy`. All v2 training starts from random init. This is more honest than the parallel-hedge approach — it accepts the 40,800-step sunk cost cleanly. |
| 2.1 | GPU memory contention for LID + Whisper | Phase 3 explicitly batched-pass: one LID sweep over all chunks, unload, then ASR sweep. No per-chunk model load/unload. |
| 2.2 | MMS license enforcement (CC-BY-NC) | Sidecar gains `data_license_class: "cc-by-nc" \| "apache" \| "mit"` field. Loader filters by license. Future commercial swap = re-filter, not retokenize. |
| 2.3 | min_speech_duration_ms=1000 erases backchannels | Lower to 300 ms. Tag chunks <1s with `lid_inherits_from_neighbor=True` — neighbor's LID transfers. Critical for Goal 4 (interruption learning). |
| 2.4 | Demucs auto-detection of music | New `scripts/detect_music.py` — spectral flux + harmonicity + tempo. Runs once over 1,845 source URLs (~30 hr CPU). Emits `music_likely: bool` per source. Demucs gating becomes data-driven. |
| 3.1 | Val set drift after re-tokenization | Pin `data/val_split_v2.json` — fixed file list captured before Phase 1 cutover. Loader reads explicit list, not deterministic seed-shuffle. |
| 3.2 | Translation pass = hidden 200 GPU-hr | Make translation lazy: sidecar `translation_en` allowed `null`. Run on sampled 10% initially. Background job fills the rest over weeks. |
| 3.3 | Stride heterogeneity in batches | Plan §6.Phase 5 explicit: "All samples in a batch share `stride`. Sampler groups by `stream_layout` field." Implement in `VoxtralDataset.__iter__`. |
| 3.4 | Quarantine rate may break 50K target | 200-chunk pilot in Phase 0 measures rejection rate. §10 success criterion revised to "≥ 30,000 token files post-quarantine" (realistic, not aspirational). |
| 3.5 | No regression test for gibberish failure mode | New `tests/test_no_gibberish.py`: Whisper LID == target language ≥ 0.9, token autocorrelation lag 50-200 < 0.5, F0 std > 30 Hz. Runs on every checkpoint. |
| 3.6 | `random.seed` global state non-determinism | Switch to `random.Random(config.seed)` local instance in `VoxtralDataset`. One-line fix, in LLD. |
| 4.1 | MarbleNet vs Silero inconsistency | Drop MarbleNet entirely. Silero v6 used in both Phase 2 and Phase 4. |
| 4.2 | chunks_v2 → dual_chunks_v2 transformation undefined | Plan §6.Phase 4 explicit: "diarize_v2.py operates on raw source audio, not chunks_v2/. Its outputs replace chunks_v2/ for conversational audio." |
| 4.3 | Phase 1 success criterion uses val_loss across drift | Replaced with "WER on 10-sample Hindi prompts ≤ 50% within 5,000 post-cutover training steps." |
| 5 (footguns) | np.save not atomic | LLD specifies `np.save(tmp); os.replace(tmp, final)` pattern explicitly. |
| 5 (footguns) | get_fake_item silent corruption | Fail-fast at >5% warning rate. Loader raises if too many sidecars missing. |
| 6.1 | Add Phase 0 pre-flight | NEW Phase 0: 1 day of validation experiments before any Phase 1 work. Listed below. |
| 6.2 | Checkpoint sunset criterion | Defined: "Checkpoint_40800 abandoned if fresh-init parallel run beats it on val WER by step 10,000." |
| 6.3 | Interruption-learning eval protocol | New `tests/test_interruption_emission.py`: synthetic dual-stream audio with known turn boundaries. Asserts model emits silence tokens during user-stream speech ≥ 70%. |
| 6.4 | Concrete rollback plan | Phase 1 rollback: `data/tokens/` retained as `data/tokens_legacy/`; trainer can resume on legacy path with one env var flip. |
| 6.5 | CI integration plan | Tests run pre-merge (manual, not CI infra in scope). PR template includes "all tests passing on local A10G" checkbox. |
| 6.6 | Per-language smoke evaluation | Phase 1 done = WER on 5 samples × 13 FLEURS languages, not just Hindi. |

## Not integrating

| # | Opus finding | Reason |
|---|---|---|
| — | "Train fresh-init from scratch with no checkpoint hedge" | Too aggressive. Fresh-init parallel is a hedge, not a replacement. Keeping both runs gives data on which approach actually wins. |
| — | "Drop YouTube from Phase 1 entirely, never put it in Phase 1" | Partially integrated. YouTube enters in Phase 3 (after LID), but it does enter — we don't lose 320 hours of Indic data. |
| — | "Replace val_loss success criterion entirely" | Partially integrated. Use WER as primary, but keep val_loss as secondary signal — it's faster to evaluate during training and the gradient information is still useful for trend detection. |
| — | "Use random.Random everywhere" | Limited to `VoxtralDataset.__iter__`. Other random usage (data augmentation if any) handled separately, not in this plan's scope. |

## New Phase 0 — Pre-flight experiments (1 day)

Added to plan as Phase 0, gating Phase 1 work. Three experiments in parallel:

1. **SP `<lang_xx>` roundtrip test.** Encode `"<lang_hi>आज मौसम बहुत अच्छा है"` through `omnivoxtral_sp.model`. Verify `<lang_hi>` becomes a single token ID in [4, 26]. If byte-fallback, document and switch language-tag injection to global prepend.

2. **Mimi silence differentiation test.** Generate 5s bit-zero, 5s -45 dB white noise, 5s -45 dB pink noise, 5s real-room-tone. Encode all four through Mimi at 8 quantizers. Verify q1 codebook IDs differ across the four conditions. If indistinguishable, switch Phase 4 silence strategy.

3. **Resume gradient sanity.** Load `checkpoint_40800.pt`. For 100 random SP token IDs in [4, 65535], compute the gradient of a dummy CE loss against a synthetic target. Verify gradient magnitude is non-zero on previously-trained ID rows AND on new ID rows. If new IDs have zero gradient, freeze non-text-embedding plan kicks in.

4. **200-chunk LID rejection pilot.** Sample 200 random YouTube chunks. Run mms-lid-2048 + Whisper LID. Compute disagreement rate, low-confidence rate, total quarantine rate. Use to set realistic Phase 3 quarantine threshold.

Phase 0 output: `planning/phase0_results.md` with all four experiment outcomes. Each Phase 1+ task is gated on the corresponding Phase 0 result.
