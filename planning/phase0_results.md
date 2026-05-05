# Phase 0 Pre-flight Results

Run timestamp: 2026-05-05T07:55:48Z

## Experiment 1 — SP `<lang_xx>` roundtrip

- Status: **pass**
- Pass rate: 100%
- Recommended `LANGUAGE_TAG_MODE`: `per_utterance`
- Findings: `[{"tag": "<|lang:asm|>", "ids": [289, 4], "ok": true}, {"tag": "<|lang:ben|>", "ids": [289, 5], "ok": true}, {"tag": "<|lang:brx|>", "ids": [289, 6], "ok": true}, {"tag": "<|lang:doi|>", "ids": [289, 7], "ok": true}, {"tag": "<|lang:guj|>", "ids": [289, 8], "ok": true}, {"...": "+18 more"}]`

## Experiment 2 — Mimi silence differentiation

- Status: **pass**
- Histogram sizes (q1 codebook unique IDs per kind): `{"bit_zero": 7, "white_noise_-45db": 6, "pink_noise_-45db": 9, "room_tone_synth": 8}`
- Pairwise Jensen-Shannon vs bit_zero: `{"bit_zero_vs_white_noise_-45db": 0.1272, "bit_zero_vs_pink_noise_-45db": 0.1866, "bit_zero_vs_room_tone_synth": 0.3085}`
- Recommended `SILENCE_STRATEGY`: `noise_floor`

## Experiment 3 — 200-chunk LID rejection pilot

- Status: **pass**
- n_sampled: 100
- n_evaluated: 100
- n_failures: 0
- low_conf_rate: 0.21
- confidence_mean: 0.833974609375
- confidence_min: 0.306884765625
- recommended_threshold: 0.7
- language_counts: `{"hi": 27, "en": 12, "ta": 20, "sa": 1, "bn": 17, "gu": 3, "te": 3, "ur": 5, "pa": 3, "ml": 2, "ne": 2, "ko": 1, "kn": 3, "mr": 1}`

## Locked env-var defaults

- `LANGUAGE_TAG_MODE=per_utterance`
- `SILENCE_STRATEGY=noise_floor`
- `LID_CONFIDENCE_THRESHOLD=0.7`
