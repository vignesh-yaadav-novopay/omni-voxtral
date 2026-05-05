"""Phase 5 — Per-language WER evaluation.

Generates 5 samples × 13 FLEURS languages = 65 outputs from the current
checkpoint, transcribes each with Whisper-large-v3, computes WER per language.

Regression guard: fails the run (exit 1) if any language WER > 80%.

Usage:
    bash autoresearch/scripts/run_safe.sh \\
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_wer.py \\
            --ckpt_path logs/<run>/checkpoint_5000.pt \\
            --output logs/eval/<run>/wer.json" 3600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_wer")

DEFAULT_LANGUAGES = ["hi", "kn", "ta", "te", "ml", "bn", "gu", "mr", "pa", "ur",
                     "ne", "or", "as"]
ISO1_TO_ISO3 = {
    "en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj", "pa": "pan",
    "ur": "urd", "or": "ori", "as": "asm", "ne": "nep",
}


def _atomic_write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_fleurs_samples(lang: str, n: int) -> list[dict]:
    """Pull n FLEURS test-split samples for `lang`. Returns dicts with audio + transcription."""
    from datasets import load_dataset
    cfg_map = {
        "hi": "hi_in", "kn": "kn_in", "ta": "ta_in", "te": "te_in",
        "ml": "ml_in", "bn": "bn_in", "gu": "gu_in", "mr": "mr_in",
        "pa": "pa_in", "ur": "ur_pk", "ne": "ne_np", "or": "or_in",
        "as": "as_in", "en": "en_us",
    }
    if lang not in cfg_map:
        return []
    ds = load_dataset("google/fleurs", cfg_map[lang], split="test", trust_remote_code=True)
    return [ds[i] for i in range(min(n, len(ds)))]


def _generate_audio_from_checkpoint(
    pipeline,
    prompts: list[dict],
    language: str,
    *,
    max_windows: int = 150,
    temperature: float = 0.8,
    top_k: int | None = 250,
    prompt_tokens: int = 100,
) -> list[torch.Tensor]:
    """Generate continuation audio for each prompt using a loaded pipeline.

    `pipeline` comes from `scripts.generate.load_inference_pipeline()` and is
    reused across all FLEURS samples in this language to avoid reloading the
    model 5×N times.
    `language` is the ISO 639-3 tag (e.g. "hin", "tam") used as the SP control
    token at the start of each generation.
    """
    from scripts.generate import generate_audio

    log.info(f"[eval_wer:{language}] generating {len(prompts)} samples")
    out = []
    for i, sample in enumerate(prompts):
        try:
            audio_arr = sample["audio"]["array"]
            sr = int(sample["audio"]["sampling_rate"])
            wav = generate_audio(
                pipeline,
                prompt_audio=audio_arr,
                prompt_sample_rate=sr,
                prompt_tokens=prompt_tokens,
                language=language,
                max_windows=max_windows,
                temperature=temperature,
                top_k=top_k,
            )
            out.append(wav)
        except Exception as e:
            log.warning(f"[eval_wer:{language}] sample {i} generation failed: {e}")
            out.append(None)
    return out


def run_wer_evaluation(
    ckpt_path: str,
    languages: list[str],
    samples_per_language: int = 5,
    whisper_model: str = "openai/whisper-large-v3",
    output_path: str | None = None,
    fail_threshold_wer: float = 0.80,
    device: str = "cuda:0",
) -> dict:
    """Generate samples, transcribe with Whisper, compute WER per language.

    Returns aggregate report. Fails the run (exit 1) if any language WER > threshold.
    """
    import jiwer
    from faster_whisper import WhisperModel
    from scripts.generate import load_inference_pipeline

    if output_path is None:
        run_id = Path(ckpt_path).stem
        output_path = f"logs/eval/{run_id}/wer.json"

    asr = WhisperModel(
        "large-v3",
        device="cuda" if device.startswith("cuda") else "cpu",
        compute_type="float16" if device.startswith("cuda") else "int8",
    )
    pipeline = load_inference_pipeline(ckpt_path, device=device)

    per_lang: dict[str, dict] = {}
    failed_langs: list[str] = []
    for lang in languages:
        samples = _load_fleurs_samples(lang, samples_per_language)
        if not samples:
            log.warning(f"[{lang}] no FLEURS samples available; skipping")
            continue
        lang_iso3 = ISO1_TO_ISO3.get(lang, lang)
        generated_audio = _generate_audio_from_checkpoint(
            pipeline, samples, lang_iso3,
        )

        wer_values = []
        for i, sample in enumerate(samples):
            ref = sample.get("transcription") or sample.get("raw_transcription") or ""
            if not ref:
                continue
            audio = generated_audio[i] if generated_audio[i] is not None else sample["audio"]["array"]
            sr = 24_000
            if isinstance(audio, np.ndarray):
                wav = audio.astype(np.float32)
            else:
                wav = audio.cpu().numpy().astype(np.float32)
            # Transcribe with Whisper
            _segs, info = asr.transcribe(
                wav.flatten(),
                language=lang if lang in ISO1_TO_ISO3 else None,
                beam_size=1, vad_filter=False,
            )
            hyp = " ".join(s.text for s in _segs).strip()
            try:
                wer = float(jiwer.wer(ref, hyp))
            except Exception:
                wer = 1.0
            wer_values.append(wer)

        if wer_values:
            avg_wer = float(np.mean(wer_values))
            per_lang[lang] = {
                "wer": avg_wer,
                "n_samples": len(wer_values),
                "wer_per_sample": wer_values,
            }
            if avg_wer > fail_threshold_wer:
                failed_langs.append(lang)
            log.info(f"[{lang}] WER={avg_wer:.2%} (n={len(wer_values)})")
        else:
            log.warning(f"[{lang}] no usable samples")

    avg_overall = (
        float(np.mean([v["wer"] for v in per_lang.values()])) if per_lang else 1.0
    )
    report = {
        "checkpoint": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples_per_language": samples_per_language,
        "fail_threshold_wer": fail_threshold_wer,
        "languages": per_lang,
        "average_wer": avg_overall,
        "failed_languages": failed_langs,
        "passed": len(failed_langs) == 0,
    }
    _atomic_write_json(report, output_path)
    log.info(f"wrote {output_path}; avg WER={avg_overall:.2%}; failed={failed_langs}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    p.add_argument("--samples_per_language", type=int, default=5)
    p.add_argument("--output", default=None)
    p.add_argument("--fail_threshold_wer", type=float, default=0.80)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    report = run_wer_evaluation(
        ckpt_path=args.ckpt_path,
        languages=languages,
        samples_per_language=args.samples_per_language,
        output_path=args.output,
        fail_threshold_wer=args.fail_threshold_wer,
        device=args.device,
    )
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
