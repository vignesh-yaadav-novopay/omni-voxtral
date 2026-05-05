"""Phase 0 pre-flight validation — gates Phase 1+ work.

Three small experiments. Each writes a section to phase0_results.md and decides
the value of an env var the rest of the pipeline keys off:

  exp1 (SP <lang_xx> roundtrip)       → LANGUAGE_TAG_MODE
  exp2 (Mimi silence differentiation) → SILENCE_STRATEGY
  exp3 (200-chunk LID rejection)      → LID_CONFIDENCE_THRESHOLD

Resume gradient sanity (formerly exp3) was dropped — fresh-init is locked.

Run via: bash autoresearch/scripts/run_safe.sh \
    "CUDA_VISIBLE_DEVICES=0 uv run scripts/phase0_preflight.py" 1800

Exit 0 = all sections written. Exit 1 = at least one fatal error (won't block
implementation; gates may degrade to fallback strategies — see plan §6.Phase 0).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SP_MODEL_PATH = REPO_ROOT / "data" / "tokenizer" / "omnivoxtral_sp.model"
PLANNING_DIR = REPO_ROOT / "autoresearch" / "planning"
RESULTS_PATH = PLANNING_DIR / "phase0_results.md"


# ---------------------------------------------------------------------------
# Experiment 1 — SentencePiece <lang_xx> roundtrip
# ---------------------------------------------------------------------------

# Actual SP tag format inspected from data/tokenizer/omnivoxtral_sp.vocab —
# user-defined symbols at IDs 4..26 (23 languages, ISO 639-3) + control tokens
# at 27..29. Phase 0 verifies each tag rounds-trips as a single token id.
LANGUAGE_TAGS = [
    "<|lang:asm|>", "<|lang:ben|>", "<|lang:brx|>", "<|lang:doi|>",
    "<|lang:guj|>", "<|lang:hin|>", "<|lang:kan|>", "<|lang:kas|>",
    "<|lang:kok|>", "<|lang:mai|>", "<|lang:mal|>", "<|lang:mni|>",
    "<|lang:mar|>", "<|lang:nep|>", "<|lang:ori|>", "<|lang:pan|>",
    "<|lang:san|>", "<|lang:sat|>", "<|lang:snd|>", "<|lang:tam|>",
    "<|lang:tel|>", "<|lang:urd|>", "<|lang:eng|>",
]
CONTROL_TAGS = ["<|silence|>", "<|overlap|>", "<|backch|>"]


def experiment_sp_roundtrip(sp_path: str) -> dict:
    """Encode `<|lang:xxx|>` strings and verify each tag becomes ONE token in [4,26].

    Pass: every tag rounds-trips as a single user-defined-symbol id, decode-clean.
    Fail: any tag falls back to byte fragments → switch to global_prepend mode.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    expected_id_range = (4, 26)
    # SP prepends a `▁` (whitespace marker, id 289) to standalone strings — that's
    # not byte-fallback, just SP's word-boundary convention. We accept either:
    # (a) exactly the tag id, or (b) `[whitespace_marker, tag_id]`.
    space_marker_id = 289
    findings: list[dict] = []
    pass_count = 0
    for wrapped in LANGUAGE_TAGS:
        ids = sp.Encode(wrapped)
        # find the tag id: any element in [4, 26] whose piece string equals the tag
        tag_indices = [
            i for i, t in enumerate(ids)
            if expected_id_range[0] <= t <= expected_id_range[1]
            and sp.IdToPiece(t) == wrapped
        ]
        # noise: anything outside {space_marker, tag} → byte fallback
        noise = [i for i in ids if i != space_marker_id and i not in {ids[j] for j in tag_indices}]
        ok = len(tag_indices) == 1 and len(noise) == 0
        if ok:
            pass_count += 1
        findings.append({"tag": wrapped, "ids": ids, "ok": ok})

    pass_rate = pass_count / len(LANGUAGE_TAGS)
    # Also probe control tokens — they should be at IDs 27..29.
    control_findings = []
    for c in CONTROL_TAGS:
        ids = sp.Encode(c)
        ctrl_indices = [
            i for i, t in enumerate(ids) if 27 <= t <= 29 and sp.IdToPiece(t) == c
        ]
        noise = [i for i in ids if i != space_marker_id and i not in {ids[j] for j in ctrl_indices}]
        control_findings.append({"tag": c, "ids": ids,
                                 "ok": len(ctrl_indices) == 1 and len(noise) == 0})

    status = "pass" if pass_rate >= 0.95 else "fail"
    recommended_mode = "per_utterance" if status == "pass" else "global_prepend"
    return {
        "status": status,
        "pass_rate": pass_rate,
        "recommended_mode": recommended_mode,
        "tag_format": "<|lang:XXX|>",
        "findings": findings[:5] + ([{"...": f"+{len(findings) - 5} more"}] if len(findings) > 5 else []),
        "control_tags": control_findings,
    }


# ---------------------------------------------------------------------------
# Experiment 2 — Mimi silence differentiation
# ---------------------------------------------------------------------------

def _gen_audio(kind: str, duration_s: float = 5.0, sr: int = 24_000) -> torch.Tensor:
    n = int(duration_s * sr)
    if kind == "bit_zero":
        return torch.zeros(1, 1, n)
    if kind == "white_noise_-45db":
        raw = torch.randn(n)
        rms = raw.pow(2).mean().sqrt()
        target = 10 ** (-45 / 20)
        return (raw * (target / rms.clamp_min(1e-8))).view(1, 1, -1)
    if kind == "pink_noise_-45db":
        # Voss-style approximation: cumsum across `rows` parallel series, summed.
        # Each row needs `n` samples so the final length matches.
        rows = 16
        a = np.random.randn(rows, n)
        c = np.cumsum(a, axis=1)
        pink = c.sum(axis=0).astype(np.float32)
        pink = pink - pink.mean()
        rms = float(np.sqrt((pink ** 2).mean()))
        target = 10 ** (-45 / 20)
        pink = pink * (target / max(rms, 1e-8))
        return torch.from_numpy(pink).view(1, 1, -1)
    if kind == "room_tone_synth":
        rows = 8
        a = np.random.randn(rows, n)
        c = np.cumsum(a, axis=1)
        pink = c.sum(axis=0).astype(np.float32)
        pink = pink - pink.mean()
        rms = float(np.sqrt((pink ** 2).mean()))
        target = 10 ** (-45 / 20)
        pink = pink * (target / max(rms, 1e-8))
        t = np.arange(n) / sr
        hum = (10 ** (-55 / 20)) * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        return torch.from_numpy(pink + hum).view(1, 1, -1)
    raise ValueError(f"unknown kind {kind}")


def experiment_mimi_silence(device: str) -> dict:
    """Encode 4 silence types through Mimi 8q. Compare q1 (semantic) histograms.

    Different histograms → SILENCE_STRATEGY=noise_floor (cheap, just synthesise).
    Identical → SILENCE_STRATEGY=room_tone (must splice real low-energy region).
    """
    from voxtral.tokenizer.mimi.models import loaders
    import huggingface_hub as hf_hub

    weight = hf_hub.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(weight, device=device)
    mimi.set_num_codebooks(8)

    kinds = ["bit_zero", "white_noise_-45db", "pink_noise_-45db", "room_tone_synth"]
    histograms: dict[str, dict[int, int]] = {}
    samples: dict[str, list[int]] = {}
    for k in kinds:
        x = _gen_audio(k).to(device)
        with torch.no_grad():
            tokens = mimi.encode(x)  # (B, num_q, T)
        q1 = tokens[0, 0].tolist()  # semantic codebook
        histograms[k] = dict(Counter(q1))
        samples[k] = q1[:50]

    # Pairwise Jensen-Shannon distance between q1 histograms
    def js(a: dict, b: dict) -> float:
        keys = set(a.keys()) | set(b.keys())
        total_a = sum(a.values()) or 1
        total_b = sum(b.values()) or 1
        m_dist = []
        for k in keys:
            pa = a.get(k, 0) / total_a
            pb = b.get(k, 0) / total_b
            pm = (pa + pb) / 2
            term_a = pa * np.log2(pa / pm) if pa > 0 and pm > 0 else 0.0
            term_b = pb * np.log2(pb / pm) if pb > 0 and pm > 0 else 0.0
            m_dist.append((term_a + term_b) / 2)
        return float(sum(m_dist))

    pairwise = {}
    base = "bit_zero"
    for k in kinds:
        if k == base:
            continue
        pairwise[f"{base}_vs_{k}"] = js(histograms[base], histograms[k])
    differs_from_zero = any(v > 0.05 for v in pairwise.values())
    recommended = "noise_floor" if differs_from_zero else "room_tone"
    return {
        "status": "pass" if differs_from_zero else "fail",
        "histogram_sizes": {k: len(v) for k, v in histograms.items()},
        "pairwise_jsd": pairwise,
        "first_50_q1_per_kind": samples,
        "recommended_strategy": recommended,
    }


# ---------------------------------------------------------------------------
# Experiment 3 — 200-chunk LID rejection pilot
# ---------------------------------------------------------------------------

def _list_chunks(chunks_dir: Path, n: int = 200, seed: int = 0xC0FFEE) -> list[Path]:
    rng = random.Random(seed)
    files: list[Path] = []
    for ext in ("*.m4a", "*.wav", "*.flac", "*.mp3", "*.webm"):
        files.extend(chunks_dir.rglob(ext))
    rng.shuffle(files)
    return files[:n]


def experiment_lid_pilot(chunks_dir: Path, n: int = 200, device: str = "cuda:0") -> dict:
    """Sample N chunks, run faster-whisper LID. mms-lid-2048 is opt-in (Phase 3).

    Phase 0 only needs an order-of-magnitude rejection rate so the Phase 3
    threshold is realistic. Whisper LID alone is sufficient for that signal.
    """
    if not chunks_dir.exists():
        return {"status": "skip", "reason": f"chunks_dir missing: {chunks_dir}"}
    chunks = _list_chunks(chunks_dir, n=n)
    if len(chunks) == 0:
        return {"status": "skip", "reason": "no chunks found"}

    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3",
        device="cuda" if device.startswith("cuda") else "cpu",
        compute_type="float16" if device.startswith("cuda") else "int8",
    )
    low_conf = 0
    confidences: list[float] = []
    detected: list[str] = []
    failures = 0
    threshold = 0.7

    for path in chunks:
        try:
            # Pass path string — faster-whisper uses av (ffmpeg) under the hood
            # and handles .m4a / .webm / .mp4 natively. Avoids the soundfile
            # AAC support gap.
            _segments, info = model.transcribe(
                str(path), language=None, beam_size=1,
                vad_filter=False, no_speech_threshold=0.6,
            )
            confidences.append(float(info.language_probability))
            detected.append(info.language)
            if info.language_probability < threshold:
                low_conf += 1
        except Exception as e:
            failures += 1
            print(f"[lid_pilot] {path.name}: {e}", file=sys.stderr)

    n_evaluated = len(confidences)
    report: dict = {
        "status": "pass" if n_evaluated > 0 else "fail",
        "n_sampled": len(chunks),
        "n_evaluated": n_evaluated,
        "n_failures": failures,
        "low_conf_rate": (low_conf / n_evaluated) if n_evaluated else None,
        "confidence_mean": float(np.mean(confidences)) if confidences else None,
        "confidence_min": float(np.min(confidences)) if confidences else None,
        "language_counts": dict(Counter(detected)),
        "threshold": threshold,
    }
    if n_evaluated:
        rate = low_conf / n_evaluated
        report["recommended_threshold"] = (
            0.6 if rate > 0.5 else 0.7
        )
    else:
        report["recommended_threshold"] = 0.7
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_report(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Phase 0 Pre-flight Results\n\n")
        f.write(f"Run timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")

        f.write("## Experiment 1 — SP `<lang_xx>` roundtrip\n\n")
        e1 = results["sp_roundtrip"]
        f.write(f"- Status: **{e1['status']}**\n")
        f.write(f"- Pass rate: {e1['pass_rate']:.0%}\n")
        f.write(f"- Recommended `LANGUAGE_TAG_MODE`: `{e1['recommended_mode']}`\n")
        f.write(f"- Findings: `{json.dumps(e1['findings'], ensure_ascii=False)}`\n\n")

        f.write("## Experiment 2 — Mimi silence differentiation\n\n")
        e2 = results["mimi_silence_diff"]
        f.write(f"- Status: **{e2['status']}**\n")
        f.write(f"- Histogram sizes (q1 codebook unique IDs per kind): "
                f"`{json.dumps(e2['histogram_sizes'])}`\n")
        f.write(f"- Pairwise Jensen-Shannon vs bit_zero: "
                f"`{json.dumps({k: round(v, 4) for k, v in e2['pairwise_jsd'].items()})}`\n")
        f.write(f"- Recommended `SILENCE_STRATEGY`: `{e2['recommended_strategy']}`\n\n")

        f.write("## Experiment 3 — 200-chunk LID rejection pilot\n\n")
        e3 = results["lid_pilot"]
        f.write(f"- Status: **{e3['status']}**\n")
        for k in ("n_sampled", "n_evaluated", "n_failures",
                  "low_conf_rate", "confidence_mean", "confidence_min",
                  "recommended_threshold"):
            if k in e3:
                f.write(f"- {k}: {e3[k]}\n")
        if "language_counts" in e3:
            f.write("- language_counts: "
                    f"`{json.dumps(e3['language_counts'])}`\n")
        f.write("\n")

        f.write("## Locked env-var defaults\n\n")
        f.write(f"- `LANGUAGE_TAG_MODE={results['sp_roundtrip']['recommended_mode']}`\n")
        f.write(f"- `SILENCE_STRATEGY={results['mimi_silence_diff']['recommended_strategy']}`\n")
        f.write(f"- `LID_CONFIDENCE_THRESHOLD={results['lid_pilot'].get('recommended_threshold', 0.7)}`\n")


def run_phase0_preflight(
    planning_dir: str | Path = PLANNING_DIR,
    chunks_dir: str | Path | None = None,
    sp_path: str | Path = SP_MODEL_PATH,
    n_lid_pilot: int = 200,
    device: str = "cuda:0",
) -> dict:
    planning_dir = Path(planning_dir)
    chunks_dir = Path(chunks_dir) if chunks_dir else REPO_ROOT / "data" / "chunks_indic_yt"
    sp_path = str(sp_path)

    print("[phase0] Experiment 1 — SP <lang_xx> roundtrip")
    e1 = experiment_sp_roundtrip(sp_path)
    print(f"  → {e1['status']} ({e1['pass_rate']:.0%}); recommended {e1['recommended_mode']}")

    print("[phase0] Experiment 2 — Mimi silence differentiation")
    e2 = experiment_mimi_silence(device=device)
    print(f"  → {e2['status']}; recommended {e2['recommended_strategy']}")

    print(f"[phase0] Experiment 3 — LID pilot on {chunks_dir} (n={n_lid_pilot})")
    e3 = experiment_lid_pilot(chunks_dir, n=n_lid_pilot, device=device)
    print(f"  → {e3['status']}; "
          f"low_conf_rate={e3.get('low_conf_rate')}; "
          f"recommended_threshold={e3.get('recommended_threshold', 0.7)}")

    results = {"sp_roundtrip": e1, "mimi_silence_diff": e2, "lid_pilot": e3}
    out = planning_dir / "phase0_results.md"
    write_report(results, out)
    print(f"[phase0] Wrote {out}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planning_dir", default=str(PLANNING_DIR))
    p.add_argument("--chunks_dir", default=str(REPO_ROOT / "data" / "chunks_indic_yt"))
    p.add_argument("--sp_path", default=str(SP_MODEL_PATH))
    p.add_argument("--n_lid_pilot", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    results = run_phase0_preflight(
        planning_dir=args.planning_dir,
        chunks_dir=args.chunks_dir,
        sp_path=args.sp_path,
        n_lid_pilot=args.n_lid_pilot,
        device=args.device,
    )
    fail = (
        results["sp_roundtrip"]["status"] == "fail"
        or results["mimi_silence_diff"]["status"] == "fail"
    )
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
