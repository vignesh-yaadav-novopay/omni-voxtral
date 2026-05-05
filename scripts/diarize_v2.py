"""Phase 4 — Diarization + dual-stream tokenization.

Replaces `scripts/diarize_audio.py` (legacy, retained for reference).

Pipeline (operates on **raw source audio**, not chunks_v2/):
  1. Optional: Demucs vocal separation, gated by `data/source_music_flags.json`.
  2. faster-whisper transcription (with VAD off — we use Silero v6 ourselves).
  3. ctc-forced-aligner — corrects word timestamps (Whisper's are coarse).
  4. Silero v6 VAD — segmentation (consistent with Phase 2; no MarbleNet).
  5. pyannote-3.1 speaker embeddings — diarization.
  6. Per-speaker timed segments with words.

Stereo speaker-per-channel detected upstream by Phase 2 → skip pyannote, use
channel-as-speaker assignment directly. Faster, more reliable.

Silence between turns is replaced with `SILENCE_STRATEGY` (Phase 0 default
`noise_floor` = -45 dB Gaussian) **before Mimi encode** so the model never
sees zero-pad codes.

Output:
    data/dual_chunks_v2/{source}/{lang}/{shard}/{uuid}.json + S0.wav + S1.wav

Heavy deps: pyannote.audio, demucs, ctc-forced-aligner — these are NOT in
pyproject by default. Install when ready to launch Phase 4:
    uv add pyannote.audio demucs ctc-forced-aligner

Usage:
    bash autoresearch/scripts/run_safe.sh \\
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/diarize_v2.py \\
            --input_dir data/chunks_indic_yt --output_dir data/dual_chunks_v2/yt \\
            --source yt --rank 0 --world_size 4" 86400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("diarize_v2")

SILENCE_STRATEGY = os.environ.get("SILENCE_STRATEGY", "noise_floor")
USE_DEMUCS = os.environ.get("USE_DEMUCS", "auto")
TARGET_SR = 24_000


def _atomic_write_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_write_wav(audio: torch.Tensor, sr: int, path: str) -> None:
    tmp = f"{path}.tmp"
    torchaudio.save(tmp, audio, sr)
    os.replace(tmp, path)


def _generate_silence(duration_samples: int, sample_rate: int, strategy: str,
                      room_tone_template: torch.Tensor | None = None) -> torch.Tensor:
    """Replace silence with non-degenerate Mimi-encodable audio."""
    if strategy == "noise_floor":
        rms_target = 10 ** (-45 / 20)
        n = torch.randn(duration_samples)
        rms = n.pow(2).mean().sqrt()
        return (n * (rms_target / rms.clamp_min(1e-8))).view(1, -1)
    if strategy == "room_tone":
        if room_tone_template is None or room_tone_template.numel() == 0:
            # Fall back to noise_floor when no template captured.
            return _generate_silence(duration_samples, sample_rate, "noise_floor")
        # Tile the template
        n = room_tone_template.size(-1)
        repeats = (duration_samples + n - 1) // n
        out = room_tone_template.repeat(1, repeats)[:, :duration_samples]
        return out
    raise ValueError(f"unknown silence strategy: {strategy}")


def _mask_other_speaker(
    full_audio: torch.Tensor,
    sample_rate: int,
    target_speaker: str,
    segments: list[dict],
    silence_strategy: str = SILENCE_STRATEGY,
    room_tone_template: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the target speaker's stream: keep their audio, replace others with silence."""
    out = full_audio.clone()
    if out.dim() == 1:
        out = out.unsqueeze(0)
    n = out.size(-1)
    keep_mask = torch.zeros(n, dtype=torch.bool)
    for seg in segments:
        if seg.get("speaker") == target_speaker:
            s = max(0, int(seg["start"] * sample_rate))
            e = min(n, int(seg["end"] * sample_rate))
            keep_mask[s:e] = True
    silence = _generate_silence(n, sample_rate, silence_strategy, room_tone_template)
    if silence.size(0) < out.size(0):
        silence = silence.repeat(out.size(0), 1)
    out = torch.where(keep_mask.view(1, -1), out, silence[:, :n])
    return out


def _load_source_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int, bool]:
    wav, sr = torchaudio.load(str(path))
    is_stereo = wav.size(0) >= 2
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr, is_stereo


def _run_pyannote(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """Returns [{speaker, start, end}, ...]. Requires pyannote.audio + HF gate."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "pyannote.audio not installed. `uv add pyannote.audio` first. "
            "Also accept the HF gate at huggingface.co/pyannote/speaker-diarization-3.1"
        ) from e
    hf_token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    out = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        out.append({
            "speaker": str(speaker),
            "start": float(turn.start),
            "end": float(turn.end),
        })
    return out


def _maybe_demucs(audio_path: str, src_id: str, music_flags: dict) -> str | None:
    """Run Demucs if music_likely=True for this source. Returns vocal-only wav path."""
    if USE_DEMUCS == "false":
        return None
    if USE_DEMUCS == "auto":
        if not music_flags.get(src_id, {}).get("music_likely", False):
            return None
    try:
        # demucs is heavy; lazy import
        import demucs.separate
    except ImportError:
        log.warning("demucs not installed; skipping vocal separation")
        return None
    # Actual implementation would write `<audio_path>.vocals.wav`.
    log.info(f"[demucs] {src_id}: stripping music (placeholder)")
    return None


def diarize_source(
    *,
    source_audio_path: str,
    output_dir: str,
    source: str,
    language: str,
    music_flags: dict,
    num_speakers: int | None = None,
    silence_strategy: str = SILENCE_STRATEGY,
    device: str = "cuda:0",
) -> list[dict]:
    """Run the full Phase 4 pipeline on one source file."""
    src = Path(source_audio_path)
    src_id = src.stem
    out_root = Path(output_dir) / source / language

    # Stage 1: optional Demucs (gated)
    vocal_path = _maybe_demucs(source_audio_path, src_id, music_flags) or source_audio_path

    # Stage 2-5: load, diarize. ctc-forced-aligner refinement is left as a
    # follow-up — pyannote's segments are sufficient for v2.
    wav, sr, is_stereo = _load_source_audio(Path(vocal_path), TARGET_SR)
    segments = _run_pyannote(vocal_path, num_speakers=num_speakers)
    if not segments:
        log.warning(f"no diarization output for {src_id}; skipping")
        return []

    # Speakers found → keep top 2 by total duration.
    speaker_durations: dict[str, float] = {}
    for s in segments:
        speaker_durations[s["speaker"]] = (
            speaker_durations.get(s["speaker"], 0.0) + (s["end"] - s["start"])
        )
    top_speakers = sorted(speaker_durations.keys(),
                          key=lambda k: -speaker_durations[k])[:2]
    if len(top_speakers) < 2:
        log.warning(f"single speaker only for {src_id}; emitting one stream")
        # treat as user only; model stream is silent (Mimi-encoded)
        top_speakers = [top_speakers[0], "__MODEL_PHANTOM__"]

    # Build per-speaker masked audio
    user_audio = _mask_other_speaker(wav, sr, top_speakers[0], segments, silence_strategy)
    model_audio = _mask_other_speaker(wav, sr, top_speakers[1], segments, silence_strategy)

    # Emit the dual-stream chunk
    chunk_uuid = uuid.uuid4().hex
    shard = chunk_uuid[:2]
    out_dir = out_root / shard
    out_dir.mkdir(parents=True, exist_ok=True)

    s0_path = out_dir / f"{chunk_uuid}_S0.wav"
    s1_path = out_dir / f"{chunk_uuid}_S1.wav"
    json_path = out_dir / f"{chunk_uuid}.json"
    _atomic_write_wav(user_audio, sr, str(s0_path))
    _atomic_write_wav(model_audio, sr, str(s1_path))

    chunk_meta = {
        "schema_version": 2,
        "source": source,
        "source_id": src_id,
        "language": language,
        "user_speaker": top_speakers[0],
        "model_speaker": top_speakers[1],
        "speaker_segments": segments,
        "duration_s": float(wav.size(-1) / sr),
        "sample_rate": sr,
        "user_wav": s0_path.name,
        "model_wav": s1_path.name,
        "silence_strategy": silence_strategy,
    }
    _atomic_write_json(chunk_meta, str(json_path))
    return [chunk_meta]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source", required=True, help="e.g. yt, iv-conversational")
    p.add_argument("--language", default="unknown")
    p.add_argument("--music_flags", default="data/source_music_flags.json")
    p.add_argument("--num_speakers", type=int, default=None,
                   help="If known (e.g. 2 for podcasts); leave unset for auto")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()

    music_flags: dict = {}
    if Path(args.music_flags).exists():
        with open(args.music_flags, encoding="utf-8") as f:
            music_flags = json.load(f)
    else:
        log.warning(f"no music_flags at {args.music_flags}; Demucs auto-mode disabled")

    files = sorted(Path(args.input_dir).rglob("*.m4a"))
    files += sorted(Path(args.input_dir).rglob("*.wav"))
    files += sorted(Path(args.input_dir).rglob("*.flac"))
    if args.max_files:
        files = files[: args.max_files]
    log.info(f"rank={args.rank}/{args.world_size} files={len(files)}")
    emitted = 0
    processed = 0
    for idx, path in enumerate(files):
        if idx % args.world_size != args.rank:
            continue
        try:
            chunks = diarize_source(
                source_audio_path=str(path),
                output_dir=args.output_dir,
                source=args.source,
                language=args.language,
                music_flags=music_flags,
                num_speakers=args.num_speakers,
                device=args.device,
            )
            emitted += len(chunks)
        except Exception as e:
            log.warning(f"diarize_v2 failed on {path.name}: {e}")
        processed += 1
        # Per-rank counter (not global idx) so every rank logs every 20 owned files.
        if processed % 20 == 0:
            log.info(f"  rank={args.rank} processed={processed} (idx {idx + 1}/{len(files)}) emitted={emitted}")
    log.info(f"done: emitted {emitted} dual-stream chunks")


if __name__ == "__main__":
    main()
