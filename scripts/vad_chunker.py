"""Phase 2 — VAD-aware variable-length chunking.

Replaces the 20s hard-cut chunker. Per-channel Silero VAD v6 with stereo
correlation detection: when L/R channels carry different speakers (correlation
< STEREO_CORRELATION_THRESHOLD, default 0.5), each region emits two chunks
tagged with `stream_role` ∈ {user, model}; otherwise audio is mixed to mono
and a single chunk is emitted.

Output:
    data/chunks_v2/{source}/{lang_or_unknown}/{shard}/{uuid}.wav
    data/chunks_v2/{source}/{lang_or_unknown}/{shard}/{uuid}.json   ← chunk meta

VAD parameters (LLD §6 / plan §4 — review-locked):
    min_speech_duration_ms = 300   # ↓ from 1000 to capture Indic backchannels
    min_silence_duration_ms = 500
    speech_pad_ms = 150
    max_speech_duration_s = 20

Usage:
    bash autoresearch/scripts/run_safe.sh \
        "uv run scripts/vad_chunker.py --input_dir data/chunks_indic_yt \
            --output_dir data/chunks_v2/yt --source yt --lang_or_unknown unknown \
            --rank 0 --world_size 4" 86400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vad_chunker")

VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", 300))
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", 500))
VAD_PAD_MS = int(os.environ.get("VAD_PAD_MS", 150))
VAD_MAX_SPEECH_S = int(os.environ.get("VAD_MAX_SPEECH_S", 20))
STEREO_CORRELATION_THRESHOLD = float(os.environ.get("STEREO_CORRELATION_THRESHOLD", 0.5))

TARGET_SR = 24_000
SILERO_SR = 16_000  # Silero v6 expects 16kHz


def _load_audio_any(path: str) -> tuple[torch.Tensor, int]:
    """Load any audio container including AAC/.m4a (which soundfile rejects).

    torchaudio.load() routes through soundfile by default, which lacks AAC
    support — every YouTube .m4a file fails with "Format not recognised".
    We detect that and fall back to PyAV (already installed via faster-whisper),
    decoding via ffmpeg's native AAC decoder.

    Returns (wav, sr). wav is float32, shape (channels, samples).
    """
    try:
        return torchaudio.load(path)
    except Exception as soundfile_err:
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".m4a", ".aac", ".mp4", ".webm", ".mkv", ".mov"}:
            raise
    # PyAV path — only invoked for AAC-family containers.
    try:
        import av
    except ImportError as e:
        raise RuntimeError(
            "PyAV not installed; cannot decode .m4a/.aac. uv add av or "
            "convert files to .wav first."
        ) from e
    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise RuntimeError(f"{path}: no audio stream found by PyAV")
    sr = int(stream.rate)
    channels: list[list[np.ndarray]] = []
    for frame in container.decode(stream):
        # frame.to_ndarray() returns shape (channels, samples) for planar formats.
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[None, :]
        for c in range(arr.shape[0]):
            if c >= len(channels):
                channels.append([])
            channels[c].append(arr[c])
    container.close()
    if not channels:
        raise RuntimeError(f"{path}: PyAV produced no audio frames")
    arr = np.stack([np.concatenate(c) for c in channels], axis=0)
    # PyAV gives int16 for most containers; normalize to float32 in [-1, 1].
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / max(abs(info.min), info.max)
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr), sr


def _load_silero():
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=False)


def detect_speaker_per_channel(stereo_waveform: torch.Tensor) -> bool:
    """Pearson correlation between L/R. < threshold → likely speaker-per-channel."""
    if stereo_waveform.size(0) < 2:
        return False
    left = stereo_waveform[0].float() - stereo_waveform[0].float().mean()
    right = stereo_waveform[1].float() - stereo_waveform[1].float().mean()
    denom = (left.norm() * right.norm()).clamp_min(1e-8)
    corr = float((left * right).sum() / denom)
    return corr < STEREO_CORRELATION_THRESHOLD


def _silero_segments(model, audio_16k_mono: torch.Tensor) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...]. min_speech, max_speech, padding all applied
    inside Silero's `get_speech_timestamps`."""
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(
        audio_16k_mono,
        model,
        sampling_rate=SILERO_SR,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_PAD_MS,
        max_speech_duration_s=VAD_MAX_SPEECH_S,
        return_seconds=True,
    )
    return [(s["start"], s["end"]) for s in ts]


def run_silero_vad_per_channel(
    waveform: torch.Tensor,
    sample_rate: int,
    model,
) -> list[list[tuple[float, float]]]:
    """For (C, T) waveform, return per-channel speech boundaries in seconds."""
    if sample_rate != SILERO_SR:
        wav = torchaudio.functional.resample(waveform, sample_rate, SILERO_SR)
    else:
        wav = waveform
    out = []
    for c in range(wav.size(0)):
        out.append(_silero_segments(model, wav[c]))
    return out


def _atomic_write_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_write_wav(audio: torch.Tensor, sr: int, path: str) -> None:
    # torchaudio.save derives the format from the extension, so we keep .wav
    # in the temp name and prepend `.partial` instead of using `.tmp`.
    tmp = path[:-4] + ".partial.wav" if path.endswith(".wav") else f"{path}.partial.wav"
    torchaudio.save(tmp, audio, sr)
    os.replace(tmp, path)


def chunk_audio_file(
    *,
    input_path: str,
    output_dir: str,
    source: str,
    language_or_unknown: str,
    silero_model,
    detect_stereo_speakers: bool = True,
    correlation_threshold: float = STEREO_CORRELATION_THRESHOLD,
) -> list[dict]:
    """Process one source audio file. Returns list of chunk-meta dicts emitted."""
    try:
        wav, sr = _load_audio_any(input_path)
    except Exception as e:
        log.warning(f"failed to load {input_path}: {e}")
        return []

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR

    is_stereo_speakers = (
        detect_stereo_speakers
        and wav.size(0) >= 2
        and detect_speaker_per_channel(wav)
    )

    chunks_meta: list[dict] = []
    src_id = Path(input_path).stem
    out_root = Path(output_dir)

    if is_stereo_speakers:
        # Two parallel VAD passes, one per channel. Each emits its own chunks.
        per_channel_segments = run_silero_vad_per_channel(wav, sr, silero_model)
        roles = ["user", "model"]  # left = user, right = model (convention)
        for ch_idx, segments in enumerate(per_channel_segments):
            for seg_idx, (start_s, end_s) in enumerate(segments):
                start = int(start_s * sr)
                end = int(end_s * sr)
                clip = wav[ch_idx:ch_idx + 1, start:end].contiguous()
                cm = _emit_chunk(
                    clip=clip, sr=sr, source=source, source_id=src_id,
                    language_or_unknown=language_or_unknown,
                    out_root=out_root, start_s=start_s, end_s=end_s,
                    chunk_index=len(chunks_meta), num_channels=1,
                    stream_role=roles[ch_idx],
                )
                chunks_meta.append(cm)
    else:
        # Mix to mono, single VAD pass.
        mono = wav.mean(dim=0, keepdim=True) if wav.size(0) > 1 else wav
        segments = _silero_segments(
            silero_model,
            torchaudio.functional.resample(mono, sr, SILERO_SR)[0]
            if sr != SILERO_SR else mono[0],
        )
        for seg_idx, (start_s, end_s) in enumerate(segments):
            start = int(start_s * sr)
            end = int(end_s * sr)
            clip = mono[:, start:end].contiguous()
            cm = _emit_chunk(
                clip=clip, sr=sr, source=source, source_id=src_id,
                language_or_unknown=language_or_unknown,
                out_root=out_root, start_s=start_s, end_s=end_s,
                chunk_index=seg_idx, num_channels=1,
                stream_role="single",
            )
            chunks_meta.append(cm)

    return chunks_meta


def _emit_chunk(
    *,
    clip: torch.Tensor,
    sr: int,
    source: str,
    source_id: str,
    language_or_unknown: str,
    out_root: Path,
    start_s: float,
    end_s: float,
    chunk_index: int,
    num_channels: int,
    stream_role: str,
) -> dict:
    duration_s = float(clip.size(-1) / sr)
    chunk_uuid = uuid.uuid4().hex
    shard = chunk_uuid[:2]
    out_dir = out_root / language_or_unknown / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{chunk_uuid}.wav"
    json_path = out_dir / f"{chunk_uuid}.json"
    _atomic_write_wav(clip, sr, str(wav_path))
    chunk_meta = {
        "source": source,
        "source_id": source_id,
        "source_url": None,
        "chunk_index": chunk_index,
        "start_s_in_source": float(start_s),
        "end_s_in_source": float(end_s),
        "duration_s": duration_s,
        "num_channels": num_channels,
        "stream_role": stream_role,
        # `lid_inherits_from_neighbor` filled by Phase 3 if duration < 1s
        "lid_inherits_from_neighbor": duration_s < 1.0,
        "language": language_or_unknown if language_or_unknown != "unknown" else None,
        "schema_version": 2,
    }
    _atomic_write_json(chunk_meta, str(json_path))
    return chunk_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--lang_or_unknown", default="unknown")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()

    silero = _load_silero()

    in_root = Path(args.input_dir)
    files: list[Path] = []
    for ext in ("*.m4a", "*.wav", "*.flac", "*.mp3", "*.webm", "*.mp4"):
        files.extend(in_root.rglob(ext))
    files.sort()
    if args.max_files:
        files = files[: args.max_files]
    log.info(f"rank={args.rank}/{args.world_size} files={len(files)}")
    total = 0
    processed = 0
    for idx, path in enumerate(files):
        if idx % args.world_size != args.rank:
            continue
        chunks = chunk_audio_file(
            input_path=str(path),
            output_dir=args.output_dir,
            source=args.source,
            language_or_unknown=args.lang_or_unknown,
            silero_model=silero,
        )
        total += len(chunks)
        processed += 1
        # Log per-rank progress. Previously gated on `(idx+1) % 50 == 0`,
        # which never fires for ranks where rank % world_size doesn't align
        # with that modulo (e.g. world_size=4 with ranks 0 or 2 — no integer
        # satisfies both idx%4==rank and idx%50==49). Switched to per-rank
        # `processed` counter so every rank logs every 50 files it owns.
        if processed % 50 == 0:
            log.info(f"  processed {processed} files (idx {idx + 1}/{len(files)}), emitted {total} chunks")
    log.info(f"done: emitted {total} chunks across {len(files)} source files")


if __name__ == "__main__":
    main()
