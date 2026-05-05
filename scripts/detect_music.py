"""Phase 4 helper — music detection per source URL.

Spectral flux + harmonicity + tempo. Cheap, CPU-only. Output is consumed by
`scripts/diarize_v2.py` to gate Demucs vocal separation: if music_likely=True,
run Demucs; otherwise skip (Demucs is the slowest stage by far, and over-strips
quiet podcasts when run unconditionally).

Output: data/source_music_flags.json mapping `{source_id: {music_likely, ...}}`.

Usage:
    uv run scripts/detect_music.py --input_dir data/chunks_indic_yt \\
        --output data/source_music_flags.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("detect_music")


def _atomic_write_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def detect_music_in_source(audio_path: str) -> dict:
    """Return {music_likely, harmonicity, tempo_detected, ...}.

    Heuristic: harmonicity ratio > 0.6 AND beat-tracker finds tempo → likely music.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=22_050, mono=True, duration=60.0)
    if len(y) < sr // 2:
        return {"music_likely": False, "reason": "too_short"}
    # Harmonicity (HPSS): ratio of harmonic energy to total energy
    y_h, y_p = librosa.effects.hpss(y)
    e_h = float(np.mean(y_h ** 2))
    e_total = float(np.mean(y ** 2)) + 1e-12
    harmonicity = e_h / e_total
    # Beat tracker
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo)
        tempo_detected = 50.0 < tempo_val < 220.0
    except Exception:
        tempo_val = 0.0
        tempo_detected = False

    music_likely = (harmonicity > 0.6) and tempo_detected
    return {
        "music_likely": bool(music_likely),
        "harmonicity": harmonicity,
        "tempo_bpm": tempo_val,
        "tempo_detected": tempo_detected,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Source audio dir; we sample N files per `source_id` for the flag")
    p.add_argument("--output", default="data/source_music_flags.json")
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()

    src_root = Path(args.input_dir)
    files: list[Path] = []
    for ext in ("*.m4a", "*.wav", "*.flac", "*.mp3", "*.webm", "*.mp4"):
        files.extend(src_root.rglob(ext))
    files.sort()
    if args.max_files:
        files = files[: args.max_files]

    # Group by source_id (filename stem prefix before "_")
    by_source: dict[str, list[Path]] = {}
    for f in files:
        # YouTube chunks named <video-id>_<chunk-index>.m4a
        stem = f.stem.split("_")[0]
        by_source.setdefault(stem, []).append(f)

    flags: dict[str, dict] = {}
    for src_id, paths in by_source.items():
        sample = paths[0]
        try:
            result = detect_music_in_source(str(sample))
        except Exception as e:
            log.warning(f"music detect failed on {sample.name}: {e}")
            result = {"music_likely": False, "error": str(e)}
        flags[src_id] = result

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(flags, args.output)
    music_count = sum(1 for v in flags.values() if v.get("music_likely"))
    log.info(f"wrote {args.output}: {len(flags)} sources, {music_count} flagged music_likely")


if __name__ == "__main__":
    main()
