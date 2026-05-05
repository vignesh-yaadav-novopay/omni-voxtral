"""Sidecar metadata I/O for v2 token files.

Every `<file>.npy` ships with a paired `<file>.meta.json` matching `ChunkMetadata`.
Both are written atomically (`<path>.tmp` → `os.replace`) so a crashed preprocess
run never produces a half-written sidecar that the trainer later mis-reads.

LLD §4 schema. Loader contract in §4.2: schema_version != 2 raises; missing
fields are tolerated (treated as `None`); a >5% rate of missing sidecars at
training time triggers a fail-fast in the dataset.
"""

import dataclasses
import datetime
import json
import os
import typing
import uuid

import numpy as np

SCHEMA_VERSION = 2
PREPROCESSING_VERSION = "v2.0"


@dataclasses.dataclass
class ChunkMetadata:
    """v2 sidecar shape — see plan.md §7 and LLD.md §4.1."""

    schema_version: int = SCHEMA_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    preprocessing_run_id: str = ""
    preprocessing_timestamp: str = ""

    source: str = "unknown"
    source_id: str = ""
    source_url: str | None = None
    chunk_index: int = 0

    language: str = "unknown"
    language_confidence: float = 1.0
    language_method: str = "caller_provided"
    language_secondary: dict | None = None
    lid_inherits_from_neighbor: bool = False

    transcript: str = ""
    translation_en: str | None = None
    transcript_method: str = "whisper_large_v3"
    transcript_avg_logprob: float = 0.0

    duration_s: float = 0.0
    sample_rate: int = 24000
    num_channels: int = 1
    num_speakers: int = 1
    speaker_segments: list[dict] = dataclasses.field(default_factory=list)

    snr_db: float | None = None
    speech_ratio: float | None = None
    clip_ratio: float | None = None
    music_likely: bool | None = None

    stream_layout: str = "single"
    tokenizer_config: dict = dataclasses.field(default_factory=dict)
    token_count: int = 0
    token_range: tuple[int, int] = (0, 0)
    valid_token_mask: list[bool] | None = None

    data_license_class: str = "research_only"
    quarantine_reason: str | None = None
    task: str = "transcribe"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def new_run_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_save_npy(arr: np.ndarray, path: str) -> None:
    """np.save with `<path>.tmp` + `os.replace` so partial writes are never visible."""
    tmp = f"{path}.tmp"
    np.save(tmp, arr)
    # np.save adds .npy if missing — detect and rename.
    actual_tmp = tmp if os.path.exists(tmp) else f"{tmp}.npy"
    target = path if path.endswith(".npy") else f"{path}.npy"
    os.replace(actual_tmp, target)


def atomic_write_json(obj: dict, path: str) -> None:
    """Same atomicity contract as atomic_save_npy."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_metadata_sidecar(
    npy_path: str,
    metadata: dict,
    *,
    valid_token_mask: list[bool] | None = None,
    run_id: str | None = None,
) -> str:
    """Write `<npy_path>.meta.json` atomically. Returns the sidecar path.

    `metadata` is the dict from `VoxtralTokenizer.encode()` plus any caller fields.
    Missing fields are filled from `ChunkMetadata` defaults so schema is uniform.
    `valid_token_mask` (Phase 1.5) marks which token positions are real audio
    (False positions are zero-padding the trainer should ignore).
    """
    base = ChunkMetadata().to_dict()
    base.update(metadata or {})
    base["preprocessing_run_id"] = run_id or base.get("preprocessing_run_id") or new_run_id()
    base["preprocessing_timestamp"] = base.get("preprocessing_timestamp") or now_iso()
    if valid_token_mask is not None:
        base["valid_token_mask"] = list(map(bool, valid_token_mask))
    sidecar_path = npy_path.replace(".npy", ".meta.json") if npy_path.endswith(".npy") else f"{npy_path}.meta.json"
    atomic_write_json(base, sidecar_path)
    return sidecar_path


def read_metadata_sidecar(npy_path: str) -> dict | None:
    """Read paired sidecar. Returns None if missing.

    Raises on schema_version mismatch (LLD §4.2: do not silently fall back).
    """
    sidecar_path = npy_path.replace(".npy", ".meta.json") if npy_path.endswith(".npy") else f"{npy_path}.meta.json"
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, encoding="utf-8") as f:
        meta = json.load(f)
    sv = meta.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"Sidecar {sidecar_path}: schema_version={sv} != {SCHEMA_VERSION}. "
            "Loader will not silently fall back."
        )
    return meta


def derive_valid_token_mask_from_audio_length(
    *,
    audio_samples: int,
    chunk_samples: int,
    token_count: int,
    stride: int,
) -> list[bool]:
    """For Phase 1a (tokens still come from 20s zero-padded chunks), mark tokens
    inside the real-audio prefix as True and the zero-pad tail as False.

    The token sequence has `token_count // stride` windows total; only
    `floor(audio_samples / chunk_samples * num_windows)` correspond to real audio.
    Used by `_save_tokens`-equivalent paths so Phase 1.5 trainer masks zero-pad.
    """
    if audio_samples >= chunk_samples or chunk_samples == 0:
        return [True] * token_count
    num_windows = token_count // stride
    real_windows = int(num_windows * audio_samples / chunk_samples)
    real_tokens = real_windows * stride
    mask = [True] * min(real_tokens, token_count) + [False] * max(0, token_count - real_tokens)
    return mask
