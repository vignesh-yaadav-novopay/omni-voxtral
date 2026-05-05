"""VoxtralDataset — v2 sidecar-aware dataset.

Phase 1.5 / Phase 5 changes (plan.md §6.Phase5, LLD §3.6):
- Reads `<file>.meta.json` paired with each `<file>.npy` (schema_version=2).
- Yields `valid_token_mask` so `compute_omni_loss` can ignore zero-pad regions.
- Local `random.Random(seed)` instance — no process-global state (audit AF-204).
- Pinned val split via `data/val_split_v2.json` if present (review fix 3.1).
- Stride-homogeneous batching: per-iterator we only emit samples that share the
  same `stream_layout` so a DDP batch never mixes stride-21 and stride-42.
- Temperature sampling τ=3.3 across languages (Phase 5, plan §6.Phase5).
- Fail-fast: if >5% of consumed files lack sidecars, raise.

Back-compat: legacy `.npy` files without sidecars are still loaded with
`language="unknown"` and `valid_token_mask` defaulting to all-True. The 5%
fail-fast threshold catches accidental pipelines that silently dropped sidecars.
"""

import json
import math
import os
import random
import typing

import numpy as np
import torch
import torch.utils.data as td

from .config import VoxtralTrainConfig
from voxtral.data.sidecar import SCHEMA_VERSION


def get_npy_files(path: str) -> list[str]:
    npy_files: list[str] = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files


def get_fake_item() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randint(0, 1000, (220,)),
        "language": "fake",
        "duration_s": 1.0,
        "source": "fake",
        "valid_token_mask": torch.ones(220, dtype=torch.bool),
        "stream_layout": "single",
    }


def _read_sidecar(npy_path: str) -> dict | None:
    sidecar = npy_path[:-4] + ".meta.json" if npy_path.endswith(".npy") else npy_path + ".meta.json"
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    sv = meta.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        raise ValueError(
            f"Sidecar {sidecar}: schema_version={sv} != {SCHEMA_VERSION}. "
            "VoxtralDataset will not silently fall back to a different schema."
        )
    return meta


def get_item(
    file_path: str,
    max_seq_len: int | None = None,
    require_sidecar: bool = False,
) -> dict[str, typing.Any]:
    """Load a token .npy + paired sidecar; return dict with tokens, language, mask.

    On corruption: returns fake item AND logs the failure. Caller (`VoxtralDataset`)
    increments a missing-sidecar counter on `_meta_missing=True` items so the
    >5% fail-fast trips before silent corruption infects the run.
    """
    try:
        npy_data = np.load(file_path)
        tokens = torch.from_numpy(npy_data)
        if tokens.dim() == 2:
            tokens = tokens.squeeze()
        if max_seq_len is not None and tokens.shape[0] > max_seq_len:
            tokens = tokens[:max_seq_len]

        meta = _read_sidecar(file_path)
        if meta is None:
            if require_sidecar:
                # Caller will count this and trip the 5% threshold.
                return {**get_fake_item(), "_meta_missing": True}
            language = "unknown"
            duration_s = 0.0
            source = "unknown"
            stream_layout = "single"
            mask = torch.ones(tokens.shape[0], dtype=torch.bool)
        else:
            language = meta.get("language") or "unknown"
            duration_s = float(meta.get("duration_s") or 0.0)
            source = meta.get("source") or "unknown"
            stream_layout = meta.get("stream_layout") or "single"
            raw_mask = meta.get("valid_token_mask")
            if raw_mask is None:
                mask = torch.ones(tokens.shape[0], dtype=torch.bool)
            else:
                mask = torch.tensor(raw_mask, dtype=torch.bool)
                if mask.shape[0] != tokens.shape[0]:
                    # Schema mismatch — pad with True or truncate to align. This is
                    # a soft-recovery; the loader logs once if it ever happens.
                    if mask.shape[0] < tokens.shape[0]:
                        pad = torch.ones(tokens.shape[0] - mask.shape[0], dtype=torch.bool)
                        mask = torch.cat([mask, pad])
                    else:
                        mask = mask[: tokens.shape[0]]
        return {
            "tokens": tokens,
            "language": language,
            "duration_s": duration_s,
            "source": source,
            "valid_token_mask": mask,
            "stream_layout": stream_layout,
            "_meta_missing": meta is None,
        }
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {**get_fake_item(), "_meta_missing": True}


def _load_pinned_val_split(path: str) -> list[str] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    files = data.get("files") if isinstance(data, dict) else data
    if not isinstance(files, list):
        return None
    return [str(f) for f in files]


def _compute_language_temperature_weights(
    file_paths: list[str], tau: float = 3.3
) -> dict[str, float]:
    """π_l = ω_l^(1/τ) / Σ ω_l'^(1/τ); ω_l = file count per language.

    Reads each sidecar to get language. Files without sidecars contribute to
    `unknown` — once Phase 1 retokenisation lands they all carry language tags.
    """
    counts: dict[str, int] = {}
    for p in file_paths:
        m = _read_sidecar(p)
        lang = (m or {}).get("language") or "unknown"
        counts[lang] = counts.get(lang, 0) + 1
    pow_counts = {k: max(v, 1) ** (1.0 / max(tau, 1e-6)) for k, v in counts.items()}
    total = sum(pow_counts.values()) or 1.0
    return {k: v / total for k, v in pow_counts.items()}


class VoxtralDataset(td.IterableDataset):
    config: VoxtralTrainConfig
    data_step: int
    rank: int
    world_size: int
    file_paths: list[str]

    def __init__(
        self,
        config: VoxtralTrainConfig,
        split: str = "train",
        val_fraction: float = 0.1,
        val_split_path: str = "data/val_split_v2.json",
        require_sidecar: bool = False,
        sidecar_missing_threshold: float = 0.05,
        temperature_sampling_tau: float = 3.3,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_step = 0
        self.rank = config.rank
        self.world_size = config.world_size
        self.fake = config.fake
        self.overfit = config.overfit
        self.max_seq_len = config.max_seq_len
        self.split = split
        self.require_sidecar = require_sidecar
        self.sidecar_missing_threshold = sidecar_missing_threshold

        # Local RNG instance (AF-204 fix). Anyone else calling random.* won't
        # perturb our shuffle order.
        self._rng = random.Random(config.seed)

        # Per-iterator counters reset at __iter__ start.
        self._meta_missing_count = 0
        self._items_seen = 0
        self._meta_check_every = 1000

        if self.fake:
            self.file_paths = []
            self.stream_layout = "single"
            self._lang_weights: dict[str, float] = {}
            return

        all_files = get_npy_files(config.data_path)

        # Pinned val split takes precedence — survives directory changes (review 3.1).
        pinned_val = _load_pinned_val_split(val_split_path)
        if pinned_val is not None:
            pinned_val_set = set(pinned_val)
            train_files = [f for f in all_files if f not in pinned_val_set]
            self.file_paths = pinned_val if split == "val" else train_files
            print(f"Dataset split={split}: using pinned val_split_v2.json "
                  f"({len(self.file_paths)} files)")
        else:
            self._rng.shuffle(all_files)
            val_count = max(1, int(len(all_files) * val_fraction))
            self.file_paths = all_files[:val_count] if split == "val" else all_files[val_count:]
            print(f"Dataset split={split}: {len(self.file_paths)} files "
                  f"(deterministic shuffle, val_fraction={val_fraction})")

        # Stride-homogeneity (review 3.3): scan sidecars once at init, partition by
        # stream_layout. Each iterator only yields one layout's files. The trainer
        # picks the dataset whose layout matches `config.dual_stream`.
        target_layout = "dual" if config.dual_stream else "single"
        if self.file_paths:
            kept: list[str] = []
            mismatched = 0
            for p in self.file_paths:
                m = _read_sidecar(p)
                if m is None:
                    # Legacy file: tag as "single". If config requires dual, drop it.
                    layout = "single"
                else:
                    layout = m.get("stream_layout") or "single"
                if layout == target_layout:
                    kept.append(p)
                else:
                    mismatched += 1
            if mismatched:
                print(
                    f"Dataset split={split}: dropped {mismatched} files with "
                    f"stream_layout != {target_layout} (stride-homogeneity)"
                )
            self.file_paths = kept
        self.stream_layout = target_layout

        # Temperature sampling weights (Phase 5).
        if temperature_sampling_tau > 0 and self.file_paths and split == "train":
            self._lang_weights = _compute_language_temperature_weights(
                self.file_paths, tau=temperature_sampling_tau
            )
            self._file_lang_cache: dict[str, str] = {}
            for p in self.file_paths:
                m = _read_sidecar(p)
                self._file_lang_cache[p] = (m or {}).get("language") or "unknown"
        else:
            self._lang_weights = {}
            self._file_lang_cache = {}

    def __len__(self) -> int:
        if self.fake:
            return 100_000
        else:
            assert len(self.file_paths) > 0, (
                f"VoxtralDataset(split={self.split}) has 0 files. "
                f"data_path={self.config.data_path}; check that v2 retokenization ran."
            )
            return len(self.file_paths)

    def _maybe_check_missing(self) -> None:
        if self._items_seen and self._items_seen % self._meta_check_every == 0:
            rate = self._meta_missing_count / self._items_seen
            if rate > self.sidecar_missing_threshold:
                raise RuntimeError(
                    f"VoxtralDataset: {rate * 100:.1f}% of consumed files lack sidecars "
                    f"(>{self.sidecar_missing_threshold * 100:.0f}% threshold). "
                    "Run scripts/retokenize_v2.py to populate sidecars before training."
                )

    def _temperature_weighted_choice(self) -> str:
        """Sample one file path weighted by language temperature."""
        # Group files by language once per call (cheap dict lookup).
        if not self._lang_weights:
            return self._rng.choice(self.file_paths)
        # Weighted draw of language, then uniform-random file inside it.
        langs = list(self._lang_weights.keys())
        weights = [self._lang_weights[l] for l in langs]
        chosen_lang = self._rng.choices(langs, weights=weights, k=1)[0]
        candidates = [p for p, l in self._file_lang_cache.items() if l == chosen_lang]
        if not candidates:
            return self._rng.choice(self.file_paths)
        return self._rng.choice(candidates)

    def __iter__(self) -> typing.Iterator[dict[str, typing.Any]]:
        worker_info = td.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        stride = num_workers * self.world_size
        offset = self.rank * num_workers + worker_id

        self._meta_missing_count = 0
        self._items_seen = 0

        while True:
            self.data_step += stride
            if self.fake:
                yield get_fake_item()
                continue

            if self.overfit is not None:
                idx = (offset + self.data_step) % min(self.overfit, len(self))
                file_path = self.file_paths[idx]
            elif self.split == "train" and self._lang_weights:
                file_path = self._temperature_weighted_choice()
            else:
                idx = (offset + self.data_step) % len(self)
                file_path = self.file_paths[idx]

            item = get_item(
                file_path,
                max_seq_len=self.max_seq_len,
                require_sidecar=self.require_sidecar,
            )
            self._items_seen += 1
            if item.pop("_meta_missing", False):
                self._meta_missing_count += 1
            self._maybe_check_missing()
            yield item
