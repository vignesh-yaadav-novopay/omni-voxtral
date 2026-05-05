"""Phase 1.5 — generate `data/val_split_v2.json`, the pinned validation set.

Locked decision #9 in the plan: phase-to-phase val comparisons need a fixed
set of files so an improvement at step 40K vs step 80K isn't being measured
through a different val sampling. This script walks `data/tokens_v2/` (or any
configured token tree), groups files by language, picks a deterministic
stratified sample, and emits a JSON file the trainer reads via
`VoxtralDataset(..., val_split_path=...)`.

Stratified sampling:
- Per-language quota: ceil(per_lang_target). Default 30 files/language.
- Files within each language are sorted, then sampled uniformly across the
  sorted list using a fixed seed (deterministic across re-runs once the
  underlying corpus stops growing).
- Any language with fewer than `min_per_lang` files is logged as "underfilled"
  but kept in the val set with whatever it has — better to have *some* val for
  Bodo than skip the language entirely.

Schema (v2):
    {
        "schema_version": 1,
        "created_at": "2026-05-05T12:34:56Z",
        "seed": 1729,
        "per_language": {"hin": 30, "ben": 30, ...},
        "files": ["data/tokens_v2/fleurs/hin/hi/hin_00000001_abc.npy", ...]
    }

Usage:
    uv run scripts/build_val_split.py
    uv run scripts/build_val_split.py --tokens_root data/tokens_v2 --per_lang 30 --output data/val_split_v2.json
    uv run scripts/build_val_split.py --tokens_root data/tokens_v2 --per_lang 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_val_split")


SCHEMA_VERSION = 1
DEFAULT_TOKENS_ROOT = "data/tokens_v2"
DEFAULT_OUTPUT = "data/val_split_v2.json"
DEFAULT_PER_LANG = 30
DEFAULT_SEED = 1729
DEFAULT_MIN_PER_LANG = 5


def _read_lang_from_sidecar(npy_path: Path) -> str | None:
    """Read the language tag from the paired .meta.json sidecar.

    The trainer needs language to be authoritative — it uses the sidecar tag
    for stride homogeneity, temperature sampling, and val grouping. So we
    DON'T derive language from the directory name (which can drift from the
    actual content if files were moved manually).
    """
    meta_path = npy_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta.get("language")


def discover_files_by_language(tokens_root: str) -> dict[str, list[str]]:
    """Walk `tokens_root` and group .npy files by their sidecar `language` tag."""
    root = Path(tokens_root)
    if not root.exists():
        log.error(f"tokens_root does not exist: {root}")
        return {}

    by_lang: dict[str, list[str]] = defaultdict(list)
    n_total = 0
    n_no_meta = 0
    for npy in root.rglob("*.npy"):
        n_total += 1
        lang = _read_lang_from_sidecar(npy)
        if lang is None:
            n_no_meta += 1
            continue
        by_lang[lang].append(str(npy))

    log.info(
        f"discovered {n_total} .npy files; {n_no_meta} missing sidecar; "
        f"{len(by_lang)} languages found: {sorted(by_lang.keys())}"
    )
    if n_no_meta > 0:
        log.warning(
            f"{n_no_meta} files have no v2 sidecar — they cannot be sampled "
            "into val. Re-run retokenize_v2.py to fix."
        )
    return dict(by_lang)


def stratified_sample(
    by_lang: dict[str, list[str]],
    per_lang: int,
    seed: int,
    min_per_lang: int = DEFAULT_MIN_PER_LANG,
) -> tuple[list[str], dict[str, int]]:
    """Pick `per_lang` files from each language deterministically.

    Files within each language are sorted to give a stable order, then the
    seeded RNG samples without replacement. If a language has fewer than
    `per_lang` files, takes everything; if fewer than `min_per_lang`, also logs
    a warning so we know which languages are underrepresented.

    Returns (selected_files, per_lang_count_dict).
    """
    rng = random.Random(seed)
    selected: list[str] = []
    counts: dict[str, int] = {}

    for lang in sorted(by_lang.keys()):
        files = sorted(by_lang[lang])  # stable ordering
        n = min(per_lang, len(files))
        if len(files) < min_per_lang:
            log.warning(
                f"[{lang}] only {len(files)} files available (< min_per_lang={min_per_lang}); "
                "val coverage for this language will be weak."
            )
        # deterministic sampling: seed + lang ensures reproducibility per-language
        lang_rng = random.Random(rng.random() * 1e9 + hash(lang))
        chosen = lang_rng.sample(files, n) if n > 0 else []
        selected.extend(sorted(chosen))  # sorted within lang for readability
        counts[lang] = n

    return selected, counts


def write_val_split(output_path: str, files: list[str], counts: dict[str, int], seed: int) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "per_language": counts,
        "n_total": len(files),
        "files": files,
    }
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(out))
    log.info(f"wrote {output_path} — {len(files)} files across {len(counts)} languages")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens_root", default=DEFAULT_TOKENS_ROOT,
                   help="Root of v2 token tree (default: data/tokens_v2)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output JSON path (default: data/val_split_v2.json)")
    p.add_argument("--per_lang", type=int, default=DEFAULT_PER_LANG,
                   help="Files per language (default: 30)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for deterministic sampling (default: 1729)")
    p.add_argument("--min_per_lang", type=int, default=DEFAULT_MIN_PER_LANG,
                   help="Languages with fewer files trigger a warning")
    args = p.parse_args()

    by_lang = discover_files_by_language(args.tokens_root)
    if not by_lang:
        log.error("no files found — nothing to write")
        raise SystemExit(2)

    selected, counts = stratified_sample(
        by_lang, per_lang=args.per_lang, seed=args.seed, min_per_lang=args.min_per_lang,
    )
    write_val_split(args.output, selected, counts, args.seed)


if __name__ == "__main__":
    main()
