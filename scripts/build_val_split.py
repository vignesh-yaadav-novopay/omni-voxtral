"""Phase 1.5 — generate `data/val_split_v2.json`, the pinned validation set.

Locked decision #9 in the plan: phase-to-phase val comparisons need a fixed
set of files so an improvement at step 40K vs step 80K isn't being measured
through a different val sampling. This script walks `data/tokens_v2/`, groups
files by language AND source (fleurs / iv / yt), picks a deterministic
stratified sample per (lang, source) cell, and emits a JSON file the trainer
reads via `VoxtralDataset(..., val_split_path=...)` (src/voxtral/trainer/data.py:131).

Two-tier stratified sampling:
- Per-language quota: `per_lang` files (default 30).
- Within a language, quota splits across available sources by ceil(per_lang/n_sources).
  Single-source languages get full quota from the only source. Trim deterministically
  if ceil rounding pushes a language above per_lang.
- Files within each (lang, source) cell are sorted, then sampled by a seeded
  RNG keyed on `f"{seed}:{lang}:{source}"` — stable across runs/processes
  regardless of PYTHONHASHSEED.
- YT files filtered by `min_lid_confidence` (default 0.7, matches LLD.md:540).
  FLEURS / IV sidecars have no `language_confidence` field — they're gold-labeled
  and treated as confidence 1.0.
- Tier-1 languages absent from the corpus are logged into `missing_tier1` in the
  output JSON. Underfilled languages (< min_per_lang after sampling) into `underfilled`.

Schema (v2):
    {
        "schema_version": 2,
        "created_at": "2026-05-06T...Z",
        "seed": 1729,
        "tokens_root": "data/tokens_v2",
        "min_lid_confidence": 0.7,
        "per_lang_target": 30,
        "per_language": {"hin": 30, "ben": 30, ...},
        "per_language_per_source": {"hin": {"fleurs": 10, "iv": 10, "yt": 10}, ...},
        "missing_tier1": ["san", "sin", ...],
        "underfilled": [],
        "n_total": 420,
        "files": ["data/tokens_v2/fleurs/hi/hi/hi_00000162_1638e872.npy", ...]
    }

Usage:
    uv run scripts/build_val_split.py
    uv run scripts/build_val_split.py --per_lang 30 --seed 1729
    uv run scripts/build_val_split.py --min_lid_confidence 0.0   # disable LID filter
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_val_split")


SCHEMA_VERSION = 2
DEFAULT_TOKENS_ROOT = "data/tokens_v2"
DEFAULT_OUTPUT = "data/val_split_v2.json"
DEFAULT_PER_LANG = 30
DEFAULT_SEED = 1729
DEFAULT_MIN_PER_LANG = 5
DEFAULT_MIN_LID_CONFIDENCE = 0.7

# Tier-1 canonical language codes (matches the retokenize_v2_batched --languages flag
# and src/voxtral/tokenizer/model.py language vocabulary).
TIER1_LANGUAGES = (
    "eng", "hin", "ben", "tam", "tel", "kan", "mal", "mar", "guj",
    "pan", "urd", "ory", "asm", "npi", "san", "sin", "snd", "bod",
    "doi", "kok", "mni", "sat", "kas",
)

# Alias codes observed in MMS-LID / Whisper sidecars. Coverage of any alias counts
# as coverage of the canonical name. Recent fix in commit 385e73e accepted these
# aliases at tokenizer time; the val split honors the same equivalence.
TIER1_ALIASES: dict[str, set[str]] = {
    "npi": {"npi", "nep"},
    "ory": {"ory", "ori"},
    "bod": {"bod", "tib"},
    "san": {"san", "sa"},
    "sin": {"sin", "si"},
}


def _read_meta(npy_path: Path) -> dict | None:
    """Load the paired .meta.json sidecar. Returns None on any failure (missing,
    truncated, unparseable). Failures are silent here; the caller tracks counts."""
    meta_path = npy_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _source_from_path(npy_path: Path, root: Path) -> str | None:
    """Derive source ('fleurs' / 'iv' / 'yt') from path layout, NOT from the
    sidecar `source` field — the path layout is canonical (controlled by the
    retokenizer), the sidecar string isn't ('fleurs' vs 'youtube' vs 'indicvoices')."""
    try:
        rel = npy_path.relative_to(root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]


def discover_files(
    tokens_root: str,
    min_lid_confidence: float,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, int]]:
    """Walk `tokens_root` and group .npy files by (language, source).

    Language is read from the sidecar's `language` field; source is derived from
    the path layout (`tokens_v2/<source>/...`). YT files with `language_confidence`
    below `min_lid_confidence` are filtered out (FLEURS/IV have no such field —
    they pass trivially as gold-labeled).

    Returns (by_lang_src, stats) where stats counts skipped categories for logging.
    """
    root = Path(tokens_root)
    if not root.exists():
        log.error(f"tokens_root does not exist: {root}")
        return {}, {}

    by_lang_src: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    stats = {"total": 0, "no_meta": 0, "no_lang": 0, "no_source": 0, "low_lid": 0}

    for npy in root.rglob("*.npy"):
        stats["total"] += 1
        meta = _read_meta(npy)
        if meta is None:
            stats["no_meta"] += 1
            continue
        lang = meta.get("language")
        if not lang:
            stats["no_lang"] += 1
            continue
        # FLEURS/IV: no language_confidence field → default 1.0 (gold-labeled).
        # YT: filter on the LID threshold to keep low-confidence noise out of val.
        conf = meta.get("language_confidence")
        if conf is not None and conf < min_lid_confidence:
            stats["low_lid"] += 1
            continue
        source = _source_from_path(npy, root)
        if source is None:
            stats["no_source"] += 1
            continue
        by_lang_src[lang][source].append(str(npy))

    sources = sorted({s for d in by_lang_src.values() for s in d})
    log.info(
        f"discovered {stats['total']} .npy files; "
        f"skipped: {stats['no_meta']} no_meta, {stats['no_lang']} no_lang, "
        f"{stats['low_lid']} low_lid (<{min_lid_confidence}), "
        f"{stats['no_source']} no_source; "
        f"{len(by_lang_src)} languages × {len(sources)} sources: {sources}"
    )
    if stats["no_meta"] > 0:
        log.warning(
            f"{stats['no_meta']} files have no v2 sidecar — re-run retokenize_v2.py to fix."
        )
    return {l: dict(d) for l, d in by_lang_src.items()}, stats


def _is_tier1_present(canonical: str, present: set[str]) -> bool:
    """Tier-1 coverage check that accepts alias codes (npi↔nep, ory↔ori, ...)."""
    accepted = TIER1_ALIASES.get(canonical, {canonical})
    return any(a in present for a in accepted)


def stratified_sample(
    by_lang_src: dict[str, dict[str, list[str]]],
    per_lang: int,
    seed: int,
    min_per_lang: int = DEFAULT_MIN_PER_LANG,
) -> tuple[list[str], dict[str, int], dict[str, dict[str, int]], list[str], list[str]]:
    """Two-tier stratified sample: per (lang, source) cell, sample min(quota, available).

    Within a language, quota splits via `ceil(per_lang / n_sources)` per cell.
    If ceil rounding pushes a language above `per_lang`, trim deterministically
    by a lang-keyed RNG. Single-source languages take full per_lang from the one
    source (no padding, no fake multi-source).

    Determinism: each cell's RNG is keyed on `f"{seed}:{lang}:{src}"` — stable
    across processes regardless of PYTHONHASHSEED. Trim RNG keyed on
    `f"{seed}:trim:{lang}"`.

    Returns:
        selected: flat list of file paths
        per_language: total per language
        per_language_per_source: breakdown {lang: {src: count}}
        underfilled: languages with total < min_per_lang
        missing_tier1: tier-1 langs absent from the corpus (alias-aware)
    """
    selected: list[str] = []
    per_language: dict[str, int] = {}
    per_language_per_source: dict[str, dict[str, int]] = {}
    underfilled: list[str] = []

    for lang in sorted(by_lang_src.keys()):
        sources = sorted(by_lang_src[lang].keys())
        n_src = len(sources)
        per_src_quota = math.ceil(per_lang / n_src) if n_src > 0 else 0

        chosen: list[str] = []
        cell_counts: dict[str, int] = {}
        path_to_src: dict[str, str] = {}

        for src in sources:
            files = sorted(by_lang_src[lang][src])
            n = min(per_src_quota, len(files))
            cell_rng = random.Random(f"{seed}:{lang}:{src}")
            cell_chosen = cell_rng.sample(files, n) if n > 0 else []
            chosen.extend(cell_chosen)
            cell_counts[src] = n
            for p in cell_chosen:
                path_to_src[p] = src

        if len(chosen) > per_lang:
            trim_rng = random.Random(f"{seed}:trim:{lang}")
            chosen = trim_rng.sample(sorted(chosen), per_lang)
            recount: dict[str, int] = collections.Counter(path_to_src[p] for p in chosen)
            cell_counts = dict(recount)

        total = len(chosen)
        if total < min_per_lang:
            underfilled.append(lang)
            log.warning(
                f"[{lang}] only {total} files across {n_src} source(s) "
                f"(< min_per_lang={min_per_lang}); val coverage is weak."
            )

        selected.extend(sorted(chosen))
        per_language[lang] = total
        per_language_per_source[lang] = cell_counts

    present = set(by_lang_src.keys())
    missing_tier1 = [l for l in TIER1_LANGUAGES if not _is_tier1_present(l, present)]
    if missing_tier1:
        log.warning(
            f"tier-1 languages with NO data in {len(present)}-lang corpus: "
            f"{missing_tier1} — val will not cover these."
        )

    return selected, per_language, per_language_per_source, underfilled, missing_tier1


def write_val_split(
    output_path: str,
    files: list[str],
    per_language: dict[str, int],
    per_language_per_source: dict[str, dict[str, int]],
    underfilled: list[str],
    missing_tier1: list[str],
    seed: int,
    tokens_root: str,
    per_lang_target: int,
    min_lid_confidence: float,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "tokens_root": tokens_root,
        "per_lang_target": per_lang_target,
        "min_lid_confidence": min_lid_confidence,
        "per_language": per_language,
        "per_language_per_source": per_language_per_source,
        "missing_tier1": missing_tier1,
        "underfilled": underfilled,
        "n_total": len(files),
        "files": files,
    }
    # Use PID + nanosecond timestamp in tmp suffix so concurrent runs don't collide.
    tmp = f"{out}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(out))
    log.info(
        f"wrote {output_path} — {len(files)} files across {len(per_language)} languages "
        f"(missing_tier1={len(missing_tier1)}, underfilled={len(underfilled)})"
    )


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
                   help="Languages with fewer files trigger an 'underfilled' warning")
    p.add_argument("--min_lid_confidence", type=float, default=DEFAULT_MIN_LID_CONFIDENCE,
                   help="Drop YT files below this LID confidence (default: 0.7); "
                        "FLEURS/IV pass through (no language_confidence field)")
    args = p.parse_args()

    if args.per_lang <= 0:
        log.error(f"--per_lang must be positive, got {args.per_lang}")
        raise SystemExit(2)
    if not (0.0 <= args.min_lid_confidence <= 1.0):
        log.error(f"--min_lid_confidence must be in [0, 1], got {args.min_lid_confidence}")
        raise SystemExit(2)

    by_lang_src, _stats = discover_files(args.tokens_root, args.min_lid_confidence)
    if not by_lang_src:
        log.error("no files found — nothing to write")
        raise SystemExit(2)

    selected, per_language, per_lang_per_src, underfilled, missing_tier1 = stratified_sample(
        by_lang_src,
        per_lang=args.per_lang,
        seed=args.seed,
        min_per_lang=args.min_per_lang,
    )
    write_val_split(
        args.output,
        files=selected,
        per_language=per_language,
        per_language_per_source=per_lang_per_src,
        underfilled=underfilled,
        missing_tier1=missing_tier1,
        seed=args.seed,
        tokens_root=args.tokens_root,
        per_lang_target=args.per_lang,
        min_lid_confidence=args.min_lid_confidence,
    )


if __name__ == "__main__":
    main()
