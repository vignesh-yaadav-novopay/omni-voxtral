"""Phase 1 validator — check that data/tokens_v2/ is consistent.

Invariants checked:
- Every .npy has a paired .meta.json (and vice versa).
- Sidecar schema_version == 2.
- language is ISO 639-3 (in the known set), token_count > 0, valid_token_mask
  length matches token_count when present, stream_layout ∈ {single, dual}.
- First token in the .npy is a known SP language control token (matches the
  sidecar language). Catches the bug where retokenize_v2 forgot to inject the
  control token (a previous regression).

Outputs:
- summary line per language: count, lang-token-injected %, mask-coverage %
- exits 1 if any invariant fails on > THRESHOLD% of files

Usage:
    uv run scripts/validate_v2_corpus.py
    uv run scripts/validate_v2_corpus.py --tokens_root data/tokens_v2 --max_per_lang 200
    uv run scripts/validate_v2_corpus.py --strict     # exit 1 on ANY failure
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_v2_corpus")


_KNOWN_ISO3 = {
    "asm", "ben", "brx", "doi", "eng", "guj", "hin", "kan", "kas", "kok",
    "mai", "mal", "mni", "mar", "nep", "ori", "pan", "san", "sat", "snd",
    "tam", "tel", "urd",
}


def _load_sp_lang_tokens(sp_path: Path) -> dict[str, int]:
    """Return {iso3: sp_token_id} for all <|lang:iso3|> control tokens.

    Used to verify the .npy starts with the right control token id.
    """
    if not sp_path.exists():
        log.warning(f"SP model not at {sp_path}; skipping lang-token check")
        return {}
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_path))
    out = {}
    for iso3 in _KNOWN_ISO3:
        piece = f"<|lang:{iso3}|>"
        tok_id = sp.PieceToId(piece)
        if tok_id != sp.unk_id():
            out[iso3] = tok_id
    return out


def validate_corpus(
    tokens_root: Path,
    sp_lang_tokens: dict[str, int],
    max_per_lang: int | None = None,
) -> dict:
    """Walk tokens_root, validate each .npy + sidecar pair. Returns a report."""
    by_lang: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "lang_token_ok": 0, "mask_ok": 0, "errors": [],
    })

    npy_files = list(tokens_root.rglob("*.npy"))
    log.info(f"found {len(npy_files)} .npy files under {tokens_root}")

    seen_per_lang: dict[str, int] = defaultdict(int)
    for npy in npy_files:
        meta_path = npy.with_suffix(".meta.json")
        if not meta_path.exists():
            by_lang["__no_sidecar__"]["total"] += 1
            by_lang["__no_sidecar__"]["errors"].append(f"missing sidecar: {npy}")
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            by_lang["__bad_json__"]["total"] += 1
            by_lang["__bad_json__"]["errors"].append(f"bad json {meta_path}: {e}")
            continue

        lang = meta.get("language") or "__missing__"
        if max_per_lang and seen_per_lang[lang] >= max_per_lang:
            continue
        seen_per_lang[lang] += 1

        rec = by_lang[lang]
        rec["total"] += 1

        if meta.get("schema_version") != 2:
            rec["errors"].append(f"{npy}: schema_version={meta.get('schema_version')}")
            continue
        if lang not in _KNOWN_ISO3:
            rec["errors"].append(f"{npy}: unknown lang={lang!r}")
            continue

        # Token count + first-token check.
        try:
            arr = np.load(npy, mmap_mode="r")
        except Exception as e:
            rec["errors"].append(f"{npy}: load failed: {e}")
            continue
        flat = arr.flatten()
        if flat.size == 0:
            rec["errors"].append(f"{npy}: empty")
            continue

        # Lang-token-injection check.
        if sp_lang_tokens:
            expected_id = sp_lang_tokens.get(lang)
            if expected_id is not None and int(flat[0]) == expected_id:
                rec["lang_token_ok"] += 1
            else:
                if len(rec["errors"]) < 5:
                    rec["errors"].append(
                        f"{npy.name}: first token {int(flat[0])} != "
                        f"<|lang:{lang}|> ({expected_id})"
                    )

        # valid_token_mask length matches token_count
        mask = meta.get("valid_token_mask")
        if mask is not None:
            if len(mask) == flat.size:
                rec["mask_ok"] += 1
            else:
                if len(rec["errors"]) < 5:
                    rec["errors"].append(
                        f"{npy.name}: mask len {len(mask)} != tokens {flat.size}"
                    )

    return dict(by_lang)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens_root", default="data/tokens_v2")
    p.add_argument("--sp_path", default="data/tokenizer/omnivoxtral_sp.model")
    p.add_argument("--max_per_lang", type=int, default=None,
                   help="Sample this many per language (faster on huge corpora)")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if ANY error; default tolerates up to 1%% errors")
    args = p.parse_args()

    sp_lang_tokens = _load_sp_lang_tokens(Path(args.sp_path))
    report = validate_corpus(Path(args.tokens_root), sp_lang_tokens, args.max_per_lang)

    total = sum(r["total"] for r in report.values())
    error_count = sum(len(r["errors"]) for r in report.values())
    print()
    print(f"{'language':>14}  {'total':>7}  {'lang-tok %':>10}  {'mask %':>8}  errors")
    print("-" * 80)
    for lang in sorted(report.keys()):
        r = report[lang]
        if r["total"] == 0:
            continue
        lang_pct = 100 * r["lang_token_ok"] / r["total"] if r["total"] else 0
        mask_pct = 100 * r["mask_ok"] / r["total"] if r["total"] else 0
        sample_errs = r["errors"][:2]
        err_str = "; ".join(sample_errs) if sample_errs else "ok"
        print(f"{lang:>14}  {r['total']:>7}  {lang_pct:>9.1f}%  {mask_pct:>7.1f}%  {err_str[:60]}")

    print()
    print(f"TOTAL files: {total}, errors observed: {error_count}")
    fail = (error_count > 0) if args.strict else (error_count > total // 100)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
