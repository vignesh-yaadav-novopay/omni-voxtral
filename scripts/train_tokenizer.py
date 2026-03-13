"""Train a multilingual SentencePiece Unigram tokenizer for OmniVoxtral.

Collects text from FLEURS transcripts across 13 Indic languages + English,
trains a SentencePiece Unigram model with 65K vocab, then benchmarks fertility.

Usage: uv run scripts/train_tokenizer.py
"""

import os
import tempfile

import sentencepiece as spm
import transformers as tr
from datasets import load_dataset
from pydantic_settings import BaseSettings
from tqdm import tqdm

FLEURS_LANG_MAP = {
    "asm": "as_in", "ben": "bn_in", "guj": "gu_in", "hin": "hi_in",
    "kan": "kn_in", "mal": "ml_in", "mar": "mr_in", "nep": "ne_np",
    "ori": "or_in", "pan": "pa_in", "tam": "ta_in", "tel": "te_in",
    "urd": "ur_pk", "eng": "en_us",
}

ALL_LANGUAGES = {
    "asm": "Assamese", "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada", "mal": "Malayalam", "mar": "Marathi", "nep": "Nepali",
    "ori": "Odia", "pan": "Punjabi", "tam": "Tamil", "tel": "Telugu",
    "urd": "Urdu", "eng": "English",
}


class TrainTokenizerConfig(BaseSettings):
    output_dir: str = "./data/tokenizer"
    vocab_size: int = 2**16  # 65536
    model_prefix: str = "omnivoxtral_sp"
    max_samples_per_lang: int = 2000
    character_coverage: float = 0.9999
    model_type: str = "unigram"
    # Proportion targets: ~50% Indic, ~30% English, ~20% cross-lingual overlap
    english_proportion: float = 0.30


def collect_transcripts(config: TrainTokenizerConfig) -> str:
    """Collect transcripts from FLEURS, write to temp file for SentencePiece."""
    corpus_path = os.path.join(config.output_dir, "training_corpus.txt")

    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            n_lines = sum(1 for _ in f)
        if n_lines > 1000:
            print(f"Corpus already exists with {n_lines} lines, reusing.")
            return corpus_path

    os.makedirs(config.output_dir, exist_ok=True)

    all_texts = []

    for lang_code, fleurs_code in tqdm(FLEURS_LANG_MAP.items(), desc="Collecting transcripts"):
        lang_name = ALL_LANGUAGES.get(lang_code, lang_code)
        print(f"  {lang_name} ({lang_code})...")

        try:
            ds = load_dataset(
                "google/fleurs", fleurs_code, split="train",
                streaming=True, trust_remote_code=True,
            )

            count = 0
            max_for_lang = config.max_samples_per_lang
            # Give English more or less weight based on proportion
            if lang_code == "eng":
                max_for_lang = int(config.max_samples_per_lang * config.english_proportion / (1 - config.english_proportion) * len(FLEURS_LANG_MAP))

            for sample in ds:
                if count >= max_for_lang:
                    break
                text = sample.get("transcription", "") or sample.get("raw_transcription", "")
                if text and len(text.strip()) > 5:
                    all_texts.append(text.strip())
                    count += 1

            print(f"    Got {count} transcripts")
        except Exception as e:
            print(f"    Failed: {e}")

    # Shuffle and write
    import random
    random.seed(42)
    random.shuffle(all_texts)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print(f"\nCorpus: {len(all_texts)} sentences written to {corpus_path}")
    return corpus_path


def train_sentencepiece(corpus_path: str, config: TrainTokenizerConfig) -> str:
    """Train SentencePiece Unigram model."""
    model_path = os.path.join(config.output_dir, config.model_prefix)

    if os.path.exists(model_path + ".model"):
        print(f"Model already exists at {model_path}.model, skipping training.")
        return model_path + ".model"

    print(f"\nTraining SentencePiece {config.model_type} tokenizer...")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Character coverage: {config.character_coverage}")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_path,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        # Key settings for multilingual
        byte_fallback=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        # Special tokens for OmniVoxtral
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # User-defined special tokens (language tags, control tokens)
        user_defined_symbols=[
            "<|lang:asm|>", "<|lang:ben|>", "<|lang:brx|>", "<|lang:doi|>",
            "<|lang:guj|>", "<|lang:hin|>", "<|lang:kan|>", "<|lang:kas|>",
            "<|lang:kok|>", "<|lang:mai|>", "<|lang:mal|>", "<|lang:mni|>",
            "<|lang:mar|>", "<|lang:nep|>", "<|lang:ori|>", "<|lang:pan|>",
            "<|lang:san|>", "<|lang:sat|>", "<|lang:snd|>", "<|lang:tam|>",
            "<|lang:tel|>", "<|lang:urd|>", "<|lang:eng|>",
            "<|silence|>", "<|overlap|>", "<|backch|>",
            "<|turn_start|>", "<|turn_end|>",
            "<|lang_switch|>",
        ],
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
    )

    print(f"  Model saved to {model_path}.model")
    return model_path + ".model"


def benchmark_comparison(sp_model_path: str, config: TrainTokenizerConfig) -> None:
    """Compare new SP tokenizer vs Mistral's BPE on Indic text."""
    print("\n" + "=" * 60)
    print("FERTILITY COMPARISON: Mistral BPE vs OmniVoxtral Unigram")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    mistral_tok = tr.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    results = []

    for lang_code, fleurs_code in tqdm(FLEURS_LANG_MAP.items(), desc="Benchmarking"):
        if lang_code == "eng":
            continue

        lang_name = ALL_LANGUAGES.get(lang_code, lang_code)
        try:
            ds = load_dataset(
                "google/fleurs", fleurs_code, split="test",
                streaming=True, trust_remote_code=True,
            )

            mistral_fertilities = []
            sp_fertilities = []

            for i, sample in enumerate(ds):
                if i >= 50:
                    break
                text = sample.get("transcription", "") or sample.get("raw_transcription", "")
                if not text or len(text.strip()) < 10:
                    continue
                text = text.strip()
                words = text.split()
                if len(words) == 0:
                    continue

                # Mistral fertility
                mistral_tokens = mistral_tok.encode(text, add_special_tokens=False)
                mistral_fertilities.append(len(mistral_tokens) / len(words))

                # SentencePiece fertility
                sp_tokens = sp.encode(text, out_type=int)
                sp_fertilities.append(len(sp_tokens) / len(words))

            if mistral_fertilities:
                m_mean = sum(mistral_fertilities) / len(mistral_fertilities)
                s_mean = sum(sp_fertilities) / len(sp_fertilities)
                improvement = (1 - s_mean / m_mean) * 100

                results.append({
                    "language": lang_code,
                    "name": lang_name,
                    "mistral_fertility": m_mean,
                    "sp_fertility": s_mean,
                    "improvement_pct": improvement,
                })

        except Exception as e:
            print(f"  {lang_name}: {e}")

    # Print comparison table
    print("\n| Language | Mistral BPE | OmniVoxtral SP | Improvement |")
    print("|----------|------------|---------------|-------------|")

    total_mistral = 0
    total_sp = 0
    for r in sorted(results, key=lambda x: x["improvement_pct"], reverse=True):
        print(
            f"| {r['name']} ({r['language']}) | "
            f"{r['mistral_fertility']:.2f} | "
            f"{r['sp_fertility']:.2f} | "
            f"**{r['improvement_pct']:+.0f}%** |"
        )
        total_mistral += r["mistral_fertility"]
        total_sp += r["sp_fertility"]

    n = len(results)
    if n > 0:
        avg_mistral = total_mistral / n
        avg_sp = total_sp / n
        avg_improvement = (1 - avg_sp / avg_mistral) * 100
        print(f"\n**Average: Mistral {avg_mistral:.2f} → OmniVoxtral {avg_sp:.2f} ({avg_improvement:+.0f}% improvement)**")

    # Save comparison
    summary_path = os.path.join(config.output_dir, "comparison.md")
    with open(summary_path, "w") as f:
        f.write("# Tokenizer Comparison: Mistral BPE vs OmniVoxtral SentencePiece Unigram\n\n")
        f.write(f"Mistral vocab: {mistral_tok.vocab_size}\n")
        f.write(f"OmniVoxtral vocab: {sp.get_piece_size()}\n\n")
        f.write("| Language | Mistral BPE | OmniVoxtral SP | Improvement |\n")
        f.write("|----------|------------|---------------|-------------|\n")
        for r in sorted(results, key=lambda x: x["improvement_pct"], reverse=True):
            f.write(
                f"| {r['name']} ({r['language']}) | "
                f"{r['mistral_fertility']:.2f} | "
                f"{r['sp_fertility']:.2f} | "
                f"{r['improvement_pct']:+.0f}% |\n"
            )
        if n > 0:
            f.write(f"\n**Average: Mistral {avg_mistral:.2f} → OmniVoxtral {avg_sp:.2f} ({avg_improvement:+.0f}%)**\n")

    print(f"\nComparison saved to {summary_path}")


def train_tokenizer(config: TrainTokenizerConfig) -> None:
    """Main entry point."""
    corpus_path = collect_transcripts(config)
    model_path = train_sentencepiece(corpus_path, config)
    benchmark_comparison(model_path, config)

    # Print some example tokenizations
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"\nFinal vocab size: {sp.get_piece_size()}")
    print(f"Model saved to: {model_path}")

    # Show language tokens
    print("\nLanguage tokens:")
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith("<|lang:"):
            print(f"  {i}: {piece}")
        if i > 100 and not piece.startswith("<|"):
            break


if __name__ == "__main__":
    config = TrainTokenizerConfig()
    train_tokenizer(config)
