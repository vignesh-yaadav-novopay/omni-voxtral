import dataclasses
import re
import string
import typing
import warnings

import numpy as np
import torch
import transformers as tr


@dataclasses.dataclass
class WordTiming:
    word: str
    tokens: list[int]
    start: float
    end: float


def clean_text(text: str) -> str:
    cleaned = re.sub(r"<\|[\d.]+\|>", "", text)
    cleaned = cleaned.strip()
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned


def split_tokens_on_unicode(
    tokens: list[int], tokenizer: typing.Any
) -> tuple[list[str], list[list[int]]]:
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "�"

    words = []
    word_tokens = []
    current_tokens = []
    unicode_offset = 0

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)]
            == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []
            unicode_offset += len(decoded)

    return words, word_tokens


def split_tokens_on_spaces(
    tokens: list[int], tokenizer: typing.Any
) -> tuple[list[str], list[list[int]]]:
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens


def tokens_to_words(
    generate_outputs: dict[str, typing.Any], tokenizer: typing.Any, language: str
) -> list[list[WordTiming]]:
    timings = []

    for batch_idx in range(generate_outputs["sequences"].shape[0]):
        predicted_ids = generate_outputs["sequences"][batch_idx].cpu().numpy()
        token_timestamps = generate_outputs["token_timestamps"][batch_idx].numpy()

        text_tokens = [token for token in predicted_ids]
        if language in {"zh", "ja", "th", "lo", "my"}:
            words, word_tokens = split_tokens_on_unicode(text_tokens, tokenizer)
        else:
            words, word_tokens = split_tokens_on_spaces(text_tokens, tokenizer)
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

        start_times = token_timestamps[word_boundaries[:-1]]
        end_times = token_timestamps[word_boundaries[1:]]

        timings.append(
            [
                WordTiming(word, tokens, start, end)
                for word, tokens, start, end in zip(
                    words, word_tokens, start_times, end_times
                )
            ]
        )

    return timings


def merge_punctuations(alignment: list[WordTiming]) -> None:
    prepend_punctuations = "\"'“¿([{-"
    append_punctuations = "\"'.。,，!！?？:：”)]}、"

    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if (
            previous.word.startswith(" ")
            and previous.word.strip() in prepend_punctuations
        ):
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in append_punctuations:
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def separate_into_buckets(
    data: list[WordTiming], bucket_size: float, total_duration: float
) -> list[list[str]]:
    buckets = []
    current_time = 0

    while current_time < total_duration:
        bucket_end_time = current_time + bucket_size
        bucket_words = []

        for entry in data[1:-1]:
            if current_time <= entry.end < bucket_end_time:
                cleaned_word = clean_text(entry.word)
                bucket_words.append(cleaned_word)

        buckets.append(bucket_words)
        current_time = bucket_end_time

    return buckets


def generate_tokens(
    processor: typing.Any,
    model: typing.Any,
    audio: torch.Tensor,
    language: str,
    task: str = "transcribe",
) -> dict[str, typing.Any]:
    """Forward Whisper generate with explicit language + task.

    `language` is REQUIRED — Defect 1 root cause was the previous default of "en"
    being silently used for Indic audio. The previous `if language != "en"` branch
    is removed; both kwargs are always forwarded.
    """
    device = next(model.parameters()).device
    input_features = processor(
        audio.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, audio.dtype)
    return model.generate(
        input_features,
        return_timestamps=True,
        return_token_timestamps=True,
        language=language,
        task=task,
    )


class TimedWhisperTokenizer(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        hertz: int,
        language: str | None = None,
        sp_tokenizer_path: str | None = None,
    ) -> None:
        super().__init__()
        self.processor = tr.WhisperProcessor.from_pretrained(model_name)
        self.model = tr.WhisperForConditionalGeneration.from_pretrained(model_name)

        # Per-call `language` always wins; this is the fallback default only.
        # Explicit `None` means caller MUST pass language at every call.
        self.language: str | None = language
        self.tokenizer: typing.Any = self.processor.tokenizer  # type: ignore
        self.hertz: int = hertz

        # Use SentencePiece tokenizer if provided, otherwise fall back to Mistral BPE
        self.sp_tokenizer = None
        if sp_tokenizer_path is not None:
            import sentencepiece as spm

            self.sp_tokenizer = spm.SentencePieceProcessor()
            self.sp_tokenizer.Load(sp_tokenizer_path)
        else:
            self.mistral_tokenizer: tr.AutoTokenizer = (
                tr.AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    padding_side="right",
                    add_prefix_space=False,
                )
            )
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

    def _resolve_language(self, language: str | None) -> str:
        effective = language if language is not None else self.language
        if effective is None:
            raise ValueError(
                "TimedWhisperTokenizer: `language` must be passed at __init__ or per-call. "
                "Defect 1 fix: no silent fallback to 'en'."
            )
        return effective

    def _tokenize_bucket(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Tokenize a list of bucket texts, padding/truncating to self.hertz tokens each."""
        if self.sp_tokenizer is not None:
            pad_id = self.sp_tokenizer.pad_id()
            if pad_id < 0:
                pad_id = 0
            all_ids = []
            for text in texts:
                ids = self.sp_tokenizer.Encode(text)
                ids = ids[: self.hertz]
                ids = ids + [pad_id] * (self.hertz - len(ids))
                all_ids.append(ids)
            return torch.tensor(all_ids, dtype=torch.long, device=device)
        else:
            tokens = self.mistral_tokenizer(
                texts,
                padding="max_length",
                max_length=self.hertz,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            return tokens["input_ids"].to(device)

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        language: str | None = None,
        task: str = "transcribe",
    ) -> torch.Tensor:
        """Bucketed text tokens via Whisper.

        audio: (batch, num_samples) at 16kHz, mono.
        Returns: (batch, num_buckets * hertz) — bucketed text tokens.
        """
        tokens, _ = self.forward_with_transcript(
            audio, sample_rate, language=language, task=task
        )
        return tokens

    def forward_with_transcript(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        language: str | None = None,
        task: str = "transcribe",
    ) -> tuple[torch.Tensor, dict[str, typing.Any]]:
        """Bucketed tokens + raw transcript metadata, in a single Whisper pass.

        Returns:
            tokens: (batch, num_buckets * hertz)
            meta: per-batch list of dicts {transcript, avg_logprob, num_words}
        """
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert audio.ndim == 2, "Audio must be 2D, batch x time"
        effective_lang = self._resolve_language(language)
        total_duration = audio.shape[1] / sample_rate

        outputs = generate_tokens(
            self.processor, self.model, audio, language=effective_lang, task=task
        )
        device = outputs["sequences"].device
        alignment = tokens_to_words(outputs, self.tokenizer, effective_lang)
        [merge_punctuations(a) for a in alignment]

        # Build bucketed token tensor + per-batch transcripts
        per_batch_meta: list[dict[str, typing.Any]] = []
        buckets_per_item: list[torch.Tensor] = []
        for batch_idx, a in enumerate(alignment):
            out = separate_into_buckets(
                a, bucket_size=1.0, total_duration=total_duration
            )
            out = [clean_text(" ".join(b)) for b in out]
            bucket_tokens = self._tokenize_bucket(out, device)
            buckets_per_item.append(bucket_tokens)

            transcript = self.tokenizer.decode(
                outputs["sequences"][batch_idx].tolist(), skip_special_tokens=True
            )
            per_batch_meta.append(
                {
                    "transcript": transcript,
                    "num_words": len([w for w in a if w.word.strip()]),
                    "task": task,
                    "language": effective_lang,
                }
            )
        buckets = torch.stack(buckets_per_item).view(len(buckets_per_item), -1).to(device)
        meta = {"per_batch": per_batch_meta}
        return buckets, meta

    @torch.no_grad()
    def translate(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        """Single-pass Whisper translate. Returns plain English string for batch[0].

        Output capped at 2000 chars (defensive — Whisper occasionally hallucinates
        long repetitive runs on noisy audio).
        """
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert audio.ndim == 2 and audio.size(0) == 1, (
            "translate() expects a single audio item (batch=1)"
        )
        effective_lang = self._resolve_language(language)
        outputs = generate_tokens(
            self.processor, self.model, audio, language=effective_lang, task="translate"
        )
        text = self.tokenizer.decode(
            outputs["sequences"][0].tolist(), skip_special_tokens=True
        )
        if len(text) > 2000:
            warnings.warn(
                f"translate() output {len(text)} chars; truncating to 2000.",
                stacklevel=2,
            )
            text = text[:2000]
        return text
