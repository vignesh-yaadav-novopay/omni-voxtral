"""Language-family LoRA adapters for OmniVoxtral.

Implements Decision 6 from ARCHITECTURE.md: three-layer language conditioning:
1. Language tokens (<|lang:kn|>) — already in the vocab (IDs 4-26)
2. Language-family LoRA adapters — this module (~100K-500K params per family)
3. Implicit detection — the model learns from context

Language families (from MMS + Indic linguistic classification):
- Indo-Aryan-Deva: Hindi, Marathi, Nepali, Konkani, Dogri, Maithili, Sanskrit, Bodo
- Indo-Aryan-Other: Bengali, Assamese, Gujarati, Punjabi, Odia, Sindhi, Kashmiri, Urdu
- Dravidian: Tamil, Telugu, Kannada, Malayalam
- Sino-Tibetan: Manipuri (Meitei)
- Austroasiatic: Santali
- English: English (default/base)

During training: all samples in a batch share the same adapter (grouped by family).
During inference: adapter is selected based on the detected/specified language.

Reference: MMS §3.2 (language adapters), ARCHITECTURE.md Decision 6.
"""

import logging

import peft
import torch.nn as nn

logger = logging.getLogger(__name__)

# Language → Family mapping (ISO 639 codes where available)
LANGUAGE_TO_FAMILY: dict[str, str] = {
    # Indo-Aryan (Devanagari script)
    "hi": "indo_aryan_deva",
    "mr": "indo_aryan_deva",
    "ne": "indo_aryan_deva",
    "kok": "indo_aryan_deva",
    "doi": "indo_aryan_deva",
    "mai": "indo_aryan_deva",
    "sa": "indo_aryan_deva",
    "brx": "indo_aryan_deva",  # Bodo uses Devanagari
    # Indo-Aryan (Other scripts)
    "bn": "indo_aryan_other",
    "as": "indo_aryan_other",
    "gu": "indo_aryan_other",
    "pa": "indo_aryan_other",
    "or": "indo_aryan_other",
    "sd": "indo_aryan_other",
    "ks": "indo_aryan_other",
    "ur": "indo_aryan_other",
    # Dravidian
    "ta": "dravidian",
    "te": "dravidian",
    "kn": "dravidian",
    "ml": "dravidian",
    # Sino-Tibetan
    "mni": "sino_tibetan",  # Manipuri (Meitei)
    # Austroasiatic
    "sat": "austroasiatic",  # Santali
    # English (base — no adapter needed, uses base model weights)
    "en": "english",
}

# All adapter family names (excluding English which uses base weights)
ADAPTER_FAMILIES = [
    "indo_aryan_deva",
    "indo_aryan_other",
    "dravidian",
    "sino_tibetan",
    "austroasiatic",
]


def get_language_family(language_code: str) -> str:
    """Get the adapter family name for a language code."""
    return LANGUAGE_TO_FAMILY.get(language_code, "english")


def create_language_adapters(
    model: nn.Module,
    adapter_rank: int = 8,
    adapter_alpha: int = 16,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Create per-language-family LoRA adapters on a model.

    Uses peft to add multiple named LoRA adapters. Each adapter is lightweight
    (~100K-500K params depending on rank and target modules).

    Args:
        model: The temporal transformer (MistralForCausalLM or PeftModel)
        adapter_rank: LoRA rank per adapter (default 8 — small since these are
            language-specific fine-tuning, not full model adaptation)
        adapter_alpha: LoRA alpha (default 2*rank = 16)
        target_modules: Which modules to apply LoRA to. Default: attention
            projections only (q_proj, k_proj, v_proj, o_proj) for minimal
            parameter overhead while still capturing language-specific patterns.

    Returns:
        PeftModel with all family adapters registered (none active by default)
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = peft.LoraConfig(
        r=adapter_rank,
        lora_alpha=adapter_alpha,
        target_modules=target_modules,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )

    # Create first adapter to convert to PeftModel
    first_family = ADAPTER_FAMILIES[0]
    if isinstance(model, peft.PeftModel):
        model.add_adapter(first_family, lora_config)
    else:
        model = peft.get_peft_model(model, lora_config, adapter_name=first_family)

    # Add remaining adapters
    for family in ADAPTER_FAMILIES[1:]:
        model.add_adapter(family, lora_config)

    # Log parameter counts
    total_adapter_params = 0
    for family in ADAPTER_FAMILIES:
        model.set_adapter(family)
        trainable = sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and family in n
        )
        total_adapter_params += trainable
        logger.info(f"Adapter '{family}': {trainable:,} params")

    logger.info(
        f"Total adapter params: {total_adapter_params:,} "
        f"({total_adapter_params / 1e6:.2f}M) across {len(ADAPTER_FAMILIES)} families"
    )

    # Disable all adapters by default (base model active)
    model.disable_adapter_layers()

    return model


def activate_adapter(model: nn.Module, language_code: str) -> str:
    """Activate the appropriate language adapter for a given language.

    Args:
        model: PeftModel with language adapters
        language_code: ISO 639 language code (e.g., "kn", "hi", "en")

    Returns:
        The family name that was activated (or "english" if base model used)
    """
    family = get_language_family(language_code)

    if not isinstance(model, peft.PeftModel):
        return "english"

    if family == "english":
        # English uses base model weights (no adapter)
        model.disable_adapter_layers()
    else:
        model.enable_adapter_layers()
        model.set_adapter(family)

    return family


def get_adapter_info(model: nn.Module) -> dict[str, int]:
    """Get parameter counts for each adapter."""
    if not isinstance(model, peft.PeftModel):
        return {}

    info = {}
    for family in ADAPTER_FAMILIES:
        model.set_adapter(family)
        params = sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and family in n
        )
        info[family] = params

    model.disable_adapter_layers()
    return info
