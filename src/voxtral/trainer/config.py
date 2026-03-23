import pydantic_settings as pyds
import wandb.util
from voxtral.tokenizer.model import VoxtralTokenizerConfig


class BaseConfig(pyds.BaseSettings):
    """Pydantic base settings but env variables take absolute priority"""

    model_config = pyds.SettingsConfigDict(env_parse_none_str="None")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pyds.BaseSettings],
        init_settings: pyds.PydanticBaseSettingsSource,
        env_settings: pyds.PydanticBaseSettingsSource,
        dotenv_settings: pyds.PydanticBaseSettingsSource,
        file_secret_settings: pyds.PydanticBaseSettingsSource,
    ) -> tuple[pyds.PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )  # switched order


class VoxtralTrainConfig(BaseConfig):
    run_id: str = wandb.util.generate_id()

    seed: int = 42
    name: str = "voxtral-test"

    mistral_pretrained_path: str = "nilq/mistral-1L-tiny"  # "mistralai/Mistral-7B-v0.3"
    mistral_kwargs: dict = {}
    voxtral_tokenizer_config: VoxtralTokenizerConfig = VoxtralTokenizerConfig()
    # text_vocab_size + num_codebooks * codebook_size = 65536 + 8*2048 = 81920
    new_vocab_size: int = 81920
    loss_weights: list[int] = [100, 1, 1]  # text, semantic, acoustic (validated: Moshi §4.4)
    lora_rank: int | None = None
    use_dora: bool = False  # Weight-Decomposed LoRA (DoRA): +1-3% quality at zero VRAM cost
    prune_layers: int | None = None  # no layer dropout
    codec_hz: int = 55

    ## ema
    ema_gamma: float = 16
    ema_every: int = 1024

    ## dataset
    data_path: str = "./data/tokens"
    fake: bool = True
    overfit: int | None = None
    max_seq_len: int | None = None  # truncate sequences to this length (None=no truncation)
    batch_size: int = 2
    num_workers: int = 4
    test_size: int = 50  # AF-203: evaluate more val batches for reliable metrics

    ## speed
    compile: bool = False
    gradient_checkpointing: bool = False

    ## opt
    lr: float = 3e-4  # validated: 3e-4 beats 1e-3 (exp039/040) and 5e-4 (exp038)
    weight_decay: float = 0.0  # validated: WD=0 beats 0.1 (exp026)
    lr_eps: float = 1e-9
    lr_betas: tuple[float, float] = (0.9, 0.95)
    grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 500
    gradient_accumulation_steps: int = 1  # now implemented in omni_train_step
    depth_lr_multiplier: float = 1.0  # Moshi uses 7-17x higher LR for depth

    ## test
    test_every: int | None = 10
    generate_kwargs: dict = {}

    ## logging and checkpointing
    log_every: int = 50
    watch_every: int | None = None
    ckpt_path: str | None = None
    save_every: int | None = None
    push_every: int | None = None
    keep_checkpoints: int = 5
    wandb_project_name: str = "voxtral"

    ## model architecture toggles
    dual_stream: bool = False
    language_adapters: bool = False

    ## depth transformer architecture (exposed for auto-research)
    depth_num_layers: int = 4  # validated: 4L beats 6L (exp029/030, -33% params)
    depth_dim: int = 1024
    depth_num_heads: int = 16
    depth_dropout: float = 0.0  # dropout in depth transformer (0.3 recommended for regularization)
    depth_label_smoothing: float = 0.0  # label smoothing for depth CE loss
    depth_mask_rate: float = 0.0  # exposure bias fix: randomly corrupt this fraction of teacher-forced tokens
    depth_q_weights: list[float] | None = None  # per-codebook loss weights (default: [100,1,1,1,1,1,1,1])
    depth_q_dropout: float = 0.0  # RVQ structured dropout: probability of dropping each codebook loss (0.3 = 30% chance)

    ## dist (picked up by env)
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
