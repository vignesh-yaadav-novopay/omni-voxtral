# Capacity Planning: GPU Memory, Compute, and Scaling

> Concrete math for every resource constraint. No hand-waving.
> Date: 2026-03-13 | Hardware: 4x A10G (23GB each), GPU:0 only for now

---

## Current Hardware

| Resource | Spec | Constraint |
|----------|------|------------|
| GPU:0 | NVIDIA A10G, 23GB VRAM | Available for OmniVoxtral |
| GPU:1-3 | NVIDIA A10G, 23GB each | Occupied by VLLM (DO NOT USE) |
| CPU | Unknown (cloud instance) | Assumed sufficient |
| RAM | Unknown | Assumed ≥64GB |
| Storage | Unknown | Assumed ≥500GB |

**All scripts must use `CUDA_VISIBLE_DEVICES=0`.**

---

## Phase 1: Codec Benchmark (fits easily on A10G)

### Mimi Codec Memory
```
Mimi model (SeaNet + Transformer): ~300MB
Whisper-large-v3 (for WER eval):   ~3GB
Audio buffer (10 samples × 15s):    ~15MB
PyTorch overhead:                   ~500MB
────────────────────────────────────────
Total:                              ~4GB
Available on A10G:                  23GB
Headroom:                           19GB ✓ (abundant)
```

### Benchmark Throughput Estimate
- Mimi encode: ~10ms per second of audio (on A10G)
- Mimi decode: ~10ms per second of audio
- Whisper-large-v3 inference: ~500ms per 15s sample
- Per sample (encode + decode + metrics): ~2 seconds
- 22 languages × 10 samples × 2 quantizer configs: 440 samples
- **Total benchmark time: ~15 minutes**

---

## Phase 2: Tokenizer Benchmark (CPU only, no GPU needed)

### SentencePiece Training
- Training corpus: ~100MB text (sampled from transcripts)
- SentencePiece Unigram training: ~10 minutes on CPU
- Fertility benchmark: milliseconds per sample
- **No GPU required**

---

## Phase 3: Model Training Memory Math

### Mistral 7B (full, no pruning)

**Model weights (bf16):**
```
Parameters: 7.24B
Bytes per param (bf16): 2
Model size: 7.24B × 2 = 14.48 GB
```

**Optimizer state (AdamW):**
```
AdamW stores: m (momentum), v (variance), params
Per param: 2 (bf16) + 4 (fp32 m) + 4 (fp32 v) = 10 bytes
Optimizer total: 7.24B × 10 = 72.4 GB
```

**Gradient storage:**
```
Gradients (bf16): 7.24B × 2 = 14.48 GB
```

**Activations (per layer, batch_size=1, seq_len=4096):**
```
Per layer activation: batch × seq × hidden × 2 = 1 × 4096 × 4096 × 2 = 33.6 MB
32 layers: 32 × 33.6 MB = 1.07 GB (without gradient checkpointing)
With gradient checkpointing: ~sqrt(32) × 33.6 MB ≈ 190 MB
```

**Full 7B training budget:**
```
Without gradient checkpointing:
  Model:       14.5 GB
  Optimizer:   72.4 GB
  Gradients:   14.5 GB
  Activations:  1.1 GB
  ──────────────────────
  Total:      102.5 GB → DOES NOT FIT on A10G (23GB)

With FSDP across 4 GPUs:
  Sharded:    102.5 / 4 = 25.6 GB → STILL TIGHT (23GB each)
```

**Conclusion: Full 7B training does NOT fit on 4× A10G even with FSDP.**

### Mistral 7B with Layer Pruning (prune_layers=2, ~3.5B effective)

**Model weights (bf16):**
```
Parameters: ~3.62B (half of 7B, 16 of 32 layers removed)
Model size: 3.62B × 2 = 7.24 GB
```

**Full pruned training budget:**
```
Without gradient checkpointing:
  Model:        7.2 GB
  Optimizer:   36.2 GB
  Gradients:    7.2 GB
  Activations:  0.5 GB
  ──────────────────────
  Total:       51.1 GB → DOES NOT FIT on single A10G

With gradient checkpointing (single GPU):
  Model:        7.2 GB
  Optimizer:   36.2 GB
  Gradients:    7.2 GB
  Activations:  0.1 GB (checkpointed)
  ──────────────────────
  Total:       50.7 GB → STILL DOES NOT FIT
```

**Conclusion: Even pruned 3.5B does NOT fit with AdamW on single A10G.**

### LoRA on Pruned 3.5B (rank=64, what Voxtral was actually designed for)

**Trainable parameters with LoRA (rank=64, all Linear layers):**
```
Per layer: 6 linear layers × (4096 × 64 + 64 × 4096) × 2 = ~6.3 MB
16 layers: 16 × 6.3 = ~100 MB trainable
Frozen model: 7.2 GB (loaded, no optimizer state)
LoRA optimizer: 100 MB × 5 (AdamW) = 500 MB
```

**LoRA training budget (single A10G):**
```
Frozen model (bf16):    7.2 GB
LoRA params:            0.1 GB
LoRA optimizer:         0.5 GB
Gradients (LoRA only):  0.1 GB
Activations (checkpt):  0.1 GB
KV cache / overhead:    1.0 GB
──────────────────────────────
Total:                  9.0 GB ✓ (fits on A10G with 14GB headroom)
```

**THIS is the viable single-GPU training configuration:**
- Mistral 7B with `prune_layers=2` (→ 3.5B)
- LoRA rank 64 on all Linear layers
- Gradient checkpointing enabled
- Batch size 1-2 (sequence length dependent)

---

## KV Cache Memory Math (for inference/serving)

**Mistral 7B (32 layers, 8 KV heads, dim 128 per head):**
```
KV cache per token = 2 (K+V) × 32 layers × 8 heads × 128 dim × 2 bytes (bf16)
                   = 131,072 bytes = 128 KB per token

Context 4096 tokens: 128 KB × 4096 = 512 MB per session
```

Note: Mistral uses GQA (8 KV heads, not 32), so KV cache is 4× smaller than full attention.

**Per-GPU inference budget (A10G 23GB):**
```
Model (pruned, bf16): 7.2 GB
Mimi codec:           0.3 GB
Overhead:             2.0 GB
Available for KV:    13.5 GB
Sessions possible:   13.5 GB / 0.5 GB = ~27 concurrent sessions
```

**Per-GPU inference budget (H100 80GB):**
```
Model (full, bf16):  14.5 GB
Mimi codec:           0.3 GB
Overhead:             5.0 GB
Available for KV:    60.2 GB
Sessions possible:   60.2 GB / 0.5 GB = ~120 concurrent sessions
```

**Per-GPU inference budget (H100, INT4 quantized):**
```
Model (INT4):         4.0 GB
Mimi codec:           0.3 GB
Overhead:             3.0 GB
Available for KV:    72.7 GB
Sessions possible:   72.7 GB / 0.5 GB = ~145 concurrent sessions
```

---

## Scaling Projections

### Training

| Configuration | Hardware | Params | Time Est. (10K steps) | Cost Est. |
|---------------|----------|--------|----------------------|-----------|
| LoRA on pruned 3.5B | 1× A10G | 100M trainable | ~8 hours | ~$25 |
| LoRA on full 7B | 1× H100 | 200M trainable | ~4 hours | ~$30 |
| Full fine-tune 7B | 4× H100 (FSDP) | 7.2B | ~12 hours | ~$140 |
| Full fine-tune 7B | 8× H100 (FSDP) | 7.2B | ~6 hours | ~$140 |
| Full train (Moshi-scale) | 128× H100 | 7.2B | ~2 weeks | ~$150K |

### Inference/Serving

| Hardware | Concurrent Sessions | Latency | Monthly Cost |
|----------|-------------------|---------|-------------|
| 1× A10G (pruned) | ~27 | ~300ms | ~$500 |
| 1× H100 (full) | ~120 | ~150ms | ~$3,000 |
| 1× H100 (INT4) | ~145 | ~180ms | ~$3,000 |
| 8× H100 (tensor parallel) | ~800+ | ~100ms | ~$24,000 |
| 32× H100 | ~4,000+ | ~100ms | ~$96,000 |

### Cost per Conversation Minute

| Configuration | Sessions/GPU | GPU $/hr | $/conversation-min |
|---------------|-------------|---------|-------------------|
| A10G (pruned) | 27 | $0.70 | $0.0004 |
| H100 (full) | 120 | $4.00 | $0.0006 |
| H100 (INT4) | 145 | $4.00 | $0.0005 |
| GPT-4o Realtime | N/A | N/A | $0.30 (API pricing) |

**OmniVoxtral is ~500-750× cheaper per minute than GPT-4o Realtime at scale.**

---

## Immediate Phase Capacity

### Phase 1: Codec Benchmark
- **GPU needed:** 1× A10G (GPU:0)
- **VRAM used:** ~4 GB
- **Time:** ~15 minutes
- **Risk:** None — abundant headroom

### Phase 2: Tokenizer
- **GPU needed:** None (CPU only)
- **Time:** ~10 minutes

### Phase 3: Initial Training (LoRA)
- **GPU needed:** 1× A10G (GPU:0)
- **VRAM used:** ~9 GB
- **Batch size:** 1-2
- **Steps/sec:** ~2-3 (estimated, depends on sequence length)
- **Time for 10K steps:** ~1-2 hours
- **Risk:** Low — fits with headroom

### Phase 4: Full Training
- **GPU needed:** 4× H100 minimum (not available locally)
- **Options:** Cloud (Lambda Labs, RunPod, AWS p5)
- **Estimated cost:** $100-500 depending on scale
- **Risk:** Medium — requires cloud provisioning

---

## Memory Optimization Techniques (ordered by impact)

| Technique | VRAM Savings | Speed Impact | Complexity |
|-----------|-------------|-------------|------------|
| LoRA (rank 64) | ~60% (frozen base) | Minimal | Low |
| Gradient checkpointing | ~40% activations | -30% speed | Low (fix existing bug) |
| Layer pruning (prune_layers=2) | ~50% model | Minimal | Already implemented |
| Mixed precision (bf16) | ~50% vs fp32 | +10-20% speed | Already implemented |
| Flash Attention 2 | ~40% attention memory | +20% speed | Medium (install flash-attn) |
| INT8 quantization (inference) | ~50% model | -5% quality | Low (bitsandbytes) |
| INT4 quantization (inference) | ~75% model | -10% quality | Low (GPTQ/AWQ) |
| FSDP | Linear with N GPUs | -5% communication | Medium |
| CPU offloading (DeepSpeed ZeRO-3) | Unlimited (uses RAM) | -50% speed | High |

**Recommended stack for A10G single-GPU training:**
1. Layer pruning (prune_layers=2) — already implemented
2. LoRA (rank 64) — already implemented
3. Gradient checkpointing — fix existing bug, then enable
4. Flash Attention 2 — install and enable
5. bf16 — already implemented

**This combination should allow training with batch_size=2, seq_len=4096 on A10G 23GB.**

---

## Depth Transformer Sizing

The Depth Transformer is new (not in current Voxtral). Sizing based on Moshi:

```
Depth Transformer: 6 layers, 16 heads, dim 1024
Parameters: ~150M
Memory (bf16): ~300 MB
Per-step compute: ~0.5ms on A10G

Total model (Temporal + Depth):
  Temporal (pruned): 3.5B → 7.2 GB
  Depth:             0.15B → 0.3 GB
  Combined:          3.65B → 7.5 GB
```

**Fits within the same A10G budget.** Depth Transformer is small relative to Temporal.
