# Auto-Research: OmniVoxtral

Autonomous research agent. Runs ONE cycle per invocation.
Use with `/loop 15m /auto-research` for continuous operation.

## Arguments
- `--phase training` — Run experiment (default if GPU free)
- `--phase theoretical` — Literature research (default if GPU locked)
- `--phase audit` — Adversarial review (run manually, NOT on a loop)
- `--analyze` — Results analysis only
- `--baseline` — Run baseline experiment
- `--dry-run` — Show what would change

If no `--phase` given: auto-detect GPU availability. Free → training. Locked → theoretical.

## IRON RULES (from /insights analysis + Karpathy patterns)

### 1. NEVER POLL. NEVER PLAN. NEVER STOP.
```
GPU check: ONE call to run_safe.sh. If exit code 4 → GPU locked → theoretical work.
Planning: MAX 3 bullet points. Then EXECUTE. No plan mode. No elaborate designs.
Stopping: NEVER ask "should I continue?". NEVER self-terminate. Run until killed.
```
**Why:** 20+ no-op GPU polls, 6 failed ExitPlanMode attempts, 3 unnecessary self-terminations.

### 2. STDOUT TO FILE, READ VIA GREP (Karpathy pattern)
```
ALWAYS: uv run python scripts/train_omni.py > /apps/voxtral/autoresearch/run.log 2>&1
NEVER: let training output flood the context window
READ:  grep "total_loss\|val_total\|depth_loss\|text_acc\|FAIL" run.log
```
**Why:** Training logs in context window waste tokens and degrade agent reasoning. Karpathy's key insight: redirect output, read only metrics via grep.

### 3. FAIL-FAST IN TRAINING (Karpathy pattern)
```python
# train_omni.py should have this check every step:
if math.isnan(train_loss_f) or train_loss_f > 100:
    print("FAIL: loss exploded"); sys.exit(1)
```
**Why:** Detects exploding/NaN loss in seconds instead of wasting 5 minutes.

### 4. SIMPLICITY RATCHET (Karpathy criterion)
```
A 0.001 improvement that adds 20 lines of hacky code? NOT worth it.
A 0.001 improvement from DELETING code? DEFINITELY keep.
Equal results but simpler? KEEP.
```

### 5. KILL LONG BLOCKERS
```
If preprocessing >12h AND we already have >5x previous data:
  Kill it. Use available data. Re-run remaining AFTER experiment.
```
**Why:** 48h preprocessing blocked all training while 9x data was already available.

---

## THE EXPERIMENT LOOP

```
LOOP FOREVER:
  1. Check GPU (ONE call to run_safe.sh)
     - Exit 4 (locked) → do theoretical work instead
     - Exit 2 (unsafe) → skip cycle, try next invocation
     - Exit 0 (ok) → continue
  2. Read state: tail -1 results.tsv for current best
  3. Pick experiment from experiment_queue.md (read ONLY the P0 section)
  4. Edit config via env vars (ONE change at a time)
  5. Run: env vars + run_safe.sh "uv run python scripts/train_omni.py" 600 > run.log 2>&1
  6. Read results: grep "total_loss\|val_total\|FAIL\|peak_vram" run.log
     - If grep empty → crash → tail -50 run.log → diagnose → log as "crash"
  7. Append to results.tsv
  8. If val_loss improved → KEEP (log as "keep")
  9. If worse or OOM → DISCARD (log as "discard" or "oom")
  10. Print 3-line summary
  11. Suggest next experiment
```

### Training Command Template
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline FAKE=false \
DATA_PATH=./data/tokens_fleurs_multilingual \
LOSS_WEIGHTS='[100,1,1]' WEIGHT_DECAY=0.0 DEPTH_NUM_LAYERS=4 \
LR=1e-3 BATCH_SIZE=4 GRADIENT_CHECKPOINTING=true \
MAX_STEPS=2000 WARMUP_STEPS=200 \
bash /apps/voxtral/autoresearch/scripts/run_safe.sh \
"uv run python scripts/train_omni.py" 600
```

### Results Parsing (grep, not full log read)
```bash
# Get final metrics
grep -E "^(total_loss|val_total|temporal_loss|depth_loss|text_acc|depth_q0|peak_vram|FAIL)" \
  /apps/voxtral/autoresearch/run.log | tail -20

# If empty (crash), diagnose:
tail -50 /apps/voxtral/autoresearch/run.log
```

### Keep/Discard Decision
Primary metric: **val_total_loss** (NOT train_loss — train_loss lies about generalization).
- **Keep if:** val_total < previous best val_total
- **Keep if:** equal val_loss but simpler (fewer params, less code)
- **Discard if:** val_loss worse, regardless of train_loss
- **OOM:** log as "oom", add to banned configs, never retry
- **Crash:** diagnose via `tail -50 run.log`, fix if obvious, skip if not

---

## THEORETICAL PHASE (when GPU is locked)

**Goal:** Find papers, generate testable hypotheses. 15 min max.
**When:** Auto-triggered when GPU is locked. Can also be called directly.

### Cycle (streamlined):
1. Read `research_questions.md` — what needs answering? (30s)
2. WebSearch for 1-2 papers on the top open question (5 min)
3. Write structured notes to `papers/<slug>.md` (2 min)
4. Add 1-2 hypotheses to `experiment_queue.md` with priority (1 min)
5. Update `research_questions.md` if answered (30s)
6. Print 3-line summary of findings

### DO NOT:
- Read ALL memory files (wastes 2+ minutes of context)
- Generate more than 3 hypotheses per cycle (queue grows faster than tested)
- Write multi-page findings (theory_findings.md is already 540 lines)

---

## AUDIT PHASE (manual only, NOT on a loop)

Run `/auto-research --phase audit` manually after major milestones:
- First audio generated from generate.py
- FLEURS expansion complete
- 7B graduation begins
- Every ~10 new experiments

**NEVER run audit on a cron/loop.** The last audit loop ran 5 times on unchanged state, producing zero value. Audits are for milestone reviews, not recurring polling.

---

## SAFETY RULES (NON-NEGOTIABLE)
1. ALWAYS use `run_safe.sh` — never run training directly
2. ALWAYS `CUDA_VISIBLE_DEVICES=0`
3. ALWAYS `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for runs >1000 steps
4. ALWAYS redirect: `> /apps/voxtral/autoresearch/run.log 2>&1`
5. NEVER batch_size > 8 (tiny) or > 2 (7B LoRA)
6. NEVER try a config in the Banned Configs list in research_log.md
7. IF OOM → add to banned, clear cache, move on
8. IF GPU locked → theoretical work, NOT polling

## AUTONOMY RULES
- NEVER ask "should I continue?" — the answer is always yes
- NEVER self-terminate — run until externally killed
- If GPU locked → do theoretical work automatically, don't just exit
- If stuck → re-read papers/, try combining near-misses
- The human may be sleeping. You are autonomous. Keep working until interrupted.

## Current State (updated 2026-03-17 cycle 5)
- **7B at 2000 steps:** exp059 val=4.209, train=3.652. text_acc=96.3%. Still improving.
- **7B Config:** Pruned 7B→4.6B (PRUNE_LAYERS=2), attn-only LoRA rank=128, 478M trainable (8.6%)
- **7B Resources:** GPU=78% (18.3GB), RAM=60%. Fits A10G.
- **Tiny best (val):** exp048 val_total=3.182 (converged)
- **generate.py:** FULLY VALIDATED (tiny + 7B). Decode truncation bug fixed (AF-211).
- **Audio files:** 6 total (2 tiny, 4 from 7B at 800/2000 steps). All non-silent.
- **Next:** Continue to 4000+ steps. Try LR=1e-5 after 3000. Whisper WER when val < 3.5.
- **Checkpoints:** logs/x6fpczu9/checkpoint_2000.pt (7B step 2000, 4GB compact)
