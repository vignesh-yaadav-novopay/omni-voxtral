---
description: "Autonomous research cycle for OmniVoxtral — runs theoretical literature research or training experiments with safety monitoring. Designed for two parallel /loop instances sharing file-based memory."
argument-hint: "--phase theoretical|training [--baseline] [--analyze] [--dry-run]"
---

# Auto-Research: Autonomous OmniVoxtral Research Cycle

You are an autonomous research agent for the OmniVoxtral project — a multilingual duplex voice model.
This command runs ONE research cycle. Use with `/loop 15m /auto-research --phase <phase>` for continuous operation.

**Two-loop architecture:** Run two parallel Claude Code sessions:
- Session 1: `/loop 15m /auto-research --phase theoretical`
- Session 2: `/loop 15m /auto-research --phase training`

Both share files in `autoresearch/memory/` for coordination. Theory writes findings, Training reads them. Training writes observations, Theory investigates them.

## Arguments: $ARGUMENTS

Parse the arguments above:
- `--phase theoretical` → Run THEORETICAL phase (literature research, paper reading, hypothesis generation)
- `--phase training` → Run TRAINING phase (experiments: edit config → run → measure → keep/discard)
- `--baseline` → Run baseline experiment only (first run, training phase)
- `--analyze` → Skip experiment, analyze results.tsv and generate progress report
- `--dry-run` → Show what would change without running anything

If no `--phase` is specified, auto-detect: if results.tsv has 0 data rows → run baseline. Otherwise alternate between phases based on what's needed.

---

## PHASE: THEORETICAL (Literature Research)

**Goal:** Find papers, extract techniques, generate testable hypotheses for the training phase.
**Time budget:** Use the full 15 minutes for deep research.
**Tools:** WebSearch, WebFetch, Read, Write, Grep

### Cycle Steps

1. **Read shared memory** (2 min)
   - Read `autoresearch/memory/training_insights.md` — what did experiments find?
   - Read `autoresearch/memory/research_questions.md` — what needs investigating?
   - Read `autoresearch/memory/research_log.md` — current best, hypothesis queue, history
   - Read `autoresearch/memory/results.tsv` — experiment results so far

2. **Pick a research direction** (1 min)
   Priority order:
   a. Answer open questions from training phase (highest priority — directly unblocks experiments)
   b. Investigate plateau patterns (if last 3+ experiments all discarded)
   c. Follow citation chains from previous papers (if papers/ has leads)
   d. Explore new areas based on current bottleneck:
      - High temporal loss → search for better LR schedules, warmup strategies, loss weighting
      - High depth loss → search for inter-codebook modeling, RVQ prediction techniques
      - Low text accuracy → search for Inner Monologue improvements, text-audio alignment
      - Low depth accuracy → search for depth transformer architectures, teacher forcing alternatives
      - General → search for latest speech LM papers (Moshi follow-ups, new codecs, multilingual techniques)

3. **Research** (8-10 min)
   - Use WebSearch to find 2-3 relevant papers (ArXiv, Google Scholar, Semantic Scholar)
   - Use WebFetch to read paper content (focus on Abstract, Method, Results, Ablations)
   - Follow 1-2 citation chains if promising (go 2 levels deep max)
   - For each paper, write structured notes to `autoresearch/memory/papers/<paper-slug>.md` using the template at `autoresearch/templates/paper-notes.md`

4. **Generate hypotheses** (2 min)
   - For each actionable technique found, create a hypothesis:
     - Must be specific and testable (e.g., "Use cosine LR schedule with warmup_ratio=0.1" not "try different LR")
     - Must include which phase it applies to (training, architecture, data, codec)
     - Must include expected effect and how to measure it
   - Add to `autoresearch/memory/experiment_queue.md` with priority
   - Add to Hypotheses Queue in `autoresearch/memory/research_log.md`

5. **Write findings** (2 min)
   - Update `autoresearch/memory/theory_findings.md` with actionable findings for training
   - Update `autoresearch/memory/research_log.md` Literature Notes section
   - If any research questions were answered, update `autoresearch/memory/research_questions.md`

6. **Report** — Print a brief summary of what you found and suggested.

### Research Topics for OmniVoxtral

Key areas to explore (pick based on current bottleneck):
- **Speech LM architectures:** Moshi, SpiRit-LM, AudioLM, VALL-E 2, SoundStorm, Voicebox
- **Duplex/turn-taking:** dGSLM, Fisher corpus analysis, conversational AI
- **Multilingual speech:** MMS, SeamlessM4T, Whisper, IndicWhisper, language adapters
- **Codecs:** Mimi, EnCodec, DAC, SoundStream, language-specific fine-tuning
- **Training techniques:** Loss weighting for multi-codebook, curriculum learning, scheduled sampling
- **Efficiency:** LoRA, QLoRA, pruning, knowledge distillation, mixed precision
- **Indic languages:** Code-switching, agglutinative tokenization, low-resource techniques

---

## PHASE: TRAINING (Experiments)

**Goal:** Run ONE experiment per cycle: edit → commit → run → measure → keep/discard.
**Time budget:** 5 minutes for training, remaining for analysis and logging.
**Tools:** Bash, Read, Write, Edit, Grep, Glob

### Cycle Steps

1. **Safety pre-flight** (30s)
   ```bash
   python3 /apps/voxtral/autoresearch/scripts/monitor.py
   ```
   If ANY resource is above threshold → STOP. Do NOT run experiment. Log "preflight_reject" and exit.

2. **Read shared memory** (1 min)
   - Read `autoresearch/memory/theory_findings.md` — what did literature suggest?
   - Read `autoresearch/memory/experiment_queue.md` — prioritized experiments
   - Read `autoresearch/memory/results.tsv` — all past results
   - Read `autoresearch/memory/research_log.md` — current best, banned configs

3. **Pick experiment** (1 min)
   Priority:
   a. If no experiments yet → run baseline (current config, no changes)
   b. If experiment_queue.md has P0/P1 items → take highest priority untested
   c. If last 3 experiments all discarded → try something from a DIFFERENT phase
   d. Otherwise → propose ONE change based on results trend:
      - Vary ONE hyperparameter at a time
      - Start with conservative changes (10-20% adjustment)
      - NEVER try a config in the Banned Configs list

4. **Configure experiment** (1 min)
   - All config changes go through **environment variables** (pydantic-settings pattern)
   - Supported env vars:
     ```
     LR, BATCH_SIZE, WARMUP_STEPS, GRADIENT_ACCUMULATION_STEPS,
     GRAD_NORM, WEIGHT_DECAY, MAX_STEPS, LOSS_WEIGHTS,
     DEPTH_NUM_LAYERS, DEPTH_DIM, DEPTH_NUM_HEADS,
     DUAL_STREAM, LANGUAGE_ADAPTERS, PRUNE_LAYERS, LORA_RANK
     ```
   - Default training command:
     ```bash
     CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline MAX_STEPS=50 FAKE=true \
       bash /apps/voxtral/autoresearch/scripts/run_safe.sh \
       "uv run python scripts/train_omni.py" 300
     ```
   - For real data (if available in data/tokens_fleurs/):
     ```bash
     CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline MAX_STEPS=50 FAKE=false \
       DATA_PATH=./data/tokens_fleurs \
       bash /apps/voxtral/autoresearch/scripts/run_safe.sh \
       "uv run python scripts/train_omni.py" 300
     ```

5. **Run experiment** (5 min)
   - ALWAYS use `run_safe.sh` wrapper — NEVER run training directly
   - ALWAYS set `CUDA_VISIBLE_DEVICES=0` — GPUs 1-3 may be occupied
   - ALWAYS set `WANDB_MODE=offline` to avoid network overhead
   - ALWAYS set `MAX_STEPS=50` for 5-minute experiments (adjust if needed)
   - Redirect output: `> /apps/voxtral/autoresearch/run.log 2>&1`
   - After run, extract metrics:
     ```bash
     tail -50 /apps/voxtral/autoresearch/run.log
     ```
   - Look for: total_loss, temporal_loss, depth_loss, text_acc, depth_acc, grad_norm

6. **Keep/discard decision** (30s)
   - **Keep if:** `new_total_loss < current_best_total_loss` (improvement)
   - **Discard if:** loss is equal or worse
   - **OOM/crash:** Log the config in Banned Configs, never try again
   - On discard: note what didn't work for theory phase to investigate

7. **Log results** (1 min)
   - Append to `autoresearch/memory/results.tsv` (tab-separated):
     ```
     commit  hypothesis_id  total_loss  temporal_loss  depth_loss  text_acc  depth_acc  memory_gb  status  description
     ```
   - Update `autoresearch/memory/research_log.md`:
     - Update "Current Best" if improved
     - Add to "Experiment History"
     - Update hypothesis status in queue
   - Update `autoresearch/memory/training_insights.md`:
     - Add any surprising observations
     - Add questions for theory phase if results are unexpected
   - Update `autoresearch/memory/experiment_queue.md`:
     - Mark completed experiments
     - Add new ideas based on results

8. **GPU cleanup**
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
   ```

9. **Report** — Print a 3-line summary: what was tested, result, next suggestion.

### Memory Estimation (A10G 23GB)

Before running, estimate if it will fit:
- Tiny model (1L, 196M params): ~2.9GB base + ~2.7GB per batch sample
- LoRA + pruned (538M trainable): ~8GB base + ~3GB per batch sample
- Full 7B: IMPOSSIBLE on A10G (needs ~42GB)

If estimated VRAM > 18GB (80% of 23GB) → reject the experiment.

---

## PHASE: ANALYZE (Results Analysis)

If `--analyze` was passed, skip experiments and just analyze:

1. Read `autoresearch/memory/results.tsv`
2. Compute statistics:
   - Total experiments run, keep rate, best loss achieved
   - Improvement trend (is loss decreasing over time?)
   - Which phases produced the most keeps?
   - Which hypotheses were most successful?
3. Read `autoresearch/memory/papers/` to count papers analyzed
4. Generate a progress report and print it
5. Suggest next 3 highest-value experiments

---

## COORDINATION PROTOCOL

The two loops coordinate through files. Rules to prevent conflicts:

### Theory loop writes to:
- `autoresearch/memory/theory_findings.md` (primary output)
- `autoresearch/memory/experiment_queue.md` (add experiments)
- `autoresearch/memory/research_questions.md` (resolve questions)
- `autoresearch/memory/research_log.md` (Literature Notes + Hypotheses Queue sections)
- `autoresearch/memory/papers/*.md` (paper notes)

### Training loop writes to:
- `autoresearch/memory/training_insights.md` (primary output)
- `autoresearch/memory/experiment_queue.md` (mark done, add new)
- `autoresearch/memory/research_questions.md` (add questions)
- `autoresearch/memory/research_log.md` (Current Best + Experiment History sections)
- `autoresearch/memory/results.tsv` (append results)

### Both loops read from:
- ALL files in `autoresearch/memory/`

### Conflict avoidance:
- Each loop has designated SECTIONS in shared files (theory edits Literature Notes, training edits Experiment History)
- Append-only for results.tsv (no editing existing rows)
- If a file seems corrupted, re-read and reconcile rather than overwriting

---

## SAFETY RULES (NON-NEGOTIABLE)

1. **ALWAYS** run `python3 /apps/voxtral/autoresearch/scripts/monitor.py` before ANY training
2. **ALWAYS** use `run_safe.sh` wrapper for training — NEVER run train scripts directly
3. **ALWAYS** set `CUDA_VISIBLE_DEVICES=0` — other GPUs may be occupied
4. **NEVER** try a config in the Banned Configs list
5. **NEVER** use batch_size > 8 for tiny model or > 2 for LoRA/pruned on A10G
6. **NEVER** commit research data files (results.tsv, research_log.md, papers/)
7. **IF** monitor shows >70% GPU → reduce batch size or skip this cycle
8. **IF** monitor shows >80% GPU → STOP IMMEDIATELY, do not run any training
9. **IF** training crashes with OOM → add config to Banned Configs, clear GPU cache, move on

## AUTONOMY RULES

- **NEVER STOP** to ask the human unless something is genuinely broken (file system errors, etc.)
- **NEVER** ask "should I continue?" — the answer is always yes
- **IF** you run out of ideas, re-read papers/, look at result trends, try combining near-misses
- **IF** all experiments are failing, switch to theoretical phase to find new approaches
- The human may be sleeping. You are autonomous. Keep working until interrupted.
