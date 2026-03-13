# Auto-Research Plugin: Design Specification

**Date:** 2026-03-14
**Project:** OmniVoxtral — Autonomous Research System
**Status:** Revised (post-review)
**Review:** 3 critical issues resolved, 5 important issues addressed

---

## 1. Problem Statement

OmniVoxtral research spans multiple domains (codecs, tokenizers, data pipelines, architecture, multi-language training) and requires both empirical experimentation and theoretical exploration. Manual research is slow, expensive (GPU time), and crash-prone (previous OOM killed the server). We need an autonomous system that:

- Runs experiments in a tight loop (edit → train → measure → keep/discard)
- Reads research papers, follows citations, extracts techniques
- Monitors system resources to prevent OOM crashes
- Coordinates multiple research phases adaptively
- Integrates with Claude Code's `/loop` for continuous operation

## 2. Architecture

### 2.1 Two-Arm Design

```
┌─────────────────────────────────────────────────────────┐
│                  /auto-research orchestrator             │
│                                                         │
│  ┌──────────────────┐       ┌──────────────────────┐   │
│  │  THEORY ARM       │──────▶│  EXPERIMENT ARM       │   │
│  │  (Literature)     │       │  (Training)           │   │
│  │                   │       │                       │   │
│  │  • WebSearch      │       │  • Edit config/code   │   │
│  │  • Read papers    │       │  • Run 5-min train    │   │
│  │  • Follow cites   │       │  • Measure metric     │   │
│  │  • Take notes     │       │  • Keep/discard       │   │
│  │  • Extract ideas  │       │  • Log results.tsv    │   │
│  └──────────────────┘       └──────────────────────┘   │
│           │                          │                   │
│           ▼                          ▼                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │           SHARED RESEARCH MEMORY                  │   │
│  │  research_log.md — findings, hypotheses, results  │   │
│  │  results.tsv — experiment metrics (Karpathy fmt)  │   │
│  │  papers/ — extracted insights per paper           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │       SAFETY WATCHDOG (real-time process)         │   │
│  │  scripts/run_safe.sh wraps training process       │   │
│  │  Background: monitor.py polls every 30s           │   │
│  │  Kills child on threshold breach via SIGTERM/KILL │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Theory Arm (explorer agent):** Reads papers via WebSearch/WebFetch, follows citation chains 2-3 levels deep, extracts techniques, generates testable hypotheses. Writes to `research_log.md` and `papers/`.

**Experiment Arm (experimenter agent):** Takes a hypothesis, edits the appropriate config/code, runs a time-budgeted training experiment, measures the target metric, decides keep/discard, logs to `results.tsv`.

**Shared Research Memory:** Both arms read/write a common set of files. These files are **never committed to git** — only code/config changes are committed. The orchestrator uses research memory to decide what to do next.

**Safety Watchdog:** A real-time bash/python process wrapper (NOT an LLM agent). `scripts/run_safe.sh` launches training as a child process, starts `monitor.py` in a background loop, and kills the child via SIGTERM/SIGKILL if any resource threshold is breached.

### 2.2 Phase Selection Logic

The orchestrator picks the highest-value phase each cycle:

```
IF no experiments run yet:
    → Phase 2 (establish baseline)
ELIF experiments plateauing (last 3 experiments all discarded):
    → Phase 1 (read papers for new ideas)
ELIF hypothesis queue has untested items:
    → Phase matching the hypothesis type (2/3/4/5)
ELIF training loss good but audio quality poor:
    → Phase 5 (codec/tokenizer tuning)
ELSE:
    → Phase 2 (continue hyperparameter search)
```

### 2.3 Sequential Execution (default)

Each 15-min cycle runs **sequentially** by default:
1. Experimenter runs training (5 min budget)
2. Explorer reads papers with remaining time (~5 min)
3. Analyzer reviews trends and updates recommendations

This avoids coordination issues (file locking, concurrent writes to research_log.md). The Theory and Experiment arms never write to the same files simultaneously.

**Optional parallel mode:** When explicitly requested via `--parallel`, the explorer runs as a background agent during training. In this mode, the explorer writes ONLY to `papers/` and the Hypotheses Queue section, while the experimenter writes ONLY to the Experiment History section and results.tsv.

## 3. Research Phases

### Phase 1: Literature Research

**Agent:** `explorer`
**Tools:** WebSearch, WebFetch, Read, Write
**Target:** `research_log.md`, `papers/`
**Metric:** Hypotheses generated (qualitative)

Workflow:
1. Generate a research question from current bottleneck (e.g., "What loss schedules work for multi-codebook speech LMs?")
2. WebSearch for relevant papers (ArXiv, Semantic Scholar)
3. WebFetch the top 2-3 results, extract key sections
4. Follow citation chains — if paper A cites B which describes a relevant technique, fetch B too (2-3 levels deep)
5. Write structured notes to `papers/<paper-slug>.md` using the paper notes template (Section 6.4)
6. Generate hypotheses and add to the queue in `research_log.md`

### Phase 2: Training Hyperparameters

**Agent:** `experimenter`
**Editable:** Environment variables passed to training script
**Metric:** Weighted total loss (temporal + depth)
**Budget:** 5 minutes training wall clock

Supported env vars (all pass through pydantic-settings):
- `LR` — learning rate (float, e.g., `3e-5`)
- `BATCH_SIZE` — per-GPU batch size (int)
- `WARMUP_STEPS` — LR warmup steps (int)
- `GRADIENT_ACCUMULATION_STEPS` — gradient accumulation (int)
- `GRAD_NORM` — gradient clipping norm (float)
- `WEIGHT_DECAY` — weight decay (float)
- `LOSS_WEIGHTS` — JSON array, e.g., `'[100, 10, 1]'` (text, semantic, acoustic)
- `MAX_STEPS` — training steps per experiment (int, default 50 for 5-min budget)

Workflow:
1. Read current best config and results history
2. Propose ONE change (vary one hyperparameter at a time)
3. Set env vars and run training via `scripts/run_safe.sh`
4. Parse output: extract total_loss, temporal_loss, depth_loss, text_acc, depth_acc
5. Compare to current best → keep or discard
6. Log to `results.tsv`

### Phase 3: Architecture Search

**Agent:** `experimenter`
**Editable:** Architecture env vars (added to VoxtralTrainConfig)
**Metric:** Loss + parameter efficiency (loss / param_count)
**Budget:** 5 minutes training

Architecture parameters exposed as env vars:
- `DEPTH_NUM_LAYERS` — depth transformer layers (int, default 6)
- `DEPTH_DIM` — depth transformer hidden dim (int, default 1024)
- `DEPTH_NUM_HEADS` — depth transformer attention heads (int, default 16)
- `DUAL_STREAM` — dual-stream mode (bool, default false)
- `LANGUAGE_ADAPTERS` — per-family LoRA adapters (bool, default false)

**Implementation note:** These fields must be added to `VoxtralTrainConfig` and threaded through to `OmniVoxtralConfig` construction in `init_fsdp_model()` / `init_omni_train_state()`. This is a prerequisite task.

Same workflow as Phase 2 but edits architecture config. Must pass safety pre-flight.

### Phase 4: Data Mix Optimization

**Agent:** `experimenter`
**Editable:** Data pipeline config (language weights, curriculum order, overfit settings)
**Metric:** Per-language loss balance (minimize max per-language loss)
**Budget:** 5 minutes training

### Phase 5: Codec/Tokenizer Tuning

**Agent:** `experimenter`
**Editable:** Codec config (mimi_num_quantizers), tokenizer config (text_vocab_size, sp_tokenizer_path)
**Metric:** PESQ (codec quality) or fertility (tokenizer efficiency)
**Budget:** Varies (codec eval is fast, tokenizer training is slow)

### Per-Phase Keep/Discard Criteria

| Phase | Keep Condition | Metric |
|-------|---------------|--------|
| 2. Training | `new_total_loss < current_best_total_loss` | total_loss |
| 3. Architecture | `new_loss / new_params < current_best_loss / current_best_params` | loss/param ratio |
| 4. Data mix | `max(per_lang_losses) < current_best_max_lang_loss` | max per-lang loss |
| 5. Codec/Tokenizer | `new_pesq > current_best_pesq` OR `new_fertility < current_best_fertility` | PESQ or fertility |

## 4. Safety Watchdog

### 4.1 Resource Thresholds

| Resource | Warning | Hard Kill | Check Interval |
|----------|---------|-----------|----------------|
| GPU VRAM | 70% | 80% | 30 seconds |
| CPU RAM | 70% | 80% | 30 seconds |
| Disk | 90% | 95% | 60 seconds |

### 4.2 Process Wrapper: `scripts/run_safe.sh`

The watchdog is a **real-time process wrapper** — NOT an LLM agent (LLM agents have 3-5s latency per call, useless for OOM prevention that happens in milliseconds).

```bash
#!/bin/bash
# scripts/run_safe.sh — Runs a command with resource monitoring
# Usage: scripts/run_safe.sh "uv run scripts/train_omni_real.py" 300
#   $1 = command to run
#   $2 = time budget in seconds (default 300 = 5 min)

COMMAND="$1"
TIME_BUDGET="${2:-300}"
MONITOR_INTERVAL=30
GPU_THRESHOLD=80
CPU_THRESHOLD=80

# Start the experiment as a background process
eval "$COMMAND" &
CHILD_PID=$!

# Monitor loop
ELAPSED=0
while kill -0 $CHILD_PID 2>/dev/null; do
    sleep $MONITOR_INTERVAL
    ELAPSED=$((ELAPSED + MONITOR_INTERVAL))

    # Check resources
    SAFETY=$(python3 scripts/monitor.py)
    IS_SAFE=$(echo "$SAFETY" | python3 -c "import sys,json; print(json.load(sys.stdin)['safe'])")

    if [ "$IS_SAFE" = "False" ]; then
        echo "SAFETY: Resource threshold breached. Killing experiment."
        echo "$SAFETY"
        kill -TERM $CHILD_PID
        sleep 5
        kill -9 $CHILD_PID 2>/dev/null
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
        exit 2  # Exit code 2 = killed by safety monitor
    fi

    # Time budget check
    if [ $ELAPSED -ge $TIME_BUDGET ]; then
        echo "TIME: Budget of ${TIME_BUDGET}s reached. Stopping experiment."
        kill -TERM $CHILD_PID
        sleep 5
        kill -9 $CHILD_PID 2>/dev/null
        exit 0  # Exit code 0 = completed (time budget reached is normal)
    fi
done

# Child finished on its own
wait $CHILD_PID
exit $?
```

### 4.3 Pre-Flight Estimation

Before each experiment, the orchestrator estimates memory. The estimation is calibrated against known configurations:

**Known configurations (measured on A10G):**

| Config | Model | Batch | Peak VRAM |
|--------|-------|-------|-----------|
| Tiny (1L) | 196M | 1 | 2.9 GB |
| Tiny (1L) | 196M | 2 | 5.5 GB |
| Tiny (1L) | 196M | 4 | 10.5 GB |
| Tiny (1L) | 196M | 8 | 20.6 GB |
| LoRA+pruned | 538M trainable | 2 | ~14 GB |

**Estimation heuristic:**
```python
def estimate_memory_gb(param_count_m, batch_size, is_lora=False):
    """Rough GPU memory estimate calibrated against A10G measurements."""
    # Base: ~2.7 GB per batch sample for tiny model
    # Scales roughly linearly with param count (for same hidden dim)
    base_per_sample = 2.7  # GB, from tiny model measurements
    model_overhead = 1.0 if is_lora else 2.0  # LoRA uses less optimizer memory
    param_scale = max(1.0, param_count_m / 196)  # relative to tiny model
    return model_overhead + base_per_sample * batch_size * (param_scale ** 0.5)
```

If estimate exceeds 75% of GPU total, the experiment is rejected with status `preflight_reject`.

### 4.4 `scripts/monitor.py`

```python
import subprocess, sys, json

def get_gpu_usage():
    """Get GPU memory usage percentage. Handles multi-GPU correctly."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        max_pct = 0.0
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(", ")
            if len(parts) == 2:
                used, total = float(parts[0]), float(parts[1])
                pct = used / total * 100 if total > 0 else 0
                max_pct = max(max_pct, pct)
        return max_pct
    except Exception:
        return 0.0

def get_cpu_usage():
    """Get CPU RAM usage percentage."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 1)
            available = info.get("MemAvailable", total)
            return (1 - available / total) * 100
    except Exception:
        return 0.0

def get_disk_usage():
    """Get disk usage percentage."""
    try:
        import os
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        return (1 - free / total) * 100 if total > 0 else 0.0
    except Exception:
        return 0.0

def check_safety(gpu_threshold=80, cpu_threshold=80, disk_threshold=95):
    gpu = get_gpu_usage()
    cpu = get_cpu_usage()
    disk = get_disk_usage()
    safe = gpu < gpu_threshold and cpu < cpu_threshold and disk < disk_threshold
    return {
        "safe": safe,
        "gpu_pct": round(gpu, 1),
        "cpu_pct": round(cpu, 1),
        "disk_pct": round(disk, 1),
        "message": f"GPU={gpu:.0f}% CPU={cpu:.0f}% Disk={disk:.0f}%",
    }

if __name__ == "__main__":
    result = check_safety()
    print(json.dumps(result))
    sys.exit(0 if result["safe"] else 1)
```

**Note:** Uses `/proc/meminfo` instead of `psutil` to avoid adding a dependency. Falls back gracefully on non-Linux.

## 5. Project-Local Plugin Structure

The plugin uses Claude Code's **project-local configuration** at `/apps/voxtral/.claude/`, which is the standard mechanism for project-specific commands, skills, and agents. This avoids the marketplace infrastructure that installed plugins require.

```
/apps/voxtral/.claude/
├── commands/
│   └── auto-research.md               # /auto-research slash command
├── agents/
│   ├── experimenter.md                # Edit→commit→run→measure→decide
│   ├── explorer.md                    # Literature research + paper traversal
│   └── analyzer.md                    # Post-experiment analysis + trends
└── hooks.json                         # PreToolUse safety check

/apps/voxtral/autoresearch/
├── research_log.md                    # Shared research memory (NEVER committed)
├── results.tsv                        # Experiment metrics (NEVER committed)
├── papers/                            # Paper notes (NEVER committed)
├── templates/
│   └── paper-notes.md                 # Structured paper extraction template
├── scripts/
│   ├── run_safe.sh                    # Process wrapper with safety watchdog
│   ├── monitor.py                     # GPU/CPU/RAM monitoring utility
│   ├── analyze_results.py             # Progress report generation
│   └── setup.sh                       # Initialize research workspace
├── prepare.py                         # Karpathy's data prep (READ-ONLY)
├── train.py                           # Karpathy's model (EDITABLE by agent)
└── program.md                         # Karpathy's agent instructions (READ-ONLY)
```

### 5.1 Git Discipline

**CRITICAL:** Research data files are NEVER committed to git.

Add to `.gitignore`:
```
autoresearch/research_log.md
autoresearch/results.tsv
autoresearch/papers/
autoresearch/run.log
```

Experiment commits contain ONLY the code/config change being tested. This makes `git reset --hard HEAD~1` safe — it only reverts the experiment's code change, never research data.

### 5.2 Command: `/auto-research`

```markdown
---
description: "Run autonomous research cycle: literature + experiments with safety monitoring"
argument-hint: "[--phase literature|training|architecture|data|codec] [--baseline] [--analyze] [--max-cycles N] [--dry-run]"
allowed-tools: ["Bash", "Read", "Write", "Edit", "WebSearch", "WebFetch", "Agent", "Grep", "Glob"]
---

[Command body: orchestrator instructions — see Section 7]
```

Arguments:
- `--phase <name>` — Force a specific phase (literature, training, architecture, data, codec)
- `--baseline` — Run baseline experiment only (first run)
- `--analyze` — Skip experiment, just analyze results and generate report
- `--max-cycles N` — Stop after N experiment cycles (default: unlimited)
- `--dry-run` — Show what would change + pre-flight estimate, without running training

### 5.3 Agents

#### `experimenter.md`
- **Tools:** Bash, Read, Write, Edit, Grep, Glob
- **Model:** sonnet (fast, good at code editing)
- Receives a hypothesis (e.g., "Try LR=5e-5 instead of 2e-5")
- Checks safety via `scripts/monitor.py` (pre-flight)
- Sets env vars for the training script
- Commits the experiment config to git (code changes only)
- Runs training via `scripts/run_safe.sh` (process wrapper with watchdog)
- Parses metrics from output
- Makes keep/discard decision based on per-phase criteria (Section 3)
- Reverts commit on discard/crash: `git reset --hard HEAD~1`
- Appends to results.tsv (untracked file, safe from git reset)

#### `explorer.md`
- **Tools:** WebSearch, WebFetch, Read, Write, Grep
- **Model:** sonnet (good at synthesis)
- Receives a research question derived from current bottleneck
- Uses WebSearch to find relevant papers (ArXiv, Semantic Scholar, Google Scholar)
- Uses WebFetch to read paper content (abstracts, methods, results)
- Follows citation chains (2-3 levels deep)
- Writes structured notes to `papers/<paper-slug>.md` using the template (Section 6.4)
- Generates hypotheses and appends to `research_log.md` Hypotheses Queue

#### `analyzer.md`
- **Tools:** Read, Write, Bash, Grep
- **Model:** haiku (lightweight analysis)
- Runs every 3rd cycle (not every cycle, to save overhead)
- Reads results.tsv history
- Identifies trends (improving? plateauing? oscillating?)
- Computes statistics (best loss, improvement rate, keep rate)
- Updates "Current Best" section of research_log.md
- Generates recommendations for next experiments

### 5.4 Hooks

`/apps/voxtral/.claude/hooks.json`:
```json
{
  "hooks": [
    {
      "matcher": {
        "tool": "Bash",
        "command_pattern": "train_omni|train\\.py|run_safe"
      },
      "hooks": [{
        "type": "command",
        "command": "python3 /apps/voxtral/autoresearch/scripts/monitor.py",
        "timeout": 10
      }]
    }
  ]
}
```

This hook fires before any Bash command that matches training scripts. If `monitor.py` exits non-zero (unsafe), the tool call is blocked.

## 6. Shared Research Memory Format

### 6.1 research_log.md

```markdown
# OmniVoxtral Auto-Research Log

## Current Best
- **Total Loss:** 18.42 (step 500, exp007)
- **Config:** LR=5e-5, batch=4, warmup=300, depth_layers=6

## Hypotheses Queue
| ID | Source | Hypothesis | Phase | Status |
|----|--------|-----------|-------|--------|
| H001 | Moshi paper | Use cosine LR schedule instead of linear | training | UNTESTED |
| H002 | Experiment plateau | Double depth transformer layers to 12 | architecture | TESTED:DISCARD |
| H003 | SpiRit-LM paper | Interleave semantic and acoustic loss | training | UNTESTED |

## Banned Configs (caused OOM/crash)
- depth_num_layers=12 with batch_size=4 (OOM at 20.6GB, exp003)
- batch_size=8 with gradient_accumulation_steps=1 (OOM, exp005)

## Literature Notes
### [Moshi: Full-Duplex Speech LM](arxiv.org/abs/2410.00037)
- **Technique:** Equal weighting of temporal + depth loss
- **Applicability:** We already use this. Consider their warmup strategy.
- **Citations to follow:** [Inner Monologue], [Soundstorm]

## Experiment History (last 10)
| Exp | Hypothesis | Change | Loss | Status | Time |
|-----|------------|--------|------|--------|------|
| 001 | — | baseline | 21.80 | keep | 5:02 |
| 002 | H004 | LR 2e-5→5e-5 | 20.10 | keep | 5:01 |
| 003 | H002 | depth=6→12 | OOM | oom_risk | 0:45 |
```

### 6.2 results.tsv

Extended Karpathy-compatible format with hypothesis tracking:
```
commit	hypothesis_id	val_loss	temporal_loss	depth_loss	text_acc	depth_acc	memory_gb	status	description
a1b2c3d	—	21.8000	14.0000	7.8000	0.0000	0.0003	5.46	keep	baseline
b2c3d4e	H004	20.1000	12.5000	7.6000	0.0120	0.0050	5.50	keep	LR 2e-5 to 5e-5
c3d4e5f	H002	0.0000	0.0000	0.0000	0.0000	0.0000	20.63	oom_risk	depth 6 to 12
```

### 6.3 papers/ directory

One markdown file per paper with structured extraction.

### 6.4 Paper Notes Template

`autoresearch/templates/paper-notes.md`:
```markdown
# [Paper Title] ([Year])

**Source:** [URL or arXiv ID]
**Authors:** [Authors]
**Read date:** [YYYY-MM-DD]

## Key Techniques
1. [Technique 1]
2. [Technique 2]

## Results
- [Key result 1]
- [Key result 2]

## Applicability to OmniVoxtral
- [How technique 1 maps to our architecture]
- [What we could adopt or adapt]

## Generated Hypotheses
- H[ID]: [Specific testable hypothesis derived from this paper]

## Citations to Follow
- [Paper A] — [why it's relevant]
- [Paper B] — [why it's relevant]
```

## 7. Loop Integration

### Usage

```bash
# Continuous autonomous research (recommended)
/loop 15m /auto-research

# Focus on literature exploration
/loop 15m /auto-research --phase literature

# Quick hyperparameter sweep
/loop 15m /auto-research --phase training

# Analyze results only (no experiments)
/loop 30m /auto-research --analyze

# Cap total cycles
/loop 15m /auto-research --max-cycles 20

# Dry run (show what would happen without running)
/loop 15m /auto-research --dry-run
```

### Cycle Timeline (15 minutes)

Increased from 10 to 15 minutes to account for LLM agent overhead (~30s per agent call, ~8 calls per cycle = ~4 min overhead).

```
0:00  Safety pre-flight check (monitor.py)
0:30  Read research memory, select phase
1:00  Commit experiment config to git
1:30  Launch training via run_safe.sh (5-min budget)
6:30  Training complete, parse metrics
7:00  Keep/discard decision, git revert if needed
7:30  Update research_log.md + results.tsv
8:00  Explorer reads papers (if time permits, ~5 min)
13:00 Report summary to user
14:00 Cleanup (empty_cache, etc.)
15:00 Loop repeats

Every 3rd cycle: analyzer runs instead of explorer (trend analysis)
```

## 8. Experiment Workflow Detail

### 8.1 Baseline Run

The first experiment is always a baseline:
```bash
# Uses current config as-is, no modifications
scripts/run_safe.sh "WANDB_MODE=offline MAX_STEPS=50 uv run scripts/train_omni_real.py" 300
```

Parse output for metrics, log as "baseline" in results.tsv.

### 8.2 Subsequent Experiments

```
1. READ results.tsv → current_best (per-phase metric)
2. READ research_log.md → next hypothesis from queue, or generate one
3. SAFETY PRE-FLIGHT → run monitor.py, estimate memory for proposed config
4. IF pre-flight fails → log "preflight_reject", pick different hypothesis
5. EDIT → set env vars for the change
6. GIT COMMIT → snapshot ONLY the code/config change (NOT results.tsv, NOT research_log.md)
7. RUN → scripts/run_safe.sh with time budget (5 min)
8. PARSE → extract metrics from output (grep for key: value lines)
9. DECIDE (using per-phase criteria from Section 3):
   IF improved:
     status = "keep"
     # Stay on this commit
   ELIF crashed or OOM or safety-killed:
     status = "oom_risk" or "crash"
     git reset --hard HEAD~1  # Safe: only reverts code change
   ELSE:
     status = "discard"
     git reset --hard HEAD~1  # Safe: only reverts code change
10. LOG → append to results.tsv (untracked, survives git reset)
11. UPDATE → research_log.md hypothesis status + experiment history
```

## 9. Hardware Constraints

**Current machine:** 1x NVIDIA A10G (23GB VRAM), ~30GB CPU RAM

**Implications:**
- Full 7B Mistral: CANNOT run (needs ~42GB)
- LoRA + pruned 3.5B: CAN run (batch_size=2, ~14GB peak)
- Tiny model experiments: CAN run (batch_size up to 8, ~20GB peak)
- Safety monitor thresholds set for A10G limits

**Known good configurations (measured):**

| Config | Trainable Params | Batch | Peak VRAM | Status |
|--------|-----------------|-------|-----------|--------|
| Tiny 1L, no LoRA | 196M | 8 | 20.6 GB | OK |
| Tiny 1L, no LoRA | 196M | 16 | >23 GB | OOM |
| 7B pruned, LoRA r=64 | 538M | 2 | ~14 GB | OK |
| 7B full, no LoRA | 7.2B | 1 | >42 GB | IMPOSSIBLE |

**When upgrading to 2xH100:**
- Update thresholds in monitor.py (80GB per GPU)
- Switch experimenter to use train_omni_full.py (FSDP)
- Batch size and architecture experiments become feasible
- Full 7B training becomes the default

## 10. Prerequisites (Implementation Dependencies)

Before the plugin can be built:

1. **Extend `VoxtralTrainConfig`** with architecture fields (`depth_num_layers`, `depth_dim`, `depth_num_heads`) so Phase 3 can use env var overrides instead of code editing.

2. **Create `scripts/run_safe.sh`** — the process wrapper with safety watchdog.

3. **Create `scripts/monitor.py`** — the resource monitoring utility (spec in Section 4.4).

4. **Update `.gitignore`** — add autoresearch data files.

5. **Create `autoresearch/templates/paper-notes.md`** — the structured extraction template.

6. **Initialize `autoresearch/research_log.md`** and `autoresearch/results.tsv` with headers.

## 11. Success Criteria

1. **No OOM crashes** — safety watchdog prevents all server-killing events
2. **Measurable improvement** — val_loss decreases across experiment sessions
3. **Literature grounding** — at least 5 papers read and connected to experiments
4. **Autonomous operation** — runs via `/loop 15m /auto-research` without human intervention
5. **Full audit trail** — every experiment logged in results.tsv, every finding in research_log.md
6. **Crash resilience** — untracked data files survive git resets; experiments resume after interruption

## 12. Verification Plan

### 12.1 Unit Tests
- `monitor.py` returns correct safety status with mocked values
- `monitor.py` handles multi-GPU nvidia-smi output correctly
- results.tsv parsing handles all status types (keep, discard, oom_risk, crash, preflight_reject)
- Phase selection logic picks correct phase for each scenario

### 12.2 Integration Tests
- `scripts/run_safe.sh` kills child process when monitor reports unsafe
- `scripts/run_safe.sh` respects time budget (kills after N seconds)
- `/auto-research --baseline` produces results.tsv with baseline entry
- `/auto-research --phase literature` produces papers/ notes
- `/auto-research --phase training` runs experiment, logs result
- `/auto-research --dry-run` shows plan without running anything
- `git reset --hard HEAD~1` does NOT destroy results.tsv or research_log.md

### 12.3 End-to-End Test
- Run `/loop 15m /auto-research` for 3 cycles
- Verify: 3 entries in results.tsv, research_log.md updated, no OOM
- Verify: at least one keep and one discard (proves decision logic works)
- Verify: explorer produced at least one paper note in papers/
