# Claude Code — My_creation (PRISM / InfoGate)

Instructions for agent sessions working in this directory.

## One-liner

Multimodal **PRISM / InfoGate**: sentiment **regression** (CMU-MOSI, CMU-MOSEI, CH-SIMS v2) and binary **classification** (MUStARD, UR-FUNNY). Regression uses **DeBERTa**; classification uses **ALBERT + HCF** (HKT-aligned)—they are **different** code paths and backbones.

## Training entry points

| Track | Script | Optuna driver | Backbone |
|-------|--------|---------------|----------|
| Regression | `train.py` | `optuna_search_v2.py` | DeBERTa-v3 (`deberta_infogate.py`; see `README.md`) |
| Classification | `train_classify.py` | `optuna_search_classify.py` | ALBERT-base-v2 (`albert_infogate.py`, default `./albert-base-v2`) |

## Shell launchers (cwd must be `My_creation`)

Scripts use `cd "$(dirname "$0")"` — run them from anywhere, but **relative paths** in child processes assume repo layout under `My_creation/`.

- **`run_optuna_4090d_restart.sh`**: progressive MOSI/MOSEI/SIMS v2 Optuna (`phase1` … `phase4`, `phase4_mosi`, etc.). Logs under `logs/optuna/4090D_restart/`.
- **`run_optuna_classify.sh`**: MUSTARD (+ optional UR-FUNNY) classification search.
- **`run_optuna_ur_funny_v2.sh`**: UR-FUNNY v2 classification slot.

Wrap parallel drivers in subshells so `cd` applies to each job, e.g. `( cd .../My_creation && ONLY=mosi ./run_optuna_4090d_restart.sh phase4_mosi ) &`.

## Python environment

Use a conda env that has **Optuna** (and project deps). On this machine, runs have used:

`PYTHON=/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python`

If that path is missing on your host, point `PYTHON` at any interpreter where `import optuna` works.

## Verification after editing search / train logic

Run at least one quick check before considering the task done:

```bash
cd My_creation   # from repo root; agents should cwd here before running Python
python -m py_compile optuna_search_v2.py optuna_search_classify.py train.py train_classify.py
```

Optional sanity:

```bash
python train.py --help
python train_classify.py --help
```

Full training is GPU-heavy—do not kick off long jobs unless the user asks.

## Logs and context discipline

- `logs/optuna/` can be **huge**.
- **Do not** blanket-grep the entire tree without narrowing paths.
- Prefer: `**/run/*.log` (tail / last “Best is trial”), or specific `train_logs/*trial_<n>.log`, or SQLite study DBs when diagnosing Optuna (see `.claude/CLAUDE.md`).

## Editing norms

- Match existing style; minimal diffs scoped to the request.
- Do **not** commit or rewrite massive logs/checkpoints; `.gitignore` already excludes typical artefacts.
- For large refactors or ambiguous design, **explore/plan first**, then implement.

## Model provider note

Hosted LLM choice (e.g. DeepSeek) is configured in **your** Claude Code user/local settings, not in this file.
