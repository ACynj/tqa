# TQA — Transferable Query Algebra

Code, generated query benchmarks, trained checkpoints, and paper sources for
**"Learnable and Transferable Logical Operators for Complex Query Answering on
Knowledge Graph Foundation Models"**.

TQA proves that fixed fuzzy operators sit at the bottom of an aggregation
hierarchy over witness structures, and instantiates the theory as **SQA**
(Structural Query Algebra): gated Aczél–Alsina operators over witness
evidence that remain transferable across vocabularies and degenerate exactly
to the fixed executor at trivial gates.

## Repository layout

| Path | Contents |
|---|---|
| `src/tqa/` | Model, training, evaluation, and analysis code |
| `scripts/` | Every shell script that produces a number or figure in the paper |
| `runs/` | Trained checkpoints (`.pt`) + per-run result JSONs + `FINAL_REPORT.md` |
| `logs/` | Raw training/evaluation logs backing every table |
| `data/queries/` | Generated query benchmarks (seeded; see `src/tqa/querygen.py`) |
| `paper/` | LaTeX source (`main.tex`, `supplement.tex`), figures (`figs_final/`), compiled PDFs |

## Setup

Two environments are used (see `scripts/setup_ultra_env.sh`):

```bash
# main environment (training + our evals)
pip install -r requirements.txt   # torch, numpy, scipy, ...

# UltraQuery baseline (third-party)
git clone https://github.com/DeepGraphLearning/ULTRA external/ULTRA
```

Datasets (BetaE suites, inductive FB15k-237 query versions, WikiTopics) are
downloaded automatically on first use into `~/git/ULTRA/query-datasets/`.
Every archive is md5-verified on fetch; corrupted archives are detected and
rejected (`scripts/verified_fetch.sh`).

## Reproducing the paper

| Paper artifact | Command |
|---|---|
| Controlled 5-seed study | `scripts/run_phase2.sh` |
| Zero-shot LOO transfer (Fig. 4 / Table VII) | `scripts/run_phase3.sh` |
| Official-protocol head-to-head (Table V) | `src/tqa/ft5.py`, logs in `logs/ft5_*.log` |
| NELL995 / FB15k breadth | `scripts/run_nell.sh`, `scripts/breadth_full.sh` |
| Gate exponent histogram (Fig. 5) | `paper/makefig_theta_cpu.py` |
| Two-world separation study | `src/tqa/synthetic2.py` |
| Router / fusion study | `src/tqa/route.py` |
| Aggregated report | `python scripts/final_report.py` → `runs/FINAL_REPORT.md` |

`paper/make_tqa_figures` style modules regenerate all figures from the JSONs
in `runs/`.

## Checkpoints

`runs/` contains every checkpoint referenced in the paper:
`ft5_gated.pt` / `ft5_fixed.pt` (official protocol), `p2_*_50k.pt`
(controlled five-seed family), `loo_*` (leave-one-graph-out), and ablations
(`abl_noscore / abl_nowitness`).

## Data integrity note

The public FB-122 archive is occasionally served truncated; the fetch scripts
verify md5 and the evaluation pipeline asserts graph/answer consistency
(`scripts/finalize_122.sh` regenerates the full FB-122 column of Table VI
from a verified archive).
