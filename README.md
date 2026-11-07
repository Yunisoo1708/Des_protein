# Binder design: RFdiffusion / RFD4 → AlphaFold 2 or AlphaFold 3

Designs protein binders for a target and checks them by predicting the complex again. Three routes share one settings file:

| `ROUTE` | Designs | Checked with | Gives |
|---|---|---|---|
| `af2` | RFdiffusion → ProteinMPNN | AlphaFold2 "initial guess" (Bennett lab) | `results_af2.csv`: `pae_interaction`, `plddt_binder`, `rmsd_binder` (AF2 has no iPTM) |
| `af3` | RFdiffusion → ProteinMPNN | local AlphaFold 3, optional target template | `results_af3.csv`: also `iptm` |
| `rfd4_af3` (default) | RFD4 (backbone + sequence together) | local AlphaFold 3 with the target template, 5 samples | `results_af3.csv` |

```
./setup.sh                  # what is installed, what to run next (changes nothing)
./setup.sh af3              # AF2 machine -> add AlphaFold 3
./setup.sh rfd4             # add RFD4
./setup.sh example          # make the two CIF files the models read (work/)
./setup.sh run-rfd4 --n 50  # RFD4 generate -> convert -> AF3 check -> report   (add --search for reward-guided search)
ROUTE=af2 python design_pipeline.py      # the original AF2 route, unchanged
```
`./setup.sh <step> --dry-run` prints the commands without running them. Read [GUIDE.md](GUIDE.md) for how to change things.

## Files
| File | Role |
|---|---|
| `setup.sh` | install / check / run: the one script for switching a machine from AF2 to AF3 and RFD4 |
| `settings.py` | the one settings file: target, routes, tool paths, thresholds |
| `design_pipeline.py` | the pipeline stages (`rfd`, `mpnn`, `af2`, `score_af2`, `af3`, `score_af3`, optional AlphaFold Server stages) |
| `make_af3_template.py`, `make_rfd4_input.py` | write the **CIF files** AF3 (target template) and RFD4 (input + task spec) read from the target PDB |
| `rfd4_to_af3_jobs.py`, `rfd4_af3_report.py` | convert RFD4 designs for the AF3 stage (refuses a design whose target changed) / summarise the check |
| `log_iteration.py`, `vault_pull.sh`, `AGENTS.md` | result notes for the Obsidian vault (`binder-design/`: one note per iteration, one per passing design, the results table, an overview). A run started with `--log` queues them when it ends; `vault_pull.sh`, scheduled on the laptop, copies them into the vault. You write the interpretation; `AGENTS.md` is the rule set for agents |
| `rfd4/` | reward-guided search config, the Apptainer stand-in (`shim/`), `select_best.py`; see `rfd4/README.md` |
| `prepared_protein_structure/5YXH.pdb` | the example target |
| `tests/` | `python tests/test_workflow.py` runs the pipeline with fake tools (no GPU); it checks plumbing and score maths, not design quality |

## What the numbers mean
| column | meaning | good |
|---|---|---|
| `iptm` | AF3's confidence in the binder–target interface (AF3 routes) | ≥ 0.8 |
| `pae_interaction` | expected error (Å) in the binder's position relative to the target | < 10 |
| `plddt_binder` | confidence in the binder's own fold (0–100) | > 80 |
| `rmsd_binder` | Å between the predicted binder and the designed one, complexes lined up on the target | < 2 |

Passing means the design is self-consistent in silico, not that it binds in the lab. AF2 and AF3 are different predictors, so their numbers are not interchangeable. AF3 needs a GPU with compute capability ≥ 8.0 and the model parameters (`af3.bin.zst`, which you obtain from DeepMind under their terms; the Pairformer is part of that file).

## Honest notes
- **AF3 as both guide and judge.** In reward-guided search the AF3 oracle steers generation and the final check is AF3 as well, so they are not independent. OpenFold3 would be a more independent guide, but it needs RFD4's patched AF3 build and converted weights that this repo does not install (`./setup.sh openfold3` reports what is missing).
- **The template matters.** AF3 cannot fold the small 5YXH target alone (≈16 Å error without the template, ≈1.2 Å with it), so `rmsd_binder` without a template is not meaningful.
- **Example result (5YXH, one run each, AF3 + template, 5 samples):** plain RFD4 0/50 pass; RFD4 with an AF3-pLDDT beam search 1/50; 51 partial re-noising variants around that passing design 11/51 (near copies of one parent, so correlated). One design reproduced across 20 seeds (97/100 samples with iPTM ≥ 0.8 and RMSD ≤ 2 Å) and fell apart without the template.
