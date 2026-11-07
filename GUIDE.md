# Guide: how to run and modify this workflow

```
RFdiffusion route:  target.pdb → [rfd] backbones → [mpnn] sequences → [af2 | af3] predict binder + target → [score] → results_af2.csv | results_af3.csv
RFD4 route:         target.pdb → make_rfd4_input.py (CIF + spec) → rfd4-generate → rfd4_to_af3_jobs.py → [af3] → [score_af3] → report
```
Almost every change needs **only `settings.py`**, not code.

## 1. Run it
```bash
./setup.sh                     # status of every component and the next step
ROUTE=af2 python design_pipeline.py              # RFdiffusion + MPNN + AF2   (use a Python with numpy and biopython)
ROUTE=af3 python design_pipeline.py              # RFdiffusion + MPNN + AF3
./setup.sh run-rfd4 --n 50                       # RFD4 + AF3 (add --search for reward-guided search)
```
| `design_pipeline.py` option | Meaning |
|---|---|
| `--settings` | another settings file (default `settings.py`) |
| `--stages` | steps to run, comma separated (default: the `stages` of the chosen `ROUTE`) |
| `--output_dir` | write here instead of `output_dir` from the settings (also `OUTPUT_DIR=...`) |
| `--afserver_zips` | folder of `fold_*.zip` downloaded from alphafoldserver.com (only for `afs_import`) |

Stages: `rfd`, `mpnn`, `af2`, `score_af2` · `af3`, `score_af3` · `afs_export`, `afs_import`, `score_afs` (optional AlphaFold Server website route, the only one with iPTM besides AF3).
Re-scoring needs no GPU, so new pass limits can be tried in seconds: `python design_pipeline.py --output_dir <finished run> --stages score_af3`.

## 2. The CIF files (`./setup.sh example`)
The models read **CIF**, not your PDB. Two small scripts write them from the PDB in `settings.py` and put them in `work/` (git-ignored):
- `make_af3_template.py` → `work/target_template.cif`: the target alone, as an AF3 *template*. AF3 is told what the target looks like and only has to place the binder.
- `make_rfd4_input.py` → `work/rfd4_input.cif`, `rfd4_spec.yaml`, `rfd4_target_seq.txt`: one chain per unbroken target piece, the spec that fixes the target's coordinates and sequence, and the sequence the converter checks against. `--check` loads the spec like RFD4 does and stops if the target is not fully fixed.

## 2b. Result notes in Obsidian
Start a run with `--log` (`./setup.sh run-rfd4 --n 50 --log --title "..." --goal "..." --parent <previous slug>`, or `python design_pipeline.py --log --title ...`) and, when it
ends, its result notes are queued on the GPU machine (`~/vault_outbox`). A scheduled task on the laptop (`vault_pull.sh`, every 5 minutes) copies finished ones into the vault
folder `binder-design/` and rebuilds the overview. Runs without `--log` are never pulled; draft notes then stay in `<output_dir>/iteration_note/`.
The notes hold generated numbers (setup, results, top and passing designs, provenance). *Findings*, *Caveats* and *Next step* are left for you to write.
Each output folder carries `run_info.json` (every `design_pipeline.py` invocation) and, for `run-rfd4`, `run_meta.json`; notes cite those as provenance.
Details, troubleshooting and the laptop install are in [AGENTS.md](AGENTS.md).
## 3. Gotchas that cost time
- **Fresh `output_dir` for every run.** Old files in a reused folder get mixed into new results (ProteinMPNN reads every `design_*.pdb`).
- **Tool Pythons are separate.** Each tool points at the Python of its own conda env as a **full path** ending in `/bin/python`; `~` is not expanded.
- **Job names must be unique.** `rfd4_to_af3_jobs.py` names a job after the number at the end of its sample folder and now stops on duplicates (an earlier run silently collapsed 50 designs onto 4 names).
- **AF3 on an old second GPU:** AF3 needs compute capability ≥ 8.0; `af3_env` hides the other card with `CUDA_VISIBLE_DEVICES`.
- **AF2 is CPU-bound** (about 16 of every 20 s per prediction on one core): use several workers per card (see the commented `af2_workers` example in `settings.py`; measured 2.5× faster with 4 workers) and a fixed `binder_length` so JAX compiles once. AF2 is not exactly repeatable: designs sitting on a limit can flip pass/fail.
- **WSL:** `nvidia-smi` is `/usr/lib/wsl/lib/nvidia-smi`. Linux paths are case sensitive.

## 4. Reading `design_pipeline.py`
Open `main()` first; it runs the stages in order. In VS Code, **Ctrl+Shift+O** lists functions.

| Part | Function(s) |
|---|---|
| Run commands, GPU workers | `run`, `workers_for`, `split_by_weight`, `run_parallel` |
| Read structures | `read_chains`, `find_binder_and_target` |
| RFdiffusion / ProteinMPNN | `step1_rfdiffusion`, `step2_mpnn`, `design_sequences`, `read_mpnn_fasta` |
| AF2 | `step3_af2`, `make_af2_pdb`, `read_af2_scores`, `score_job_af2`, `step4_score_af2` |
| AF3 | `step3_af3`, `score_job_afs` (shared with AlphaFold Server), `step4_score_afs` |
| AlphaFold Server | `step3_afserver_export`, `step3_afserver_import`, `afs_files` |
| RMSD maths | `make_fitter`, `rmsd` |

## 5. "I want to change…"
| Change | Where |
|---|---|
| target, residues, hotspots, binder length | `settings.py`: the TARGET block |
| route (AF2 / AF3 / RFD4) | `ROUTE` in `settings.py` or the environment |
| number of designs | `num_backbones`, `seqs_per_backbone` (RFdiffusion); `rfd4_num_designs` or `--n` (RFD4) |
| predictions per design in AF3 | `af3_diffusion_samples` |
| what counts as a pass | `min_iptm`, `min_plddt`, `max_rmsd`, `max_pae_interaction`; the rule itself is the `ok = ...` line in `step4_score_*` |
| reward-guided search (beam width, branches, when to score, which metric) | `rfd4/hydra/inference_experiment/search.yaml` |
| more varied ProteinMPNN sequences | the `"--sampling_temp", "0.1"` line in `design_sequences` (try 0.2–0.3) |
| a new result column | the dictionary returned by `score_job_af2` / `score_job_afs` |

## 6. Safe editing habits
1. Change one thing at a time and run `python tests/test_workflow.py` after code edits.
2. `git diff` shows what changed; `git checkout -- <file>` undoes an uncommitted edit.
3. Copy files between machines through WSL (`scp`); the server's `100.x` address is not reachable from PowerShell.
