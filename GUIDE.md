# Guide: how to run and modify this workflow

The workflow designs protein binders in four steps:

```
target.pdb -> [1 RFdiffusion] -> [2 ProteinMPNN] -> [3 AF2 initial guess (Bennett)] -> [4 score] -> results_af2.csv
              draw backbones     write sequences      predict binder + target          pAE_interaction, pLDDT, RMSD
```

Most changes need **only a settings file**, not the code.

## 1. The run command

```bash
python design_pipeline.py --settings <settings file> --stages <steps>
```

| Option | Meaning |
|---|---|
| `--settings` | which settings file to use (default: `settings.py` next to the script) |
| `--stages` | comma-separated steps to run (default: `rfd,mpnn,af2,score_af2`) |
| `--output_dir` | write to this folder instead of the one in the settings file |
| `--afserver_zips` | folder of `fold_*.zip` files downloaded from alphafoldserver.com (only for `afs_import`) |

Stage names: `rfd`, `mpnn`, `af2`, `score_af2` (main route) - `afs_export`, `afs_import`, `score_afs` (optional AlphaFold Server route).

### The two ways to check designs (step 3)
| Route | Command | Gives |
|---|---|---|
| AF2 "initial guess" (Bennett lab), the main route | `--stages rfd,mpnn,af2,score_af2` | `results_af2.csv`: pAE_interaction (instead of iPTM), pLDDT, RMSD |
| AlphaFold Server (website), optional, the only route with iPTM | `--stages rfd,mpnn,afs_export`, upload `afserver_jobs.json` at alphafoldserver.com, download the zips, then `--stages afs_import,score_afs --afserver_zips <folder>` | `results_afserver.csv`: iPTM, pLDDT, RMSD |

### Example (GPU server, AF2 route)
```bash
cd ~/workflows/Protein_design_workflow
~/miniconda3/envs/rfd_bw/bin/python design_pipeline.py --settings settings_server_parallel.py --stages rfd,mpnn,af2,score_af2
```

### A small trial first
Make a tiny settings file (4 backbones x 2 sequences) that reuses a bigger one:
```python
import runpy
from pathlib import Path
S = runpy.run_path(str(Path.home() / "workflows/Protein_design_workflow/settings_server_parallel.py"))["SETTINGS"]
S.update({"num_backbones": 4, "seqs_per_backbone": 2, "output_dir": str(Path.home() / "trial_run")})
SETTINGS = S
```
Save it outside the git folder (for example `~/scripts/settings_trial.py`) and pass it with `--settings`. Expect a few minutes.

### Self-test (no GPU needed, a few seconds)
```bash
python tests/test_workflow.py
```
It uses pretend tools to check the plumbing and the score maths. It cannot judge design quality.

## 2. Gotchas that cost time
- **Use a Python that has numpy and Biopython** to run `design_pipeline.py`. A plain conda `(base)` environment often lacks Biopython (`No module named 'Bio'`). On the GPU server use the `rfd_bw` environment.
- **The tool Pythons are separate.** The settings file points each tool (RFdiffusion, ProteinMPNN, AF2) at the Python of its own conda environment, as a **full path** ending in `/bin/python`. `~` is not expanded there; write `/home/<user>/...` or build the path with `Path.home()`.
- **Use a fresh `output_dir` for every run.** Re-running into an old folder leaves old files behind, and ProteinMPNN reads every `design_*.pdb` in `1_rfdiffusion/`, including stale ones.
- **Linux paths are case sensitive**: `Protein_design_workflow` and `protein_design_workflow` are different folders.
- **GPU numbers** come from `nvidia-smi`. On the WSL server it is not on PATH: use `/usr/lib/wsl/lib/nvidia-smi`.
- **AF2 is limited by the CPU**, not the GPU (about 16 of every 20 s per prediction runs on one CPU core). Use several workers per card (see below) and a fixed `binder_length` (for example `"55-55"`) so JAX compiles only once.

## 3. Files
| File | Role | Edit it? |
|---|---|---|
| `design_pipeline.py` | the workflow (all steps) | rarely |
| `settings_server_parallel.py` | two-GPU server; `WORKERS_PER_GPU` at the top | often |
| `settings_big_gpu.py` | one big GPU (L40S / H200); worker counts can be set per run, e.g. `AF2_WORKERS=16` | often |
| `settings.py` | beginner template for one GPU (AF2 route) | yes, to start |
| `settings_server.py`, `settings_server_large.py` | base / older server settings the others build on | rarely |
| `tests/` | `test_workflow.py` (run it), `fake_tools.py` (pretend tools), `settings_test.py` | run, do not edit |
| `real_run/` | example results and timings copied here by hand; the pipeline never writes to it | read only |

## 4. Reading `design_pipeline.py`
Open `main()` first: it is the table of contents and runs the stages in order. Then jump to a step. In VS Code, **Ctrl+Shift+O** lists all functions, **Ctrl+G** goes to a line, **F12** jumps to a definition.

| Part | Function(s) |
|---|---|
| Run commands, GPU workers | `run`, `workers_for`, `split_by_weight`, `run_parallel` |
| Read structures | `read_chains`, `find_binder_and_target` |
| Step 1: RFdiffusion | `step1_rfdiffusion` |
| Step 2: ProteinMPNN | `step2_mpnn`, `design_sequences`, `read_mpnn_fasta` |
| Step 3: AF2 (main route) | `step3_af2`, `make_af2_pdb`, `read_af2_scores` |
| Step 3: AlphaFold Server (optional) | `step3_afserver_export`, `step3_afserver_import`, `afs_files` |
| Step 4: scoring (AF2) | `score_job_af2`, `step4_score_af2` |
| Step 4: scoring (AlphaFold Server) | `score_job_afs`, `step4_score_afs` |
| RMSD maths | `make_fitter`, `rmsd` |

## 5. "I want to change..." cheat sheet
| Change | Where |
|---|---|
| target file, residues, hotspots, binder length | settings: `target_pdb`, `target_range`, `hotspots`, `binder_length` |
| number of designs | settings: `num_backbones`, `seqs_per_backbone` |
| workers per GPU | `WORKERS_PER_GPU` at the top of the settings file (or `"copies": N` on a worker) |
| what counts as a pass | settings: `min_plddt`, `max_rmsd`, `max_pae_interaction`, `min_iptm` |
| AF2 effort | settings: `af2_recycles` |
| more varied sequences | the `"--sampling_temp", "0.1"` line in `design_sequences` (try 0.2-0.3) |
| ranking of the table | the `rows.sort(...)` lines in `step4_score_af2` / `step4_score_afs` |
| the pass rule itself | the `ok = ...` lines in `step4_score_af2` / `step4_score_afs` |
| a new result column | add it to the dictionary returned by `score_job_af2` / `score_job_afs` |

Output folders inside `output_dir`: `1_rfdiffusion/` (backbones), `2_mpnn/sequences.csv`, `3_af2_inputs/` and `3_af2_outputs/` (one `w0`, `w1`... folder per worker), and finally `results_af2.csv` (the AlphaFold Server route uses `3_afserver_outputs/` and `results_afserver.csv`).

## 6. Safe editing habits
1. Change one thing at a time.
2. Run the self-test after code edits.
3. Try new pass thresholds without re-running anything (seconds, no GPU):
   ```bash
   python design_pipeline.py --settings settings_server_parallel.py --output_dir <finished run folder> --stages score_af2
   ```
4. `git diff` shows what you changed; `git checkout -- <file>` undoes an edit you have not committed.

## 7. Moving files between computers
The server's `100.x` address is reachable only from WSL on the laptop, not from PowerShell.
```bash
scp <file> <user>@<server-ip>:~/workflows/Protein_design_workflow/      # laptop -> server
scp <user>@<server-ip>:~/trial_run/results_af2.csv .                    # server -> laptop
```
From PowerShell, put `wsl` in front, and write `C:\Research` as `/mnt/c/Research`.

## 8. Reading the results
`pae_interaction` (lower is better, below 10 is good) - `plddt_binder` (above 80) - `rmsd_binder` in angstrom (below 2). AF2 is not exactly repeatable: designs that sit right on a limit can flip between runs, so trust designs that pass comfortably. Passing means the design is self-consistent in silico, not that it binds in the lab.
