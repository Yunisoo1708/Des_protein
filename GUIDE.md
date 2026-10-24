# Guide: how to run and modify this workflow

The workflow designs protein binders in four steps:

```
target.pdb -> [1 RFD4-Proteina] -> [3 AF2 initial guess (Bennett)] -> [4 score] -> results_af2.csv
              backbone + sequence   predict binder + target          pAE_interaction, pLDDT, RMSD
```

RFD4-Proteina is the default and designs the sequence too, so step 2 (ProteinMPNN) is skipped. With `BACKBONE_TOOL=rfd3` (or
automatically on a card older than Ampere) the older route is used: RFdiffusion 3 -> [2 ProteinMPNN] -> AF2 -> score.

Most changes need **only a settings file**, not the code.

## 1. The run command

```bash
python design_pipeline.py --settings <settings file> --stages <steps>
```

| Option | Meaning |
|---|---|
| `--settings` | which settings file to use (default: `settings.py` next to the script) |
| `--stages` | comma-separated steps to run (default: `glyco,rfd,mpnn,af2,score_af2`) |
| `--netnglyc_output` | text output of NetNGlyc 1.0 for `0_glyco/target.fasta` (stage `glyco`) |
| `--output_dir` | write to this folder instead of the one in the settings file |
| `--afserver_zips` | folder of `fold_*.zip` files downloaded from alphafoldserver.com (only for `afs_import`) |

Stage names: `glyco` (target glycosylation scan, on by default, no GPU), `rfd`, `mpnn`, `af2`, `score_af2` (main route) - `afs_export`, `afs_import`, `score_afs` (optional AlphaFold Server route).

### The two ways to check designs (step 3)
| Route | Command | Gives |
|---|---|---|
| AF2 "initial guess" (Bennett lab), the main route | `--stages rfd,mpnn,af2,score_af2` | `results_af2.csv`: pAE_interaction (instead of iPTM), pLDDT, RMSD |
| AlphaFold Server (website), optional, the only route with iPTM | `--stages rfd,mpnn,afs_export`, upload `afserver_jobs.json` at alphafoldserver.com, download the zips, then `--stages afs_import,score_afs --afserver_zips <folder>` | `results_afserver.csv`: iPTM, pLDDT, RMSD |

### The one settings file: `settings.py`
Everything about your job is at the top of `settings.py`, in the **YOUR JOB** block (target file, residues, hotspots, binder length, number of designs, GPUs). Either edit that block and run `python design_pipeline.py --output_dir <new folder>`, or leave the file alone and put the same values in front of the command as environment variables (they win over the block). `settings.py` looks at each card with `nvidia-smi` and works out the tool, the AF2 environment, the worker counts and the batch sizes for you. Its docstring lists every variable.

RFD4-Proteina, the default, on an Ampere or newer card:
```bash
cd ~/workflows/Protein_design_workflow
GPUS=0 TARGET_PDB=~/targets/5YXH.pdb TARGET_RANGE=A16-108,A113-151 HOTSPOTS=A31,A123,A122,A32 BINDER_LENGTH=55-55 \
  NUM_BACKBONES=8 python design_pipeline.py --output_dir ~/runs/trial1
```
RFD3 + ProteinMPNN (chosen automatically on a card older than Ampere, e.g. the GTX 1660 SUPER, or force it with `BACKBONE_TOOL=rfd3`):
```bash
GPUS=1 TARGET_PDB=~/targets/5YXH.pdb TARGET_RANGE=A16-108,A113-151 HOTSPOTS=A31,A123,A122,A32 BINDER_LENGTH=55-55 \
  NUM_BACKBONES=4 SEQS_PER_BACKBONE=2 python design_pipeline.py --output_dir ~/runs/trial2
```
Both cards: `GPUS=0,1`. `GPUS=1` makes every tool use only the second GPU. The first full test of the RFD3 example (4 backbones x 2 sequences) took 7 min 38 s on a 6 GB GTX 1660 SUPER. Leaving out `--stages` runs all of them.

### A small trial first
Add `NUM_BACKBONES=4 SEQS_PER_BACKBONE=2` to the command (or edit those two lines in the YOUR JOB block) and use a new `--output_dir`. Expect a few minutes. (Four designs say little about a target; this only checks that everything runs.)

### Self-test (no GPU needed, a few seconds)
```bash
python tests/test_workflow.py
```
It uses pretend tools to check the plumbing and the score maths. It cannot judge design quality.

## 2. Gotchas that cost time
- **Use a Python that has numpy and Biopython** to run `design_pipeline.py`. A plain conda `(base)` environment often lacks Biopython (`No module named 'Bio'`). On the GPU server use the `rfd_bw` environment.
- **The tool Pythons are separate.** `settings.py` points each tool (RFD4, RFD3, ProteinMPNN, AF2) at the program of its own conda environment under `ENVS_DIR` (default `~/miniconda3/envs`); if you installed somewhere else, set `ENVS_DIR` and `TOOLS_DIR`. Paths you type into the file must be **full paths** (`~` is not expanded there).
- **Use a fresh `output_dir` for every run.** Re-running into an old folder leaves old files behind, and ProteinMPNN reads every `design_*.pdb` in `1_rfdiffusion/`, including stale ones. The backbone step now refuses to run into a folder that already holds backbones.
- **`target_range` uses commas between pieces** (`A16-108,A113-151`) and every residue in it must exist in the PDB file. Leave out any stretch that is missing from the file.
- **Never use `low_memory_mode=True` for RFD3 on a small card:** it was far slower in a test.
- **Linux paths are case sensitive**: `Protein_design_workflow` and `protein_design_workflow` are different folders.
- **GPU numbers** come from `nvidia-smi`. On the WSL server it is not on PATH: use `/usr/lib/wsl/lib/nvidia-smi`.
- **AF2 is limited by the CPU**, not the GPU (about 16 of every 20 s per prediction runs on one CPU core). Use several workers per card (see below) and a fixed `binder_length` (for example `"55-55"`) so JAX compiles only once.

## 3. Files
| File | Role | Edit it? |
|---|---|---|
| `design_pipeline.py` | the workflow (all steps) | rarely |
| `settings.py` | **the one settings file**: your job at the top, the GPU/tool logic below (one card or several) | the top block, often |
| `install_envs.sh` | installs the tools and conda environments once on a new machine | run it, do not edit |
| `slurm_job.sh`, `slurm_job_tillicum.sh` | Slurm job scripts for Hyak Klone / Tillicum; they just set the variables and call `design_pipeline.py` | edit the `<-- EDIT` lines |
| `tests/` | `test_workflow.py` (run it), `fake_tools.py` (pretend tools), `settings_test.py` | run, do not edit |
| `real_run/` | example results and timings copied here by hand; the pipeline never writes to it | read only |

## 4. Reading `design_pipeline.py`
Open `main()` first: it is the table of contents and runs the stages in order. Then jump to a step. In VS Code, **Ctrl+Shift+O** lists all functions, **Ctrl+G** goes to a line, **F12** jumps to a definition.

| Part | Function(s) |
|---|---|
| Run commands, GPU workers | `run`, `workers_for`, `split_by_weight`, `run_parallel` |
| Read structures | `read_chains`, `find_binder_and_target` |
| Step 0: target glycosylation scan | `step0_glyco`, `glyco_scan_baseline`, `reglyco_scan`, `parse_netnglyc` |
| Step 1: backbones | `step1_backbones` (picks the tool), `step1_rfd3`, `rfd3_spec`, `convert_rfd3_output`, `fresh_backbone_folder`; `step1_rfdiffusion` is the older RFdiffusion 1 |
| Step 1 (optional RFD4-Proteina) | `step1_rfd4`, `rfd4_target` (target -> CIF, one chain per piece), `rfd4_spec`, `convert_rfd4_output` |
| Step 2: ProteinMPNN | `step2_mpnn`, `design_sequences`, `read_mpnn_fasta`; `native_sequences` (RFD4: its own sequences, no ProteinMPNN) |
| Step 3: AF2 (main route) | `step3_af2`, `make_af2_pdb`, `read_af2_scores` |
| Step 3: AlphaFold Server (optional) | `step3_afserver_export`, `step3_afserver_import`, `afs_files` |
| Step 4: scoring (AF2) | `score_job_af2`, `step4_score_af2` |
| Step 4: scoring (AlphaFold Server) | `score_job_afs`, `step4_score_afs` |
| RMSD maths | `make_fitter`, `rmsd` |

## 5. "I want to change..." cheat sheet
| Change | Where |
|---|---|
| target file, residues, hotspots, binder length | the YOUR JOB block in `settings.py`, or `TARGET_PDB`, `TARGET_RANGE`, `HOTSPOTS`, `BINDER_LENGTH` |
| number of designs | `NUM_BACKBONES`, `SEQS_PER_BACKBONE` (same two places) |
| which GPU(s) | `GPUS` (`0`, `1` or `0,1`) in the YOUR JOB block or in front of the command |
| workers per GPU | `AF2_WORKERS`, `RFD_WORKERS`, `MPNN_WORKERS` (every card), or with a card suffix, e.g. `AF2_WORKERS_1=2` |
| what counts as a pass | settings dictionary (bottom of `settings.py`): `min_plddt`, `max_rmsd`, `max_pae_interaction`, `min_iptm` |
| glycosylation scan | `GLYCO_WARN_A` (warning distance to hotspots, default 15 A), `GLYCO_CACHE`, `REGLYCO_OFFLINE=1`, `NETNGLYC_OUTPUT` / `NETNGLYC_BIN`, `REGLYCO_BIN` |
| AF2 effort | `af2_recycles` in the settings dictionary |
| how varied the RFD3 backbones are | `RFD3_OVERRIDES`, e.g. `inference_sampler.step_scale=1.5` (higher = less diverse, more designable), `inference_sampler.gamma_0=0` (deterministic) |
| designs RFD3 makes at once | `RFD3_BATCH` (default 4 on a card with 12+ GB, else 2; 2 was ~1.8x faster than 1 on a 6 GB card, using 5 GB) |
| choose the backbone tool | RFD4-Proteina is the default and is picked automatically on Ampere or newer cards; `BACKBONE_TOOL=rfd3` or `rfdiffusion` to change; a GTX 16xx falls back to RFD3. Install RFD4 with `bash install_envs.sh rfd4_env` |
| designs RFD4 makes at once / ProteinMPNN on top of RFD4 | `RFD4_BATCH` (default 8 on a card with 12+ GB, else 2), `RFD4_MPNN=1` |
| use both GPUs of the two-card server | `GPUS=0,1`; worth it for big AF2 batches only, see the README |
| AF2 on an RTX 50xx card | nothing to do: `settings.py` picks `af2_bw` for Blackwell cards (the `af2_bd` environment cannot compile for them); `AF2_ENV` forces one |
| more varied sequences | the `"--sampling_temp", "0.1"` line in `design_sequences` (try 0.2-0.3) |
| ranking of the table | the `rows.sort(...)` lines in `step4_score_af2` / `step4_score_afs` |
| the pass rule itself | the `ok = ...` lines in `step4_score_af2` / `step4_score_afs` |
| a new result column | add it to the dictionary returned by `score_job_af2` / `score_job_afs` |

Output folders inside `output_dir`: `0_glyco/glyco_sites.csv` (target glycosylation scan), `1_rfd3_raw/` (RFD3's own files; `1_rfd4_raw/` with RFD4, plus its `target.cif` and `spec.yaml`), `1_rfdiffusion/` (backbones `design_N.pdb` + `design_N.json` side files), `2_mpnn/sequences.csv`, `3_af2_inputs/` and `3_af2_outputs/` (one `w0`, `w1`... folder per worker), and finally `results_af2.csv` (the AlphaFold Server route uses `3_afserver_outputs/` and `results_afserver.csv`).

## 6. Safe editing habits
1. Change one thing at a time.
2. Run the self-test after code edits.
3. Try new pass thresholds without re-running anything (seconds, no GPU):
   ```bash
   python design_pipeline.py --output_dir <finished run folder> --stages score_af2
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
