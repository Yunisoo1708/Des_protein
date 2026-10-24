# Binder design: RFD4 or RFD3 + ProteinMPNN → AlphaFold2 validation

Designs protein binders against a target structure, then checks each design by predicting the binder + target complex
with AlphaFold2 ("initial guess", Bennett lab).

Which route runs depends on the GPU, and `settings.py` picks it automatically:

| GPU | Route |
|---|---|
| **Modern: Ampere or newer** (compute capability 8.0+, e.g. RTX 30/40/50, A100, H100) | **RFD4-Proteina** (designs backbone and sequence together) → **AF2** → score |
| **Older than Ampere** (e.g. GTX 16xx, RTX 20xx) | **RFdiffusion 3** → **ProteinMPNN** (sequences) → **AF2** → score |

```
modern:  target.pdb ─► [1 RFD4-Proteina] ─► binders (backbone + sequence) ──────────────► [3 AF2] ─► [4 Score] ─► results_af2.csv
older:   target.pdb ─► [1 RFdiffusion 3] ─► backbones ─► [2 ProteinMPNN] ─► sequences ─► [3 AF2] ─► [4 Score] ─► results_af2.csv
```
Force a route with `BACKBONE_TOOL=rfd4` or `BACKBONE_TOOL=rfd3`.

## Files
| File | Role |
|---|---|
| `design_pipeline.py` | The workflow. `main()` runs the stages in order. |
| `settings.py` | Your job (target, residues, hotspots, ...) and the GPU logic: chooses the tool, the AF2 environment and the worker counts per card. |
| `install_envs.sh` | Installs the tools and conda environments once. |
| `slurm_job.sh`, `slurm_job_tillicum.sh` | Run the pipeline on a Slurm cluster. |

## Install (once, Linux or WSL)
Everything goes under `$BASE`: `$BASE/tools`, `$BASE/envs`, `$BASE/pip_cache`. Pick a folder with plenty of space
(about 30 GB in total). Run the steps one at a time, in this order, with the same `BASE`:
```bash
export BASE=$HOME/binder
bash install_envs.sh tools        # clone ProteinMPNN + dl_binder_design
bash install_envs.sh weights      # AF2 weights (~5.5 GB)
bash install_envs.sh rfd3_env     # env "rfd3_env": RFdiffusion 3 + ProteinMPNN + RFD3 weights (~10 GB)
bash install_envs.sh af2_env      # env "af2_bd": AF2 + PyRosetta (~8 GB)
bash install_envs.sh rfd4_env     # RFD4-Proteina (needs rfd3_env first; ~2.5 GB)
bash install_envs.sh check        # import tests (no GPU needed)
```
Optional: `reglyco` (glycosylation scan). The pipeline script itself only needs `pip install numpy biopython`.

| Environment | Used for | Notes |
|---|---|---|
| `rfd3_env` | RFdiffusion 3, ProteinMPNN | needs `atomworks<3` (the script handles it) |
| `rfd4_env` | RFD4-Proteina | Python 3.12, CUDA 13-compatible driver (`nvidia-smi` shows "CUDA Version: 13.x"); a small overlay on `rfd3_env`, so keep `rfd3_env`; the model weights (1.7 GB) come from Hugging Face |
| `af2_bd` | AF2 initial guess | PyRosetta is free for academic, non-commercial use only |

The pipeline looks for the tools in `$ENVS_DIR` (default `~/miniconda3/envs`) and `$TOOLS_DIR` (default `~/tools`).
If you installed with `BASE=$HOME/binder`, set `ENVS_DIR=$HOME/binder/envs TOOLS_DIR=$HOME/binder/tools` when you run.

To make the environments show up in `conda env list` and `conda activate`, add the folder to conda's search path:
`conda config --append envs_dirs $BASE/envs` (and `mkdir -p $BASE/envs/rfd4_env/conda-meta` for the RFD4 venv).

## Run
Describe the target with environment variables in front of the command (they win over the "YOUR JOB" block at the top of
`settings.py`, which you can edit instead):
```bash
ENVS_DIR=$HOME/binder/envs TOOLS_DIR=$HOME/binder/tools \
GPUS=0 \
TARGET_PDB=/full/path/to/target.pdb \
TARGET_RANGE=A16-108,A113-151 \
HOTSPOTS=A31,A32 \
NUM_BACKBONES=8 \
python design_pipeline.py --output_dir runs/trial1
```
Notes on the command:
- Each line ends with a single `\` and nothing after it (a space before it, none after).
- `TARGET_PDB` is a full path or a path relative to where you run the command, not starting with `/` unless it is a full path.
- **Use a new `--output_dir` for every run.** The backbone tools skip designs that already exist, so the pipeline refuses to
  write backbones into a folder that already has designs.
- `TARGET_RANGE`: the target residues to keep, as comma-separated pieces (`A16-108,A113-151`). Leave out residues that are
  missing from your PDB file (for example a numbering gap). Hotspots are residues such as `A31,A32` and should lie inside the range.
- Other settings: `BINDER_LENGTH` (default `55-55`; a fixed length makes AF2 compile only once), `NUM_BACKBONES`
  (default 8), `SEQS_PER_BACKBONE` (default 4; ProteinMPNN route only).
- Start small (`NUM_BACKBONES=2` to check that it runs); a handful of designs is too few to judge a target.
- Re-run one stage with `--stages` (`glyco,rfd,mpnn,af2,score_af2`). Re-scoring (`--stages score_af2`) needs no GPU.

## Results
`<output_dir>/results_af2.csv`, best design first:

| column | meaning | good |
|---|---|---|
| `pae_interaction` | AF2's expected error in the binder's position relative to the target (lower is better) | < 10 |
| `plddt_binder` | AF2's confidence in the binder's fold (0-100) | > 80 |
| `rmsd_binder` | Å between the AF2 binder and the designed backbone, complexes aligned on the target | < 2 |

`rmsd_binder` asks whether AF2 agrees with what was designed. AF2 here has **no iPTM**. Passing means the design is
self-consistent in silico, not that it binds in the lab.

Folders created in the output directory:
`0_glyco/` target glycosylation scan · `1_rfd4_raw/` or `1_rfd3_raw/` the design tool's own files (one `w0`, `w1`, ... folder per
worker; `1_rfd4_raw/` also holds `target.cif` and `spec.yaml`) · `1_rfdiffusion/` converted backbones `design_N.pdb` + `design_N.json` ·
`2_mpnn/sequences.csv` sequences (with RFD4 these are RFD4's own, named `design_N_native`) · `3_af2_inputs/` · `3_af2_outputs/` · `results_af2.csv`

## RFD4-Proteina route (modern GPUs)
- Designs the binder's sequence and structure together, so ProteinMPNN is skipped. Set `RFD4_MPNN=1` to redesign the sequences with ProteinMPNN anyway.
- RFD4 reads only CIF and does not bond across gaps, so the pipeline writes `1_rfd4_raw/target.cif` with every piece of
  `TARGET_RANGE` as its own chain, writes `spec.yaml`, and converts each result to `design_N.pdb` (target = chain A with the
  original numbering, binder = chain B).
- `TARGET_RANGE` must be kept pieces only (no bare numbers that design new residues in a gap), and every residue in it must exist in the PDB.
- The first run compiles the model (a few minutes); later runs reuse the cache.
- Variables: `RFD4_BATCH` (designs at once; choose `NUM_BACKBONES` as a multiple of it), `RFD4_SEED`, `RFD4_CKPT`,
  `RFD4_OVERRIDES` (for example `hf_ckpt_revision=<commit>` to pin the model), `RFD4_MPNN`.

## RFdiffusion 3 + ProteinMPNN route (older GPUs)
- RFD3 makes backbones; ProteinMPNN writes `SEQS_PER_BACKBONE` sequences for each; AF2 then scores them.
- RFD3 renumbers the target from 1; the pipeline restores the original residue numbers in `design_N.pdb`.
- A bare number between pieces designs that many new residues in the gap: `A16-108,4,A113-151`.
- Hotspots use each residue's tip atom by default; set `"rfd3_hotspot_atoms": "ALL"` to use every atom.
- Tuning: `rfd3_overrides`, for example `["inference_sampler.step_scale=1.5"]` (higher = less diverse, more designable).
- Do not use `low_memory_mode=True` on a small card; it was far slower in testing.
- Variable: `RFD3_BATCH` (designs at once).
- RFdiffusion 1 is still available with `BACKBONE_TOOL=rfdiffusion` (needs its own `rfdiffusion_cmd`/`rfdiffusion_workers`).

## AF2 validation (both routes)
- Runs dl_binder_design's `predict.py` with the model_1_ptm weights; one prediction takes roughly a minute on a laptop GPU, mostly compile and CPU work.
- AF2 recompiles for every new input length, so use a fixed `BINDER_LENGTH`; the JAX compile cache is kept between runs.
- Environment `af2_bd` uses JAX 0.4.23 with cuDNN 8.9 and the CUDA 12.2 compiler (`ptxas`). `settings.py` passes the
  `XLA_FLAGS` and `PATH` that point JAX at it. JAX 0.4.20 failed under WSL with "Failed to load PTX text as a module",
  and 0.4.24 or newer cannot be used because dm-haiku 0.0.10 needs `jax.linear_util`.
- RTX 50xx (Blackwell) cards need a newer-JAX environment named `af2_bw`, which `install_envs.sh` does not create; select it with `AF2_ENV=af2_bw`.

## GPUs and workers
- `GPUS=0` picks the card by its `nvidia-smi` number; `GPUS=0,1` uses both. Without it the pipeline uses `CUDA_VISIBLE_DEVICES`
  (Slurm sets this), else card 0.
- `settings.py` asks `nvidia-smi` about each card to choose the route and the AF2 environment.
- A *worker* is one command on one GPU. Work is shared between workers by their relative speed, and each tool can have several workers per card.
  Set the number per run: `AF2_WORKERS=4`, `RFD_WORKERS=1`, `MPNN_WORKERS=1` (all cards), or per card, for example `AF2_WORKERS_1=2`.
- AF2 is CPU-bound, so more AF2 workers per card gives more throughput, until GPU memory or CPU cores run out. The default is 8 per card
  on a single GPU. On a card with 8 GB or less, try `AF2_WORKERS=2` first.

## Target glycosylation scan (stage `glyco`, runs first, no GPU)
Reports every N-glycosylation motif (Asn-X-Ser/Thr, X not Pro) in the target chain: exposure of the Asn, distance to the nearest hotspot,
and whether it lies inside the kept `TARGET_RANGE`. Ignore it if your target is intracellular. Output: `0_glyco/glyco_sites.csv`.
Optional extra predictors: ReGlyco (`bash install_envs.sh reglyco`) and NetNGlyc (a manual web step: submit `0_glyco/target.fasta`, then re-run with
`--stages glyco --netnglyc_output <file>`). These are predictions only; the design is not changed by this stage.

## Optional: iPTM from AlphaFold Server (website)
1. `--stages rfd,mpnn,afs_export` writes `afserver_jobs.json`.
2. Upload it at alphafoldserver.com (needs a Google login; there is a daily job limit, so send only your best designs) and download the result `.zip` files into one folder.
3. `--stages afs_import,score_afs --afserver_zips <that folder>` writes `results_afserver.csv` with `iptm`, `plddt_binder`, `rmsd_binder`.

## Troubleshooting
- **"The target is not set: ..."**: a variable was not passed. Check for a `\\` (use a single `\`), a missing space before a `\`, or a typo in the variable name.
- **`FileNotFoundError` for `rfd4-generate`, `rfd3` or `predict.py`**: set `ENVS_DIR` and `TOOLS_DIR` to where you installed (`$BASE/envs`, `$BASE/tools`).
- **AF2 prints "No GPU detected running AF2 on CPU"**: check that `af2_bd` has JAX 0.4.23 + `nvidia-cudnn-cu12==8.9.7.29` + `nvidia-cuda-nvcc-cu12==12.2.140`
  (cuDNN 9 makes JAX report "cuDNN version 0").
- **Disk full while installing RFD4**: the Hugging Face repo is about 84 GB, but only `config.json` and `model.safetensors` (1.7 GB) are needed; `install_envs.sh` downloads just those.
