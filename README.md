# Binder design: RFD4-Proteina → AlphaFold2 (Bennett "initial guess")

```
target.pdb ─► [1 RFD4-Proteina] ─► binders (backbone + sequence) ─► [3 AF2 initial guess] ─► [4 Score] ─► results_af2.csv
              designs both together                                  predict binder+target    pAE_interaction, pLDDT, RMSD
```
**RFD4-Proteina is the default.** On a card older than Ampere (e.g. GTX 1660 SUPER) `settings.py` falls back to the older route
`RFdiffusion 3 → ProteinMPNN → AF2`, or choose it with `BACKBONE_TOOL=rfd3` (see "RFdiffusion 3 notes" below):
```
target.pdb ─► [1 RFdiffusion 3] ─► backbones ─► [2 ProteinMPNN] ─► sequences ─► [3 AF2 initial guess] ─► [4 Score] ─► results_af2.csv
```

The whole program is `design_pipeline.py` (the workflow) and `settings.py` (your job + the GPU logic); `install_envs.sh` installs the tools once, and `slurm_job.sh` / `slurm_job_tillicum.sh` run it on a cluster. Open `main()` in `design_pipeline.py` first: it runs the stages in order.

## Use it
1. Install the tools (once) with `install_envs.sh`, one step at a time: `tools`, `weights`, `rfd3_env`, `af2_env`, `rfd4_env`, `check`
   (RFdiffusion 3, ProteinMPNN and [dl_binder_design](https://github.com/nrbennet/dl_binder_design); needs PyRosetta and the public AF2 `params_model_1_ptm` weights). The pipeline script itself needs only `pip install numpy biopython`. Optional extra steps: `rfd4_env` (RFD4-Proteina, see below), `reglyco`.
2. Tell it your target: edit the **"YOUR JOB" block at the top of `settings.py`** (target file, residues, hotspots, binder length, number of designs, GPUs), **or** put the same values in front of the command as environment variables (they win over the block), e.g. `GPUS=0 TARGET_PDB=... TARGET_RANGE=... HOTSPOTS=... python design_pipeline.py`. `settings.py` also works out the tool, the AF2 environment and the worker counts for each card by itself.
3. Run inside WSL/Linux: `python design_pipeline.py`
4. Open `runs/run1/results_af2.csv` (best design first). The run starts with a quick scan of the target for N-glycosylation sites (`0_glyco/glyco_sites.csv`, see below).

## Output columns
| column | meaning | good |
|---|---|---|
| `pae_interaction` | AF2's expected error in the binder's position relative to the target (lower is better) | < 10 |
| `plddt_binder` | AF2 confidence in the binder's fold (0–100, mean over the binder) | > 80 |
| `rmsd_binder` | Å between the AF2 binder and the backbone ProteinMPNN was given (the RFD3 design), both complexes lined up on the target | < 2 |

`rmsd_binder` is the "does AF2 agree with what I designed?" check. ProteinMPNN only writes sequences, so its "structure" is the RFD3 backbone it was given. The AF2 model has **no iPTM**; for that, use the optional AlphaFold Server route below.

## Folders it creates
`0_glyco/` target glycosylation scan · `1_rfd3_raw/` RFD3's own files (one `w0`, `w1`… folder per worker; `1_rfd4_raw/` with RFD4, which also holds `target.cif` and `spec.yaml`) · `1_rfdiffusion/` the converted backbones `design_N.pdb` + `design_N.json` (which chain is the binder, RFD3's quality metrics) · `2_mpnn/` sequences (`sequences.csv`) · `3_af2_inputs/` · `3_af2_outputs/` · `results_af2.csv`

## Target glycosylation scan (stage `glyco`, runs first, no GPU, seconds)
N-glycans attach to Asn in an **Asn-X-Ser/Thr** motif (X is not Pro), and only on proteins that enter the secretory pathway (secreted and membrane proteins). If your target is intracellular, ignore this stage. Otherwise a glycan can sit on or beside the surface your binder should touch, so the stage reports, for every motif in the target chain:
- **the built-in scan** (always on): where the motifs are, how exposed the Asn is (Shrake-Rupley), the distance to the nearest hotspot, and whether the Asn is inside the kept `target_range`;
- **[ReGlyco](https://github.com/Ojas-Singh/Re-Glyco) `scan`** (automatic once installed with `bash install_envs.sh reglyco`; `settings.py` finds `$TOOLS_DIR/reglyco/bin/reglyco` by itself, or set `REGLYCO_BIN`): attaches a GlcNAc to each motif and checks it fits without clashing, a structure-based occupancy prediction. Its own wording: "structural accessibility only; not proof of biological glycosylation". It downloads the GlcNAc data from glycoshape.org on its first run and caches it (`reglyco_cache`); on a machine without internet run it once online first, then set `reglyco_offline`. Licence: AGPL-3.0.
- **NetNGlyc 1.0** (sequence-based, trained on human proteins, from 2002; its website needs a DTU sign-in, so this one is a manual step): run `--stages glyco`, submit `0_glyco/target.fasta` to NetNGlyc (or run its standalone program), save its text output, then re-run `--stages glyco --netnglyc_output <file>` (or set `NETNGLYC_OUTPUT`; `netnglyc_cmd` runs a standalone program for you). The file reader was written from NetNGlyc's public description and is **not verified against a real output file**: it says so if it recognises no result line.

Output: `0_glyco/glyco_sites.csv` with one row per motif: `exposed`, `reglyco_accessible`, `netnglyc_potential` (above 0.5 counts as glycosylated), `support` (how many of the predictors that ran agree), `verdict` (`likely` / `possible` / `unlikely`, or `candidate` when only the exposure check ran) and a `warning` (a glycan within `glyco_warn_A` = 15 Å of a hotspot, or the Asn lies outside the kept range, so the design target lacks that attachment point). Predictions only: a design is not changed by this stage, you decide what to do (for example move a hotspot, or extend `target_range`).

## RFdiffusion 3 notes
- **Run it with a NEW output folder every time.** RFD3 (like RFdiffusion 1) skips designs that already exist, and later steps would mix old and new backbones, so the pipeline refuses to run the backbone step into a folder that already has designs.
- **`target_range` uses commas between pieces:** `A16-108,A113-151`. Leave out residues that are missing from your PDB file (RFdiffusion 1 stops with an error on them; I did not test what RFD3 does if you include them). A bare number between pieces designs that many NEW residues in the gap: `A16-108,4,A113-151`.
- **Hotspots** are residues (`A31,A32`); the pipeline gives RFD3 each residue's tip atom (`TIP`). Set `"rfd3_hotspot_atoms": "ALL"` to use every atom.
- RFD3 renumbers the target from 1; the pipeline puts the **original residue numbers back** in `design_N.pdb`.
- Randomness knobs: `rfd3_overrides`, e.g. `["inference_sampler.step_scale=1.5"]` (higher = less diverse, more designable) or `inference_sampler.gamma_0=0` (deterministic sampling).
- **Measured on a 6 GB GTX 1660 SUPER** (5YXH target, 132 target + 55 binder residues): 94 s for a batch of 2 designs (about 47 s per design, 5.0 GB of GPU memory); the whole pipeline for 4 backbones x 2 sequences took 7 min 38 s. RFdiffusion 1 took about 2 minutes per design on the same card (a different, similar-sized target).
- **Do not use `low_memory_mode=True`** on a small card: in a test it was far slower (the run had not finished after 15 minutes, while the same design took 8.6 s with 20 steps and the mode off).
- Needs `atomworks<3`: `rc-foundry 0.2.0` calls a function that atomworks 3.x removed (`install_envs.sh rfd3_env` handles this).
- RFdiffusion 1 is still available: set `"backbone_tool": "rfdiffusion"` (or `BACKBONE_TOOL=rfdiffusion`) and give it `rfdiffusion_cmd`/`rfdiffusion_workers`.

## RFD4-Proteina (the default backbone tool, `backbone_tool: "rfd4"`)
[RFD4-Proteina](https://huggingface.co/nvidia/RFD4-Proteina) (IPD + NVIDIA, announced 2026-10-21) designs the binder's **sequence and all-atom structure together**, so the ProteinMPNN stage is skipped: `2_mpnn/sequences.csv` is filled with RFD4's own sequences (jobs are named `design_N_native`). Set `"rfd4_mpnn": True` (`RFD4_MPNN=1`) to redesign them with ProteinMPNN anyway.
- **Install:** `bash install_envs.sh rfd4_env` (after `rfd3_env`). It needs a **CUDA 13 driver** (`nvidia-smi` shows "CUDA Version: 13.x") and Python 3.12. It is a small overlay on `rfd3_env` (re-uses its PyTorch, about 0.7 GB extra) because RFD4 needs atomworks 3.0.4+ while RFD3 needs atomworks below 3. The weights (1.7 GB) come from Hugging Face; download them on a machine with internet.
- **Run:** `python design_pipeline.py ...` (RFD4 is picked automatically on Ampere or newer cards; `BACKBONE_TOOL=rfd3` selects the older route; variables: `RFD4_BATCH`, `RFD4_SEED`, `RFD4_CKPT`, `RFD4_OVERRIDES`, `RFD4_MPNN`; pin the exact model with `RFD4_OVERRIDES="hf_ckpt_revision=<commit>"`).
- **What the pipeline does for you:** RFD4 only reads CIF and will not bond across a gap in the target, so the pipeline writes `1_rfd4_raw/target.cif` with **every piece of `target_range` as its own chain** (`A16-108,A113-151` becomes chains A and B, original residue numbers kept), writes `spec.yaml` (contig, target backbone + sequence, hotspots), then converts each result to the usual `design_N.pdb` (target = chain A with the original numbers, binder = chain B). Each worker gets its own `seed` (RFD4's default seed is fixed, so two workers would repeat each other).
- **`target_range` must be kept pieces only** (`A16-108,A113-151`); a bare number that designs new residues in a gap (`A16-108,4,A113-151`) is not supported with RFD4, and every residue in the range must exist in the PDB file.
- The first run compiles the model (about 4 minutes for batch 8); later runs reuse the cache. `cuEquivariance` kernels are not installed, which RFD4 only warns about.
- **Measured on 5YXH** (RTX 5060 Ti, 55-residue binder, 24 designs, one run each, so treat small differences as noise): generation took 237 s with RFD4 (batch 8, 5.4 GB) and 273 s with RFD3 (batch 8, 6.2 GB), i.e. about the same speed (NVIDIA's blog claims a much bigger gap). AF2 pass rate (pAE ≤ 10, pLDDT ≥ 80, RMSD ≤ 2): **RFD4 3/24 and 2/24 in two runs with its own sequences, 9/96 with ProteinMPNN sequences; RFD3 + ProteinMPNN 0/96** (also 0/96 in an earlier run).- **Second target** (insulin receptor L1 domain from RFdiffusion's examples, `A1-150`, hotspots `A59,A83,A91`, 24 designs per tool, one run each): **RFD4 12/24 pass with its own sequences (50%) and 46/96 with ProteinMPNN sequences; RFD3 7/96 (from only 2 backbones); RFdiffusion 1 27/200 in an earlier run.** So RFD3 does work here, and RFD4 still passes about 4-7x more often. Two targets and single runs: promising, not proof.
- AF2 on an **RTX 50xx (Blackwell)** card needs the newer-JAX environment: `AF2_ENV=af2_bw` (the default `af2_bd` crashes there with a `ptxas` error).

## Optional: iPTM from AlphaFold Server (website)
The only route that gives iPTM, and it needs nothing installed:
1. `--stages rfd,mpnn,afs_export` writes `afserver_jobs.json`.
2. Upload it at alphafoldserver.com (Upload JSON; needs a Google login; the site has a daily job limit, so send only your best designs) and download the result `.zip` files into one folder.
3. `--stages afs_import,score_afs --afserver_zips <that folder>` gives `results_afserver.csv` with `iptm`, `plddt_binder`, `rmsd_binder`.

## Choosing the GPU(s)
- `GPUS=0` (or the "GPUS" line in the YOUR JOB block) picks the card by its `nvidia-smi` number; `GPUS=0,1` uses both. Without it, `CUDA_VISIBLE_DEVICES` is used (a Slurm job sets that for you), else card 0. To use only ONE card (for example the other one is busy training): `GPUS=1 python design_pipeline.py ...`.
- `settings.py` asks `nvidia-smi` about each card: Ampere or newer (compute capability 8.0+) runs RFD4; an older card such as the GTX 1660 SUPER uses RFD3 + ProteinMPNN; RTX 50xx (Blackwell) cards get the newer-JAX AF2 environment `af2_bw`, all others `af2_bd`.
- AF2 recompiles for every new input length (slow, CPU-bound, GPU idle). Use a fixed `BINDER_LENGTH` such as `55-55`; the JAX compile cache is kept between runs.
- RFdiffusion 1 on a 5060 Ti needs a small PyTorch-only stand-in for DGL (`~/tools/dgl_shim` on the server), because DGL has no kernels for that card. RFD3 and RFD4 do not need it.

## Using several GPUs at once
Give a tool a list of *workers* (a command + the GPU it runs on + a `weight` = relative speed) and the work is
shared between them, running at the same time; `settings.py` builds these lists for you from `GPUS`. Measured on the server
(50 backbones x 4 sequences, RFdiffusion 1): RFdiffusion 31 min on both cards, ProteinMPNN 2.5 min, AF2 51 min.
AF2 is limited by the CPU (about 16 s of every 20 s per prediction is PyRosetta / TensorFlow / file writing on one
core), not the GPU, so the way to speed it up is *more workers per card*, not a faster card. Measured: 200 AF2
predictions took 51 min with 1 worker per card and **20.6 min with 4 workers on the 5060 Ti + 2 on the 1660 SUPER**
(measured on the dev server). AF2 is not exactly repeatable: the two runs gave the same scores to ~0.1
(median), but 5 of 200 designs flipped pass/fail, mostly designs sitting right on the RMSD limit.
`--output_dir` re-runs a stage into a different folder without touching earlier results.

**Both cards of the two-GPU server with the current tools: `GPUS=0,1`** (plus per-card counts such as `RFD_WORKERS_1`, `MPNN_WORKERS_1`, `AF2_WORKERS_1`). `settings.py` picks the right AF2 environment for each card (`af2_bw` for the 5060 Ti, `af2_bd` for the 1660 SUPER), shares work by measured speed and gives each card its own batch size. Measured on 5YXH, same inputs, one run each:
- **AF2, 96 predictions:** 584 s with 6 workers on the 5060 Ti alone, **510 s with 6 on the 5060 Ti + 2 on the 1660 SUPER** (1.15x). Identical pass set (9 of 96) and pAE within 0.06 between the cards. The gain is small because AF2 is CPU-bound (16 cores) and the 1660's workers are slower (about 0.7x per worker, and they start about 75 s later). With only 24 predictions the dual setup was *slower* (the 1660 workers' start-up dominates), so use it for big batches.
- **RFD3:** about 4x slower per design on the 1660 SUPER (47 s vs 11.4 s), so it gets a 4 : 1 share.
- **RFD4: leave the 1660 SUPER out** (the default). It cannot run RFD4's default bf16 attention (error `Invalid backend`), so it needs `precision=32-true` and no compile, and then needs about 125 s per design against 9.9 s on the 5060 Ti. Adding it (`RFD_WORKERS_1=1`, the per-card variable) made the 24-design run no faster: 22 designs on the 5060 Ti are split 8+8+6, and the odd batch of 6 forces a recompile, so choose `NUM_BACKBONES` as a multiple of `RFD4_BATCH`.
- **Whole pipeline, 24 RFD4 designs:** 432 s on the 5060 Ti alone vs 539 s dual (cold AF2 cache on the 1660 and the odd batch).

**How many workers per GPU?** Set them per run, e.g. `AF2_WORKERS=16 python design_pipeline.py ...` (every card) or `AF2_WORKERS_1=2` (card 1 only); the same for `RFD_WORKERS` and `MPNN_WORKERS`. (Under the hood a worker with `"copies": N` runs N identical processes on one card and JAX's memory grab is switched off automatically.)
The defaults (8 AF2 workers on one card; 6 + 2 on two cards) were measured on the dev server only: on a new machine time 4, 8 and 16 AF2 workers and keep the fastest.

## Tips
- Re-run one step only: `--stages score_af2` (options: `rfd`, `mpnn`, `af2`, `score_af2`). Re-scoring needs no GPU, so you can try new pass limits in seconds.
- Start small (`num_backbones: 4`) and look at the table before scaling up. Four designs are far too few to judge a target: in a 4-design RFD3 trial on 5YXH none passed, which says little either way.
- Passing means the design is self-consistent in silico, not that it binds in the lab.

(The self-test suite with fake tools used to live in `tests/`; it was removed from the repo to keep it minimal, but it is still in the git history, e.g. `git checkout 221fd85 -- tests`.)
