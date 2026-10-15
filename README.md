# Binder design: RFdiffusion → ProteinMPNN → AlphaFold2 (Bennett "initial guess")

```
target.pdb ─► [1 RFdiffusion] ─► backbones ─► [2 ProteinMPNN] ─► sequences ─► [3 AF2 initial guess] ─► [4 Score] ─► results_af2.csv
              draw skeletons                  write sequences                 predict binder+target    pAE_interaction, pLDDT, RMSD
```

New here? Read [GUIDE.md](GUIDE.md): the run command, a map of the code, and a "how do I change X" cheat sheet.

## Use it
1. Install the tools (once): RFdiffusion, ProteinMPNN and [dl_binder_design](https://github.com/nrbennet/dl_binder_design) (needs PyRosetta and the public AF2 `params_model_1_ptm` weights). The pipeline script itself needs only `pip install numpy biopython`.
2. Edit `settings.py` (target file, hotspots, tool paths).
3. Run inside WSL/Linux: `python design_pipeline.py`
4. Open `runs/run1/results_af2.csv` (best design first).

## Output columns
| column | meaning | good |
|---|---|---|
| `pae_interaction` | AF2's expected error in the binder's position relative to the target (lower is better) | < 10 |
| `plddt_binder` | AF2 confidence in the binder's fold (0–100, mean over the binder) | > 80 |
| `rmsd_binder` | Å between the AF2 binder and the RFdiffusion/ProteinMPNN backbone, both complexes lined up on the target | < 2 |

`rmsd_binder` is the "does AF2 agree with what I designed?" check. ProteinMPNN only writes sequences, so its "structure" is the RFdiffusion backbone it was given. The AF2 model has **no iPTM**; for that, use the optional AlphaFold Server route below.

## Folders it creates
`1_rfdiffusion/` backbones · `2_mpnn/` sequences (`sequences.csv`) · `3_af2_inputs/` · `3_af2_outputs/` (one `w0`, `w1`… folder per worker) · `results_af2.csv`

## Optional: iPTM from AlphaFold Server (website)
The only route that gives iPTM, and it needs nothing installed:
1. `--stages rfd,mpnn,afs_export` writes `afserver_jobs.json`.
2. Upload it at alphafoldserver.com (Upload JSON; needs a Google login; the site has a daily job limit, so send only your best designs) and download the result `.zip` files into one folder.
3. `--stages afs_import,score_afs --afserver_zips <that folder>` gives `results_afserver.csv` with `iptm`, `plddt_binder`, `rmsd_binder`.

## GPU server notes (`settings_server.py`, `settings_server_large.py`)
- RTX 5060 Ti (new) runs RFdiffusion and AF2; GTX 1660 SUPER (old) can run ProteinMPNN. Pick a card per tool with `*_env: {"CUDA_VISIBLE_DEVICES": "0"}`.
- RFdiffusion on the 5060 Ti uses a small PyTorch-only stand-in for DGL (`~/tools/dgl_shim` on the server), because DGL has no kernels for that card.
- AF2 recompiles for every new input length (slow, CPU-bound, GPU idle). Use a fixed `binder_length` such as `"55-55"` and keep the JAX compile cache setting.

## Using several GPUs at once
Give a tool a list of *workers* (a command + the GPU it runs on + a `weight` = relative speed) and the work is
shared between them, running at the same time. See `settings_server_parallel.py`. Measured on the server
(50 backbones x 4 sequences): RFdiffusion 31 min on both cards, ProteinMPNN 2.5 min, AF2 51 min.
AF2 is limited by the CPU (about 16 s of every 20 s per prediction is PyRosetta / TensorFlow / file writing on one
core), not the GPU, so the way to speed it up is *more workers per card*, not a faster card. Measured: 200 AF2
predictions took 51 min with 1 worker per card and **20.6 min with 4 workers on the 5060 Ti + 2 on the 1660 SUPER**
(`real_run/timing_af2_6workers.txt`). AF2 is not exactly repeatable: the two runs gave the same scores to ~0.1
(median), but 5 of 200 designs flipped pass/fail, mostly designs sitting right on the RMSD limit.
`--output_dir` re-runs a stage into a different folder without touching earlier results.

**How many workers per GPU?** Add `"copies": N` to a worker to run N identical workers on one card (JAX's
memory grab is switched off automatically). The easy way is the `WORKERS_PER_GPU` block at the top of
`settings_server_parallel.py` (two cards) or `settings_big_gpu.py` (one big card, e.g. L40S / H200; numbers can
also be set per run, e.g. `AF2_WORKERS=16 python design_pipeline.py --settings settings_big_gpu.py ...`).
The defaults for the big card are untested starting guesses: time 4, 8 and 16 AF2 workers and keep the fastest.

## Tips
- Re-run one step only: `--stages score_af2` (options: `rfd`, `mpnn`, `af2`, `score_af2`). Re-scoring needs no GPU, so you can try new pass limits in seconds.
- Use a new `output_dir` for every run; old files in a reused folder get mixed into the new results.
- Start small (`num_backbones: 10`) and look at the table before scaling up.

## Test it (no GPU, no tools needed)
`python tests/test_workflow.py` runs the whole pipeline with fake tools (`tests/fake_tools.py`) that copy the real tools' command lines and output formats. It checks the commands, file parsing, the multi-worker splitting and the pAE/iPTM/pLDDT/RMSD maths. It does **not** judge design quality.
