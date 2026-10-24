"""Settings for ONE GPU (a laptop card, a cluster GPU, an L40S, an H200 ...), running several worker processes on it.
Backbones come from RFD4-Proteina by default (it designs the sequence too, so ProteinMPNN is skipped). BACKBONE_TOOL=rfd3 uses
RFdiffusion 3 + ProteinMPNN instead, BACKBONE_TOOL=rfdiffusion the older RFdiffusion 1. On a card older than Ampere (compute
capability below 8.0, e.g. the GTX 1660 SUPER) the default falls back to RFD3, because RFD4 cannot run there.

Everything is set with environment variables, so the same file runs any target on any machine:

    GPU=1 TARGET_PDB=/path/target.pdb TARGET_RANGE=A16-108,A113-151 HOTSPOTS=A31,A32 BINDER_LENGTH=55-55 \
    NUM_BACKBONES=4 SEQS_PER_BACKBONE=2 AF2_WORKERS=2 RFD_WORKERS=1 RFD3_BATCH=2 \
    python design_pipeline.py --settings settings_big_gpu.py --output_dir runs/trial1 --stages rfd,mpnn,af2,score_af2

Required:  TARGET_PDB, TARGET_RANGE, HOTSPOTS        (TARGET_RANGE uses commas between pieces: A16-108,A113-151;
                                                      a bare number between pieces, A16-108,4,A113-151, designs 4 new residues)
Optional:  GPU (which card, default 0)   BINDER_LENGTH (55-55)   NUM_BACKBONES (50)   SEQS_PER_BACKBONE (4)
           RFD_WORKERS (2)  AF2_WORKERS (8)  MPNN_WORKERS (1)  RFD3_BATCH (1)  RFD3_OVERRIDES ("")
           ENVS_DIR, TOOLS_DIR, RFD3_BIN, RFD3_CKPT, MPNN_PY, BACKBONE_TOOL (rfd3 | rfd4 | rfdiffusion)
           RFD4_BIN ($ENVS_DIR/rfd4_env/bin/rfd4-generate)  RFD4_BATCH (8)  RFD4_MPNN (1 = also redesign with ProteinMPNN)  RFD4_SEED
           AF2_ENV (conda env that runs AF2: af2_bd by default; use af2_bw on an RTX 50xx / Blackwell card)

Expected layout (what install_envs.sh creates):
    $TOOLS_DIR/ProteinMPNN, $TOOLS_DIR/dl_binder_design           (and $TOOLS_DIR/RFdiffusion only for BACKBONE_TOOL=rfdiffusion)
    conda envs in $ENVS_DIR:  "rfd3_env" (RFD3 + ProteinMPNN) and "af2_bd" (AF2 + PyRosetta)
    RFD3 weights:             $RFD3_CKPT   (default ~/.foundry/checkpoints/rfd3_latest.ckpt)
"""
import os
import subprocess
from pathlib import Path


def need(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} first (see the top of settings_big_gpu.py), e.g.  {name}=...  python design_pipeline.py ...")
    return value


# ---- how many worker processes to run on the GPU ------------------------------------------------------------------
#  - AF2 spends ~16 of every ~20 s per prediction on ONE CPU core, so it can use about as many workers as you have
#    free CPU cores (each worker used 1-1.5 GB of GPU memory on the RTX 5060 Ti).
#  - RFD3 is GPU-heavy: on a 6 GB GTX 1660 SUPER one worker with RFD3_BATCH=2 already used 5 GB.
WORKERS_PER_GPU = {
    "rfdiffusion": int(os.environ.get("RFD_WORKERS", 2)),     # backbone workers (RFD3 or RFdiffusion 1)
    "mpnn":        int(os.environ.get("MPNN_WORKERS", 1)),
    "af2":         int(os.environ.get("AF2_WORKERS", 8)),
}
GPU = os.environ.get("GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")   # which card; a cluster job sets it for you


def pick_default_tool(compute_capability):
    """RFD4-Proteina is the default backbone tool. Its attention needs bf16, which cards older than Ampere (compute capability
    below 8.0, e.g. the GTX 1660 SUPER = 7.5) do not have: RFD4 fails there with 'Invalid backend', so those cards default to RFD3."""
    if compute_capability is not None and compute_capability < 8.0:
        print(f"  note: GPU {GPU} has compute capability {compute_capability} (older than Ampere): using RFD3 instead of RFD4 "
              f"(set BACKBONE_TOOL=rfd4 to force RFD4)")
        return "rfd3"
    return "rfd4"


def gpu_compute_capability():
    """Compute capability of the chosen card, e.g. 7.5; None when nvidia-smi cannot tell (then the default stays RFD4)."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", GPU.split(",")[0], "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None

HOME = Path.home()
ENVS = Path(os.environ.get("ENVS_DIR", HOME / "miniconda3/envs"))
TOOLS = Path(os.environ.get("TOOLS_DIR", HOME / "tools"))
RFD3_BIN = os.environ.get("RFD3_BIN", str(ENVS / "rfd3_env/bin/rfd3"))
RFD3_CKPT = os.environ.get("RFD3_CKPT", str(HOME / ".foundry/checkpoints/rfd3_latest.ckpt"))
MPNN_PY = os.environ.get("MPNN_PY", str(ENVS / "rfd3_env/bin/python"))
RFD4_BIN = os.environ.get("RFD4_BIN", str(ENVS / "rfd4_env/bin/rfd4-generate"))     # made by: bash install_envs.sh rfd4_env
REGLYCO_BIN = os.environ.get("REGLYCO_BIN", str(TOOLS / "reglyco/bin/reglyco"))       # glycosylation scan (optional)
AF2_ENV = os.environ.get("AF2_ENV", "af2_bd")        # "af2_bw" = newer JAX, needed by Blackwell cards (RTX 50xx)
CACHE = {"JAX_COMPILATION_CACHE_DIR": str(HOME / (".cache/jax_af2" if AF2_ENV == "af2_bd" else f".cache/jax_{AF2_ENV}")),
         "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


def workers(program, script, n, extra_env=None):
    """n identical workers on the one card (the pipeline turns off JAX's memory grab when n > 1).
    The env's own lib folder goes first in LD_LIBRARY_PATH, so its newer C++ runtime is used on clusters
    whose system libstdc++ is too old (error: GLIBCXX_3.4.xx not found). script=None: the program needs no script."""
    env_lib = Path(program).parent.parent / "lib"
    lib_path = f"{env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return [{"cmd": [str(program)] + ([str(script)] if script else []),
             "env": {"CUDA_VISIBLE_DEVICES": GPU, "LD_LIBRARY_PATH": lib_path, **(extra_env or {})},
             "weight": 1, "copies": n}]


SETTINGS = {
    "backbone_tool": os.environ.get("BACKBONE_TOOL") or pick_default_tool(gpu_compute_capability()),   # rfd4 (default) | rfd3 | rfdiffusion
    "target_pdb": need("TARGET_PDB"),
    "target_range": need("TARGET_RANGE"),
    "hotspots": need("HOTSPOTS"),
    "binder_length": os.environ.get("BINDER_LENGTH", "55-55"),   # a fixed length ("55-55") lets AF2 compile only once
    "num_backbones": int(os.environ.get("NUM_BACKBONES", 50)),
    "seqs_per_backbone": int(os.environ.get("SEQS_PER_BACKBONE", 4)),
    "output_dir": str(HOME / "run_big_gpu"),    # override on the command line with --output_dir

    # RFdiffusion 3: "design" is the sub-command of the rfd3 program
    "rfd3_workers": workers(RFD3_BIN, "design", WORKERS_PER_GPU["rfdiffusion"]),
    "rfd3_ckpt": RFD3_CKPT,
    "rfd3_batch_size": int(os.environ.get("RFD3_BATCH", 1)),    # designs made at once (2 = ~1.8x faster, needs more GPU memory)
    "rfd3_overrides": os.environ.get("RFD3_OVERRIDES", "").split(),   # e.g. "inference_sampler.step_scale=1.5"

    # RFD4-Proteina (only used with BACKBONE_TOOL=rfd4): designs backbone AND sequence, so ProteinMPNN is skipped
    # unless RFD4_MPNN=1. RFD4_BATCH designs are made at once (8 used 5.4 GB for a 55-residue binder; the first run compiles
    # the model for about 4 minutes, later runs reuse the cache).
    "rfd4_workers": workers(RFD4_BIN, None, WORKERS_PER_GPU["rfdiffusion"]),
    "rfd4_ckpt": os.environ.get("RFD4_CKPT", "nvidia/RFD4-Proteina"),         # Hugging Face name of the weights
    "rfd4_batch_size": int(os.environ.get("RFD4_BATCH", 8)),
    "rfd4_mpnn": os.environ.get("RFD4_MPNN", "") == "1",
    "rfd4_overrides": os.environ.get("RFD4_OVERRIDES", "").split(),           # e.g. "hf_ckpt_revision=<commit>" or "seed=7"
    **({"rfd4_seed": int(os.environ["RFD4_SEED"])} if os.environ.get("RFD4_SEED") else {}),

    # RFdiffusion 1 (only used with BACKBONE_TOOL=rfdiffusion)
    "rfd_noise": 0,
    "rfdiffusion_workers": workers(ENVS / "rfdiffusion/bin/python", TOOLS / "RFdiffusion/scripts/run_inference.py",
                                   WORKERS_PER_GPU["rfdiffusion"]),

    "mpnn_workers": workers(MPNN_PY, TOOLS / "ProteinMPNN/protein_mpnn_run.py", WORKERS_PER_GPU["mpnn"]),
    "af2_workers": workers(ENVS / AF2_ENV / "bin/python", TOOLS / "dl_binder_design/af2_initial_guess/predict.py",
                           WORKERS_PER_GPU["af2"], CACHE),
    "af2_recycles": 3,

    "min_iptm": 0.8, "min_plddt": 80, "max_rmsd": 2.0, "max_pae_interaction": 10,

    # Target glycosylation scan (stage "glyco"): the built-in sequon + exposure scan always runs; ReGlyco is added when its
    # program exists (install_envs.sh reglyco), NetNGlyc when you give its result file (NETNGLYC_OUTPUT) or its program.
    "glyco_warn_A": float(os.environ.get("GLYCO_WARN_A", 15)),              # warn if a sequon is this close (A) to a hotspot
    "reglyco_cache": os.environ.get("GLYCO_CACHE", str(HOME / ".cache/reglyco")),   # keep between runs: one download only
    "reglyco_offline": os.environ.get("REGLYCO_OFFLINE", "") == "1",         # on machines without internet (after one online run)
    **({"reglyco_cmd": [REGLYCO_BIN]} if Path(REGLYCO_BIN).exists() else {}),
    **({"netnglyc_output": os.environ["NETNGLYC_OUTPUT"]} if os.environ.get("NETNGLYC_OUTPUT") else {}),
    **({"netnglyc_cmd": [os.environ["NETNGLYC_BIN"]]} if os.environ.get("NETNGLYC_BIN") else {}),
}
