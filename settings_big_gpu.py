"""Settings for ONE big GPU (for example an L40S or an H200), running several worker processes on it.

Quick timing sweep (no need to edit this file):
    AF2_WORKERS=4  python design_pipeline.py --settings settings_big_gpu.py --stages af2,score_af2 --output_dir runs/af2_x4
    AF2_WORKERS=8  python design_pipeline.py --settings settings_big_gpu.py --stages af2,score_af2 --output_dir runs/af2_x8
    AF2_WORKERS=16 python design_pipeline.py --settings settings_big_gpu.py --stages af2,score_af2 --output_dir runs/af2_x16
(copy the finished 1_rfdiffusion and 2_mpnn folders into each --output_dir first, so only AF2 is re-run).

Expected tools layout (what the install scripts in this project create):
    ~/tools/RFdiffusion, ~/tools/ProteinMPNN, ~/tools/dl_binder_design
    conda envs: "rfdiffusion" (PyTorch 2.4 + real DGL; fine on Ada/Hopper cards) and "af2_bd" (JAX 0.4.20 + PyRosetta)
"""
import os
from pathlib import Path

# ---- EDIT THESE: how many worker processes to run on the GPU ----------------------------------------------------
# These are untested starting guesses for a 48 GB card. Why:
#  - AF2 spends ~16 of every ~20 s per prediction on ONE CPU core, so it can use roughly as many workers as you
#    have free CPU cores (each worker used under 5 GB of GPU memory on the RTX 5060 Ti).
#  - RFdiffusion is GPU-heavy, so a few workers are enough to keep the card busy (about 4 GB each there).
WORKERS_PER_GPU = {
    "rfdiffusion": int(os.environ.get("RFD_WORKERS", 3)),
    "mpnn":        int(os.environ.get("MPNN_WORKERS", 1)),
    "af2":         int(os.environ.get("AF2_WORKERS", 8)),
}
GPU = os.environ.get("GPU", "0")          # which card (as listed by nvidia-smi)
# -----------------------------------------------------------------------------------------------------------------

HOME = Path.home()
ENVS, TOOLS = HOME / "miniconda3/envs", HOME / "tools"
CACHE = {"JAX_COMPILATION_CACHE_DIR": str(HOME / ".cache/jax_af2"), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


def workers(python, script, n, extra_env=None):
    """n identical workers on the one card (the pipeline turns off JAX's memory grab when n > 1)."""
    return [{"cmd": [str(python), str(script)], "env": {"CUDA_VISIBLE_DEVICES": GPU, **(extra_env or {})},
             "weight": 1, "copies": n}]


SETTINGS = {
    "target_pdb": str(TOOLS / "RFdiffusion/examples/input_pdbs/insulin_target.pdb"),   # example target
    "target_range": "A1-150",
    "hotspots": "A59,A83,A91",
    "binder_length": "55-55",      # fixed length: AF2 compiles only once
    "num_backbones": 50,
    "seqs_per_backbone": 4,
    "rfd_noise": 0,
    "output_dir": str(HOME / "run_big_gpu"),

    "rfdiffusion_workers": workers(ENVS / "rfdiffusion/bin/python", TOOLS / "RFdiffusion/scripts/run_inference.py",
                                   WORKERS_PER_GPU["rfdiffusion"]),
    "mpnn_workers": workers(ENVS / "rfdiffusion/bin/python", TOOLS / "ProteinMPNN/protein_mpnn_run.py",
                            WORKERS_PER_GPU["mpnn"]),
    "af2_workers": workers(ENVS / "af2_bd/bin/python", TOOLS / "dl_binder_design/af2_initial_guess/predict.py",
                           WORKERS_PER_GPU["af2"], CACHE),
    "af2_recycles": 3,

    "min_iptm": 0.8, "min_plddt": 80, "max_rmsd": 2.0, "max_pae_interaction": 10,
}
