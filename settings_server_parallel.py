"""Server settings that use BOTH GPUs at the same time (each tool gets one worker per card).
weight = relative speed, measured on this machine: the RTX 5060 Ti is ~2.3x faster than the GTX 1660 SUPER for
RFdiffusion (0.87 vs 2.0 min per design) and ~3x for AF2 (4 vs 12 s per prediction), so it gets more of the work.
Compare with settings_server_large.py (same job, 5060 Ti only)."""
import runpy
from pathlib import Path

HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"
NEW, OLD = {"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "1"}      # 5060 Ti, 1660 SUPER

SETTINGS = runpy.run_path(str(Path(__file__).with_name("settings_server.py")))["SETTINGS"]
SETTINGS.update({"binder_length": "55-55", "num_backbones": 50, "seqs_per_backbone": 4,
                 "output_dir": str(HOME / "run_large_par")})

rfd = str(TOOLS / "RFdiffusion/scripts/run_inference.py")
mpnn = str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")
af2 = str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")


def cache(name):  # JAX compile cache; separate per environment because the JAX versions differ
    return {"JAX_COMPILATION_CACHE_DIR": str(HOME / f".cache/jax_{name}"), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


SETTINGS["rfdiffusion_workers"] = [
    {"cmd": [str(ENVS / "rfd_bw/bin/python"), rfd], "env": NEW, "weight": 7},        # new PyTorch + DGL stand-in
    {"cmd": [str(ENVS / "rfdiffusion/bin/python"), rfd], "env": OLD, "weight": 3},   # real DGL (old card)
]
SETTINGS["mpnn_workers"] = [
    {"cmd": [str(ENVS / "mpnn/bin/python"), mpnn], "env": NEW, "weight": 1},
    {"cmd": [str(ENVS / "rfdiffusion/bin/python"), mpnn], "env": OLD, "weight": 1},
]
# AF2 is limited by the CPU, not the GPU: of ~20 s per prediction, only ~4 s run on the GPU; the rest is PyRosetta /
# TensorFlow / file writing on ONE CPU core. So run SEVERAL workers per card and let their CPU parts overlap
# (the machine has 16 cores). JAX normally grabs 75% of GPU memory per process, so turn that off.
# MEASURED (200 predictions): 2 workers 3088 s -> these 6 workers 1237 s (2.5x faster). Peak GPU memory only
# 4.7 GB (5060 Ti) / 3.0 GB (1660 SUPER), mean GPU use 50% / 68%, so a few more workers per card may still help.
SHARE_GPU = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
SETTINGS["af2_workers"] = (
    # RTX 5060 Ti (JAX 0.6): 4 workers; weight 1.0 each (~20 s per prediction per worker)
    [{"cmd": [str(ENVS / "af2_bw/bin/python"), af2], "env": {**NEW, **SHARE_GPU, **cache("af2_bw")}, "weight": 1.0}] * 4
    # GTX 1660 SUPER (JAX 0.4.20): 2 workers; weight 0.7 each (~28 s per prediction per worker)
    + [{"cmd": [str(ENVS / "af2_bd/bin/python"), af2], "env": {**OLD, **SHARE_GPU, **cache("af2_bd")}, "weight": 0.7}] * 2
)
