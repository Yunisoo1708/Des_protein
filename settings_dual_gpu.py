"""Settings that use BOTH cards of the two-GPU server at once: GPU 0 = RTX 5060 Ti (16 GB), GPU 1 = GTX 1660 SUPER (6 GB).
It takes everything from settings_big_gpu.py (same environment variables: TARGET_PDB, TARGET_RANGE, HOTSPOTS, BINDER_LENGTH,
NUM_BACKBONES, BACKBONE_TOOL ...) and only replaces the worker lists, so every tool runs on each card with the right environment.

    TARGET_PDB=... TARGET_RANGE=A16-108,A113-151 HOTSPOTS=A31,A32 BACKBONE_TOOL=rfd4 \
    python design_pipeline.py --settings settings_dual_gpu.py --output_dir ~/runs/dual1 --stages rfd,mpnn,af2,score_af2

Workers per card (set to 0 to keep a card out of a tool):   RFD_WORKERS_0 / RFD_WORKERS_1   MPNN_WORKERS_0 / _1   AF2_WORKERS_0 / _1
Designs made at once per card:                              RFD3_BATCH / RFD3_BATCH_1       RFD4_BATCH / RFD4_BATCH_1

What the two cards need (measured on this machine, 5YXH target, 55-residue binder):
  - AF2: the 5060 Ti (Blackwell) needs the newer JAX in "af2_bw"; the 1660 SUPER uses "af2_bd". Both help: AF2 is CPU-bound.
  - RFD3: 11.4 s per design on the 5060 Ti (batch 8) and about 47 s on the 1660 SUPER (batch 2): a 4 : 1 share.
  - RFD4: 9.9 s per design on the 5060 Ti (batch 8, compiled) but about 125 s on the 1660 SUPER, which cannot run RFD4's default bf16
    attention (error "Invalid backend"), so it runs in fp32 without compiling: a 12 : 1 share. The 1660 SUPER is left out of RFD4
    by default (RFD_WORKERS_1=0 when BACKBONE_TOOL=rfd4); setting it to 1 adds only about 8% speed.
"""
import os
import runpy
from pathlib import Path

BASE = runpy.run_path(str(Path(__file__).with_name("settings_big_gpu.py")))
SETTINGS, HOME, ENVS, TOOLS = BASE["SETTINGS"], BASE["HOME"], BASE["ENVS"], BASE["TOOLS"]
RFD3_BIN, RFD4_BIN, MPNN_PY = BASE["RFD3_BIN"], BASE["RFD4_BIN"], BASE["MPNN_PY"]
tool = SETTINGS["backbone_tool"]


def count(name, gpu, default):
    return int(os.environ.get(f"{name}_WORKERS_{gpu}", default))


def cache(name):     # JAX compile cache; separate per environment because the JAX versions differ
    return {"JAX_COMPILATION_CACHE_DIR": str(HOME / (".cache/jax_af2" if name == "af2_bd" else f".cache/jax_{name}")),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


def make_workers(program, script, gpu, n, weight, extra_args=(), extra_env=None, batch=None):
    """n identical workers on card `gpu`. The env's own lib folder goes first in LD_LIBRARY_PATH (see settings_big_gpu.py)."""
    env_lib = Path(program).parent.parent / "lib"
    lib_path = f"{env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return {"cmd": [str(program)] + ([str(script)] if script else []) + list(extra_args),
            "env": {"CUDA_VISIBLE_DEVICES": gpu, "LD_LIBRARY_PATH": lib_path, **(extra_env or {})},
            "weight": weight, "copies": n, "batch": batch}


# card 0 = RTX 5060 Ti, card 1 = GTX 1660 SUPER
rfd = {t: [] for t in ("rfd3", "rfd4")}
if count("RFD", 0, 1):
    n = count("RFD", 0, 1)
    rfd["rfd3"].append(make_workers(RFD3_BIN, "design", "0", n, 4 / n, batch=int(os.environ.get("RFD3_BATCH", 8))))
    rfd["rfd4"].append(make_workers(RFD4_BIN, None, "0", n, 12 / n, batch=int(os.environ.get("RFD4_BATCH", 8))))
n1 = count("RFD", 1, 0 if tool == "rfd4" else 1)
if n1:
    rfd["rfd3"].append(make_workers(RFD3_BIN, "design", "1", n1, 1 / n1, batch=int(os.environ.get("RFD3_BATCH_1", 2))))
    # fp32 (bf16 attention is not available on this card) and no torch.compile ("quickstart")
    rfd["rfd4"].append(make_workers(RFD4_BIN, None, "1", n1, 1 / n1, ["precision=32-true", "inference_experiment=quickstart"],
                                    batch=int(os.environ.get("RFD4_BATCH_1", 2))))
SETTINGS["rfd3_workers"], SETTINGS["rfd4_workers"] = rfd["rfd3"], rfd["rfd4"]

mpnn = []
for gpu in ("0", "1"):
    n = count("MPNN", gpu, 1)
    if n:
        mpnn.append(make_workers(MPNN_PY, TOOLS / "ProteinMPNN/protein_mpnn_run.py", gpu, n, 1 / n))
SETTINGS["mpnn_workers"] = mpnn

af2_script = TOOLS / "dl_binder_design/af2_initial_guess/predict.py"
af2 = []
for gpu, env_name, speed, default in (("0", "af2_bw", 1.0, 4), ("1", "af2_bd", 0.7, 2)):
    n = count("AF2", gpu, default)
    if n:       # the weight of a card = its speed per worker x its workers (AF2 is CPU-bound, so more workers = more throughput)
        af2.append(make_workers(ENVS / env_name / "bin/python", af2_script, gpu, n, speed, extra_env=cache(env_name)))
SETTINGS["af2_workers"] = af2
