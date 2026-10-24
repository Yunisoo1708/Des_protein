"""
THE settings file. Together with design_pipeline.py it is all you need to run (install_envs.sh installs the tools once).

HOW TO USE IT - pick one:
  A. Edit the "YOUR JOB" block just below (target, residues, hotspots, ...), then run:      python design_pipeline.py
  B. Leave the file alone and set environment variables in front of the command (they win over the block below):

       GPUS=0 TARGET_PDB=/path/target.pdb TARGET_RANGE=A16-108,A113-151 HOTSPOTS=A31,A32 \\
       NUM_BACKBONES=8 python design_pipeline.py --output_dir runs/trial1

  The same works for a Slurm job (see slurm_job.sh): the job script just exports the variables.

WHAT IT DECIDES FOR YOU (by asking nvidia-smi about each card in GPUS):
  - Tool:  RFD4-Proteina (designs backbone AND sequence, ProteinMPNN is skipped) on cards of Ampere or newer
           (compute capability 8.0+). Cards older than that, e.g. the GTX 1660 SUPER, cannot run RFD4 (error "Invalid backend"),
           so the pipeline uses RFdiffusion 3 + ProteinMPNN there. Force one with BACKBONE_TOOL=rfd4 | rfd3 | rfdiffusion.
  - AF2:   "af2_bw" (newer JAX) on RTX 50xx / Blackwell cards (compute capability 10+), "af2_bd" on every other card.
  - Several cards (GPUS=0,1): every tool runs on each card that can run it, and the work is shared by measured speed.
  - Batch sizes and worker counts have defaults that were measured on the dev server; override them with the variables below.

ALL VARIABLES (every one is optional except the three target ones):
  Target (required):   TARGET_PDB   TARGET_RANGE (commas between pieces: A16-108,A113-151; every residue must exist in the file)
                       HOTSPOTS (2-4 residues, same numbering)
  Design:              BINDER_LENGTH (55-55; a fixed length makes AF2 compile only once)   NUM_BACKBONES (8: use a multiple of the RFD4 batch)
                       SEQS_PER_BACKBONE (4; only used when ProteinMPNN runs: RFD3, or RFD4 with RFD4_MPNN=1)
  Cards:               GPUS (0 | 1 | 0,1; GPU is the old name)   BACKBONE_TOOL   AF2_ENV (force one AF2 env on every card)
  Workers per card:    RFD_WORKERS (1)   MPNN_WORKERS (1)   AF2_WORKERS (8 on one card; with several cards 6 on the first, 2 on the others)
                       any of them with a card suffix, e.g. AF2_WORKERS_1=2 or RFD_WORKERS_1=1, applies to that card only
  Designs at once:     RFD4_BATCH (8 on a card with 12+ GB, else 2)   RFD3_BATCH (4, else 2)   (also with a card suffix: RFD3_BATCH_1)
  RFD4 extras:         RFD4_MPNN=1 (also redesign RFD4's sequences with ProteinMPNN)   RFD4_SEED   RFD4_CKPT   RFD4_OVERRIDES
  RFD3 extras:         RFD3_CKPT   RFD3_OVERRIDES
  Where things are:    ENVS_DIR (~/miniconda3/envs)   TOOLS_DIR (~/tools)   RFD4_BIN   RFD3_BIN   MPNN_PY   REGLYCO_BIN
  Glycosylation scan:  GLYCO_WARN_A (15)   GLYCO_CACHE   REGLYCO_OFFLINE=1   NETNGLYC_OUTPUT   NETNGLYC_BIN
  Testing only:        GPU_INFO="0:12.0:16311,1:7.5:6144" (gpu:compute capability:MiB) instead of asking nvidia-smi

Expected layout (what install_envs.sh creates):
    $TOOLS_DIR/ProteinMPNN, $TOOLS_DIR/dl_binder_design           (and $TOOLS_DIR/RFdiffusion only for BACKBONE_TOOL=rfdiffusion)
    conda envs in $ENVS_DIR:  "rfd4_env" (RFD4-Proteina), "rfd3_env" (RFD3 + ProteinMPNN), "af2_bd" / "af2_bw" (AF2 + PyRosetta)
    RFD3 weights:             $RFD3_CKPT (default ~/.foundry/checkpoints/rfd3_latest.ckpt)
    RFD4 weights:             Hugging Face cache (HF_HOME), downloaded by  install_envs.sh rfd4_env
"""
import os
import subprocess
from pathlib import Path

# =====================================================================================================================
# 1. YOUR JOB: edit these (or set the environment variable of the same name, which wins)
# =====================================================================================================================
JOB = {
    "TARGET_PDB": "",              # full path of the PDB file of the protein you want to bind, e.g. /home/me/targets/5YXH.pdb
    "TARGET_RANGE": "",            # chain letter + residues to keep; separate pieces with COMMAS, e.g. A16-108,A113-151
    "HOTSPOTS": "",                # 2-4 target residues (same numbering) the binder should touch, e.g. A31,A32
    "BINDER_LENGTH": "55-55",      # "55-55" = fixed length; "60-80" = the tool picks a length in that range
    "NUM_BACKBONES": "8",          # how many designs; a multiple of 8 suits RFD4's batch size
    "SEQS_PER_BACKBONE": "4",      # ProteinMPNN sequences per backbone (only used with RFD3, or RFD4 with RFD4_MPNN=1)
    "GPUS": "",                    # which card(s), numbers from nvidia-smi: "0", "1" or "0,1". Empty = CUDA_VISIBLE_DEVICES, else 0
}

# =====================================================================================================================
# 2. Everything below works things out from the block above. You normally do not need to read it.
# =====================================================================================================================
HOME = Path.home()
ENVS = Path(os.environ.get("ENVS_DIR", HOME / "miniconda3/envs"))
TOOLS = Path(os.environ.get("TOOLS_DIR", HOME / "tools"))


def get(name, default=""):
    """A value: the environment variable first, then the YOUR JOB block, then the default."""
    return os.environ.get(name) or JOB.get(name) or default


def per_card(name, gpu, default):
    """NAME_<gpu> (this card only) first, then NAME (every card), then the default."""
    return os.environ.get(f"{name}_{gpu}") or os.environ.get(name) or default


# ---- the cards ----------------------------------------------------------------------------------------------------
GPUS = [g.strip() for g in (os.environ.get("GPUS") or os.environ.get("GPU") or JOB["GPUS"]
                            or os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",") if g.strip()]


def card_info(gpu):
    """{"cc": compute capability (e.g. 12.0), "mem": MiB} of one card; None values when nvidia-smi cannot tell."""
    for entry in os.environ.get("GPU_INFO", "").split(","):             # testing hook: GPU_INFO="0:12.0:16311,1:7.5:6144"
        if entry.count(":") == 2 and entry.split(":")[0] == str(gpu):
            _, cc, mem = entry.split(":")
            return {"cc": float(cc), "mem": float(mem)}
    if os.environ.get("GPU_INFO"):
        return {"cc": None, "mem": None}
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=compute_cap,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0].split(",")
        return {"cc": float(out[0]), "mem": float(out[1])}
    except Exception:
        return {"cc": None, "mem": None}


CARDS = {g: card_info(g) for g in GPUS}
modern = lambda gpu: CARDS[gpu]["cc"] is None or CARDS[gpu]["cc"] >= 8.0        # Ampere or newer: can run RFD4
blackwell = lambda gpu: CARDS[gpu]["cc"] is not None and CARDS[gpu]["cc"] >= 10.0
big = lambda gpu: CARDS[gpu]["mem"] is None or CARDS[gpu]["mem"] >= 12000        # 12 GB or more


def choose_tool(forced, any_modern_card, show=True):
    """The backbone tool: the one you forced, else RFD4 when any card can run it, else RFD3 (with a notice)."""
    if forced:
        return forced
    if any_modern_card:
        return "rfd4"
    if show:
        print(f"  note: no GPU in {GPUS} is Ampere or newer (compute capability 8.0+): using RFD3 + ProteinMPNN instead of RFD4 "
              f"(set BACKBONE_TOOL=rfd4 to force RFD4)")
    return "rfd3"


TOOL = choose_tool(os.environ.get("BACKBONE_TOOL"), any(modern(g) for g in GPUS))


def make_workers(program, script, gpu, n, weight, extra_args=(), extra_env=None, batch=None):
    """n identical workers on card `gpu`. The env's own lib folder goes first in LD_LIBRARY_PATH, so its newer C++ runtime
    is used on clusters whose system libstdc++ is too old (error: GLIBCXX_3.4.xx not found)."""
    env_lib = Path(program).parent.parent / "lib"
    lib_path = f"{env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return {"cmd": [str(program)] + ([str(script)] if script else []) + list(extra_args),
            "env": {"CUDA_VISIBLE_DEVICES": str(gpu), "LD_LIBRARY_PATH": lib_path, **(extra_env or {})},
            "weight": weight, "copies": n, "batch": batch}


def cache(env_name):     # JAX compile cache; one per AF2 environment because the JAX versions differ
    return {"JAX_COMPILATION_CACHE_DIR": str(HOME / (".cache/jax_af2" if env_name == "af2_bd" else f".cache/jax_{env_name}")),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


RFD4_BIN = os.environ.get("RFD4_BIN", str(ENVS / "rfd4_env/bin/rfd4-generate"))
RFD3_BIN = os.environ.get("RFD3_BIN", str(ENVS / "rfd3_env/bin/rfd3"))
MPNN_PY = os.environ.get("MPNN_PY", str(ENVS / "rfd3_env/bin/python"))        # ProteinMPNN only needs PyTorch
RFD3_CKPT = os.environ.get("RFD3_CKPT", str(HOME / ".foundry/checkpoints/rfd3_latest.ckpt"))
REGLYCO_BIN = os.environ.get("REGLYCO_BIN", str(TOOLS / "reglyco/bin/reglyco"))        # glycosylation scan (optional)


def backbone_workers(tool):
    """Workers for a backbone tool, one entry per card that runs it."""
    workers = []
    for g in GPUS:
        runs_here = tool != "rfd4" or modern(g)                              # RFD4 stays off cards that cannot run it ...
        if os.environ.get(f"RFD_WORKERS_{g}"):                               # ... unless you ask for THAT card (RFD_WORKERS_1=1)
            n = int(os.environ[f"RFD_WORKERS_{g}"])
        else:
            n = int(os.environ.get("RFD_WORKERS", 1)) if runs_here else 0
        if n < 1:
            continue
        if tool == "rfd4":
            old = not modern(g)                                              # ... unless you ask for it: fp32, no compile (slow)
            workers.append(make_workers(RFD4_BIN, None, g, n, (1 if old else 12) / n,
                                        ["precision=32-true", "inference_experiment=quickstart"] if old else [],
                                        batch=int(per_card("RFD4_BATCH", g, 8 if big(g) else 2))))
        elif tool == "rfd3":
            workers.append(make_workers(RFD3_BIN, "design", g, n, (4 if modern(g) else 1) / n,
                                        batch=int(per_card("RFD3_BATCH", g, 4 if big(g) else 2))))
        else:   # "rfdiffusion": the older RFdiffusion 1
            workers.append(make_workers(ENVS / "rfdiffusion/bin/python", TOOLS / "RFdiffusion/scripts/run_inference.py", g, n,
                                        (7 if modern(g) else 3) / n))
    if not workers and tool == "rfd4":           # forced onto cards that cannot run it: one slow fp32 worker each
        for g in GPUS:
            workers.append(make_workers(RFD4_BIN, None, g, 1, 1, ["precision=32-true", "inference_experiment=quickstart"], batch=2))
    return workers


def af2_workers():
    workers = []
    for i, g in enumerate(GPUS):
        env_name = os.environ.get("AF2_ENV") or ("af2_bw" if blackwell(g) else "af2_bd")
        default_n = 8 if len(GPUS) == 1 else (6 if i == 0 else 2)
        n = int(per_card("AF2_WORKERS", g, default_n))
        if n >= 1:      # weight = the card's speed per worker (older cards ~0.7x); AF2 is CPU-bound, so more workers = more throughput
            workers.append(make_workers(ENVS / env_name / "bin/python", TOOLS / "dl_binder_design/af2_initial_guess/predict.py", g, n,
                                        1.0 if modern(g) else 0.7, extra_env=cache(env_name)))
    return workers


mpnn_workers = [make_workers(MPNN_PY, TOOLS / "ProteinMPNN/protein_mpnn_run.py", g, int(per_card("MPNN_WORKERS", g, 1)),
                             1 / int(per_card("MPNN_WORKERS", g, 1)))
                for g in GPUS if int(per_card("MPNN_WORKERS", g, 1)) >= 1]

SETTINGS = {
    "backbone_tool": TOOL,                       # rfd4 (default) | rfd3 | rfdiffusion
    "target_pdb": get("TARGET_PDB"),             # (design_pipeline.py complains if a stage that needs the target finds these empty)
    "target_range": get("TARGET_RANGE"),
    "hotspots": get("HOTSPOTS"),
    "binder_length": get("BINDER_LENGTH", "55-55"),
    "num_backbones": int(get("NUM_BACKBONES", "8")),
    "seqs_per_backbone": int(get("SEQS_PER_BACKBONE", "4")),
    "output_dir": "runs/run1",                   # use --output_dir on the command line; a NEW folder every run

    # RFD4-Proteina: designs backbone AND sequence, so ProteinMPNN is skipped unless RFD4_MPNN=1
    "rfd4_workers": backbone_workers("rfd4"),
    "rfd4_ckpt": os.environ.get("RFD4_CKPT", "nvidia/RFD4-Proteina"),          # Hugging Face name of the weights
    "rfd4_mpnn": os.environ.get("RFD4_MPNN", "") == "1",
    "rfd4_overrides": os.environ.get("RFD4_OVERRIDES", "").split(),            # e.g. "hf_ckpt_revision=<commit>"
    **({"rfd4_seed": int(os.environ["RFD4_SEED"])} if os.environ.get("RFD4_SEED") else {}),

    # RFdiffusion 3 (+ ProteinMPNN)
    "rfd3_workers": backbone_workers("rfd3"),
    "rfd3_ckpt": RFD3_CKPT,
    "rfd3_overrides": os.environ.get("RFD3_OVERRIDES", "").split(),            # e.g. "inference_sampler.step_scale=1.5"

    # RFdiffusion 1 (only with BACKBONE_TOOL=rfdiffusion)
    "rfd_noise": 0,
    "rfdiffusion_workers": backbone_workers("rfdiffusion"),

    "mpnn_workers": mpnn_workers,
    "af2_workers": af2_workers(),
    "af2_recycles": 3,

    # what counts as a good design (rules of thumb)
    "min_iptm": 0.8, "min_plddt": 80, "max_rmsd": 2.0, "max_pae_interaction": 10,

    # Target glycosylation scan (stage "glyco"): the built-in sequon + exposure scan always runs; ReGlyco is added when its
    # program exists (install_envs.sh reglyco), NetNGlyc when you give its result file (NETNGLYC_OUTPUT) or its program.
    "glyco_warn_A": float(os.environ.get("GLYCO_WARN_A", 15)),                 # warn if a sequon is this close (A) to a hotspot
    "reglyco_cache": os.environ.get("GLYCO_CACHE", str(HOME / ".cache/reglyco")),   # keep between runs: one download only
    "reglyco_offline": os.environ.get("REGLYCO_OFFLINE", "") == "1",           # on machines without internet (after one online run)
    **({"reglyco_cmd": [REGLYCO_BIN]} if Path(REGLYCO_BIN).exists() else {}),
    **({"netnglyc_output": os.environ["NETNGLYC_OUTPUT"]} if os.environ.get("NETNGLYC_OUTPUT") else {}),
    **({"netnglyc_cmd": [os.environ["NETNGLYC_BIN"]]} if os.environ.get("NETNGLYC_BIN") else {}),
}
