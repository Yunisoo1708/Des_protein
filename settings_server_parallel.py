"""Server settings that use BOTH GPUs at the same time, with several worker processes per card.
Edit WORKERS_PER_GPU below to change how many workers run on each card (the work is shared out by speed).
Compare with settings_server_large.py (same job, 5060 Ti only)."""
import runpy
from pathlib import Path

# ---- EDIT THESE: number of worker processes per card, per tool --------------------------------------------------
WORKERS_PER_GPU = {
    "5060 Ti":    {"rfdiffusion": 2, "mpnn": 1, "af2": 4},
    "1660 SUPER": {"rfdiffusion": 1, "mpnn": 1, "af2": 2},
}
# MEASURED on this machine (200 AF2 predictions): 1 worker/card 3088 s -> 4 + 2 workers 1237 s (2.5x faster).
# AF2 is limited by the CPU (about 16 of every 20 s per prediction runs on one CPU core), so more workers per card
# help until the CPU cores run out (16 here). Peak GPU memory was only 4.7 GB (5060 Ti) / 3.0 GB (1660 SUPER).
# -----------------------------------------------------------------------------------------------------------------

HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"

SETTINGS = runpy.run_path(str(Path(__file__).with_name("settings_server.py")))["SETTINGS"]
SETTINGS.update({"binder_length": "55-55", "num_backbones": 50, "seqs_per_backbone": 4,
                 "output_dir": str(HOME / "run_large_par")})

rfd = str(TOOLS / "RFdiffusion/scripts/run_inference.py")
mpnn = str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")
af2 = str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")


def cache(name):  # JAX compile cache; separate per environment because the JAX versions differ
    return {"JAX_COMPILATION_CACHE_DIR": str(HOME / f".cache/jax_{name}"), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


# Per card: which GPU number, which python environment for each tool, and how fast the card is for each tool
# ("speed" = the card's share of the work; a card with speed 7 gets 7 parts for every 3 of a card with speed 3).
CARDS = {
    "5060 Ti": {"gpu": "0",
                "rfdiffusion": (ENVS / "rfd_bw/bin/python", 7, {}),             # new PyTorch + DGL stand-in
                "mpnn":        (ENVS / "mpnn/bin/python", 1, {}),
                "af2":         (ENVS / "af2_bw/bin/python", 1.0, cache("af2_bw"))},   # JAX 0.6
    "1660 SUPER": {"gpu": "1",
                   "rfdiffusion": (ENVS / "rfdiffusion/bin/python", 3, {}),     # real DGL (old card)
                   "mpnn":        (ENVS / "rfdiffusion/bin/python", 1, {}),
                   "af2":         (ENVS / "af2_bd/bin/python", 0.7, cache("af2_bd"))},   # JAX 0.4.20
}
SCRIPTS = {"rfdiffusion": rfd, "mpnn": mpnn, "af2": af2}

for tool in ("rfdiffusion", "mpnn", "af2"):
    workers = []
    for card, info in CARDS.items():
        n = WORKERS_PER_GPU[card][tool]
        if n < 1:
            continue
        python, speed, extra_env = info[tool]
        # "copies" = run n identical workers on this card; each copy gets an equal share of the card's speed
        workers.append({"cmd": [str(python), SCRIPTS[tool]], "env": {"CUDA_VISIBLE_DEVICES": info["gpu"], **extra_env},
                        "weight": speed if tool == "af2" else speed / n, "copies": n})
    SETTINGS[f"{tool}_workers"] = workers
