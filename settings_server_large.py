"""Larger batch on the GPU server: 50 backbones, fixed binder length (AF2 compiles once), 4 sequences each.
MPNN_GPU=0|1 (environment variable) picks which card runs ProteinMPNN, so the two can be timed."""
import os
import runpy
from pathlib import Path

SETTINGS = runpy.run_path(str(Path(__file__).with_name("settings_server.py")))["SETTINGS"]
ENVS = Path.home() / "miniconda3/envs"
TOOLS = Path.home() / "tools"

SETTINGS.update({
    "binder_length": "55-55",
    "num_backbones": 50,
    "seqs_per_backbone": 4,
    "output_dir": str(Path.home() / "run_large"),
})

gpu = os.environ.get("MPNN_GPU", "1")
# the 5060 Ti needs the newer-PyTorch env ("mpnn"); the older GTX 1660 SUPER uses the older one ("rfdiffusion")
mpnn_py = ENVS / ("mpnn" if gpu == "0" else "rfdiffusion") / "bin/python"
SETTINGS["mpnn_cmd"] = [str(mpnn_py), str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")]
SETTINGS["mpnn_env"] = {"CUDA_VISIBLE_DEVICES": gpu}
