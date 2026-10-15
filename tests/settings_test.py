"""Settings for the test run: same pipeline, but the three tools are the fakes in fake_tools.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
FAKE = [sys.executable, str(HERE / "fake_tools.py")]

SETTINGS = {
    "target_pdb": str(HERE / "_work" / "toy_target.pdb"),   # made by test_workflow.py
    "target_range": "A1-40",
    "hotspots": "A10,A20",
    "binder_length": "20-30",
    "num_backbones": 3,
    "seqs_per_backbone": 2,
    "rfd_noise": 0,
    "output_dir": str(HERE / "_work" / "run"),
    "rfdiffusion_cmd": FAKE + ["rfdiffusion"],
    "mpnn_cmd": FAKE + ["mpnn"],
    "af3_cmd": FAKE + ["af3"],
    "af2_cmd": FAKE + ["af2"],
    "af3_model_dir": "unused",
    "af3_use_msa": False,
    "af3_db_dir": "unused",
    "min_iptm": 0.8,
    "min_plddt": 80,
    "max_rmsd": 2.0,
}
