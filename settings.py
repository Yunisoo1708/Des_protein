"""
The only file a beginner needs to edit: settings for ONE GPU, using AF2 (Bennett lab) to check the designs.
Run the pipeline inside WSL/Linux. Paths are built from your home folder, so you do not need to type your user name.

Each tool lives in its own conda environment: point to THAT environment's python (full path, ending in /bin/python).
"""
from pathlib import Path

HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"                      # where RFdiffusion, ProteinMPNN and dl_binder_design were cloned

RFD_PY = str(ENVS / "rfdiffusion/bin/python")     # environment that has RFdiffusion's packages
MPNN_PY = str(ENVS / "rfdiffusion/bin/python")    # ProteinMPNN only needs PyTorch, so it can share that one
AF2_PY = str(ENVS / "af2_bd/bin/python")          # environment with JAX + PyRosetta (dl_binder_design)

SETTINGS = {
    # ---- Your target ---------------------------------------------------------
    "target_pdb": str(HOME / "targets/my_target.pdb"),     # PDB file of the protein you want to bind
    "target_range": "A1-120",      # chain letter + residues to keep, e.g. A1-120 (trim floppy ends!)
    "hotspots": "A30,A33,A34",     # 2-4 target residues (same numbering) the binder should touch
    "binder_length": "60-80",      # RFdiffusion picks a length in this range ("55-55" = fixed; AF2 then compiles once)

    # ---- How many designs ----------------------------------------------------
    "num_backbones": 10,           # step 1: skeletons from RFdiffusion
    "seqs_per_backbone": 4,        # step 2: sequences per skeleton (total AF2 runs = 10 x 4 = 40)
    "rfd_noise": 0,                # 0 = cleanest backbones, 1 = most varied

    # ---- Output folder (use a NEW folder for every run) ------------------------
    "output_dir": "runs/run1",

    # ---- Tool commands (command + path to the tool's script) -----------------
    "rfdiffusion_cmd": [RFD_PY, str(TOOLS / "RFdiffusion/scripts/run_inference.py")],
    "mpnn_cmd": [MPNN_PY, str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")],
    "af2_cmd": [AF2_PY, str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")],
    "af2_recycles": 3,

    # ---- Optional: which GPU each tool uses (0, 1, ... as listed by nvidia-smi) ----
    # "rfdiffusion_env": {"CUDA_VISIBLE_DEVICES": "0"},
    # "mpnn_env": {"CUDA_VISIBLE_DEVICES": "0"},
    # "af2_env": {"CUDA_VISIBLE_DEVICES": "0"},
    # For several workers or several GPUs see settings_server_parallel.py / settings_big_gpu.py.

    # ---- What counts as a good design (rules of thumb) -----------------------
    "max_pae_interaction": 10,     # lower is better
    "min_plddt": 80,
    "max_rmsd": 2.0,               # Angstrom
    "min_iptm": 0.8,               # only used by the optional AlphaFold Server route
}
