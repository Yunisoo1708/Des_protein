"""
The only file a beginner needs to edit.
Run the pipeline inside WSL/Linux, so all paths below are Linux paths (e.g. /home/you/...).
"""

# Each tool may live in its own conda environment: point to THAT environment's python.
RFD_PY = "/home/<user>/miniconda3/envs/SE3nv/bin/python"
MPNN_PY = "/home/<user>/miniconda3/envs/SE3nv/bin/python"      # ProteinMPNN only needs PyTorch
AF3_PY = "/home/<user>/miniconda3/envs/af3/bin/python"

SETTINGS = {
    # ---- Your target ---------------------------------------------------------
    "target_pdb": "/home/<user>/targets/my_target.pdb",   # PDB file of the protein you want to bind
    "target_range": "A1-120",      # chain letter + residues to keep, e.g. A1-120 (trim floppy ends!)
    "hotspots": "A30,A33,A34",     # 2-4 target residues (same numbering) the binder should touch
    "binder_length": "60-80",      # RFdiffusion picks a length in this range

    # ---- How many designs ----------------------------------------------------
    "num_backbones": 10,           # step 1: skeletons from RFdiffusion
    "seqs_per_backbone": 4,        # step 2: sequences per skeleton (total AF3 runs = 10 x 4 = 40)
    "rfd_noise": 0,                # 0 = cleanest backbones, 1 = most varied

    # ---- Output folder -------------------------------------------------------
    "output_dir": "runs/run1",

    # ---- Tool commands (command + path to the tool's script) -----------------
    "rfdiffusion_cmd": [RFD_PY, "/home/<user>/RFdiffusion/scripts/run_inference.py"],
    "mpnn_cmd": [MPNN_PY, "/home/<user>/ProteinMPNN/protein_mpnn_run.py"],
    "af3_cmd": [AF3_PY, "/home/<user>/alphafold3/run_alphafold.py"],
    "af3_model_dir": "/home/<user>/af3_weights",        # AF3 weights (request them from Google DeepMind)
    "af3_use_msa": False,          # False = fast single-sequence mode. True needs the big AF3 databases:
    "af3_db_dir": "/home/<user>/af3_databases",         # (only used when af3_use_msa is True)

    # ---- Optional: which GPU each tool uses (0, 1, ... as listed by nvidia-smi) ----
    # "rfdiffusion_env": {"CUDA_VISIBLE_DEVICES": "0"},
    # "mpnn_env": {"CUDA_VISIBLE_DEVICES": "0"},
    # "af3_env": {"CUDA_VISIBLE_DEVICES": "0"},

    # ---- What counts as a good design (rules of thumb) -----------------------
    "min_iptm": 0.8,
    "min_plddt": 80,
    "max_rmsd": 2.0,               # Angstrom
}
