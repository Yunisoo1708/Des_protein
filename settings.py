"""
The only file a beginner needs to edit: settings for ONE GPU. Backbones come from RFdiffusion 3 (RFD3), sequences from
ProteinMPNN, and AF2 (Bennett lab) checks the designs.
Run the pipeline inside WSL/Linux. Paths are built from your home folder, so you do not need to type your user name.

Each tool lives in a conda environment: point to the program INSIDE that environment (full path, ending in /bin/...).
"""
from pathlib import Path

HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"                      # where ProteinMPNN and dl_binder_design were cloned

RFD3_BIN = str(ENVS / "rfd3_env/bin/rfd3")        # the rfd3 program (RFdiffusion 3)
MPNN_PY = str(ENVS / "rfd3_env/bin/python")       # ProteinMPNN only needs PyTorch, so it can share the RFD3 environment
AF2_PY = str(ENVS / "af2_bd/bin/python")          # environment with JAX + PyRosetta (dl_binder_design)

SETTINGS = {
    "backbone_tool": "rfd3",       # "rfd3" (default) or "rfdiffusion" (the older RFdiffusion 1; needs its own settings)

    # ---- Your target ---------------------------------------------------------
    "target_pdb": str(HOME / "targets/my_target.pdb"),     # PDB file of the protein you want to bind
    # Which target residues to keep: chain letter + range; separate pieces with COMMAS. Every residue must exist in the PDB
    # file (a missing stretch is an error): leave it out, e.g. "A16-108,A113-151". A bare number between pieces
    # designs that many NEW residues in the gap: "A16-108,4,A113-151".
    "target_range": "A1-120",
    "hotspots": "A30,A33,A34",     # 2-4 target residues (same numbering) the binder should touch
    "binder_length": "60-80",      # RFD3 picks a length in this range ("55-55" = fixed; AF2 then compiles only once)

    # ---- How many designs ----------------------------------------------------
    "num_backbones": 10,           # step 1: backbones from RFD3
    "seqs_per_backbone": 4,        # step 2: sequences per backbone (total AF2 runs = 10 x 4 = 40)

    # ---- Output folder (use a NEW folder for every run) ------------------------
    "output_dir": "runs/run1",

    # ---- Tool commands -----------------------------------------------------------
    "rfd3_cmd": [RFD3_BIN, "design"],
    "rfd3_ckpt": str(HOME / ".foundry/checkpoints/rfd3_latest.ckpt"),   # RFD3 weights
    "rfd3_batch_size": 1,          # designs RFD3 makes at once: 2 is ~1.8x faster but needs more GPU memory (5 GB on a 6 GB card)
    # "rfd3_overrides": ["inference_sampler.step_scale=1.5"],           # advanced: any other RFD3 setting
    "mpnn_cmd": [MPNN_PY, str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")],
    "af2_cmd": [AF2_PY, str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")],
    "af2_recycles": 3,

    # ---- Optional: which GPU each tool uses (0, 1, ... as listed by nvidia-smi) ----
    # "rfd3_env": {"CUDA_VISIBLE_DEVICES": "1"},     # e.g. use only the second GPU for all three tools
    # "mpnn_env": {"CUDA_VISIBLE_DEVICES": "1"},
    # "af2_env": {"CUDA_VISIBLE_DEVICES": "1"},
    # For several workers or several GPUs, and a settings file driven by environment variables, see settings_big_gpu.py.

    # ---- Optional: RFD4-Proteina instead of RFD3 (designs the binder's sequence too, so ProteinMPNN is skipped) ----
    # Install: bash install_envs.sh rfd4_env   (needs a CUDA 13 driver). Then uncomment these and set backbone_tool to "rfd4":
    # "backbone_tool": "rfd4",
    # "rfd4_cmd": [str(ENVS / "rfd4_env/bin/rfd4-generate")],
    # "rfd4_ckpt": "nvidia/RFD4-Proteina",    # Hugging Face name of the weights
    # "rfd4_batch_size": 8,                   # designs made at once (the first run compiles the model for ~4 minutes)
    # "rfd4_mpnn": False,                     # True = also redesign RFD4's sequences with ProteinMPNN

    # ---- Target glycosylation scan (stage "glyco", runs first; the built-in sequon + exposure scan needs nothing) ----
    # "reglyco_cmd": [str(TOOLS / "reglyco/bin/reglyco")],      # adds ReGlyco's scan (install: bash install_envs.sh reglyco)
    # "netnglyc_output": "netnglyc_output.txt",                  # adds NetNGlyc: its text output for 0_glyco/target.fasta
    # "glyco_warn_A": 15,                                        # warn if a glycosylation motif is this close (A) to a hotspot

    # ---- What counts as a good design (rules of thumb) -----------------------
    "max_pae_interaction": 10,     # lower is better
    "min_plddt": 80,
    "max_rmsd": 2.0,               # Angstrom
    "min_iptm": 0.8,               # only used by the optional AlphaFold Server route
}
