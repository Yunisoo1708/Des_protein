"""Settings for the GPU server (<user>@<server-ip>). Example target = RFdiffusion's insulin_target.pdb."""
from pathlib import Path

HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"

# GPU 0 = RTX 5060 Ti (16 GB, fast, new)   GPU 1 = GTX 1660 SUPER (6 GB, slower, older)
# - "rfd_bw" env: PyTorch for the new card + a tiny stand-in for DGL (see ~/tools/dgl_shim)
# - "rfdiffusion" env: older PyTorch (works on the old card only)
RFD_PY = str(ENVS / "rfd_bw/bin/python")
MPNN_PY = str(ENVS / "rfdiffusion/bin/python")

SETTINGS = {
    "target_pdb": str(TOOLS / "RFdiffusion/examples/input_pdbs/insulin_target.pdb"),
    "target_range": "A1-150",
    "hotspots": "A59,A83,A91",
    "binder_length": "50-60",
    "num_backbones": 4,
    "seqs_per_backbone": 4,
    "rfd_noise": 0,
    "output_dir": str(HOME / "real_run"),

    "rfdiffusion_cmd": [RFD_PY, str(TOOLS / "RFdiffusion/scripts/run_inference.py")],
    "rfdiffusion_env": {"CUDA_VISIBLE_DEVICES": "0"},     # heavy job -> fast card
    "mpnn_cmd": [MPNN_PY, str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")],
    "mpnn_env": {"CUDA_VISIBLE_DEVICES": "1"},            # light job -> old card

    # Step 3: AF2 "initial guess" (Bennett lab, dl_binder_design). "af2_bw" env = newer JAX for the new card.
    "af2_cmd": [str(ENVS / "af2_bw/bin/python"), str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")],
    # JAX recompiles for every new input length (slow, CPU-bound, GPU sits idle). The cache remembers compiled
    # programs between runs; also try a fixed binder_length (e.g. "55-55") so a batch compiles only once.
    "af2_env": {"CUDA_VISIBLE_DEVICES": "0",
                "JAX_COMPILATION_CACHE_DIR": str(HOME / ".cache/jax_af2"),
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"},
    "af2_recycles": 3,

    # What counts as a good design (rules of thumb). min_iptm is used only by the optional AlphaFold Server route.
    "min_iptm": 0.8, "min_plddt": 80, "max_rmsd": 2.0, "max_pae_interaction": 10,
}
