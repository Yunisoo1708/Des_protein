"""
settings.py -- the ONE settings file for the whole workflow (AF2, AF3 and RFD4 routes).

Pick the route with ROUTE (or an environment variable):   ROUTE=af2 python design_pipeline.py
    "af2"       RFdiffusion -> ProteinMPNN -> AlphaFold2 initial guess          (the original route)
    "af3"       RFdiffusion -> ProteinMPNN -> local AlphaFold 3 (+ target template)
    "rfd4_af3"  RFD4 designs backbone + sequence together -> local AlphaFold 3  (default; run it with ./setup.sh run-rfd4)

Edit the TARGET block for your own protein; the rest already points at the conda environments and tool folders that
./setup.sh creates. Paths are built from your home folder, so you do not need to type your user name.
Each tool lives in its own conda environment: point to THAT environment's python (full path, ending in /bin/python).
"""
import os
from pathlib import Path

ROUTE = os.environ.get("ROUTE", "rfd4_af3")

HERE = Path(__file__).resolve().parent
HOME = Path.home()
ENVS = HOME / "miniconda3/envs"
TOOLS = HOME / "tools"                       # RFdiffusion, ProteinMPNN, dl_binder_design, alphafold3 are cloned here

# ---- TARGET: the protein to bind (residue numbers are those of the PDB file) ------------------------------------------
TARGET_PDB = HERE / "prepared_protein_structure/5YXH.pdb"
TARGET_CHAIN = "A"
TARGET_RANGES = "16-108,113-151"             # residues kept as the target (trim floppy ends; leave out unresolved gaps)
HOTSPOTS = "31,32,122,123"                   # 2-4 target residues the binder should touch
BINDER_LENGTH = "50-80"                      # a range; "55-55" = fixed (AF2 then compiles only once)

# ---- GENERATED CIFs (made by ./setup.sh example, or by the two make_*.py scripts) -------------------------------------
WORK = HERE / "work"                         # generated files go here (git-ignored)
AF3_TEMPLATE_CIF = WORK / "target_template.cif"          # make_af3_template.py  (AF3 reads the target's structure from it)
RFD4_INPUT_CIF = WORK / "rfd4_input.cif"                 # make_rfd4_input.py    (RFD4 reads the target from it)
RFD4_SPEC = WORK / "rfd4_spec.yaml"                      # written next to the CIF by make_rfd4_input.py

_ch = TARGET_CHAIN
SETTINGS = {
    # ---- target, in the format RFdiffusion wants ----------------------------------------------------------------------
    "target_pdb": str(TARGET_PDB),
    "target_range": "/".join(f"{_ch}{r}" for r in TARGET_RANGES.split(",")),     # e.g. A16-108/A113-151
    "hotspots": ",".join(f"{_ch}{h}" for h in HOTSPOTS.split(",")),              # e.g. A31,A32,A122,A123
    "binder_length": BINDER_LENGTH,

    # ---- how many designs ---------------------------------------------------------------------------------------------
    "num_backbones": 10,           # RFdiffusion route: skeletons
    "seqs_per_backbone": 5,        # RFdiffusion route: ProteinMPNN sequences per skeleton
    "rfd_noise": 0,                # 0 = cleanest backbones, 1 = most varied

    # ---- output folder (use a NEW folder for every run) ---------------------------------------------------------------
    "output_dir": os.environ.get("OUTPUT_DIR", str(HOME / "runs" / ROUTE)),     # or:  OUTPUT_DIR=~/runs/x ./setup.sh run-rfd4

    # ---- RFdiffusion + ProteinMPNN ------------------------------------------------------------------------------------
    "rfdiffusion_cmd": [str(ENVS / "rfd_bw/bin/python"), str(TOOLS / "RFdiffusion/scripts/run_inference.py")],
    "mpnn_cmd": [str(ENVS / "rfdiffusion/bin/python"), str(TOOLS / "ProteinMPNN/protein_mpnn_run.py")],
    # "rfdiffusion_env": {"CUDA_VISIBLE_DEVICES": "0"},       # which GPU a tool uses (numbers from nvidia-smi)
    # "mpnn_env": {"CUDA_VISIBLE_DEVICES": "0"},

    # ---- AlphaFold 2 initial guess (dl_binder_design) -----------------------------------------------------------------
    "af2_cmd": [str(ENVS / "af2_bw/bin/python"), str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")],
    "af2_recycles": 3,
    # "af2_env": {"CUDA_VISIBLE_DEVICES": "0", "JAX_COMPILATION_CACHE_DIR": str(HOME / ".cache/jax_af2_bw")},
    # Several workers / GPUs: give a tool a list of workers instead of one command. "copies" = identical workers on one
    # card, "weight" = relative speed (a card with weight 2 gets twice the work). AF2 is CPU-bound, so 4 workers on a
    # fast card were 2.5x faster than 1 (measured: 200 predictions, 3088 s -> 1237 s). Example:
    # "af2_workers": [{"cmd": [str(ENVS / "af2_bw/bin/python"), str(TOOLS / "dl_binder_design/af2_initial_guess/predict.py")],
    #                  "env": {"CUDA_VISIBLE_DEVICES": "0"}, "weight": 1.0, "copies": 4}],   # same idea: rfdiffusion_workers, mpnn_workers

    # ---- AlphaFold 3, run locally (google-deepmind/alphafold3 in the "af3" env) ---------------------------------------
    "af3_cmd": [str(ENVS / "af3/bin/python"), str(TOOLS / "alphafold3/run_alphafold.py")],
    # AF3 needs compute capability >= 8.0. On a machine with an older second card, show AF3 only the new one:
    "af3_env": {"CUDA_VISIBLE_DEVICES": "0", "JAX_COMPILATION_CACHE_DIR": str(HOME / ".cache/jax_af3")},
    "af3_model_dir": str(HOME / "af3_models"),    # folder with af3.bin.zst (you obtain it from DeepMind under their terms)
    "af3_db_dir": None,                           # None = single-sequence mode (no 630 GB database download)
    # The target's known structure, so AF3 only has to place the binder (strongly recommended; None = no template).
    "af3_target_template_cif": str(AF3_TEMPLATE_CIF) if ROUTE in ("af3", "rfd4_af3") else None,
    "af3_diffusion_samples": 5 if ROUTE == "rfd4_af3" else 1,    # predictions per design; a stable design agrees across them
    "af3_flash_attention": "xla",                 # "triton" is faster on datacenter GPUs; "xla" is safe everywhere
    "af3_seed": 1,

    # ---- RFD4 (used by ./setup.sh run-rfd4) ---------------------------------------------------------------------------
    "rfd4_bin": str(HOME / "rfd4_venv/bin/rfd4-generate"),
    "rfd4_python": str(HOME / "rfd4_venv/bin/python"),                    # has gemmi, used by the converter and the report
    "rfd4_hf_repo": "nvidia/RFD4-Proteina",
    "rfd4_hf_revision": "a06e2509c7033b010fc43614e9fd840303588502",       # pinned weights
    "rfd4_num_designs": 50,
    "rfd4_gpu": "0",
    # Reward-guided search (./setup.sh run-rfd4 --search): partly denoised designs are folded by an AF3 "oracle" and only the
    # best candidates continue. Needs the apptainer shim (./setup.sh shim). Set the oracle's weights here:
    "rfd4_oracle_model_dir": str(HOME / "af3_models"),   # AF3 weights; converted OpenFold3 weights would go here (see ./setup.sh openfold3)

    # ---- Obsidian vault queue: runs started with --log leave their result notes here; vault_pull.sh (on the laptop) copies them -------
    "vault_outbox": str(HOME / "vault_outbox"),

    # ---- what counts as a good design (rules of thumb) ----------------------------------------------------------------
    "max_pae_interaction": 10,     # lower is better
    "min_plddt": 80,
    "max_rmsd": 2.0,               # Angstrom
    "min_iptm": 0.8,               # AF3 / AlphaFold Server routes
    "stages": {"af2": "rfd,mpnn,af2,score_af2", "af3": "rfd,mpnn,af3,score_af3", "rfd4_af3": "af3,score_af3"}[ROUTE],
}
