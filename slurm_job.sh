#!/bin/bash
# Slurm job script for the binder-design workflow (one GPU, several workers on it).
#
#   1. Edit the lines marked  <-- EDIT.
#   2. Check the request is valid without running it:   sbatch --test-only slurm_job.sh
#   3. Submit:                                          sbatch slurm_job.sh
#   4. Watch it:   squeue -u $USER      log: tail -f slurm-<jobnumber>.out      cancel: scancel <jobnumber>
#
# The #SBATCH lines are instructions to the scheduler. They must stay at the top, before any other command.
# <-- EDIT: your account (hyakalloc lists the ones you can use)
#SBATCH --account=stf
# <-- EDIT: a GPU partition from hyakalloc (this word makes the job fail until you change it, on purpose)
#SBATCH --partition=CHANGE_ME
# one GPU
#SBATCH --gres=gpu:1
# AF2 needs about one CPU core per worker, plus a few
#SBATCH --cpus-per-task=12
# roughly 2 GB per AF2 worker plus headroom
#SBATCH --mem=32G
# <-- EDIT: the job is killed when this runs out
#SBATCH --time=04:00:00
#SBATCH --job-name=binder
# log file; %j = job number
#SBATCH --output=slurm-%j.out

set -eo pipefail      # (no "-u": conda's activation scripts use variables that may be unset)

BASE=/gscratch/scrubbed/legia8/binder                         # <-- EDIT: where install_envs.sh put things
PROJECT=/gscratch/scrubbed/legia8/Protein_design_workflow     # <-- EDIT: the project folder
RUN=$BASE/runs/trial_$(date +%Y%m%d_%H%M)                     # a NEW output folder for every run

# --- the job description (read by settings_big_gpu.py) ---
export TARGET_PDB=/gscratch/scrubbed/legia8/targets/5YXH.pdb  # <-- EDIT
export TARGET_RANGE="A16-108,A113-151"                         # <-- EDIT (commas between pieces; every residue must exist in the PDB)
export HOTSPOTS="A31,A123,A122,A32"                            # <-- EDIT
export BINDER_LENGTH="55-55"
export NUM_BACKBONES=4                                         # a small trial first!
export SEQS_PER_BACKBONE=2
export AF2_WORKERS=8                                           # workers on the GPU: roughly one per CPU core you asked for
export RFD_WORKERS=1                                           # RFD4: ONE worker with a big batch (each extra worker compiles the model again)
# Backbones: RFD4-Proteina is the default (needs: bash install_envs.sh rfd4_env). For RFD3 + ProteinMPNN use: export BACKBONE_TOOL=rfd3
export RFD4_BATCH=8                                          # designs RFD4 makes at once; use NUM_BACKBONES as a multiple of it for real runs
export RFD3_BATCH=4                                         # designs RFD3 makes at once (more GPU memory = bigger batch)
export RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt           # the RFD3 weights that install_envs.sh rfd3_env downloaded

# --- where the environments and tools are ---
export ENVS_DIR=$BASE/envs
export TOOLS_DIR=$BASE/tools
export HF_HOME=$BASE/hf_cache                  # RFD4's weights (1.7 GB, Hugging Face): install_envs.sh rfd4_env downloads them here
export GLYCO_CACHE=$BASE/reglyco_cache         # ReGlyco's glycan data: run  --stages glyco  once on the login node (it needs internet)

# --- activate conda and the light environment that runs the pipeline script ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binder_pipeline

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --- preflight: stop NOW (not after a model download or a long compile) if RFD4 cannot work on this node ---
if [ "${BACKBONE_TOOL:-rfd4}" = rfd4 ]; then
    "$ENVS_DIR/rfd4_env/bin/python" -c "import sys, torch; ok = torch.cuda.is_available(); print('RFD4 env: torch', torch.__version__, 'cuda', torch.version.cuda, '| GPU visible:', ok); sys.exit(0 if ok else 1)" \
        || { echo "ERROR: PyTorch cannot see the GPU (is the node's NVIDIA driver new enough for this PyTorch build?). Try: export BACKBONE_TOOL=rfd3"; exit 1; }
    ls -d "$HF_HOME"/hub/models--nvidia--RFD4-Proteina >/dev/null 2>&1 \
        || { echo "ERROR: RFD4 weights not found in $HF_HOME: run  bash install_envs.sh rfd4_env  on the login node (compute nodes may have no internet)"; exit 1; }
    export HF_HUB_OFFLINE=1                    # use the downloaded weights; never wait for the network
fi

cd "$PROJECT"
echo "output folder: $RUN"

python design_pipeline.py --settings settings_big_gpu.py --output_dir "$RUN" --stages glyco,rfd,mpnn,af2,score_af2

echo "finished $(date)"
