#!/bin/bash
# Slurm job script for Tillicum (UW Hyak): one full H200, 8 CPU cores (the fixed ratio there is 8 CPUs per GPU).
# Written from https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs  -- NOT yet run on Tillicum.
#
#   1. Edit the lines marked  <-- EDIT.
#   2. Submit:   sbatch slurm_job_tillicum.sh
#      Tillicum bills GPU hours (about $0.90 per GPU-hour x the QoS and partition multipliers); the submission prints
#      a cost estimate, and cancelling right away is free:   scancel <jobnumber>
#   3. Watch it:   squeue -u $USER      log: tail -f slurm-<jobnumber>.out      usage: hyakusage
#
# QoS "debug": max 1 hour, 1 GPU, 8 CPUs, 1 job at a time -- ideal for a small trial. For a bigger run use
# --qos=normal (max 24 hours) and a longer --time.
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
# one full H200; a job gets 8 CPUs per GPU at most (to get more CPUs you must request more GPUs)
#SBATCH --gres=gpu:1
# AF2 needs about one CPU core per worker: 6 AF2 workers + the pipeline script + headroom
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=binder
#SBATCH --output=slurm-%j.out
# <-- EDIT only if sbatch complains about the account (find yours with:  hyakusage ) -- remove one # and fill in:
##SBATCH --account=YOUR_ACCOUNT

set -eo pipefail      # (no "-u": conda's activation scripts use variables that may be unset)

BASE=CHANGE_ME_TO_YOUR_BIG_FOLDER      # <-- EDIT: where install_envs.sh put envs/ and tools/ on Tillicum
PROJECT=CHANGE_ME_TO_THE_PROJECT       # <-- EDIT: the project folder (contains design_pipeline.py)
TARGET=CHANGE_ME_TO_THE_PDB            # <-- EDIT: full path of your target PDB, e.g. .../targets/5YXH.pdb
RUN=$BASE/runs/trial_$(date +%Y%m%d_%H%M)    # a NEW output folder for every run

for p in "$BASE/envs" "$PROJECT/design_pipeline.py" "$TARGET"; do
    [ -e "$p" ] || { echo "ERROR: $p not found: edit the CHANGE_ME lines at the top of this script"; exit 1; }
done

# --- the job description (read by settings_big_gpu.py) ---
export TARGET_PDB=$TARGET
export TARGET_RANGE="A16-108,A113-151"       # <-- EDIT (commas between pieces; every residue must exist in the PDB)
export HOTSPOTS="A31,A123,A122,A32"          # <-- EDIT
export BINDER_LENGTH="55-55"
export NUM_BACKBONES=4                       # a small trial first!
export SEQS_PER_BACKBONE=2
export AF2_WORKERS=6                         # 8 cores per H200: 6 AF2 workers leave room for the rest
export RFD_WORKERS=1                         # RFD4: ONE worker with a big batch (each extra worker compiles the model again and takes more GPU memory)
# Backbones: RFD4-Proteina is the default (needs: bash install_envs.sh rfd4_env). For RFD3 + ProteinMPNN use: export BACKBONE_TOOL=rfd3
export RFD4_BATCH=8                                          # designs RFD4 makes at once; use NUM_BACKBONES as a multiple of it for real runs
export RFD3_BATCH=8                                         # designs RFD3 makes at once (more GPU memory = bigger batch)
export RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt           # the RFD3 weights that install_envs.sh rfd3_env downloaded

# --- where the environments and tools are ---
export ENVS_DIR=$BASE/envs
export TOOLS_DIR=$BASE/tools
export HF_HOME=$BASE/hf_cache                # RFD4's weights (1.7 GB, Hugging Face): install_envs.sh rfd4_env downloads them here
export GLYCO_CACHE=$BASE/reglyco_cache       # ReGlyco's glycan data: run  --stages glyco  once on the login node (it needs internet)

# --- activate conda (Tillicum provides it as a module) and the light environment that runs the pipeline script ---
module load conda 2>/dev/null || true
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
    export HF_HUB_OFFLINE=1                  # use the downloaded weights; never wait for the network
fi

cd "$PROJECT"
echo "output folder: $RUN"

python design_pipeline.py --settings settings_big_gpu.py --output_dir "$RUN" --stages glyco,rfd,mpnn,af2,score_af2

echo "finished $(date)"
