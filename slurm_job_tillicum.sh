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
export RFD_WORKERS=2
export RFD3_BATCH=8                                         # designs RFD3 makes at once (more GPU memory = bigger batch)
export RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt           # the RFD3 weights that install_envs.sh rfd3_env downloaded

# --- where the environments and tools are ---
export ENVS_DIR=$BASE/envs
export TOOLS_DIR=$BASE/tools

# --- activate conda (Tillicum provides it as a module) and the light environment that runs the pipeline script ---
module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binder_pipeline

cd "$PROJECT"
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "output folder: $RUN"

python design_pipeline.py --settings settings_big_gpu.py --output_dir "$RUN" --stages glyco,rfd,mpnn,af2,score_af2

echo "finished $(date)"
