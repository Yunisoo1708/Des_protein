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
export RFD_WORKERS=2
export RFD3_BATCH=4                                         # designs RFD3 makes at once (more GPU memory = bigger batch)
export RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt           # the RFD3 weights that install_envs.sh rfd3_env downloaded

# --- where the environments and tools are ---
export ENVS_DIR=$BASE/envs
export TOOLS_DIR=$BASE/tools

# --- activate conda and the light environment that runs the pipeline script ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate binder_pipeline

cd "$PROJECT"
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "output folder: $RUN"

python design_pipeline.py --settings settings_big_gpu.py --output_dir "$RUN" --stages glyco,rfd,mpnn,af2,score_af2

echo "finished $(date)"
