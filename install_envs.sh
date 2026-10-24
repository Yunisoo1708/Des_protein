#!/bin/bash
# Installs the tools this workflow needs, plus their conda environments, on a Linux machine with conda.
#   RFdiffusion 3 (backbones), ProteinMPNN (sequences), dl_binder_design (AF2 "initial guess", Bennett lab)
#   (RFdiffusion 1 is optional: run every step with  WITH_RFD1=1  to install it too)
#
# Usage:   BASE=/path/with/lots/of/space  bash install_envs.sh <step>
# Steps, one at a time and in this order (each can be re-run):
#   tools     clone ProteinMPNN + dl_binder_design (+ RFdiffusion 1 with WITH_RFD1=1)   (internet, ~0.1 GB)
#   weights   download the AF2 weights (+ RFdiffusion 1 weights with WITH_RFD1=1)       (internet, ~5.5 GB)
#   rfd3_env  conda env "rfd3_env": RFdiffusion 3 + ProteinMPNN, and the RFD3 weights    (internet, ~10 GB)
#   af2_env   conda env "af2_bd": AF2 + PyRosetta                                        (internet, ~8 GB)
#   rfd4_env  OPTIONAL RFD4-Proteina (backbone + sequence), overlay on rfd3_env + weights (internet, ~2.5 GB)
#   reglyco   OPTIONAL ReGlyco (glycosylation scan of the target): Rust build, ~3 min   (internet, ~1.5 GB)
#   rfd_env   OPTIONAL conda env "rfdiffusion": RFdiffusion 1 (only with WITH_RFD1=1)    (internet, ~6 GB)
#   check     import tests that work without a GPU
#
# Everything goes under $BASE:  $BASE/tools  $BASE/envs  $BASE/pip_cache
# The versions below are the ones that worked on the GPU server (RTX 5060 Ti / GTX 1660 SUPER):
# RFD3 + ProteinMPNN: the newest PyTorch that pip picks (it worked on both cards; atomworks must stay below 3).
# AF2: JAX 0.4.20 + CUDA 12 (Turing/Ampere/Ada/Hopper cards, but NOT the newest Blackwell cards).
# PyRosetta (used by dl_binder_design) is free for academic, non-commercial use only.
set -euo pipefail

BASE=${BASE:?Set BASE first, e.g.   BASE=/gscratch/scrubbed/YOUR_NAME/binder bash install_envs.sh tools}
STEP=${1:?Give a step: tools | weights | rfd3_env | af2_env | rfd4_env | reglyco | rfd_env | check}
WITH_RFD1=${WITH_RFD1:-0}
TOOLS=$BASE/tools
ENVS=$BASE/envs
export PIP_CACHE_DIR=$BASE/pip_cache          # keep big pip downloads out of your home folder
mkdir -p "$TOOLS" "$ENVS" "$PIP_CACHE_DIR"
command -v conda >/dev/null || { echo "conda not found: load/activate conda first"; exit 1; }

case "$STEP" in
tools)
    cd "$TOOLS"
    if [ "$WITH_RFD1" = 1 ]; then [ -d RFdiffusion ] || git clone https://github.com/RosettaCommons/RFdiffusion.git; fi
    [ -d ProteinMPNN ]      || git clone https://github.com/dauparas/ProteinMPNN.git
    [ -d dl_binder_design ] || git clone https://github.com/nrbennet/dl_binder_design.git
    ls "$TOOLS"
    ;;

weights)
    # AF2 weights: only params_model_1_ptm.npz is used (public, CC BY 4.0)
    mkdir -p "$TOOLS/dl_binder_design/af2_initial_guess/model_weights/params"
    cd "$TOOLS/dl_binder_design/af2_initial_guess/model_weights/params"
    if [ ! -f params_model_1_ptm.npz ]; then
        wget -nv -c https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
        tar -xf alphafold_params_2022-12-06.tar params_model_1_ptm.npz
        rm -f alphafold_params_2022-12-06.tar
    fi
    ls -la "$TOOLS/dl_binder_design/af2_initial_guess/model_weights/params"
    ;;

rfd3_env)
    [ -d "$ENVS/rfd3_env" ] || conda create -y -p "$ENVS/rfd3_env" -c conda-forge --override-channels python=3.12 pip libstdcxx-ng
    PY=$ENVS/rfd3_env/bin/python
    $PY -m pip install --upgrade pip
    # atomworks 3.x removed a function that RFD3 (rc-foundry 0.2.0) imports, so keep atomworks on the 2.x series
    $PY -m pip install "rc-foundry[rfd3]" "atomworks<3"
    # RFD3 weights (~2.7 GB). Use them with   RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt
    "$ENVS/rfd3_env/bin/foundry" install rfd3 --checkpoint-dir "$BASE/checkpoints"
    ls -la "$BASE/checkpoints"
    echo "Now set  RFD3_CKPT=$BASE/checkpoints/rfd3_latest.ckpt  when you run the pipeline."
    ;;

rfd4_env)
    # RFD4-Proteina (designs backbone AND sequence). Needs Python 3.12 and a CUDA 13 compatible driver (check: nvidia-smi).
    # It is a thin overlay on rfd3_env: the new env re-uses rfd3_env's PyTorch and only adds the few packages RFD4 needs.
    # (It cannot share rfd3_env itself: RFD4 needs atomworks 3.0.4 or newer, RFD3 needs atomworks below 3.)
    [ -x "$ENVS/rfd3_env/bin/python" ] || { echo "run the 'rfd3_env' step first"; exit 1; }
    [ -d "$ENVS/rfd4_env" ] || "$ENVS/rfd3_env/bin/python" -m venv --system-site-packages "$ENVS/rfd4_env"
    "$ENVS/rfd4_env/bin/python" -m pip install --upgrade pip
    "$ENVS/rfd4_env/bin/python" -m pip install rfd4-proteina
    # weights (~1.7 GB) go to the Hugging Face cache; do this on a machine WITH internet (compute nodes often have none)
    # they go to $BASE/hf_cache, which the Slurm scripts also use (HF_HOME): re-use it with  export HF_HOME=$BASE/hf_cache
    # Only model.safetensors + config.json are fetched: the whole repo is ~84 GB (OpenFold3 images/weights, AF2, benchmarks)
    # and is not needed for generation.
    HF_HOME=${HF_HOME:-$BASE/hf_cache} "$ENVS/rfd4_env/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/RFD4-Proteina', allow_patterns=['config.json', 'model.safetensors']))"
    echo "Weights are in ${HF_HOME:-$BASE/hf_cache}. The pipeline uses RFD4 by default on Ampere or newer cards (RFD4_BIN=$ENVS/rfd4_env/bin/rfd4-generate if ENVS_DIR is not $ENVS)."
    ;;

reglyco)
    # ReGlyco is a Rust program; conda-forge provides the compiler. AGPL-3.0 licence (a commercial licence exists).
    [ -d "$ENVS/reglyco_build" ] || conda create -y -p "$ENVS/reglyco_build" -c conda-forge --override-channels "rust>=1.97"
    export PATH=$ENVS/reglyco_build/bin:$PATH CARGO_HOME=$TOOLS/reglyco/cargo_home
    mkdir -p "$TOOLS/reglyco"
    cargo install reglyco --locked --version 0.2.3 --root "$TOOLS/reglyco"
    "$TOOLS/reglyco/bin/reglyco" --version
    echo "ReGlyco downloads glycan data from glycoshape.org on its first scan (then caches it): run one scan on a machine WITH internet."
    ;;

rfd_env)
    [ "$WITH_RFD1" = 1 ] || { echo "rfd_env installs the OPTIONAL RFdiffusion 1; run it with WITH_RFD1=1 if you want it"; exit 0; }
    [ -d "$TOOLS/RFdiffusion" ] || { echo "run the 'tools' step (with WITH_RFD1=1) first"; exit 1; }
    [ -d "$ENVS/rfdiffusion" ] || conda create -y -p "$ENVS/rfdiffusion" -c conda-forge --override-channels python=3.10 pip
    # a recent C++ runtime inside the env (clusters with an old system libstdc++ need it for DGL / JAX)
    conda install -y -p "$ENVS/rfdiffusion" -c conda-forge --override-channels libstdcxx-ng
    PY=$ENVS/rfdiffusion/bin/python
    $PY -m pip install --upgrade pip
    $PY -m pip install "numpy<2" torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
    $PY -m pip install "numpy<2" hydra-core==1.3.2 pyrsistent omegaconf decorator pynvml e3nn==0.3.3 wandb icecream \
        scipy pandas biopython pyyaml torchdata==0.8.0 pydantic
    $PY -m pip install git+https://github.com/NVIDIA/dllogger.git
    # the DGL graph library (this also settles PyTorch at 2.4.0)
    $PY -m pip install "dgl==2.4.0" -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    $PY -m pip install "numpy<2"
    $PY -m pip install --no-deps -e "$TOOLS/RFdiffusion/env/SE3Transformer"
    $PY -m pip install --no-deps -e "$TOOLS/RFdiffusion"
    ;;

af2_env)
    [ -d "$TOOLS/dl_binder_design" ] || { echo "run the 'tools' step first"; exit 1; }
    # PyRosetta comes from the public RosettaCommons conda channel
    [ -d "$ENVS/af2_bd" ] || conda create -y -p "$ENVS/af2_bd" --override-channels \
        -c https://conda.rosettacommons.org -c conda-forge \
        python=3.11 pyrosetta biopython ml-collections ml_dtypes mock pip libstdcxx-ng
    PY=$ENVS/af2_bd/bin/python
    $PY -m pip install "numpy<2" "tensorflow-cpu==2.15.1" dm-haiku dm-tree absl-py scipy
    # exact versions that work together (newer haiku / biopython / numpy break the AF2 code):
    # JAX 0.4.23 (not 0.4.20): 0.4.20 failed on an RTX 4060 under WSL with "Failed to load PTX text as a module".
    # 0.4.24 and newer cannot be used: they removed jax.linear_util, which dm-haiku 0.0.10 needs.
    $PY -m pip install "numpy<2" "jax[cuda12_pip]==0.4.23" "jaxlib==0.4.23+cuda12.cudnn89" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    # jaxlib above is built for cuDNN 8.9; pip would otherwise pick cuDNN 9, and JAX then falls back to the CPU
    # ("Found cuDNN version 0, but JAX was built against version 8904")
    # ptxas from CUDA 12.2 (the version jaxlib was built with); settings.py points XLA at it (XLA_FLAGS, PATH)
    $PY -m pip install "nvidia-cudnn-cu12==8.9.7.29" "nvidia-cuda-nvcc-cu12==12.2.140"
    $PY -m pip install --no-deps "dm-haiku==0.0.10" "biopython==1.81"
    $PY -m pip install "ml_dtypes<0.3" "numpy<2"
    ;;

check)
    # Each env's own lib folder goes FIRST in the library search path, so its newer C++ runtime is used
    # instead of an older system one. When you run the pipeline in a job, do the same:
    #   export LD_LIBRARY_PATH=$BASE/envs/rfdiffusion/lib:$BASE/envs/af2_bd/lib:$LD_LIBRARY_PATH
    echo "== RFD3 + ProteinMPNN env (CPU-side imports; the GPU is hidden)"
    [ -x "$ENVS/rfd3_env/bin/python" ] || { echo "rfd3_env not found: run the 'rfd3_env' step first"; exit 1; }
    CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$ENVS/rfd3_env/lib:${LD_LIBRARY_PATH:-} $ENVS/rfd3_env/bin/python -c "import torch, numpy, atomworks, rfd3.inference.input_parsing; from importlib.metadata import version; print('torch', torch.__version__, '| atomworks', version('atomworks'), '| rc-foundry', version('rc-foundry'), '| imports OK')" 2>&1 | grep -v -i -E "warn|CCD_MIRROR|PDB_MIRROR|\(1\)|\(2\)|\(3\)" | tail -2
    if [ -x "$ENVS/rfd4_env/bin/python" ]; then
        echo "== RFD4-Proteina env (CPU-side imports)"
        CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$ENVS/rfd3_env/lib:${LD_LIBRARY_PATH:-} $ENVS/rfd4_env/bin/python -c "import torch, rfproteina; from importlib.metadata import version; print('torch', torch.__version__, '| rfd4-proteina', version('rfd4-proteina'), '| atomworks', version('atomworks'), '| imports OK')" 2>&1 | grep -v -i -E "warn|CCD_MIRROR|PDB_MIRROR|\(1\)|\(2\)|\(3\)" | tail -2
    fi
    if [ "$WITH_RFD1" = 1 ] && [ -x "$ENVS/rfdiffusion/bin/python" ]; then
        echo "== RFdiffusion 1 env (CPU-side imports)"
        LD_LIBRARY_PATH=$ENVS/rfdiffusion/lib:${LD_LIBRARY_PATH:-} $ENVS/rfdiffusion/bin/python -c "import torch, dgl, rfdiffusion, se3_transformer; print('torch', torch.__version__, '| dgl', dgl.__version__, '| imports OK')" 2>&1 | grep -v -i -E "warn|pynvml|DGL backend|Setting the default" || true
    fi
    echo "== AF2 env (CPU-side imports)"
    [ -x "$ENVS/af2_bd/bin/python" ] || { echo "af2_bd env not found: run the 'af2_env' step first"; exit 1; }
    LD_LIBRARY_PATH=$ENVS/af2_bd/lib:${LD_LIBRARY_PATH:-} $ENVS/af2_bd/bin/python -c "import jax, haiku, tensorflow, pyrosetta, Bio; print('jax', jax.__version__, '| haiku', haiku.__version__, '| biopython', Bio.__version__, '| imports OK')" 2>&1 | tail -1
    echo "== files"
    ls "$TOOLS/dl_binder_design/af2_initial_guess/model_weights/params/params_model_1_ptm.npz" "$BASE/checkpoints/rfd3_latest.ckpt" 2>&1 || echo "(RFD3 weights: if they are somewhere else, set RFD3_CKPT to that file)"
    echo
    echo "GPU checks need a GPU job. After requesting a GPU (salloc/sbatch), run:"
    echo "  $ENVS/rfd3_env/bin/python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
    echo "  $ENVS/af2_bd/bin/python -c \"import jax; print(jax.devices())\""
    ;;

*)  echo "unknown step '$STEP' (use: tools | weights | rfd3_env | af2_env | rfd4_env | reglyco | rfd_env | check)"; exit 1 ;;
esac
echo "step '$STEP' finished"
