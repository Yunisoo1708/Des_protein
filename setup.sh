#!/usr/bin/env bash
# setup.sh -- move a machine that runs the AF2 binder-design workflow over to AlphaFold 3, and then to RFD4 (+ an AF3 / OpenFold3 reward).
#
#   ./setup.sh                 show what is installed and what to run next (changes nothing)
#   ./setup.sh af3             1. local AlphaFold 3: conda env "af3", the code at a pinned commit, build_data, the model parameters
#   ./setup.sh rfd4            2. RFD4-Proteina: python venv, the package, the pinned weights from Hugging Face
#   ./setup.sh shim            3. let RFD4's reward-guided search call your native AF3 (no Apptainer needed): checks and prints the exports
#   ./setup.sh openfold3       4. optional: check whether OpenFold3 can be the search reward (it needs things not installed by this script)
#   ./setup.sh example         make the two CIF files the models read (AF3 template, RFD4 input) for the target in settings.py
#   ./setup.sh run-rfd4 [--n 50] [--search] [--log --title "..." --goal "..." --parent <slug>]
#                              RFD4 generate -> convert -> AF3 check -> report. --search = reward-guided beam search (about 5x slower).
#                              --log queues the result notes for the Obsidian vault when the run ends (pulled by vault_pull.sh on the laptop)
#
# Flags: --dry-run (print the commands instead of running them), --n N (number of RFD4 designs), --search, --log (+ --title/--goal/--parent).
# Everything you may want to change is an environment variable below, or lives in settings.py.
#
# STATUS OF THIS SCRIPT: `check`, `shim`, `example` and `run-rfd4` were run on the machine this workflow was built on. The installing steps
# (`af3`, `rfd4`) reproduce the commands used there, but were not re-run from an empty machine; use --dry-run first and read what they do.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
TOOLS="${TOOLS:-$HOME/tools}"
PYTHON="${PYTHON:-python3}"                                   # any python3: only used to read settings.py
AF3_ENV="${AF3_ENV:-af3}"
AF3_REPO="${AF3_REPO:-https://github.com/google-deepmind/alphafold3.git}"
AF3_COMMIT="${AF3_COMMIT:-5e23d45bff3a4c1993516ead2ec848cb42d64c1a}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-$HOME/af3_models}"            # must contain af3.bin.zst
AF3_WEIGHTS_SRC="${AF3_WEIGHTS_SRC:-}"                        # path or URL of af3.bin.zst, obtained from DeepMind under their terms
RFD4_VENV="${RFD4_VENV:-$HOME/rfd4_venv}"
RFD4_BASE_ENV="${RFD4_BASE_ENV:-rfd3_env}"                    # the venv is built ON TOP of this conda env (--system-site-packages): do not delete it
RFD4_PACKAGE="${RFD4_PACKAGE:-rfd4_proteina==0.0.3}"
RFD4_BASE_PIP="${RFD4_BASE_PIP:-rc-foundry==0.2.0 atomworks==2.2.1}"   # what the base env held when RFD4 was installed here
OF3_MODEL_DIR="${OF3_MODEL_DIR:-$HOME/of3_models}"
PIPELINE_PY="${PIPELINE_PY:-$CONDA_ROOT/envs/rfd_bw/bin/python}"      # needs numpy + biopython (runs design_pipeline.py)

DRY=0; N=""; SEARCH=0; STEP=""; LOG=0; TITLE=""; GOAL=""; PARENT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;; --search) SEARCH=1 ;; --n) N="$2"; shift ;; --log) LOG=1 ;;
    --title) TITLE="$2"; shift ;; --goal) GOAL="$2"; shift ;; --parent) PARENT="$2"; shift ;;
    -h|--help|help) STEP=help ;; *) STEP="${STEP:-$1}" ;;
  esac; shift
done
STEP="${STEP:-check}"

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '  [ ok ] %s\n' "$*"; }
miss() { printf '  [MISS] %s\n         -> %s\n' "$1" "$2"; }
run()  { printf '  + %s\n' "$*"; [ "$DRY" = 1 ] || "$@"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
cfg()  { "$PYTHON" - "$1" "$HERE/settings.py" <<'PY'
import runpy, sys
ns = runpy.run_path(sys.argv[2]); v = ns["SETTINGS"].get(sys.argv[1], ns.get(sys.argv[1], ""))
print("" if v is None else v)
PY
}
conda_run() { [ -x "$CONDA_ROOT/bin/conda" ] || die "conda not found at $CONDA_ROOT (set CONDA_ROOT)"; run "$CONDA_ROOT/bin/conda" "$@"; }
af3_py()    { echo "$CONDA_ROOT/envs/$AF3_ENV/bin/python"; }
gpu_cc()    { nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null || true; }

# ------------------------------------------------------------------------------------------------------------------------------
step_check() {
  say "settings.py"
  ROUTE_NOW="$(cfg ROUTE)"; ok "loads; ROUTE=$ROUTE_NOW (af2 | af3 | rfd4_af3), output_dir=$(cfg output_dir)"
  say "GPU (AlphaFold 3 needs compute capability >= 8.0)"
  if [ -n "$(gpu_cc)" ]; then gpu_cc | while IFS=, read -r i n c; do
      if "$PYTHON" -c "import sys; sys.exit(0 if float('$c'.strip())>=8.0 else 1)"; then ok "GPU $i $n  cc $c  (can run AF3)"; else printf '  [ -- ] GPU %s %s  cc %s  (too old for AF3; AF2 / RFdiffusion only)\n' "$i" "$n" "$c"; fi; done
  else miss "no nvidia-smi output" "install the NVIDIA driver; on WSL use /usr/lib/wsl/lib/nvidia-smi"; fi

  say "AF2 route (what you run today)"
  for e in af2_bw rfd_bw mpnn; do [ -d "$CONDA_ROOT/envs/$e" ] && ok "conda env $e" || miss "conda env $e" "see the dl_binder_design / RFdiffusion install notes"; done
  for t in RFdiffusion ProteinMPNN dl_binder_design; do [ -d "$TOOLS/$t" ] && ok "$TOOLS/$t" || miss "$TOOLS/$t" "clone it into $TOOLS"; done

  say "AF3 route   (./setup.sh af3)"
  [ -x "$(af3_py)" ] && ok "conda env $AF3_ENV" || miss "conda env $AF3_ENV" "./setup.sh af3"
  [ -d "$TOOLS/alphafold3/.git" ] && ok "$TOOLS/alphafold3 at $(git -C "$TOOLS/alphafold3" rev-parse --short HEAD)" || miss "$TOOLS/alphafold3" "./setup.sh af3"
  if [ -x "$(af3_py)" ]; then "$(af3_py)" -c "import alphafold3, jax; print('  [ ok ] alphafold3 imports; jax', jax.__version__)" 2>/dev/null || miss "alphafold3 does not import in env $AF3_ENV" "./setup.sh af3"; fi
  [ -x "$CONDA_ROOT/envs/$AF3_ENV/bin/build_data" ] && ok "build_data is installed" || miss "build_data" "./setup.sh af3"
  [ -f "$AF3_MODEL_DIR/af3.bin.zst" ] && ok "AF3 parameters: $AF3_MODEL_DIR/af3.bin.zst ($(du -h "$AF3_MODEL_DIR/af3.bin.zst" | cut -f1)); the Pairformer is inside this file" \
     || miss "AF3 parameters ($AF3_MODEL_DIR/af3.bin.zst)" "AF3_WEIGHTS_SRC=<path or URL> ./setup.sh af3   (obtain the file from DeepMind under their terms)"

  say "RFD4 route   (./setup.sh rfd4)"
  [ -x "$RFD4_VENV/bin/rfd4-generate" ] && ok "RFD4 venv: $RFD4_VENV" || miss "RFD4 venv ($RFD4_VENV)" "./setup.sh rfd4"
  [ -x "$CONDA_ROOT/envs/$RFD4_BASE_ENV/bin/python" ] && ok "base env $RFD4_BASE_ENV (the venv depends on it)" || miss "base env $RFD4_BASE_ENV" "./setup.sh rfd4"
  ls "$HOME/.cache/huggingface/hub" 2>/dev/null | grep -q "RFD4-Proteina" && ok "RFD4 weights in the Hugging Face cache" || miss "RFD4 weights" "./setup.sh rfd4"
  [ -x "$HERE/rfd4/shim/apptainer" ] && ok "apptainer shim in the repo (rfd4/shim)" || miss "rfd4/shim/apptainer" "git checkout the repo again"
  command -v apptainer >/dev/null 2>&1 && ok "a real apptainer is installed too (the shim is only needed without it)" || ok "no real apptainer: the shim stands in for it (./setup.sh shim)"
  [ -f work/rfd4_input.cif ] && [ -f work/target_template.cif ] && ok "CIF files in work/" || miss "CIF files in work/" "./setup.sh example"

  say "OpenFold3 reward   (./setup.sh openfold3, optional)"
  if [ -f "$OF3_MODEL_DIR/of3_ported_weights.provenance.json" ]; then ok "converted OpenFold3 weights in $OF3_MODEL_DIR"; else miss "converted OpenFold3 weights" "see ./setup.sh openfold3 (not installed by this script)"; fi

  say "next"
  [ -x "$(af3_py)" ] && [ -f "$AF3_MODEL_DIR/af3.bin.zst" ] || echo "  1. ./setup.sh af3"
  [ -x "$RFD4_VENV/bin/rfd4-generate" ] || echo "  2. ./setup.sh rfd4"
  [ -f work/rfd4_input.cif ] || echo "  3. ./setup.sh example"
  echo "  then:  ./setup.sh run-rfd4 --n 50          (add --search for reward-guided search)"
  echo "         ROUTE=af3 $PIPELINE_PY design_pipeline.py     (RFdiffusion + MPNN + AF3 instead of AF2)"
}

# ------------------------------------------------------------------------------------------------------------------------------
step_af3() {
  say "AlphaFold 3 (local)"
  command -v gcc >/dev/null && command -v g++ >/dev/null || die "gcc/g++ are needed to build AF3 (sudo apt install build-essential)"
  [ -x "$CONDA_ROOT/envs/$AF3_ENV/bin/python" ] || conda_run create -n "$AF3_ENV" python=3.12 -y
  if [ ! -d "$TOOLS/alphafold3/.git" ]; then run mkdir -p "$TOOLS"; run git clone "$AF3_REPO" "$TOOLS/alphafold3"; fi
  run git -C "$TOOLS/alphafold3" checkout "$AF3_COMMIT"
  run "$(af3_py)" -m pip install --upgrade pip
  ( cd "$TOOLS/alphafold3" && run "$(af3_py)" -m pip install . )
  run "$CONDA_ROOT/envs/$AF3_ENV/bin/build_data"                   # one-time: builds AF3's chemical-components data
  run mkdir -p "$AF3_MODEL_DIR" "$HOME/.cache/jax_af3"
  if [ ! -f "$AF3_MODEL_DIR/af3.bin.zst" ]; then
    [ -n "$AF3_WEIGHTS_SRC" ] || die "AF3 parameters missing. Get af3.bin.zst from DeepMind (you must accept their terms), then:  AF3_WEIGHTS_SRC=<path or URL> ./setup.sh af3"
    case "$AF3_WEIGHTS_SRC" in http*) run curl -L --fail -o "$AF3_MODEL_DIR/af3.bin.zst" "$AF3_WEIGHTS_SRC" ;; *) run cp "$AF3_WEIGHTS_SRC" "$AF3_MODEL_DIR/af3.bin.zst" ;; esac
  fi
  say "done. Try:  ROUTE=af3 $PIPELINE_PY design_pipeline.py   (RFdiffusion + MPNN + AF3), or go on with ./setup.sh rfd4"
}

step_rfd4() {
  say "RFD4-Proteina"
  if [ ! -x "$CONDA_ROOT/envs/$RFD4_BASE_ENV/bin/python" ]; then
    echo "  base env $RFD4_BASE_ENV is missing; creating it (the venv sits on top of it and borrows its packages)"
    conda_run create -n "$RFD4_BASE_ENV" python=3.12 -y
    # shellcheck disable=SC2086
    run "$CONDA_ROOT/envs/$RFD4_BASE_ENV/bin/python" -m pip install $RFD4_BASE_PIP
  fi
  [ -x "$RFD4_VENV/bin/python" ] || run "$CONDA_ROOT/envs/$RFD4_BASE_ENV/bin/python" -m venv --system-site-packages "$RFD4_VENV"
  run "$RFD4_VENV/bin/python" -m pip install "$RFD4_PACKAGE"
  run "$RFD4_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$(cfg rfd4_hf_repo)", revision="$(cfg rfd4_hf_revision)", allow_patterns=["config.json", "model.safetensors"]))
PY
  say "done. Next: ./setup.sh example   then   ./setup.sh run-rfd4 --n 50"
}

shim_env() {   # the environment RFD4's AF3 oracle needs; the fake apptainer then runs the worker with the af3 env
  export PATH="$HERE/rfd4/shim:$PATH" AF3_IMAGE="${AF3_IMAGE:-dummy.sif}" AF3_MODEL_DIR="${1:-$AF3_MODEL_DIR}" \
         AF3_JAX_CACHE_DIR="$HOME/.cache/jax_af3" AF3_FLASH_ATTENTION_IMPLEMENTATION=xla AF3_ENV_PYTHON="$(af3_py)"
}

step_shim() {
  say "native-AF3 stand-in for Apptainer (reward-guided search)"
  [ -x "$HERE/rfd4/shim/apptainer" ] || die "rfd4/shim/apptainer is missing or not executable"
  [ -x "$(af3_py)" ] || die "no $AF3_ENV env: run ./setup.sh af3 first"
  "$(af3_py)" -c "import alphafold3, tokamax, absl" || die "the $AF3_ENV env lacks alphafold3 / tokamax / absl"
  ok "RFD4's worker will run inside the $AF3_ENV env; the one container-only argument (of3_weights) is dropped by rfd4/shim/patched_worker.py"
  echo "  limits: stock AF3 only (no OpenFold3); a tokamax autotune traceback in the log is harmless (RFD4 falls back to default kernels)."
  echo "  ./setup.sh run-rfd4 --search sets these for you; to use the shim by hand:"
  echo "    export PATH=$HERE/rfd4/shim:\$PATH AF3_IMAGE=dummy.sif AF3_MODEL_DIR=$AF3_MODEL_DIR AF3_JAX_CACHE_DIR=$HOME/.cache/jax_af3 AF3_FLASH_ATTENTION_IMPLEMENTATION=xla"
}

step_openfold3() {
  say "OpenFold3 as the reward model for the search (optional, NOT set up by this script)"
  local patched=no
  [ -x "$(af3_py)" ] && "$(af3_py)" -c "
import inspect
from alphafold3.data import featurisation as f
raise SystemExit(0 if 'of3_weights' in inspect.signature(f.featurise_input).parameters else 1)" 2>/dev/null && patched=yes
  [ "$patched" = yes ] && ok "this AF3 build understands of3_weights (patched build)" \
     || miss "a patched AF3 build" "RFD4's OpenFold3 oracle needs the AF3 fork inside RFD4's container image; the stock AF3 installed by ./setup.sh af3 refuses OpenFold3 weights"
  [ -f "$OF3_MODEL_DIR/of3_ported_weights.provenance.json" ] && ok "converted weights found in $OF3_MODEL_DIR" \
     || miss "converted OpenFold3 weights ($OF3_MODEL_DIR/of3_ported_weights.provenance.json)" "convert the OpenFold3 parameters with the converter that ships with RFD4's container, into $OF3_MODEL_DIR"
  if [ "$patched" = yes ] && [ -f "$OF3_MODEL_DIR/of3_ported_weights.provenance.json" ]; then
    echo "  both present (UNTESTED here): set \"rfd4_oracle_model_dir\" in settings.py to $OF3_MODEL_DIR and run ./setup.sh run-rfd4 --search."
    echo "  Keep the final check on AF3 (af3_model_dir) so the model that guides the search is not the one that grades it."
  else
    echo "  Until then the search uses AF3 as its reward and the final check also uses AF3: they are not independent. Say so when you report results."
  fi
}

# ------------------------------------------------------------------------------------------------------------------------------
step_example() {
  say "CIF files for the target in settings.py  ->  work/"
  local PY; PY="$(cfg rfd4_python)"
  [ -x "$PY" ] || die "needs gemmi: run ./setup.sh rfd4 first (or set rfd4_python in settings.py)"
  run mkdir -p work
  run "$PY" make_af3_template.py --pdb "$(cfg TARGET_PDB)" --chain "$(cfg TARGET_CHAIN)" --ranges "$(cfg TARGET_RANGES)" --out work/target_template.cif
  run "$PY" make_rfd4_input.py --pdb "$(cfg TARGET_PDB)" --chain "$(cfg TARGET_CHAIN)" --ranges "$(cfg TARGET_RANGES)" \
        --hotspots "$(cfg HOTSPOTS)" --binder "$(cfg BINDER_LENGTH)" --out_dir work --check
}

step_run_rfd4() {
  local PY OUT NDES GEN SAMPLES SEQ
  PY="$(cfg rfd4_python)"; OUT="$(cfg output_dir)"; NDES="${N:-$(cfg rfd4_num_designs)}"; GEN="$OUT/rfd4_generated"
  [ -x "$(cfg rfd4_bin)" ] || die "RFD4 is not installed: ./setup.sh rfd4"
  [ -f work/rfd4_input.cif ] && [ -f work/target_template.cif ] || step_example
  [ "$(cfg ROUTE)" = rfd4_af3 ] || die "settings.py ROUTE is '$(cfg ROUTE)'; run this with ROUTE=rfd4_af3 (the default)"
  [ -e "$OUT/results_af3.csv" ] || [ ! -e "$OUT/1_rfdiffusion" ] || die "$OUT already holds a run; set a new output_dir (old files would mix into the new results)"
  run mkdir -p "$OUT"
  # facts about HOW this run was made, for log_iteration.py (the AF3 stage records its own settings in run_info.json)
  [ "$DRY" = 1 ] || "$PYTHON" - "$OUT/run_meta.json" "$NDES" "$SEARCH" "$(cfg rfd4_hf_revision)" "$(cfg af3_diffusion_samples)" <<'PY'
import json, sys, datetime
p, n, search, rev, samples = sys.argv[1:6]
json.dump({"route": "rfd4_af3", "generator": "RFD4-Proteina" + (" + reward-guided beam search" if search == "1" else ""),
           "n_designs_requested": int(n), "rfd4_weights_revision": rev[:12],
           "search_reward": "AF3 pLDDT of the designed chain (stock AF3 via the apptainer shim)" if search == "1" else "none",
           "checker": f"AF3 with target template, {samples} diffusion samples per design",
           "created": datetime.datetime.now().isoformat(timespec="seconds")}, open(p, "w"), indent=1)
PY
  say "1/4 RFD4 generates $NDES designs$([ "$SEARCH" = 1 ] && echo ' with reward-guided search')"
  local cmd=("$(cfg rfd4_bin)" inputs="$HERE/work/rfd4_spec.yaml" hf_ckpt_path="$(cfg rfd4_hf_repo)" hf_ckpt_revision="$(cfg rfd4_hf_revision)"
             num_replicates="$NDES" out_dir="$GEN" compile_lengths=null)
  # rfd4-generate drops hydra_outputs/ and runs/ in the folder it starts from: start it inside the output folder, not the repo
  inwd() { [ "$DRY" = 1 ] || { mkdir -p "$1" && cd "$1"; }; }
  if [ "$SEARCH" = 1 ]; then
    cmd+=(inference_experiment=search "hydra.searchpath=[file://$HERE/rfd4/hydra]")
    ( inwd "$OUT/rfd4_cwd"; shim_env "$(cfg rfd4_oracle_model_dir)"; CUDA_VISIBLE_DEVICES="$(cfg rfd4_gpu)" HF_HUB_OFFLINE=1 run "${cmd[@]}" )
    run "$PY" rfd4/select_best.py "$GEN" "$GEN/best"; SAMPLES="$GEN/best/inference_samples"
  else
    cmd+=('~metrics')
    ( inwd "$OUT/rfd4_cwd"; CUDA_VISIBLE_DEVICES="$(cfg rfd4_gpu)" HF_HUB_OFFLINE=1 run "${cmd[@]}" ); SAMPLES="$GEN/inference_samples"
  fi
  say "2/4 convert to the layout the AF3 stage reads (refuses a design whose target changed)"
  run "$PY" rfd4_to_af3_jobs.py --rfd4_samples "$SAMPLES" --out "$OUT" --target_seq "$(cat work/rfd4_target_seq.txt)"
  say "3/4 AlphaFold 3 check (target template, $(cfg af3_diffusion_samples) samples per design)"
  run "$PIPELINE_PY" design_pipeline.py --settings settings.py
  say "4/4 report"
  run "$PY" rfd4_af3_report.py --run "$OUT" --hotspots "$(cfg HOTSPOTS)" --target_ranges "$(cfg TARGET_RANGES)"
  local t="${TITLE:-RFD4 run $(date +%F): $NDES designs$([ "$SEARCH" = 1 ] && echo ', reward-guided search')}"
  local extra=(--title "$t" --target "$(basename "$(cfg TARGET_PDB)" .pdb)")
  [ -n "$GOAL" ] && extra+=(--goal "$GOAL"); [ -n "$PARENT" ] && extra+=(--parent "$PARENT")
  if [ "$LOG" = 1 ]; then
    say "queueing result notes for the Obsidian vault (the scheduled puller on the laptop copies them)"
    run "$PYTHON" log_iteration.py --run "$OUT" --outbox "$(cfg vault_outbox)" "${extra[@]}"
  else
    say "draft notes (kept here, NOT sent to the vault; add --log to queue them) -> $OUT/iteration_note/binder-design/"
    run "$PYTHON" log_iteration.py --run "$OUT" "${extra[@]}"
  fi
}

case "$STEP" in
  check) step_check ;; af3) step_af3 ;; rfd4) step_rfd4 ;; shim) step_shim ;; openfold3) step_openfold3 ;;
  example) step_example ;; run-rfd4) step_run_rfd4 ;;
  help) awk 'NR>1 && /^#/ {print substr($0,3)} NR>1 && !/^#/ {exit}' "$0" ;;
  *) die "unknown step '$STEP' (try ./setup.sh help)" ;;
esac
