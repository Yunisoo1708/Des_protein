# rfd4/ -- reward-guided search for RFD4

Everything else about the RFD4 route is in `../setup.sh` (`rfd4`, `example`, `run-rfd4`) and `../settings.py`. This folder holds the three pieces
that only the reward-guided search (`./setup.sh run-rfd4 --search`) needs:

| File | What it does |
|---|---|
| `hydra/inference_experiment/search.yaml` | the search: beam of 2 with 2 branches; at steps 30 and 60 of 60 the partly denoised designs are folded by an AF3 oracle and scored by the pLDDT of the designed chain (`oracle_ppi.plddt_coi`), and only the best candidates continue. Edit this file to score `oracle_ppi.min_ipae_coi` (interface error, lower is better; use `weight: -1.0`) or to change the beam. Passed to RFD4 through `hydra.searchpath`. |
| `shim/apptainer`, `shim/patched_worker.py` | a fake `apptainer`: RFD4's oracle normally runs AF3 inside `apptainer run af3-open.sif`; the shim runs RFD4's AF3 worker with the `af3` conda env instead and drops the one argument (`of3_weights`) only RFD4's patched AF3 understands. |
| `select_best.py` | a search leaves 8 candidates per design; this keeps the one with the highest search reward, under the design's name, so each design is checked once. |

**Limits.** Stock AF3 only: OpenFold3 weights are refused (`./setup.sh openfold3` says what is missing). A tokamax autotune traceback in the log is harmless.
Cost: about 264 s per design on one RTX 5060 Ti (50 designs ≈ 2.5 h) against about 21 s per design without search.
The AF3 oracle that guides the search and the AF3 check that grades it are the same predictor, so a pass is not independent evidence.
