"""
Test flow.  Run:  python tests/test_workflow.py

  Test 1: the RMSD maths gives the right answer on points we moved by a known amount.
  Test 2: the whole pipeline runs end-to-end with fake tools and writes results.csv.
  Test 3: the RMSD the pipeline reports matches the exact error the fake AF3 secretly added.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from design_pipeline import make_fitter, read_csv, rmsd          # noqa: E402
from fake_tools import helix, write_pdb                          # noqa: E402

work = HERE / "_work"
shutil.rmtree(work, ignore_errors=True)
work.mkdir()

# ---- Test 1: RMSD maths --------------------------------------------------------------------
pts = np.random.default_rng(0).normal(size=(50, 3)) * 10
q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
q *= np.sign(np.linalg.det(q))
moved = pts @ q + [5, -3, 8]                                     # same shape, new position/orientation
assert rmsd(make_fitter(moved, pts)(moved), pts) < 1e-6, "identical shapes should give RMSD ~ 0"
shifted = pts.copy()
shifted[:, 0] += 2.0                                             # every point 2 Å away (no fitting)
assert abs(rmsd(shifted, pts) - 2.0) < 1e-9, "uniform 2 Å shift should give RMSD = 2"
print("Test 1 passed: RMSD maths")

# ---- Test 2: full pipeline with fake tools -------------------------------------------------
names = ["ALA", "LEU", "GLU", "LYS"]                              # toy target = 40-residue helix
write_pdb(work / "toy_target.pdb", {"A": [(names[i % 4], xyz) for i, xyz in enumerate(helix(40))]})
subprocess.run([sys.executable, str(HERE.parent / "design_pipeline.py"),
                "--settings", str(HERE / "settings_test.py")], check=True)
rows = read_csv(work / "run" / "results.csv")
assert len(rows) == 3 * 2, f"expected 6 results, got {len(rows)}"
for r in rows:
    assert 0 <= float(r["iptm"]) <= 1 and 0 <= float(r["plddt_binder"]) <= 100 and float(r["rmsd_binder"]) >= 0
    assert r["pass"] in ("PASS", "fail") and set(r["binder_seq"]) <= set("ACDEFGHIKLMNPQRSTVWY")
print("Test 2 passed: pipeline ran end to end,", len(rows), "rows in results.csv")

# ---- Test 3: reported RMSD vs the error the fake AF3 added ---------------------------------
for r in rows:
    truth = json.loads((work / "run" / "3_af3_outputs" / r["job"] / "fake_truth.json").read_text())
    expected, got = truth["expected_rmsd_binder"], float(r["rmsd_binder"])
    print(f"  {r['job']}: expected ~{expected:.2f} Å, pipeline says {got:.2f} Å")
    assert abs(got - expected) < 0.02, "RMSD does not match the known error"
print("Test 3 passed: RMSD matches the known error")

# ---- Test 4: AlphaFold Server route (export jobs, import website-style zips, same scores) ---
import zipfile
run, run2, zips = work / "run", work / "run_afs", work / "afs_zips"
shutil.copytree(run, run2, ignore=shutil.ignore_patterns("3_af3_*", "results.csv"))
zips.mkdir()
for r in rows:                                                   # pretend the website returned these zips
    n, src = r["job"], run / "3_af3_outputs" / r["job"]
    with zipfile.ZipFile(zips / f"fold_{n}.zip", "w") as z:
        z.write(src / f"{n}_model.cif", f"fold_{n}_model_0.cif")
        z.write(src / f"{n}_summary_confidences.json", f"fold_{n}_summary_confidences_0.json")
        z.write(src / f"{n}_confidences.json", f"fold_{n}_full_data_0.json")
base = [sys.executable, str(HERE.parent / "design_pipeline.py"), "--settings", str(HERE / "settings_test.py")]
import re
cfg = (HERE / "settings_test.py").read_text().replace('"run"', '"run_afs"')   # point output_dir at run_afs
(HERE / "_work" / "settings_afs.py").write_text(cfg.replace("HERE = Path(__file__).parent", f"HERE = Path(r'{HERE}')"))
afs = base[:3] + [str(work / "settings_afs.py")]
subprocess.run(afs + ["--stages", "afs_export"], check=True)
jobs_json = json.loads((run2 / "afserver_jobs.json").read_text())
assert len(jobs_json) == 6 and jobs_json[0]["dialect"] == "alphafoldserver" and len(jobs_json[0]["sequences"]) == 2
subprocess.run(afs + ["--stages", "afs_import,score", "--afserver_zips", str(zips)], check=True)
rows2 = {r["job"]: r for r in read_csv(run2 / "results.csv")}
assert all(rows2[r["job"]]["iptm"] == r["iptm"] and rows2[r["job"]]["rmsd_binder"] == r["rmsd_binder"] for r in rows)
print("Test 4 passed: AlphaFold Server export/import gives identical scores")

# ---- Test 5: AF2 initial-guess route (Bennett): input prep (binder first, unique numbering) + scoring ----
subprocess.run(base + ["--stages", "af2,score_af2"], check=True)
first = (run / "3_af2_inputs" / "w0" / "design_0_s1.pdb").read_text().splitlines()
res_numbers = [(l[21], int(l[22:26])) for l in first if l.startswith("ATOM")]
assert res_numbers[0][0] == "A", "binder must be the first chain"
assert len({n for _, n in res_numbers}) == len(res_numbers), "residue numbers must be unique across both chains"
rows5 = read_csv(run / "results_af2.csv")
assert len(rows5) == 6 and all(float(r["pae_interaction"]) > 0 for r in rows5)
for r in rows5:
    truth = json.loads(next((run / "3_af2_outputs").glob(f"w*/{r['job']}_truth.json")).read_text())["expected_rmsd_binder"]
    assert abs(float(r["rmsd_binder"]) - truth) < 0.02, "AF2-route RMSD does not match the known error"
print("Test 5 passed: AF2 initial-guess route")

# ---- Test 6: both GPUs at once (2 workers per tool, weights 2:1) must give the SAME answers as one worker ----
(work / "settings_par.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
def two(tool):
    return [{{"cmd": S[tool + "_cmd"], "env": {{"WORKER": "0"}}, "weight": 2}},
            {{"cmd": S[tool + "_cmd"], "env": {{"WORKER": "1"}}, "weight": 1}}]
S.update(output_dir=r"{work / 'run_par'}", rfdiffusion_workers=two("rfdiffusion"), mpnn_workers=two("mpnn"), af2_workers=two("af2"))
SETTINGS = S
""")
subprocess.run(base[:3] + [str(work / "settings_par.py"), "--stages", "rfd,mpnn,af2,score_af2"], check=True)
par = work / "run_par"
assert (par / "3_af2_outputs" / "w0").exists() and (par / "3_af2_outputs" / "w1").exists(), "both workers must be used"
assert len(list((par / "1_rfdiffusion").glob("design_*.pdb"))) == 3, "split designs must not overwrite each other"
serial = {r["job"]: r for r in read_csv(run / "results_af2.csv")}
parallel = {r["job"]: r for r in read_csv(par / "results_af2.csv")}
assert serial.keys() == parallel.keys() and all(
    serial[j][c] == parallel[j][c] for j in serial for c in ("pae_interaction", "plddt_binder", "rmsd_binder", "binder_seq")
), "parallel run must match the single-worker run"
print("Test 6 passed: two workers share the work and give identical results")
print("\nALL TESTS PASSED")
