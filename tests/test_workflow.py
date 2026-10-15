"""
Test flow.  Run:  python tests/test_workflow.py     (a few seconds, no GPU, no real tools)

  Test 1: the RMSD maths gives the right answer on points we moved by a known amount.
  Test 2: the main route (RFdiffusion -> ProteinMPNN -> AF2 -> score) runs end to end with fake tools,
          prepares the AF2 inputs correctly, and the RMSD matches the exact error the fake AF2 secretly added.
  Test 3: the optional AlphaFold Server route (export jobs, import website-style zips, score) works the same way.
  Test 4: two workers (two GPUs) share the work and give the same answers as one worker.
  Test 5: "copies" (several workers on ONE GPU) gives the same answers too.
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
from design_pipeline import make_fitter, read_csv, rmsd, workers_for    # noqa: E402
from fake_tools import helix, write_pdb                                 # noqa: E402

work = HERE / "_work"
shutil.rmtree(work, ignore_errors=True)
work.mkdir()
PIPELINE = str(HERE.parent / "design_pipeline.py")


def pipeline(settings, *args):
    subprocess.run([sys.executable, PIPELINE, "--settings", str(settings), *args], check=True)


# ---- Test 1: RMSD maths --------------------------------------------------------------------
pts = np.random.default_rng(0).normal(size=(50, 3)) * 10
q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
q *= np.sign(np.linalg.det(q))
moved = pts @ q + [5, -3, 8]                                     # same shape, new position/orientation
assert rmsd(make_fitter(moved, pts)(moved), pts) < 1e-6, "identical shapes should give RMSD ~ 0"
shifted = pts.copy()
shifted[:, 0] += 2.0                                             # every point 2 A away (no fitting)
assert abs(rmsd(shifted, pts) - 2.0) < 1e-9, "uniform 2 A shift should give RMSD = 2"
print("Test 1 passed: RMSD maths")

# ---- Test 2: main route end to end ---------------------------------------------------------
names = ["ALA", "LEU", "GLU", "LYS"]                              # toy target = 40-residue helix
write_pdb(work / "toy_target.pdb", {"A": [(names[i % 4], xyz) for i, xyz in enumerate(helix(40))]})
run = work / "run"
pipeline(HERE / "settings_test.py", "--stages", "rfd,mpnn,af2,score_af2")
rows = read_csv(run / "results_af2.csv")
assert len(rows) == 3 * 2, f"expected 6 results, got {len(rows)}"
for r in rows:
    assert float(r["pae_interaction"]) > 0 and 0 <= float(r["plddt_binder"]) <= 100 and float(r["rmsd_binder"]) >= 0
    assert r["pass"] in ("PASS", "fail") and set(r["binder_seq"]) <= set("ACDEFGHIKLMNPQRSTVWY")
first = (run / "3_af2_inputs" / "w0" / "design_0_s1.pdb").read_text().splitlines()
res_numbers = [(l[21], int(l[22:26])) for l in first if l.startswith("ATOM")]
assert res_numbers[0][0] == "A", "binder must be the first chain"
assert len({n for _, n in res_numbers}) == len(res_numbers), "residue numbers must be unique across both chains"
for r in rows:
    truth = json.loads(next((run / "3_af2_outputs").glob(f"w*/{r['job']}_truth.json")).read_text())["expected_rmsd_binder"]
    print(f"  {r['job']}: expected ~{truth:.2f} A, pipeline says {float(r['rmsd_binder']):.2f} A")
    assert abs(float(r["rmsd_binder"]) - truth) < 0.02, "RMSD does not match the known error"
print("Test 2 passed: main route ran end to end,", len(rows), "rows; inputs prepared correctly; RMSD matches")

# ---- Test 3: AlphaFold Server route ---------------------------------------------------------
afs_run, zips = work / "run_afs", work / "afs_zips"
shutil.copytree(run, afs_run, ignore=shutil.ignore_patterns("3_af2_*", "results*.csv"))
zips.mkdir()
(work / "settings_afs.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S["output_dir"] = r"{afs_run}"
SETTINGS = S
""")
pipeline(work / "settings_afs.py", "--stages", "afs_export")
jobs_json = json.loads((afs_run / "afserver_jobs.json").read_text())
assert len(jobs_json) == 6 and jobs_json[0]["dialect"] == "alphafoldserver" and len(jobs_json[0]["sequences"]) == 2
subprocess.run([sys.executable, str(HERE / "fake_tools.py"), "afserver", "--jobs", str(afs_run / "afserver_jobs.json"),
                "--run_dir", str(afs_run), "--out_zips", str(zips)], check=True)       # the "website"
pipeline(work / "settings_afs.py", "--stages", "afs_import,score_afs", "--afserver_zips", str(zips))
rows3 = read_csv(afs_run / "results_afserver.csv")
assert len(rows3) == 6 and all(0 <= float(r["iptm"]) <= 1 for r in rows3)
for r in rows3:
    truth = json.loads((zips / f"{r['job']}_truth.json").read_text())["expected_rmsd_binder"]
    assert abs(float(r["rmsd_binder"]) - truth) < 0.02, "server-route RMSD does not match the known error"
print("Test 3 passed: AlphaFold Server route (export, import, score) works; RMSD matches")

# ---- Test 4: both GPUs at once (2 workers per tool, weights 2:1) must give the SAME answers as one worker ----
(work / "settings_par.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
def two(tool):
    return [{{"cmd": S[tool + "_cmd"], "env": {{"WORKER": "0"}}, "weight": 2}},
            {{"cmd": S[tool + "_cmd"], "env": {{"WORKER": "1"}}, "weight": 1}}]
S.update(output_dir=r"{work / 'run_par'}", rfdiffusion_workers=two("rfdiffusion"), mpnn_workers=two("mpnn"), af2_workers=two("af2"))
SETTINGS = S
""")
pipeline(work / "settings_par.py", "--stages", "rfd,mpnn,af2,score_af2")
par = work / "run_par"
assert (par / "3_af2_outputs" / "w0").exists() and (par / "3_af2_outputs" / "w1").exists(), "both workers must be used"
assert len(list((par / "1_rfdiffusion").glob("design_*.pdb"))) == 3, "split designs must not overwrite each other"
serial = {r["job"]: r for r in rows}
parallel = {r["job"]: r for r in read_csv(par / "results_af2.csv")}
assert serial.keys() == parallel.keys() and all(
    serial[j][c] == parallel[j][c] for j in serial for c in ("pae_interaction", "plddt_binder", "rmsd_binder", "binder_seq")
), "parallel run must match the single-worker run"
print("Test 4 passed: two workers share the work and give identical results")

# ---- Test 5: "copies" (several workers on ONE GPU) -------------------------------------------
w = workers_for({"x_workers": [{"cmd": ["a"], "env": {"CUDA_VISIBLE_DEVICES": "0"}, "weight": 2, "copies": 3}]}, "x")
assert len(w) == 3 and all(x["weight"] == 2 and x["env"]["CUDA_VISIBLE_DEVICES"] == "0"
                           and x["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for x in w), "copies must expand to 3 workers"
assert workers_for({"x_workers": [{"cmd": ["a"], "env": {}, "weight": 1}]}, "x")[0]["env"] == {}, "1 worker: env untouched"
assert workers_for({"x_cmd": ["a"]}, "x")[0]["weight"] == 1, "old single-GPU keys still work"
(work / "settings_copies.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_copies'}", af2_workers=[{{"cmd": S["af2_cmd"], "env": {{}}, "weight": 1, "copies": 3}}])
SETTINGS = S
""")
cp = work / "run_copies"
shutil.copytree(run, cp, ignore=shutil.ignore_patterns("3_af*", "results*.csv"))
pipeline(work / "settings_copies.py", "--stages", "af2,score_af2")
assert sorted(p.name for p in (cp / "3_af2_outputs").iterdir()) == ["w0", "w1", "w2"], "3 copies -> 3 worker folders"
same = {r["job"]: r for r in read_csv(cp / "results_af2.csv")}
assert same.keys() == serial.keys() and all(same[j]["rmsd_binder"] == serial[j]["rmsd_binder"] for j in serial)
print("Test 5 passed: 'copies' runs several workers on one GPU with the same results")
print("\nALL TESTS PASSED")
