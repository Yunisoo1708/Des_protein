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

# ---- Test 6: local AlphaFold 3 route (af3 -> score_af3), fake run_alphafold.py -----------------------------
af3_run = work / "run_af3"
shutil.copytree(run, af3_run, ignore=shutil.ignore_patterns("3_af*", "results*.csv"))
(work / "settings_af3.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S.update(output_dir=r"{af3_run}", af3_cmd=[r"{sys.executable}", r"{HERE / 'fake_tools.py'}", "af3"],
         af3_model_dir=r"{work / 'af3_models'}", af3_db_dir=None, max_pae_interaction=10)
SETTINGS = S
""")
pipeline(work / "settings_af3.py", "--stages", "af3,score_af3")
rows6 = read_csv(af3_run / "results_af3.csv")
assert len(rows6) == 6 and all(0 <= float(r["iptm"]) <= 1 for r in rows6)
for r in rows6:
    truth = json.loads((af3_run / "3_af3_outputs" / r["job"].lower() / f"{r['job'].lower()}_truth.json").read_text())
    assert abs(float(r["rmsd_binder"]) - truth["expected_rmsd_binder"]) < 0.02, "AF3-route RMSD does not match the known error"
    assert abs(float(r["pae_interaction"]) - truth["expected_pae_interaction"]) < 0.01, "pae_interaction does not match the known value"
    ok = float(r["iptm"]) >= 0.8 and float(r["plddt_binder"]) >= 80 and float(r["rmsd_binder"]) <= 2.0 and float(r["pae_interaction"]) <= 10
    assert r["pass"] == ("PASS" if ok else "fail"), "verdict must use iptm, plddt, rmsd AND pae_interaction"
assert not (af3_run / "results_afserver.csv").exists(), "AF3 route must not overwrite the AlphaFold Server table"
print("Test 6 passed: local AlphaFold 3 route (inputs, flags, collect, score incl. pae_interaction) works")

# ---- Test 7: target template on the AF3 route -----------------------------------------------------------------
from fake_tools import write_cif                                  # noqa: E402
tmpl = work / "target_template.cif"
write_cif(tmpl, "tmpl", {"A": [(names[i % 4], xyz) for i, xyz in enumerate(helix(40))]})   # 40 residues = the toy target
af3_t = work / "run_af3_tmpl"
shutil.copytree(run, af3_t, ignore=shutil.ignore_patterns("3_af*", "results*.csv"))
for tag, cif, out_dir in (("ok", tmpl, af3_t), ("wrong_length", None, work / "run_af3_badtmpl")):
    path = cif
    if cif is None:                                               # a template with 39 residues instead of 40
        path = work / "bad_template.cif"
        write_cif(path, "tmpl", {"A": [(names[i % 4], xyz) for i, xyz in enumerate(helix(39))]})
        shutil.copytree(run, out_dir, ignore=shutil.ignore_patterns("3_af*", "results*.csv"))
    (work / f"settings_af3_{tag}.py").write_text(f"""
import runpy
S = runpy.run_path(r"{work / 'settings_af3.py'}")["SETTINGS"]
S.update(output_dir=r"{out_dir}", af3_target_template_cif=r"{path}")
SETTINGS = S
""")
pipeline(work / "settings_af3_ok.py", "--stages", "af3,score_af3")
sent = json.loads(next((af3_t / "3_af3_inputs").glob("*.json")).read_text())["sequences"]
assert sent[0]["protein"]["templates"] == [] and len(sent[1]["protein"]["templates"]) == 1, "template must be on the target (B) only"
assert len(read_csv(af3_t / "results_af3.csv")) == 6
bad = subprocess.run([sys.executable, PIPELINE, "--settings", str(work / "settings_af3_wrong_length.py"), "--stages", "af3"],
                     capture_output=True, text=True)
assert bad.returncode != 0 and "39 residues but the target sequence has 40" in bad.stderr, "a wrong-length template must stop the run"
print("Test 7 passed: target template goes on chain B only; a wrong-length template is refused")

# ---- Test 8: iteration notes for the Obsidian vault (log_iteration.py)
run8, vault = work / "run_logged", work / "vault"
run8.mkdir()
(run8 / "results_af3.csv").write_text(
    "job,iptm,plddt_binder,plddt_complex,rmsd_binder,pae_interaction,mpnn_score,binder_seq,pass\n"
    "rfd4_0,0.82,90.4,86.3,1.23,6.62,NA,AAAAACCCCCDDDDD,PASS\n"
    "rfd4_1,0.31,70.0,70.0,30.0,20.0,NA,EEEEEFFFFFGGGGG,fail\n"
    "rfd4_2,0.12,60.0,60.0,45.0,25.0,NA,HHHHHIIIIIKKKKK,fail\n")
(run8 / "run_info.json").write_text(json.dumps({"invocations": [{"time": "t", "stages": ["af3", "score_af3"], "code_version": "abc1234",
                                                                   "settings": {"af3_diffusion_samples": 5, "binder_length": "50-80"}}]}))
LOG = str(HERE.parent / "log_iteration.py")
cmd8 = [sys.executable, LOG, "--run", str(run8), "--title", "Test iteration", "--goal", "G", "--vault", str(vault), "--parent", "earlier-run"]
subprocess.run(cmd8, check=True, capture_output=True)
base = vault / "binder-design"
notes = list((base / "iterations").glob("*.md"))
assert len(notes) == 1, notes
text = notes[0].read_text(encoding="utf-8")
assert "type: design-iteration" in text and "status: draft" in text and "n_designs: 3" in text and "n_pass: 1" in text, text[:400]
assert 'parent: "[[earlier-run]]"' in text and "abc1234" in text and "TODO" in text and "sha256" in text
slug = notes[0].stem
assert (base / "designs" / f"{slug}--rfd4_0.md").exists() and not (base / "designs" / f"{slug}--rfd4_1.md").exists(), "only the passing design gets a note"
assert (base / "data" / f"{slug}-results.csv").exists() and f"[[{slug}]]" in (base / "iterations.md").read_text(encoding="utf-8")
assert sorted(p.name for p in vault.iterdir()) == ["binder-design"], "nothing may be written outside binder-design/"
notes[0].write_text(text + "\nMY EDIT\n", encoding="utf-8")
subprocess.run(cmd8, check=True, capture_output=True)                         # run again: the edited note must survive
assert "MY EDIT" in notes[0].read_text(encoding="utf-8"), "an existing note must never be overwritten"
print("Test 8 passed: iteration notes (properties, passing-design note, results copy, index) are made and never overwrite edits")

# ---- Test 9: only runs started with --log are queued for the vault puller; DONE is written last and never re-queues a pulled run
outbox = work / "outbox"
shutil.rmtree(outbox, ignore_errors=True)
subprocess.run([sys.executable, str(HERE.parent / "design_pipeline.py"), "--settings", str(HERE / "settings_test.py"), "--stages", "score_af2"], check=True, capture_output=True)
assert not outbox.exists(), "a run without --log must not be queued"
(work / "settings_log.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S["vault_outbox"] = r"{outbox}"
SETTINGS = S
""")
logged = subprocess.run([sys.executable, str(HERE.parent / "design_pipeline.py"), "--settings", str(work / "settings_log.py"), "--stages", "score_af2",
                         "--log", "--title", "Logged run", "--goal", "G"], check=True, capture_output=True, text=True)
queued = [d for d in outbox.iterdir() if d.is_dir()]
assert len(queued) == 1 and (queued[0] / "DONE").exists() and (queued[0] / "binder-design" / "iterations").is_dir(), "a run started with --log must be queued with a DONE marker"
assert "Logged run" in next((queued[0] / "binder-design" / "iterations").glob("*.md")).read_text(encoding="utf-8")
(queued[0] / "PULLED").write_text("x")
(queued[0] / "DONE").unlink()
subprocess.run([sys.executable, str(HERE.parent / "design_pipeline.py"), "--settings", str(work / "settings_log.py"), "--stages", "score_af2", "--log", "--title", "Logged run"],
               check=True, capture_output=True)
assert not (queued[0] / "DONE").exists(), "a run that was already pulled must not be queued again"
print("Test 9 passed: --log queues the notes with a DONE marker; runs without --log and already-pulled runs are not queued")
print("\nALL TESTS PASSED")
