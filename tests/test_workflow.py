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

# ---- Test 6: RFdiffusion 3 route end to end (fake rfd3 -> convert -> ProteinMPNN -> AF2 -> score) --------------------
# The target range has a GAP (A21-A24 left out), like a real PDB with missing residues; RFD3 renumbers the target from
# 1, so the pipeline must put the ORIGINAL numbers back.
(work / "settings_rfd3.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd3'}", target_range="A5-20,A25-40", hotspots="A10,A30",
         rfd3_cmd=[S["rfdiffusion_cmd"][0], S["rfdiffusion_cmd"][1], "rfd3", "design"], rfd3_batch_size=2)
S.pop("rfdiffusion_cmd")
SETTINGS = S
""")
r3 = work / "run_rfd3"
pipeline(work / "settings_rfd3.py", "--stages", "rfd,mpnn,af2,score_af2")
designs = sorted((r3 / "1_rfdiffusion").glob("design_*.pdb"))
assert len(designs) == 3, f"expected 3 RFD3 backbones, got {len(designs)}"
from design_pipeline import read_chains                                        # noqa: E402
chains = read_chains(designs[0])
side = json.loads(designs[0].with_suffix(".json").read_text())
assert (side["binder_chain"], side["target_chain"]) == ("B", "A") and "rfd3_metrics" in side
assert len(chains["B"]) in range(20, 31) and len({n for n, _ in chains["B"]}) > 3, "binder should carry a real sequence"
numbers = [int(l[22:26]) for l in designs[0].read_text().splitlines() if l.startswith("ATOM") and l[21] == "A" and l[12:16].strip() == "CA"]
assert numbers == list(range(5, 21)) + list(range(25, 41)), f"original target numbering not restored: {numbers}"
rows6 = read_csv(r3 / "results_af2.csv")
assert len(rows6) == 6
for r in rows6:
    truth = json.loads(next((r3 / "3_af2_outputs").glob(f"w*/{r['job']}_truth.json")).read_text())["expected_rmsd_binder"]
    assert abs(float(r["rmsd_binder"]) - truth) < 0.02, "RFD3-route RMSD does not match the known error"
print("Test 6 passed: RFD3 route (convert, original numbering restored, binder found from the side file, scored)")

# ---- Test 7: refuses to mix runs (RFdiffusion and RFD3 both SKIP designs that already exist) -------------------------
again = subprocess.run([sys.executable, PIPELINE, "--settings", str(work / "settings_rfd3.py"), "--stages", "rfd"],
                       capture_output=True, text=True)
assert again.returncode != 0 and "NEW --output_dir" in (again.stdout + again.stderr), "must refuse a folder with old backbones"
print("Test 7 passed: re-running 'rfd' into a used folder is refused")

# ---- Test 8: RFD3 with 2 workers (two GPUs) shares the designs and keeps them apart ----------------------------------
(work / "settings_rfd3_par.py").write_text(f"""
import runpy
S = runpy.run_path(r"{work / 'settings_rfd3.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd3_par'}", num_backbones=5)
S["rfd3_workers"] = [{{"cmd": S["rfd3_cmd"], "env": {{"WORKER": "0"}}, "weight": 2}},
                     {{"cmd": S["rfd3_cmd"], "env": {{"WORKER": "1"}}, "weight": 1}}]
SETTINGS = S
""")
pipeline(work / "settings_rfd3_par.py", "--stages", "rfd")
p3 = work / "run_rfd3_par"
assert (p3 / "1_rfd3_raw" / "w0").exists() and (p3 / "1_rfd3_raw" / "w1").exists(), "both workers must be used"
made = sorted((p3 / "1_rfdiffusion").glob("design_*.pdb"))
assert len(made) == 5 and len({p.read_text() for p in made}) == 5, "5 distinct backbones expected"
print("Test 8 passed: RFD3 designs split across two workers, 5 distinct backbones")

# ---- Test 9: target glycosylation scan (stage "glyco") ---------------------------------------------------------------
# Toy target: residues 10-12 = N-E-T (a sequon), 20-22 = N-P-S (NOT a sequon: Pro), 30-32 = N-K-S (a sequon)
names = ["ALA"] * 40
names[9:12], names[19:22], names[29:32] = ["ASN", "GLU", "THR"], ["ASN", "PRO", "SER"], ["ASN", "LYS", "SER"]
write_pdb(work / "glyco_target.pdb", {"A": list(zip(names, helix(40)))})
from design_pipeline import step0_glyco, parse_netnglyc                       # noqa: E402
from Bio.SeqUtils import seq1                                                  # noqa: E402
S9 = {"target_pdb": str(work / "glyco_target.pdb"), "target_range": "A5-25,A28-40", "hotspots": "A8,A33"}
rows9 = step0_glyco(S9, work / "glyco_run")
assert [r["residue"] for r in rows9] == [10, 30] and [r["motif"] for r in rows9] == ["NET", "NKS"], "Pro motif must be skipped"
assert all(r["in_target_range"] for r in rows9)
assert 4.0 < rows9[0]["hotspot_dist_A"] < 7.0 and rows9[0]["nearest_hotspot"] == "A8", "distance residue 10 -> hotspot A8 (2 residues apart)"
assert rows9[0]["verdict"].startswith("candidate") and "A8" in rows9[0]["warning"], "a sequon next to a hotspot must warn"
assert (work / "glyco_run" / "0_glyco" / "target.fasta").read_text().splitlines()[1] == "".join(seq1(n) for n in names)
# a sequon left out of the kept range is reported as such
rows9b = step0_glyco({**S9, "target_range": "A5-25"}, work / "glyco_run_b")
assert [r["in_target_range"] for r in rows9b] == [True, False] and "outside the kept target range" in rows9b[1]["warning"]
# NetNGlyc result file (website or standalone), given back with --netnglyc_output
(work / "netnglyc.txt").write_text("""# NetNGlyc 1.0 toy output (format assumed)
target_chain_A       10 NET   0.6531  (8/9)  ++
target_chain_A       30 NKS   0.1210  (9/9)  --
""")
(work / "settings_glyco.py").write_text(f"SETTINGS = {S9!r}\n")
subprocess.run([sys.executable, PIPELINE, "--settings", str(work / "settings_glyco.py"), "--output_dir", str(work / "glyco_cli"),
                "--stages", "glyco", "--netnglyc_output", str(work / "netnglyc.txt")], check=True, capture_output=True)
cli = read_csv(work / "glyco_cli" / "0_glyco" / "glyco_sites.csv")               # the command-line route gives the same table
assert [r["netnglyc_potential"] for r in cli] == ["0.6531", "0.121"] and [r["verdict"] for r in cli] == ["likely", "possible"]
rows9c = step0_glyco({**S9, "netnglyc_output": str(work / "netnglyc.txt")}, work / "glyco_run_c")
assert [r["netnglyc_glycosylated"] for r in rows9c] == [True, False]
assert [r["verdict"] for r in rows9c] == ["likely", "possible"] and [r["support"] for r in rows9c] == ["2/2", "1/2"]
assert parse_netnglyc("nothing useful here", {1: 1}) == {}
print("Test 9 passed: sequon scan (Pro rule, numbering, hotspot distance, range check) and NetNGlyc import")

# ---- Test 10: ReGlyco scan merged into the table (fake `reglyco scan` that writes the real tool's scan.json layout) -----
FAKE = [sys.executable, str(HERE / "fake_tools.py"), "reglyco"]
rows10 = step0_glyco({**S9, "reglyco_cmd": FAKE, "netnglyc_output": str(work / "netnglyc.txt")}, work / "glyco_run_d")
assert [r["reglyco_accessible"] for r in rows10] == [True, False] and rows10[0]["reglyco_steric_score"] == 1.0
assert [r["support"] for r in rows10] == ["3/3", "1/3"] and [r["verdict"] for r in rows10] == ["likely", "possible"], rows10
assert (work / "glyco_run_d" / "0_glyco" / "reglyco_scan" / "scan.json").exists()
rows10b = step0_glyco({**S9, "reglyco_cmd": FAKE}, work / "glyco_run_e")               # ReGlyco alone (no NetNGlyc)
assert [r["support"] for r in rows10b] == ["2/2", "1/2"]
print("Test 10 passed: ReGlyco scan results merged with exposure and NetNGlyc into one verdict")

# ---- Test 11: RFD4-Proteina route (fake rfd4-generate): target -> CIF with one chain per piece, spec, convert, native sequences ----
# Same gap as Test 6 (A21-A24 left out). RFD4 refuses to bond across a gap, so each piece becomes its own chain (A, B) and the
# binder gets the next letter (C); the pipeline must read that back as target = A (original numbers) and binder = B, and must
# use RFD4's own sequences (no ProteinMPNN).
(work / "settings_rfd4.py").write_text(f"""
import runpy
S = runpy.run_path(r"{HERE / 'settings_test.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd4'}", target_range="A5-20,A25-40", hotspots="A10,A30", backbone_tool="rfd4",
         rfd4_cmd=S["rfdiffusion_cmd"][:2] + ["rfd4"], rfd4_batch_size=2, rfd4_ckpt="nvidia/RFD4-Proteina")
S.pop("rfdiffusion_cmd")
S["mpnn_cmd"] = ["this-must-not-run"]                 # proves ProteinMPNN is skipped
SETTINGS = S
""")
r4 = work / "run_rfd4"
pipeline(work / "settings_rfd4.py", "--stages", "rfd,mpnn,af2,score_af2")
spec = (r4 / "1_rfd4_raw" / "spec.yaml").read_text()
assert 'contig: "20-30, /0, A5-20, /0, B25-40"' in spec, spec
assert "C_HOT: {true: [{select: \"sel('A/*/[10]')\"}, {select: \"sel('B/*/[30]')\"}]}" in spec, spec     # hotspots follow their piece
assert "C_CRDGRP: {1: [" in spec and "C_SEQ: {true: [" in spec
from Bio.PDB.MMCIF2Dict import MMCIF2Dict                                        # noqa: E402
tc = MMCIF2Dict(str(r4 / "1_rfd4_raw" / "target.cif"))
assert sorted(set(tc["_atom_site.auth_asym_id"])) == ["A", "B"] and sorted({int(n) for n in tc["_atom_site.auth_seq_id"]}) == \
    list(range(5, 21)) + list(range(25, 41)), "target.cif: one chain per piece, original residue numbers"
assert tc["_atom_site.label_seq_id"] == tc["_atom_site.auth_seq_id"], "RFD4 reads label_seq_id: it must hold the original numbers too"
designs4 = sorted((r4 / "1_rfdiffusion").glob("design_*.pdb"))
assert len(designs4) == 3
side4 = json.loads(designs4[0].with_suffix(".json").read_text())
assert (side4["binder_chain"], side4["target_chain"]) == ("B", "A")
chains4 = read_chains(designs4[0])
numbers4 = [int(l[22:26]) for l in designs4[0].read_text().splitlines() if l.startswith("ATOM") and l[21] == "A"]
assert numbers4 == list(range(5, 21)) + list(range(25, 41)), f"original target numbering not restored: {numbers4}"
assert len(chains4["B"]) in range(20, 31) and len({n for n, _ in chains4["B"]}) > 3, "binder should carry RFD4's sequence"
seq4 = read_csv(r4 / "2_mpnn" / "sequences.csv")
assert [r["job"] for r in seq4] == ["design_0_native", "design_1_native", "design_2_native"]
assert all(r["binder_seq"] == "".join(seq1(n) for n, _ in read_chains(r["backbone"])["B"]) for r in seq4), "native sequence = the PDB's"
assert not list((r4 / "2_mpnn").glob("design_*")), "ProteinMPNN must not run for RFD4"
rows11 = read_csv(r4 / "results_af2.csv")
assert len(rows11) == 3
for r in rows11:
    truth = json.loads(next((r4 / "3_af2_outputs").glob(f"w*/{r['job']}_truth.json")).read_text())["expected_rmsd_binder"]
    assert abs(float(r["rmsd_binder"]) - truth) < 0.02, "RFD4-route RMSD does not match the known error"

# two workers must NOT repeat each other's designs (RFD4's default seed is fixed, so each worker gets its own seed)
(work / "settings_rfd4_par.py").write_text(f"""
import runpy
S = runpy.run_path(r"{work / 'settings_rfd4.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd4_par'}", num_backbones=5)
S["rfd4_workers"] = [{{"cmd": S["rfd4_cmd"], "env": {{"WORKER": "0"}}, "weight": 2}},
                     {{"cmd": S["rfd4_cmd"] + ["precision=32-true", "inference_experiment=quickstart"], "env": {{"WORKER": "1"}},
                      "weight": 1, "batch": 1}}]          # a small card: its own batch size and extra RFD4 arguments
SETTINGS = S
""")
pipeline(work / "settings_rfd4_par.py", "--stages", "rfd")
made4 = sorted((work / "run_rfd4_par" / "1_rfdiffusion").glob("design_*.pdb"))
assert len(made4) == 5 and len({p.read_text() for p in made4}) == 5, "5 distinct RFD4 designs expected from 2 workers"
a0 = (work / "run_rfd4_par" / "1_rfd4_raw" / "w0" / "args.txt").read_text().splitlines()
a1 = (work / "run_rfd4_par" / "1_rfd4_raw" / "w1" / "args.txt").read_text().splitlines()
assert "batch_size=2" in a0 and "~metrics" in a0 and "batch_size=1" in a1 and "precision=32-true" in a1, (a0, a1)
assert "~metrics" not in a1 and "seed=1" in a0 and "seed=1001" in a1, "per-worker batch, extra arguments and seeds"

# optional: redesign the sequences with ProteinMPNN anyway
(work / "settings_rfd4_mpnn.py").write_text(f"""
import runpy
S = runpy.run_path(r"{work / 'settings_rfd4.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd4_mpnn'}", rfd4_mpnn=True, mpnn_cmd=[r"{sys.executable}", r"{HERE / 'fake_tools.py'}", "mpnn"])
SETTINGS = S
""")
pipeline(work / "settings_rfd4_mpnn.py", "--stages", "rfd,mpnn")
assert len(read_csv(work / "run_rfd4_mpnn" / "2_mpnn" / "sequences.csv")) == 6, "3 designs x 2 ProteinMPNN sequences"

# clear messages for inputs RFD4 cannot take
for bad, words in (({"target_range": "A5-20,4,A25-40"}, "bare number"), ({"hotspots": "A10,A60"}, "not inside target_range"),
                   ({"target_range": "A5-45"}, "not in the PDB file")):
    (work / "settings_rfd4_bad.py").write_text(f"""
import runpy
S = runpy.run_path(r"{work / 'settings_rfd4.py'}")["SETTINGS"]
S.update(output_dir=r"{work / 'run_rfd4_bad'}", **{bad!r})
SETTINGS = S
""")
    shutil.rmtree(work / "run_rfd4_bad", ignore_errors=True)
    res = subprocess.run([sys.executable, PIPELINE, "--settings", str(work / "settings_rfd4_bad.py"), "--stages", "rfd"],
                         capture_output=True, text=True)
    assert res.returncode != 0 and words in (res.stdout + res.stderr), (bad, res.stdout[-300:], res.stderr[-300:])
print("Test 11 passed: RFD4 route (CIF with one chain per piece, spec, original numbering, native sequences, per-worker seeds)")

# ---- Test 12: settings_dual_gpu.py builds the right workers for the two-card server -------------------------------------------
import os                                                                      # noqa: E402
import runpy                                                                   # noqa: E402
from design_pipeline import split_by_weight                                    # noqa: E402
env12 = {"TARGET_PDB": "t.pdb", "TARGET_RANGE": "A1-10", "HOTSPOTS": "A5", "NUM_BACKBONES": "24"}
saved = {k: os.environ.get(k) for k in [*env12, "BACKBONE_TOOL", "RFD_WORKERS_1"]}
try:
    os.environ.update(env12)
    for tool_name, expect_cards in (("rfd4", ["0"]), ("rfd3", ["0", "1"])):          # RFD4 leaves the 1660 SUPER out by default
        os.environ["BACKBONE_TOOL"] = tool_name
        S12 = runpy.run_path(str(HERE.parent / "settings_dual_gpu.py"))["SETTINGS"]
        bb = workers_for(S12, f"{tool_name}_workers".replace("_workers", ""))
        assert [w["env"]["CUDA_VISIBLE_DEVICES"] for w in bb] == expect_cards, (tool_name, bb)
    os.environ.update(BACKBONE_TOOL="rfd4", RFD_WORKERS_1="1")                       # ... and a user can switch it on
    S12 = runpy.run_path(str(HERE.parent / "settings_dual_gpu.py"))["SETTINGS"]
    bb = workers_for(S12, "rfd4")
    assert [len(c) for c in split_by_weight(range(24), bb)] == [22, 2], "12 : 1 share of 24 designs"
    assert bb[1]["batch"] == 2 and "precision=32-true" in bb[1]["cmd"] and "inference_experiment=quickstart" in bb[1]["cmd"]
    assert bb[0]["batch"] == 8 and bb[0]["cmd"][0].endswith("rfd4-generate") and bb[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    split24 = [len(c) for c in split_by_weight(range(24), [{"weight": 1}] * 6 + [{"weight": 0.7}] * 2)]
    assert sum(split24) == 24 and max(split24[:6]) <= 4 and max(split24[6:]) <= 3, split24      # no leftover dumped on the last worker
    assert sorted(len(c) for c in split_by_weight(range(7), [{"weight": 1}] * 3)) == [2, 2, 3]
    af = workers_for(S12, "af2")
    assert [w["env"]["CUDA_VISIBLE_DEVICES"] for w in af] == ["0"] * 4 + ["1"] * 2
    assert "af2_bw" in af[0]["cmd"][0] and "af2_bd" in af[-1]["cmd"][0], "Blackwell card needs the newer-JAX environment"
    assert all(w["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for w in af)
finally:
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
# RFD4 is the default tool; cards older than Ampere (compute capability < 8) fall back to RFD3
os.environ["TARGET_PDB"], os.environ["TARGET_RANGE"], os.environ["HOTSPOTS"] = "t.pdb", "A1-10", "A5"
os.environ.pop("BACKBONE_TOOL", None)
ns12 = runpy.run_path(str(HERE.parent / "settings_big_gpu.py"))
assert ns12["pick_default_tool"](None) == "rfd4" and ns12["pick_default_tool"](8.9) == "rfd4" and ns12["pick_default_tool"](12.0) == "rfd4"
assert ns12["pick_default_tool"](7.5) == "rfd3", "GTX 1660 SUPER (7.5) cannot run RFD4"
from design_pipeline import backbone_tool                                     # noqa: E402
assert backbone_tool({"rfd4_cmd": ["x"], "rfd3_cmd": ["y"]}) == "rfd4" and backbone_tool({"rfd3_cmd": ["y"]}) == "rfd3"
assert backbone_tool({"backbone_tool": "rfd3", "rfd4_cmd": ["x"]}) == "rfd3", "an explicit setting always wins"
print("Test 12 passed: dual-GPU settings (RFD4 on the big card only, RFD3 on both, AF2 environment per card, 12:1 / 4:1 shares)")
print("\nALL TESTS PASSED")
