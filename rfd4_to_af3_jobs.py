"""
rfd4_to_af3_jobs.py -- turn RFD4 binder designs into the folder layout design_pipeline.py's AF3 stages expect.

RFD4 writes, per design, a CIF with the target pieces first (chains A, B, ...) and the designed binder LAST. This script:
  * joins the target chains into one chain "A" and calls the binder chain "B"  (a backbone PDB per design,
    used by the scorer for the binder-RMSD)
  * checks that the target really came out unchanged (sequence identical to --target_seq, else it refuses)
  * names each job <prefix>_<last number of its sample folder> and stops if two designs would get the same name
  * writes <out>/1_rfdiffusion/<job>.pdb  and  <out>/2_mpnn/sequences.csv   (same columns the pipeline already reads)

    python rfd4_to_af3_jobs.py --rfd4_samples <rfd4 out_dir>/inference_samples --out <output_dir> \
        --target_seq "$(cat work/rfd4_target_seq.txt)"   [--prefix rfd4]

./setup.sh run-rfd4 runs this for you; work/rfd4_target_seq.txt is written by make_rfd4_input.py.

Run with a Python that has gemmi (the RFD4 environment does). The "mpnn_score" column does not exist for RFD4 (it designs
sequence and structure together) and is filled with "NA".
"""
import argparse
import csv
import sys
from pathlib import Path

import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--rfd4_samples", required=True, help=".../inference_samples")
ap.add_argument("--out", required=True)
ap.add_argument("--target_seq", required=True, help="the known target sequence (what the designs were conditioned on)")
ap.add_argument("--prefix", default="rfd4")
a = ap.parse_args()

out = Path(a.out)
(out / "1_rfdiffusion").mkdir(parents=True, exist_ok=True)
(out / "2_mpnn").mkdir(parents=True, exist_ok=True)

rows, bad = [], []
for cif in sorted(Path(a.rfd4_samples).glob("*/*_generated.cif"), key=lambda p: int(p.parent.name.rsplit("_", 1)[1])):
    idx = cif.parent.name.rsplit("_", 1)[1]
    job = f"{a.prefix}_{idx}"
    if any(r["job"] == job for r in rows):
        sys.exit(f"duplicate job name {job} (from {cif.parent.name}): sample folders must end in a unique _<number>")
    model = gemmi.read_structure(str(cif))[0]
    chains = [c for c in model]
    if len(chains) < 2:
        bad.append((job, "fewer than 2 chains")); continue
    binder, target_chains = chains[-1], chains[:-1]
    target_seq = "".join(gemmi.one_letter_code([r.name for r in c]) for c in target_chains)
    if target_seq != a.target_seq:
        bad.append((job, "target sequence differs from --target_seq (the design did not keep the target fixed)")); continue
    binder_seq = gemmi.one_letter_code([r.name for r in binder])

    st = gemmi.Structure(); m = gemmi.Model("1")
    ta, bb = gemmi.Chain("A"), gemmi.Chain("B")
    n = 0
    for c in target_chains:
        for r in c:
            n += 1
            r.seqid = gemmi.SeqId(n, " ")
            ta.add_residue(r)
    for i, r in enumerate(binder, start=1):
        r.seqid = gemmi.SeqId(i, " "); bb.add_residue(r)
    m.add_chain(ta); m.add_chain(bb); st.add_model(m)
    st.write_pdb(str(out / "1_rfdiffusion" / f"{job}.pdb"))
    rows.append({"job": job, "backbone": str(out / "1_rfdiffusion" / f"{job}.pdb"), "binder_chain": "B", "target_chain": "A",
                 "binder_seq": binder_seq, "target_seq": target_seq, "mpnn_score": "NA"})

with open(out / "2_mpnn" / "sequences.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["job", "backbone", "binder_chain", "target_chain", "binder_seq", "target_seq", "mpnn_score"])
    w.writeheader(); w.writerows(rows)
lens = sorted(len(r["binder_seq"]) for r in rows)
print(f"wrote {len(rows)} jobs to {out}  (binder lengths {lens[0]}-{lens[-1]}, median {lens[len(lens)//2]})" if rows else "NO JOBS WRITTEN")
for job, why in bad:
    print(f"  SKIPPED {job}: {why}")
sys.exit(0 if rows and not bad else 1)
