"""
rfd4_af3_report.py -- summarise an AF3 check of RFD4 binder designs (the folder made by rfd4_to_af3_jobs.py + the af3,score_af3 stages).

Prints how many designs pass, the iPTM / pAE / pLDDT spread, how many AF3 binders touch the hotspot residues, and a table of the
best designs with the iPTM of each diffusion sample (a design whose samples all agree is a stable prediction).

    python rfd4_af3_report.py --run <output_dir>
    python rfd4_af3_report.py --run <output_dir> --top 20 --hotspots 31,32,122,123 --target_ranges 16-108,113-151

Needs gemmi and numpy (the RFD4 environment has both). "hot_d" is the smallest heavy-atom distance in AF3's own model between the
binder and the hotspot residues; <= 4.5 A counts as touching. The 5YXH numbers are the defaults, so pass --hotspots and
--target_ranges for another target (residue numbers of the ORIGINAL target file; the ranges are the 132 residues kept as the target).
"""
import argparse
import csv
import json
import pathlib

import gemmi
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", required=True, help="pipeline output folder that holds results_af3.csv")
ap.add_argument("--top", type=int, default=15, help="how many designs to list")
ap.add_argument("--hotspots", default="31,32,122,123", help="hotspot residue numbers in the original target file")
ap.add_argument("--target_ranges", default="16-108,113-151", help="residue ranges of the original file that form the target chain")
a = ap.parse_args()

run = pathlib.Path(a.run).expanduser()
rows = list(csv.DictReader(open(run / "results_af3.csv")))
hot_orig = {int(x) for x in a.hotspots.split(",")}
orig = [n for lo, hi in (r.split("-") for r in a.target_ranges.split(",")) for n in range(int(lo), int(hi) + 1)]
hot_idx = [i for i, o in enumerate(orig) if o in hot_orig]


def hotspot_contact(job):
    """AF3's own model: smallest heavy-atom distance from the binder (chain A) to the hotspot residues of the target (chain B)."""
    cif = run / "3_afserver_outputs" / job / f"{job}_model.cif"
    m = gemmi.read_structure(str(cif))[0]
    chains = {c.name: c for c in m}
    tgt = [r for r in chains["B"]]
    hat = np.array([at.pos.tolist() for i in hot_idx for at in tgt[i]])
    bat = np.array([at.pos.tolist() for r in chains["A"] for at in r])
    return float(np.sqrt(((hat[:, None] - bat[None]) ** 2).sum(-1)).min())


def samples(job):
    d = run / "3_af3_outputs" / job.lower()
    return [json.loads(next(s.glob("*_summary_confidences.json")).read_text())["iptm"] for s in sorted(d.glob("seed-*_sample-*"))]


for r in rows:
    r["hot_dist"] = hotspot_contact(r["job"])
    r["blen"] = len(r["binder_seq"])
f = lambda k: np.array([float(r[k]) for r in rows])
iptm, pae, pl, rm, hd = f("iptm"), f("pae_interaction"), f("plddt_binder"), f("rmsd_binder"), f("hot_dist")
n = len(rows)
print(f"{run.name}: {n} designs checked with AF3")
print(f"  PASS at the pipeline's cutoffs (iptm>=0.8, plddt_b>=80, rmsd<=2.0, pae<=10): {sum(r['pass'] == 'PASS' for r in rows)}/{n}")
loose = (iptm >= 0.8) & (pl >= 80) & (pae <= 10) & (rm <= 4.0)
print(f"  same with rmsd<=4.0: {int(loose.sum())}/{n}")
print(f"  iptm>=0.8: {int((iptm >= 0.8).sum())} | iptm>=0.7: {int((iptm >= 0.7).sum())} | iptm>=0.5: {int((iptm >= 0.5).sum())}")
print(f"  median iptm {np.median(iptm):.2f} (max {iptm.max():.2f}) | median pae_interaction {np.median(pae):.1f} (best {pae.min():.1f}) | "
      f"median binder pLDDT {np.median(pl):.1f} | best rmsd {rm.min():.2f}")
print(f"  AF3 binder touches the hotspot residues (<=4.5 A): {int((hd <= 4.5).sum())}/{n}; median distance {np.median(hd):.1f} A\n")
print(f"{'job':10s}{'len':>4}{'iptm':>6}{'pae':>7}{'plddt_b':>8}{'rmsd':>7}{'hot_d':>7}  verdict   iPTM of the samples")
for r in sorted(rows, key=lambda r: -float(r["iptm"]))[:a.top]:
    print(f"{r['job']:10s}{r['blen']:>4}{r['iptm']:>6}{r['pae_interaction']:>7}{r['plddt_binder']:>8}{r['rmsd_binder']:>7}{r['hot_dist']:>7.1f}  {r['pass']:8s}  "
          + " ".join(f"{x:.2f}" for x in samples(r['job'])))
