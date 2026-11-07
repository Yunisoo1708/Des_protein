"""
make_rfd4_input.py -- turn part of a PDB file into what RFD4 needs to read a target: an input CIF, a task spec (yaml) and the target sequence.

RFD4 only reads CIF files, with one chain per unbroken piece of the target, and it is told in the spec to keep the target EXACTLY as it is
(fixed coordinates and sequence) and to design a binder next to it. This script writes all three from the original PDB:

    python make_rfd4_input.py --pdb prepared_protein_structure/5YXH.pdb --chain A --ranges 16-108,113-151 \
        --hotspots 31,32,122,123 --binder 50-80 --out_dir work --check

    work/rfd4_input.cif        the target pieces as chains A, B, ... (residue numbers of the PDB are kept)
    work/rfd4_spec.yaml        contig + conditions: target fixed, hotspots, a binder of 50-80 residues
    work/rfd4_target_seq.txt   the target sequence (the converter rfd4_to_af3_jobs.py uses it to prove the target did not change)

--ranges   residues kept as the target, comma separated; every range becomes one chain (leave unresolved gaps between ranges)
--hotspots residue numbers of the PDB the binder should touch (they must lie inside the ranges)
--check    afterwards load the spec the way RFD4 does and stop if the target is not fully fixed (needs the RFD4 python)

Needs `gemmi` (the RFD4 environment has it). The PDB must contain SEQRES records (the full chain sequence); hydrogens, water and ligands are dropped.
"""
import argparse
import sys
from pathlib import Path

import gemmi

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pdb", required=True)
ap.add_argument("--chain", default="A")
ap.add_argument("--ranges", required=True, help="e.g. 16-108,113-151")
ap.add_argument("--hotspots", required=True, help="e.g. 31,32,122,123")
ap.add_argument("--binder", default="50-80", help="binder length range, e.g. 50-80 (or 60-60 for a fixed length)")
ap.add_argument("--name", default="rfd4")
ap.add_argument("--out_dir", default="work")
ap.add_argument("--check", action="store_true", help="verify with RFD4's own loader that the target is fully fixed")
a = ap.parse_args()

ranges = [tuple(int(x) for x in r.split("-")) for r in a.ranges.split(",")]
hotspots = [int(h) for h in a.hotspots.split(",")]
kept = {n for lo, hi in ranges for n in range(lo, hi + 1)}
if not set(hotspots) <= kept:
    sys.exit(f"hotspot residues {sorted(set(hotspots) - kept)} are not inside --ranges {a.ranges}")

st = gemmi.read_structure(a.pdb)
st.setup_entities()
st.remove_hydrogens()
st.remove_ligands_and_waters()
if not st.entities or not st.entities[0].full_sequence:
    sys.exit(f"{a.pdb} has no SEQRES records, so the full chain sequence is unknown; add them or give a CIF.")
ent = st.entities[0]
src = {r.seqid.num: r for r in st[0][a.chain]}

letters = [chr(ord("A") + i) for i in range(len(ranges))]
model = gemmi.Model("1")
seq = []
for letter, (lo, hi) in zip(letters, ranges):
    missing = [n for n in range(lo, hi + 1) if n not in src]
    if missing:
        sys.exit(f"residues {missing[:8]}{'...' if len(missing) > 8 else ''} of range {lo}-{hi} are missing from chain {a.chain}; "
                 f"split the range around the gap")
    ch = gemmi.Chain(letter)
    for n in range(lo, hi + 1):
        res = src[n]
        res.subchain = letter
        res.entity_id = ent.name
        res.label_seq = res.seqid.num                 # label_seq_id == author residue number (position in the full sequence)
        ch.add_residue(res)
        seq.append(gemmi.find_tabulated_residue(res.name).one_letter_code.upper())
    model.add_chain(ch)
out = gemmi.Structure()
out.cell, out.spacegroup_hm, out.name = st.cell, st.spacegroup_hm, st.name
out.add_model(model)
out.entities.append(ent)
ent.subchains = letters
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
cif = out_dir / f"{a.name}_input.cif"
out.make_mmcif_document().write_file(str(cif))

contig = ",/0,".join(f"{l}{lo}-{hi}" for l, (lo, hi) in zip(letters, ranges)) + f",/0,{a.binder}"
chains = " or ".join(f"chain_id == '{l}'" for l in letters)
hot = " or ".join(f"res_id == {h}" for h in hotspots)
spec = out_dir / f"{a.name}_spec.yaml"
spec.write_text(f"""{a.name}:
  input: {cif.name}
  contig: "{contig}"
  conditions:
    C_CRD:                       # keep the target's coordinates FIXED
      true:
        - select: "{chains}"
    C_SEQ:                       # keep the target's sequence FIXED
      true:
        - select: "{chains}"
    hotspot:
      true:
        - select: "{hot}"
""")
(out_dir / f"{a.name}_target_seq.txt").write_text("".join(seq) + "\n")
print(f"wrote {cif}, {spec}, {a.name}_target_seq.txt  | target = {len(seq)} residues in {len(ranges)} chain(s) | contig {contig}")

if a.check:
    import numpy as np
    import yaml
    from rfproteina.inference_utils.condition_spec import build_annotated_atom_array_from_condition_spec
    arr = build_annotated_atom_array_from_condition_spec(yaml.safe_load(spec.read_text())[a.name], spec_dir=out_dir, example_id=a.name)["atom_array"]
    tgt = np.isin(np.asarray(arr.chain_id), letters)
    cats = list(arr.get_annotation_categories())
    ok = True
    for col in ("condition_coordinate_1_atom", "condition_sequence_1_residue"):
        if col not in cats:
            sys.exit(f"RFD4's loader has no annotation {col}: the spec does not fix the target (or RFD4 changed its names)")
        v = np.asarray(arr.get_annotation(col)).astype(bool)
        v = v.reshape(len(tgt), -1).any(axis=1)
        # RFD4 inserts one placeholder atom for the expandable binder segment; it copies its neighbour's flags, so <= 1 is normal
        good = int((v & tgt).sum()) == int(tgt.sum()) and int((v & ~tgt).sum()) <= 1
        print(f"  {col}: fixed on {int((v & tgt).sum())}/{int(tgt.sum())} target atoms, {int((v & ~tgt).sum())} binder atoms")
        ok = ok and good
    print("TARGET FULLY FIXED, BINDER FREE:", ok)
    sys.exit(0 if ok else 1)
