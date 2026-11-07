"""
make_af3_template.py -- turn part of a PDB file into a single-chain mmCIF that AlphaFold 3 accepts as a TEMPLATE.

Why: if AF3 is told what the target protein looks like, it only has to work out where the binder goes. Use the
ORIGINAL experimental target (with side chains), not RFdiffusion's output: RFdiffusion keeps only N/CA/C/O, and AF3's
template features need the C-beta atom.

    python make_af3_template.py --pdb 5YXH.pdb --chain A --ranges 16-108,113-151 --out target_template.cif \
        --expect_seq SVGKSSLTIQ...      # optional: stop if the extracted sequence differs

Needs `gemmi` (pip install gemmi; it is already inside the RFD4 environment). The CIF gets a dummy release date,
because AF3 insists on one.
"""
import argparse
import sys

import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--chain", required=True)
ap.add_argument("--ranges", required=True, help="residue numbers to keep, e.g. 16-108,113-151")
ap.add_argument("--out", required=True)
ap.add_argument("--expect_seq", help="if given, the extracted sequence must match exactly")
a = ap.parse_args()

ranges = [tuple(int(x) for x in r.split("-")) for r in a.ranges.split(",")]
keep = lambda n: any(lo <= n <= hi for lo, hi in ranges)

src = gemmi.read_structure(a.pdb)
src.remove_hydrogens()
src.remove_alternative_conformations()
chain = src[0][a.chain]

out = gemmi.Structure()
out.name = "target_template"
model = gemmi.Model("1")
new = gemmi.Chain("A")
for res in chain:
    if res.het_flag == "A" and keep(res.seqid.num):          # protein residues in the requested ranges
        new.add_residue(res)
model.add_chain(new)
out.add_model(model)

seq = gemmi.one_letter_code([r.name for r in new])
print(f"kept {len(new)} residues; sequence: {seq}")
missing_cb = [r.seqid.num for r in new if r.name != "GLY" and r.find_atom("CB", "*") is None]
print("residues without C-beta (non-Gly):", missing_cb or "none")
if a.expect_seq and seq != a.expect_seq:
    sys.exit("SEQUENCE MISMATCH with --expect_seq; not writing a template")

out.setup_entities()
out.assign_label_seq_id(True)  # force: numbers every residue 1..N
text = out.make_mmcif_document().as_string()
text += """
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 2000-01-01
"""
open(a.out, "w").write(text)
print("wrote", a.out)
