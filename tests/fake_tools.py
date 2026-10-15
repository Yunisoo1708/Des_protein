"""
FAKE versions of RFdiffusion, ProteinMPNN and AlphaFold3, used only to test the pipeline.

They copy the real tools' command-line options and output file names/formats, but the "designs" are
random helices and the "predictions" are the input backbone plus known noise. So this tests the
plumbing (commands, file parsing, iPTM / pLDDT / RMSD maths), NOT design quality.

Usage:  python fake_tools.py rfdiffusion|mpnn|af3  <the same arguments the real tool takes>
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from Bio.SeqUtils import seq1, seq3

AA = "ACDEFGHIKLMNPQRSTVWY"


def helix(n):
    """Ideal alpha-helix C-alpha positions (2.3 Å radius, 1.5 Å rise, 100° per residue)."""
    i = np.arange(n)
    return np.stack([2.3 * np.cos(np.radians(100 * i)), 2.3 * np.sin(np.radians(100 * i)), 1.5 * i], axis=1)


def write_pdb(path, chains):
    """chains = {chain_id: [(resname, xyz), ...]}"""
    lines, k = [], 1
    for cid, residues in chains.items():
        for i, (name, xyz) in enumerate(residues, start=1):
            lines.append(f"ATOM  {k:5d}  CA  {name:>3s} {cid}{i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C")
            k += 1
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def read_pdb(path):
    chains = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains.setdefault(line[21], []).append((line[17:20].strip(), xyz))
    return chains


# ---------------------------------------------------------------- RFdiffusion
def rfdiffusion(args):
    kv = dict(a.split("=", 1) for a in args)                       # Hydra style:  key=value
    prefix, n = kv["inference.output_prefix"], int(kv["inference.num_designs"])
    start = int(kv.get("inference.design_startnum", 0))       # real RFdiffusion option: first design number
    target_part, binder_part = kv["contigmap.contigs"].strip("[]").split()
    chain, lo, hi = re.match(r"([A-Z])(\d+)-(\d+)", target_part).groups()
    target = read_pdb(kv["inference.input_pdb"])[chain][int(lo) - 1:int(hi)]
    b_lo, b_hi = map(int, binder_part.split("-"))
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for d in range(start, start + n):
        rng = np.random.default_rng(d)
        length = int(rng.integers(b_lo, b_hi + 1))
        centre = np.array([0, 0, 1.5 * (len(target) - length) / 2])
        binder_xyz = helix(length) + centre + np.array([10.0, 0, 0])   # a helix lying next to the target
        # Real RFdiffusion output: target keeps its residues (chain A); binder = Glycine only (chain B)
        write_pdb(f"{prefix}_{d}.pdb", {"A": target, "B": [("GLY", x) for x in binder_xyz]})
    print(f"fake RFdiffusion wrote {n} backbones")


# ---------------------------------------------------------------- ProteinMPNN
def mpnn(args):
    p = argparse.ArgumentParser()
    for flag in ["--pdb_path", "--pdb_path_chains", "--out_folder", "--num_seq_per_target",
                 "--sampling_temp", "--seed", "--batch_size"]:
        p.add_argument(flag)
    a, _ = p.parse_known_args(args)
    chains = read_pdb(a.pdb_path)
    designed = a.pdb_path_chains.split()
    order = designed + [c for c in chains if c not in designed]    # designed chains first, then fixed ones
    native = {c: "".join(seq1(n) for n, _ in chains[c]) for c in order}
    name = Path(a.pdb_path).stem
    rng = np.random.default_rng(int(a.seed))
    out = [f">{name}, score=1.5000, global_score=1.5000, fixed_chains=[], designed_chains={designed}, seed={a.seed}",
           "/".join(native[c] for c in designed)]          # real tool lists ONLY the designed chain(s)
    for k in range(1, int(a.num_seq_per_target) + 1):
        parts = ["".join(rng.choice(list(AA), len(native[c]))) for c in designed]
        out += [f">T={a.sampling_temp}, sample={k}, score={rng.uniform(0.6, 1.2):.4f}, global_score=1.0000, seq_recovery=0.1", "/".join(parts)]
    (Path(a.out_folder) / "seqs").mkdir(parents=True, exist_ok=True)
    (Path(a.out_folder) / "seqs" / f"{name}.fa").write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- AlphaFold3
def write_cif(path, name, chains):
    head = ["data_" + name, "loop_"] + ["_atom_site." + c for c in
            "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id "
            "label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id "
            "auth_asym_id pdbx_PDB_model_num".split()]
    rows, k = [], 1
    for cid, residues in chains.items():
        for i, (res, xyz) in enumerate(residues, start=1):
            rows.append(f"ATOM {k} C CA . {res} {cid} 1 {i} ? {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} 1.00 90.00 {i} {cid} 1")
            k += 1
    Path(path).write_text("\n".join(head + rows) + "\n#\n")


def af3(args):
    p = argparse.ArgumentParser()
    for flag in ["--input_dir", "--json_path", "--output_dir", "--model_dir", "--db_dir"]:
        p.add_argument(flag)
    a, _ = p.parse_known_args(args)
    files = sorted(Path(a.input_dir).glob("*.json")) if a.input_dir else [Path(a.json_path)]
    for f in files:
        job = json.loads(f.read_text())
        name = job["name"]
        seqs = {s["protein"]["id"]: s["protein"]["sequence"] for s in job["sequences"]}
        backbone = read_pdb(Path(a.output_dir).parent / "1_rfdiffusion" / (name.rsplit("_s", 1)[0] + ".pdb"))
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        sigma = float(rng.uniform(0.1, 3.0))                       # how wrong this "prediction" is (Å per axis)
        gly = {c: np.mean([r == "GLY" for r, _ in res]) for c, res in backbone.items()}
        b_src = max(gly, key=gly.get)
        t_src = [c for c in backbone if c != b_src][0]
        noise = rng.normal(0, sigma, (len(seqs["A"]), 3))
        binder = np.array([x for _, x in backbone[b_src]]) + noise
        target = np.array([x for _, x in backbone[t_src]])
        # AF3 models come out in an arbitrary position/orientation -> rotate + shift everything
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        shift = rng.normal(0, 20, 3)
        model = {"A": [(seq3(a_).upper(), x) for a_, x in zip(seqs["A"], binder @ q + shift)],
                 "B": [(seq3(a_).upper(), x) for a_, x in zip(seqs["B"], target @ q + shift)]}
        d = Path(a.output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        write_cif(d / f"{name}_model.cif", name, model)
        iptm = float(np.clip(0.95 - 0.12 * sigma + rng.normal(0, 0.03), 0, 1))
        (d / f"{name}_summary_confidences.json").write_text(json.dumps(
            {"iptm": iptm, "ptm": 0.8, "ranking_score": iptm, "fraction_disordered": 0.0, "has_clash": 0.0}))
        n_a, n_b = len(seqs["A"]), len(seqs["B"])
        plddt = np.clip(np.concatenate([rng.normal(92 - 8 * sigma, 2, n_a), rng.normal(90, 2, n_b)]), 0, 100)
        (d / f"{name}_confidences.json").write_text(json.dumps(
            {"atom_chain_ids": ["A"] * n_a + ["B"] * n_b, "atom_plddts": plddt.tolist()}))
        (d / "fake_truth.json").write_text(json.dumps({"expected_rmsd_binder": float(np.sqrt((noise ** 2).sum(axis=1).mean()))}))


# ---------------------------------------------------------------- AF2 initial guess (dl_binder_design predict.py)
def af2(args):
    p = argparse.ArgumentParser()
    for flag in ["-pdbdir", "-outpdbdir", "-scorefilename", "-checkpoint_name", "-recycle"]:
        p.add_argument(flag)
    a, _ = p.parse_known_args(args)
    out = Path(a.outpdbdir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["SCORE:     binder_aligned_rmsd pae_binder pae_interaction pae_target plddt_binder plddt_target "
             "plddt_total target_aligned_rmsd time description"]
    for f in sorted(Path(a.pdbdir).glob("*.pdb")):
        chains = read_pdb(f)                                        # the real tool wants: binder first (A), target (B)
        rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
        sigma = float(rng.uniform(0.1, 3.0))
        noise = rng.normal(0, sigma, (len(chains["A"]), 3))
        binder = [(r, x + d) for (r, x), d in zip(chains["A"], noise)]
        write_pdb(out / f"{f.stem}_af2pred.pdb", {"A": binder, "B": chains["B"]})   # target stays in place (initial guess)
        (out / f"{f.stem}_truth.json").write_text(json.dumps({"expected_rmsd_binder": float(np.sqrt((noise ** 2).sum(axis=1).mean()))}))
        lines.append(f"SCORE: 0.5 2.0 {4 + 3 * sigma:.3f} 2.0 {min(100, 93 - 7 * sigma):.3f} 95.0 90.0 0.1 10.0 {f.stem}_af2pred")
    Path(a.scorefilename).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"rfdiffusion": rfdiffusion, "mpnn": mpnn, "af3": af3, "af2": af2}[sys.argv[1]](sys.argv[2:])
