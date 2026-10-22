"""
FAKE versions of RFdiffusion, ProteinMPNN, the AF2 predictor and the AlphaFold Server website, used only to
test the pipeline.

They copy the real tools' command-line options and output file names/formats, but the "designs" are
random helices and the "predictions" are the input backbone plus known noise. So this tests the
plumbing (commands, file parsing, pAE / iPTM / pLDDT / RMSD maths), NOT design quality.

Usage:  python fake_tools.py rfdiffusion|rfd3|rfd4|mpnn|af2|afserver|reglyco  <the same arguments the real tool takes>
"""
import argparse
import gzip
import json
import re
import sys
import zipfile
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


# ---------------------------------------------------------------- AlphaFold Server (the website)
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


def afserver(args):
    """Pretend to be the alphafoldserver.com website: read the exported afserver_jobs.json and write one
    fold_<name>.zip per job, named and filled the way the website does it."""
    p = argparse.ArgumentParser()
    for flag in ["--jobs", "--run_dir", "--out_zips"]:
        p.add_argument(flag)
    a, _ = p.parse_known_args(args)
    for job in json.loads(Path(a.jobs).read_text()):
        name = job["name"]
        seqs = {"A": job["sequences"][0]["proteinChain"]["sequence"],      # binder = chain A
                "B": job["sequences"][1]["proteinChain"]["sequence"]}      # target = chain B
        backbone = read_pdb(Path(a.run_dir) / "1_rfdiffusion" / (name.rsplit("_s", 1)[0] + ".pdb"))
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        sigma = float(rng.uniform(0.1, 3.0))                       # how wrong this "prediction" is (Å per axis)
        gly = {c: np.mean([r == "GLY" for r, _ in res]) for c, res in backbone.items()}
        b_src = max(gly, key=gly.get)
        t_src = [c for c in backbone if c != b_src][0]
        noise = rng.normal(0, sigma, (len(seqs["A"]), 3))
        binder = np.array([x for _, x in backbone[b_src]]) + noise
        target = np.array([x for _, x in backbone[t_src]])
        # predicted models come out in an arbitrary position/orientation -> rotate + shift everything
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        shift = rng.normal(0, 20, 3)
        model = {"A": [(seq3(a_).upper(), x) for a_, x in zip(seqs["A"], binder @ q + shift)],
                 "B": [(seq3(a_).upper(), x) for a_, x in zip(seqs["B"], target @ q + shift)]}
        d = Path(a.out_zips) / "_tmp" / name
        d.mkdir(parents=True, exist_ok=True)
        write_cif(d / f"{name}_model.cif", name, model)
        iptm = float(np.clip(0.95 - 0.12 * sigma + rng.normal(0, 0.03), 0, 1))
        (d / f"{name}_summary_confidences.json").write_text(json.dumps(
            {"iptm": iptm, "ptm": 0.8, "ranking_score": iptm, "fraction_disordered": 0.0, "has_clash": 0.0}))
        n_a, n_b = len(seqs["A"]), len(seqs["B"])
        plddt = np.clip(np.concatenate([rng.normal(92 - 8 * sigma, 2, n_a), rng.normal(90, 2, n_b)]), 0, 100)
        (d / f"{name}_confidences.json").write_text(json.dumps(
            {"atom_chain_ids": ["A"] * n_a + ["B"] * n_b, "atom_plddts": plddt.tolist()}))
        (Path(a.out_zips) / f"{name}_truth.json").write_text(json.dumps(
            {"expected_rmsd_binder": float(np.sqrt((noise ** 2).sum(axis=1).mean()))}))
        with zipfile.ZipFile(Path(a.out_zips) / f"fold_{name}.zip", "w") as z:   # the website returns 5 models; we write model 0
            z.write(d / f"{name}_model.cif", f"fold_{name}_model_0.cif")
            z.write(d / f"{name}_summary_confidences.json", f"fold_{name}_summary_confidences_0.json")
            z.write(d / f"{name}_confidences.json", f"fold_{name}_full_data_0.json")


# ---------------------------------------------------------------- RFdiffusion 3  (rfd3 design ...)
AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def rfd3(args):
    """Pretend to be `rfd3 design out_dir=... inputs=spec.json diffusion_batch_size=.. n_batches=..`.
    Like the real tool: files are <specname>_<key>_<batch>_model_<n>.cif.gz + .json; the BINDER is chain A with a real
    sequence; the TARGET is chain B renumbered from 1, and the JSON's diffused_index_map says A16 -> B1 and so on."""
    kv = dict(a.split("=", 1) for a in args if "=" in a)
    out = Path(kv["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    spec_file = Path(kv["inputs"])
    key, spec = next(iter(json.loads(spec_file.read_text()).items()))
    parts = [p for p in spec["contig"].split(",") if p != "/0"]            # e.g. ["20-30", "A5-20", "A25-40"]
    bits = parts[0].split("-")
    lo, hi = int(bits[0]), int(bits[-1])
    source = read_pdb(spec["input"])
    target, index_map = [], {}                                                  # kept target residues
    for p in parts[1:]:
        m = re.fullmatch(r"([A-Za-z])(\d+)-(\d+)", p)
        if m:
            chain, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            for number in range(a, b + 1):
                target.append(source[chain][number - 1])                        # toy PDBs are numbered from 1
                index_map[f"{chain}{number}"] = f"B{len(target)}"
    for batch in range(int(kv.get("n_batches", 1))):
        length = int(np.random.default_rng(zlib.crc32(f"{out}{batch}".encode())).integers(lo, hi + 1))   # one length per batch
        for n in range(int(kv.get("diffusion_batch_size", 8))):
            rng = np.random.default_rng(zlib.crc32(f"{out}_{batch}_{n}".encode()))
            centre = np.array([0, 0, 1.5 * (len(target) - length) / 2])
            binder = [(AA3[int(i)], x) for i, x in zip(rng.integers(0, 20, length), helix(length) + centre + np.array([10.0, 0, 0]))]
            name = f"{spec_file.stem}_{key}_{batch}_model_{n}"
            write_cif(out / f"{name}.cif", name, {"A": binder, "B": target})
            with open(out / f"{name}.cif", "rb") as src, gzip.open(out / f"{name}.cif.gz", "wb") as dst:
                dst.write(src.read())
            (out / f"{name}.cif").unlink()
            (out / f"{name}.json").write_text(json.dumps(
                {"diffused_index_map": index_map, "metrics": {"n_chainbreaks": 0, "max_ca_deviation": 0.1},
                 "specification": spec, "ckpt_path": kv.get("ckpt_path"), "seed": None}))


# ---------------------------------------------------------------- RFD4-Proteina  (rfd4-generate inputs=spec.yaml ...)
def rfd4(args):
    """Pretend to be `rfd4-generate inputs=spec.yaml out_dir=.. num_replicates=.. batch_size=.. seed=.. ~metrics`.
    Like the real tool: <out_dir>/inference_samples/<name>_<i>/<name>_<i>_generated.cif, with the target pieces keeping their
    chain letters (each numbered from 1), the binder in the next free letter WITH a real sequence, and NO B-factor column."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    assert "~metrics" in args, "the pipeline should skip RFD4's metrics (they need other tools)"
    kv = dict(a.split("=", 1) for a in args if "=" in a)
    text = Path(kv["inputs"]).read_text()
    contig = re.search(r'contig: "([^"]+)"', text).group(1)                      # e.g.  20-30, /0, A5-20, /0, B25-40
    source = MMCIF2Dict(re.search(r'input: "([^"]+)"', text).group(1))
    assert "C_HOT: {true: [" in text and "C_CRDGRP: {1: [" in text, "conditions must be written the way RFD4's examples do"
    parts = [p.strip() for p in contig.split(",") if p.strip() != "/0"]
    lo, hi = int(parts[0].split("-")[0]), int(parts[0].split("-")[-1])
    pieces = {}
    for p in parts[1:]:
        letter, a, b = re.fullmatch(r"([A-Z])(\d+)-(\d+)", p).groups()
        xyz = [(int(n), np.array([float(x), float(y), float(z)])) for c, n, x, y, z in zip(
            source["_atom_site.auth_asym_id"], source["_atom_site.auth_seq_id"], source["_atom_site.Cartn_x"],
            source["_atom_site.Cartn_y"], source["_atom_site.Cartn_z"]) if c == letter]
        pieces[letter] = [x for n, x in xyz if int(a) <= n <= int(b)]
    binder_letter = next(c for c in "ABCDEFGHIJ" if c not in pieces)
    out = Path(kv["out_dir"]) / "inference_samples"
    n_total = int(kv["num_replicates"])
    target_len = sum(len(v) for v in pieces.values())
    for i in range(n_total):
        rng = np.random.default_rng(int(kv["seed"]) * 100003 + i)                 # a different seed gives different designs
        length = int(rng.integers(lo, hi + 1))
        binder = [(AA3[int(k)], x) for k, x in zip(rng.integers(0, 20, length),
                                                   helix(length) + np.array([10.0, 0, 1.5 * (target_len - length) / 2]))]
        chains = {binder_letter: binder}                                            # the real file lists the binder first
        chains.update({c: [(("ALA"), x) for x in v] for c, v in pieces.items()})
        name = f"design_{i}"
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        head = ["data_" + name + "_generated.cif", "loop_"] + ["_atom_site." + c for c in
                "group_PDB type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id "
                "pdbx_PDB_ins_code auth_seq_id auth_comp_id auth_asym_id auth_atom_id occupancy Cartn_x Cartn_y Cartn_z "
                "pdbx_PDB_model_num id".split()]
        rows, k = [], 1
        for cid, residues in chains.items():
            for r, (res, xyz) in enumerate(residues, start=1):
                rows.append(f"ATOM C CA . {res} {cid} 1 {r} . {r} {res} {cid} CA 1.0 {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} 1 {k}")
                k += 1
        (d / f"{name}_generated.cif").write_text("\n".join(head + rows) + "\n#\n")
    print(f"fake RFD4 wrote {n_total} designs (seed {kv['seed']})")


# ---------------------------------------------------------------- ReGlyco  (reglyco scan --protein X --output DIR ...)
def reglyco(args):
    """Pretend to be `reglyco scan`: writes scan.json with the real tool's layout. The FIRST sequon is called accessible,
    every other sequon is called not accessible (so a test can see both answers)."""
    kv = {a: args[i + 1] for i, a in enumerate(args) if a.startswith("--") and i + 1 < len(args)}
    chains = read_pdb(kv["--protein"])
    out = Path(kv["--output"])
    out.mkdir(parents=True, exist_ok=True)
    sites = []
    for chain, residues in chains.items():
        seq = "".join(seq1(n) for n, _ in residues)
        for m in re.finditer(r"(?=(N[^P][ST]))", seq):
            asn = {"chain": chain, "insertion_code": None, "number": m.start() + 1}      # toy PDBs are numbered from 1
            sites.append({"accessible": not sites, "cluster": 1, "generations": 9, "joint_subset": not sites, "phi": 56.2,
                          "psi": -178.9, "rotamer": None, "steric_score": 1.0 if not sites else 0.2,
                          "sequon": {"asparagine": asn, "motif": m.group(1)}})
    (out / "scan.json").write_text(json.dumps({"glycan": "G14843DJ", "status": "complete", "sites": sites,
                                               "interpretation": "structural accessibility only; not proof of biological glycosylation"}))
    print(f"Scanned {len(sites)} sequon(s)")


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
    {"rfdiffusion": rfdiffusion, "rfd3": rfd3, "rfd4": rfd4, "mpnn": mpnn, "afserver": afserver, "af2": af2,
     "reglyco": reglyco}[sys.argv[1]](sys.argv[2:])
