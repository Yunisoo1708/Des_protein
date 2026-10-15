"""
Protein binder design pipeline:   RFdiffusion  ->  ProteinMPNN  ->  AlphaFold2 "initial guess" (Bennett lab)

What each tool does (one line each):
  1. RFdiffusion : draws NEW protein backbones (just the "skeleton", no sequence) that stick to your target.
  2. ProteinMPNN : writes amino-acid sequences that should fold into each skeleton.
  3. AF2 (dl_binder_design, Bennett et al.) : predicts the structure of (binder + target) from each sequence,
                   starting from the designed backbone, to check that the design is self-consistent.

Final table (results_af2.csv) has, for every designed sequence:
  pae_interaction  how sure AF2 is about the binder's position relative to the target.  Lower = better (<10 is good)
  plddt_binder     0-100  how sure AF2 is about the binder's own fold.                  Higher = better (>80 is good)
  rmsd_binder      in A   how far AF2's binder is from the RFdiffusion/ProteinMPNN backbone, after
                          lining up the two complexes on the target.                    Lower = better (<2 is good)

AF2 has no iPTM. If you want one, the optional AlphaFold Server route (alphafoldserver.com, a website) gives
iPTM, pLDDT and RMSD: see --stages afs_export / afs_import / score_afs below.

How to run (inside WSL / Linux, where the tools live):
  python design_pipeline.py                         # all four steps, uses settings.py
  python design_pipeline.py --settings my_set.py    # use another settings file
  python design_pipeline.py --stages score_af2      # re-run only some steps (rfd, mpnn, af2, score_af2)
  python design_pipeline.py --stages rfd,mpnn,afs_export   # then upload afserver_jobs.json to alphafoldserver.com
  python design_pipeline.py --stages afs_import,score_afs --afserver_zips <folder of downloaded zips>
"""
import argparse
import csv
import json
import os
import re
import runpy
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def run(cmd, env=None):
    """Run one command line and stop the whole pipeline if it fails.
    `env` = extra environment variables for this tool only, e.g. {"CUDA_VISIBLE_DEVICES": "0"} picks GPU 0."""
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(f"{k}={v}" for k, v in (env or {}).items()), " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def workers_for(S, tool):
    """A 'worker' = one command + the GPU it runs on. Give a tool several workers (one per GPU) and the
    work is shared between them. Settings: "<tool>_workers": [{"cmd": [...], "env": {...}, "weight": 3}, ...]
    (weight = relative speed: a card twice as fast gets twice the work). Without it, one worker is built
    from the older single-GPU keys "<tool>_cmd" / "<tool>_env".
    Add "copies": N to a worker to run N identical workers on the same GPU (their CPU-heavy parts overlap)."""
    workers = S.get(f"{tool}_workers") or [{"cmd": S[f"{tool}_cmd"], "env": S.get(f"{tool}_env"), "weight": 1}]
    expanded = []
    for w in workers:
        copies = int(w.get("copies", 1))
        for _ in range(copies):
            env = dict(w.get("env") or {})
            if copies > 1:   # JAX normally grabs ~75% of GPU memory per process; share the card instead
                env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            expanded.append({"cmd": w["cmd"], "env": env, "weight": w.get("weight", 1)})
    return expanded


def split_by_weight(items, workers):
    """Cut a list into one chunk per worker, each chunk's size proportional to the worker's weight."""
    items, total, shares, start = list(items), sum(w["weight"] for w in workers), [], 0
    for k, w in enumerate(workers):
        end = len(items) if k == len(workers) - 1 else start + round(len(items) * w["weight"] / total)
        shares.append(items[start:end])
        start = end
    return shares


def run_parallel(commands):
    """Start several (cmd, env) commands at the SAME time (one per GPU) and wait for all of them."""
    procs = []
    for cmd, env in commands:
        cmd = [str(c) for c in cmd]
        print("  $", " ".join(f"{k}={v}" for k, v in (env or {}).items()), " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env={**os.environ, **(env or {})}))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise subprocess.CalledProcessError(max(codes), "one of the parallel GPU jobs failed (see messages above)")


def read_chains(path):
    """Read a .pdb or .cif file -> {chain_id: [(residue_name, CA_xyz), ...]}.
    We only keep one atom (the C-alpha, "CA") per residue; that is enough for RMSD."""
    parser = MMCIFParser(QUIET=True) if Path(path).suffix == ".cif" else PDBParser(QUIET=True)
    model = parser.get_structure("x", str(path))[0]
    chains = {}
    for chain in model:
        # r.id[0] == " " means "normal amino acid" (skips water, ions, ligands)
        chains[chain.id] = [(r.get_resname(), r["CA"].coord) for r in chain if r.id[0] == " " and "CA" in r]
    return chains


def find_binder_and_target(chains):
    """RFdiffusion writes the NEW residues as Glycine (no sequence yet), so the chain
    with the most Glycine is the binder and the other chain is the target."""
    if len(chains) != 2:
        raise ValueError(f"Expected 2 chains (binder + target), found {list(chains)}")
    gly_fraction = {c: np.mean([name == "GLY" for name, _ in res]) for c, res in chains.items()}
    binder = max(gly_fraction, key=gly_fraction.get)
    target = [c for c in chains if c != binder][0]
    return binder, target


def sequence_of(residues):
    return "".join(seq1(name) for name, _ in residues)


# ----------------------------------------------------------------------------
# Step 1: RFdiffusion  (make backbones)
# ----------------------------------------------------------------------------
def step1_rfdiffusion(S, out):
    folder = out / "1_rfdiffusion"
    folder.mkdir(parents=True, exist_ok=True)
    # "Contig" = RFdiffusion's recipe.  "A1-120/0 60-80" means:
    #   keep target residues A1-120 as they are, "/0" = start a new chain,
    #   then build a brand-new chain of 60-80 residues (the binder).
    contig = f"[{S['target_range']}/0 {S['binder_length']}]"
    workers = workers_for(S, "rfdiffusion")
    counts = [len(c) for c in split_by_weight(range(S["num_backbones"]), workers)]   # designs per GPU
    commands, first = [], 0
    for w, count in zip(workers, counts):
        if count:
            commands.append((w["cmd"] + [
                f"inference.output_prefix={folder}/design",       # files become design_0.pdb, design_1.pdb ...
                f"inference.design_startnum={first}",              # this GPU numbers its designs from here
                f"inference.input_pdb={Path(S['target_pdb']).resolve()}",
                f"contigmap.contigs={contig}",
                f"ppi.hotspot_res=[{S['hotspots']}]",              # target residues the binder should touch
                f"inference.num_designs={count}",
                f"denoiser.noise_scale_ca={S['rfd_noise']}",       # less noise = cleaner but less varied backbones
                f"denoiser.noise_scale_frame={S['rfd_noise']}",
            ], w.get("env")))
        first += count
    run_parallel(commands)
    backbones = sorted(folder.glob("design_*.pdb"))
    print(f"  -> {len(backbones)} backbones made")
    return backbones


# ----------------------------------------------------------------------------
# Step 2: ProteinMPNN  (make sequences)
# ----------------------------------------------------------------------------
def read_mpnn_fasta(fasta, binder_length):
    """ProteinMPNN writes a .fa file. Record 1 = the input, records 2.. = new sequences.
    Only the redesigned chain is listed (if several chains, they are joined by "/").
    We pick the part that is the binder."""
    records, header = [], None
    for line in Path(fasta).read_text().splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        elif line.strip():
            records[-1][1] += line.strip()
    native_parts = records[0][1].split("/")
    found = []
    for header, seq in records[1:]:
        parts = seq.split("/")
        # binder = the part with the binder's length that CHANGED compared to the input
        # (the input binder is all Glycine, the target part never changes)
        candidates = [p for i, p in enumerate(parts) if len(p) == binder_length and p != native_parts[i]]
        score = float(re.search(r"(?<!\w)score=([\d.]+)", header).group(1))  # ProteinMPNN score: lower = better
        found.append((candidates[0], score))
    return found


def design_sequences(S, worker, pdbs, folder):
    """Run ProteinMPNN on a list of backbones, one after another, on ONE worker (= one GPU)."""
    rows = []
    for pdb in pdbs:
        chains = read_chains(pdb)
        binder, target = find_binder_and_target(chains)
        run(worker["cmd"] + [
            "--pdb_path", pdb.resolve(),
            "--pdb_path_chains", binder,                   # ONLY redesign the binder; target stays fixed
            "--out_folder", (folder / pdb.stem).resolve(),
            "--num_seq_per_target", S["seqs_per_backbone"],
            "--sampling_temp", "0.1",                      # low = safer sequences, high = more varied
            "--seed", "42",
            "--batch_size", "1",
        ], env=worker.get("env"))
        fasta = folder / pdb.stem / "seqs" / f"{pdb.stem}.fa"
        for i, (seq, score) in enumerate(read_mpnn_fasta(fasta, len(chains[binder])), start=1):
            rows.append({"job": f"{pdb.stem}_s{i}", "backbone": str(pdb.resolve()),
                         "binder_chain": binder, "target_chain": target,
                         "binder_seq": seq, "target_seq": sequence_of(chains[target]),
                         "mpnn_score": score})
    return rows


def step2_mpnn(S, backbones, out):
    folder = out / "2_mpnn"
    workers = workers_for(S, "mpnn")
    shares = split_by_weight(backbones, workers)
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:   # one thread per GPU, running at the same time
        results = list(pool.map(lambda args: design_sequences(S, *args, folder), zip(workers, shares)))
    rows = sorted((r for part in results for r in part),
                  key=lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["job"])])
    write_csv(folder / "sequences.csv", rows)
    print(f"  -> {len(rows)} sequences made")
    return rows


# ----------------------------------------------------------------------------
# Step 3 (optional route): AlphaFold SERVER (alphafoldserver.com, a website; nothing to install)
#   afs_export : writes afserver_jobs.json  -> you upload it on the website (Upload JSON)
#   afs_import : reads the .zip files you download from the website, puts them where "score_afs" expects them
# The only route that gives iPTM. The site has a daily job limit, so send it only your best designs.
# ----------------------------------------------------------------------------
def afs_files(out, name):
    """Where the imported AlphaFold Server results for one job are kept."""
    d = out / "3_afserver_outputs" / name
    return d / f"{name}_model.cif", d / f"{name}_summary_confidences.json", d / f"{name}_confidences.json"



def step3_afserver_export(jobs, out):
    payload = [{"name": j["job"], "modelSeeds": [], "dialect": "alphafoldserver", "version": 1,
                "sequences": [{"proteinChain": {"sequence": j["binder_seq"], "count": 1}},    # binder = chain A
                              {"proteinChain": {"sequence": j["target_seq"], "count": 1}}]}   # target = chain B
               for j in jobs]
    path = out / "afserver_jobs.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"  -> {len(payload)} jobs written to {path}\n"
          f"     Upload it at https://alphafoldserver.com (Upload JSON). The site has a daily job limit,\n"
          f"     so for big runs export only your best ProteinMPNN sequences. Download the results (.zip files)\n"
          f"     into a folder, then run:  --stages afs_import --afserver_zips <that folder>")


def step3_afserver_import(zip_folder, jobs, out):
    import zipfile
    for j in jobs:
        name = j["job"]
        zips = list(Path(zip_folder).glob(f"fold_{name}.zip"))
        if not zips:
            print(f"  ! no zip found for {name} (expected fold_{name}.zip), skipped")
            continue
        dest = afs_files(out, name)[0].parent
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as z:   # model_0 = the best of the 5 models the server returns
            (dest / f"{name}_model.cif").write_bytes(z.read(f"fold_{name}_model_0.cif"))
            (dest / f"{name}_summary_confidences.json").write_bytes(z.read(f"fold_{name}_summary_confidences_0.json"))
            (dest / f"{name}_confidences.json").write_bytes(z.read(f"fold_{name}_full_data_0.json"))
    print("  -> imported. Now run:  --stages score_afs")


# ----------------------------------------------------------------------------
# Step 3 (main route): AlphaFold2 "initial guess" from Bennett et al. (github.com/nrbennet/dl_binder_design)
#   It reports pae_interaction (lower = better; the score the paper found most predictive) instead of
#   iPTM, because it uses the AF2 monomer model.
# ----------------------------------------------------------------------------
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS",
         "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
         "Y": "TYR", "V": "VAL"}


def make_af2_pdb(job, out_path):
    """Rewrite the RFdiffusion backbone the way the AF2 script wants it:
    binder FIRST (chain A, our new ProteinMPNN sequence), then the target (chain B),
    with residue numbers that never repeat (binder 1..N, target N+1..)."""
    atoms = {}   # chain -> list of (residue_key, line)
    for line in Path(job["backbone"]).read_text().splitlines():
        if line.startswith("ATOM"):
            atoms.setdefault(line[21], []).append((line[22:27], line))
    new_lines, number = [], 0
    for new_chain, old_chain in (("A", job["binder_chain"]), ("B", job["target_chain"])):
        seq, last_key, i = job["binder_seq"], None, -1
        for key, line in atoms[old_chain]:
            if key != last_key:                      # a new residue starts
                number, i, last_key = number + 1, i + 1, key
            name = THREE[seq[i]] if new_chain == "A" else line[17:20]
            new_lines.append(f"{line[:17]}{name} {new_chain}{number:4d} {line[27:]}")
        new_lines.append("TER")
    Path(out_path).write_text("\n".join(new_lines) + "\nEND\n")


def step3_af2(S, jobs, out):
    workers = workers_for(S, "af2")
    commands = []
    # each GPU gets its own input folder (w0, w1 ...) and its own output folder, so they never collide
    for k, (w, share) in enumerate(zip(workers, split_by_weight(jobs, workers))):
        if not share:
            continue
        in_dir, out_dir = out / "3_af2_inputs" / f"w{k}", out / "3_af2_outputs" / f"w{k}"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        for job in share:
            make_af2_pdb(job, in_dir / f"{job['job']}.pdb")
        commands.append((w["cmd"] + ["-pdbdir", in_dir, "-outpdbdir", out_dir, "-recycle", S.get("af2_recycles", 3),
                                     "-scorefilename", out_dir / "out.sc", "-checkpoint_name", out_dir / "check.point"],
                         w.get("env")))
    run_parallel(commands)


def read_af2_scores(af2_out_dir):
    """Every GPU wrote an out.sc (a Rosetta score file: 'SCORE: col1 col2 ... description').
    Merge them all -> {description: {column: value}}"""
    merged = {}
    for score_file in sorted(Path(af2_out_dir).glob("w*/out.sc")):
        lines = [l.split() for l in score_file.read_text().splitlines() if l.startswith("SCORE:")]
        header = lines[0][1:]                               # drop the "SCORE:" label
        for r in lines[1:]:
            row = dict(zip(header, r[1:]))
            merged[row["description"]] = row
    return merged


# ----------------------------------------------------------------------------
# Step 4: Score  (iPTM, pLDDT, RMSD)
# ----------------------------------------------------------------------------
def make_fitter(moving, fixed):
    """Kabsch algorithm: find the rotation + shift that best lays `moving` points on `fixed` points.
    Returns a function that applies that same move to any other set of points."""
    mc, fc = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    flip = np.sign(np.linalg.det(u @ vt))          # avoids accidentally producing a mirror image
    rot = u @ np.diag([1, 1, flip]) @ vt
    return lambda xyz: (xyz - mc) @ rot + fc


def rmsd(a, b):
    """Root-mean-square distance between matching points (smaller = more similar)."""
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def score_job_afs(job, out):
    """Score one design from its imported AlphaFold Server result (iPTM, pLDDT, RMSD)."""
    cif, summary_file, conf_file = afs_files(out, job["job"])
    summary = json.loads(summary_file.read_text())
    conf = json.loads(conf_file.read_text())

    # --- iPTM: straight from the server's summary file ---
    iptm = summary["iptm"]

    # --- pLDDT: one value per atom; average over the binder's atoms (chain A) ---
    plddt = np.array(conf["atom_plddts"])
    is_binder = np.array(conf["atom_chain_ids"]) == "A"

    # --- RMSD: predicted model vs the backbone that went into ProteinMPNN ---
    ref = read_chains(job["backbone"])                      # RFdiffusion / ProteinMPNN backbone
    pred = read_chains(cif)                                 # prediction (binder = A, target = B)
    ref_binder = np.array([xyz for _, xyz in ref[job["binder_chain"]]])
    ref_target = np.array([xyz for _, xyz in ref[job["target_chain"]]])
    pred_binder = np.array([xyz for _, xyz in pred["A"]])
    pred_target = np.array([xyz for _, xyz in pred["B"]])
    if pred_binder.shape != ref_binder.shape or pred_target.shape != ref_target.shape:
        raise ValueError(f"{job['job']}: prediction and backbone have different residue counts")
    move = make_fitter(pred_target, ref_target)             # 1) line up the TARGETS
    rmsd_binder = rmsd(move(pred_binder), ref_binder)       # 2) then measure how far the BINDER moved

    return {"job": job["job"], "iptm": round(iptm, 3),
            "plddt_binder": round(float(plddt[is_binder].mean()), 1),
            "plddt_complex": round(float(plddt.mean()), 1),
            "rmsd_binder": round(rmsd_binder, 2),
            "mpnn_score": job["mpnn_score"], "binder_seq": job["binder_seq"]}


def step4_score_afs(S, jobs, out):
    # only score the jobs whose results were imported (a daily limit means you may send just a few)
    jobs = [j for j in jobs if afs_files(out, j["job"])[1].exists()]
    rows = [score_job_afs(j, out) for j in jobs]
    for r in rows:  # rule-of-thumb pass mark; change the limits in the settings file
        ok = r["iptm"] >= S.get("min_iptm", 0.8) and r["plddt_binder"] >= S["min_plddt"] and r["rmsd_binder"] <= S["max_rmsd"]
        r["pass"] = "PASS" if ok else "fail"
    rows.sort(key=lambda r: r["iptm"], reverse=True)   # best first
    write_csv(out / "results_afserver.csv", rows)
    print(f"\n{'job':<16}{'iptm':>6}{'plddt_binder':>14}{'rmsd_binder':>13}  verdict")
    for r in rows:
        print(f"{r['job']:<16}{r['iptm']:>6}{r['plddt_binder']:>14}{r['rmsd_binder']:>13}  {r['pass']}")
    print(f"\nSaved: {out / 'results_afserver.csv'}   ({sum(r['pass'] == 'PASS' for r in rows)}/{len(rows)} pass)")


def score_job_af2(job, out, af2_scores):
    """Pass/fail ideas as in score_job_afs, but from the AF2 'initial guess' outputs."""
    sc = af2_scores[f"{job['job']}_af2pred"]
    ref = read_chains(job["backbone"])
    pred_file = next((out / "3_af2_outputs").glob(f"w*/{job['job']}_af2pred.pdb"))      # whichever GPU made it
    pred = read_chains(pred_file)                                                       # chain A = binder, B = target
    move = make_fitter(np.array([x for _, x in pred["B"]]), np.array([x for _, x in ref[job["target_chain"]]]))
    rmsd_binder = rmsd(move(np.array([x for _, x in pred["A"]])), np.array([x for _, x in ref[job["binder_chain"]]]))
    return {"job": job["job"], "iptm": "n/a",              # AF2 monomer model has no iPTM
            "pae_interaction": round(float(sc["pae_interaction"]), 2),
            "plddt_binder": round(float(sc["plddt_binder"]), 1),
            "plddt_complex": round(float(sc["plddt_total"]), 1),
            "rmsd_binder": round(rmsd_binder, 2),
            "mpnn_score": job["mpnn_score"], "binder_seq": job["binder_seq"]}


def step4_score_af2(S, jobs, out):
    af2_scores = read_af2_scores(out / "3_af2_outputs")
    rows = [score_job_af2(j, out, af2_scores) for j in jobs if f"{j['job']}_af2pred" in af2_scores]
    for r in rows:
        ok = (r["pae_interaction"] <= S.get("max_pae_interaction", 10) and r["plddt_binder"] >= S["min_plddt"]
              and r["rmsd_binder"] <= S["max_rmsd"])
        r["pass"] = "PASS" if ok else "fail"
    rows.sort(key=lambda r: r["pae_interaction"])          # lower pAE_interaction = better, best first
    write_csv(out / "results_af2.csv", rows)
    print(f"\n{'job':<16}{'pae_inter':>10}{'plddt_binder':>14}{'rmsd_binder':>13}  verdict")
    for r in rows:
        print(f"{r['job']:<16}{r['pae_interaction']:>10}{r['plddt_binder']:>14}{r['rmsd_binder']:>13}  {r['pass']}")
    print(f"\nSaved: {out / 'results_af2.csv'}   ({sum(r['pass'] == 'PASS' for r in rows)}/{len(rows)} pass)")


# ----------------------------------------------------------------------------
# Glue
# ----------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="RFdiffusion -> ProteinMPNN -> AlphaFold2 (Bennett initial guess) binder design")
    ap.add_argument("--settings", default=str(Path(__file__).with_name("settings.py")))
    ap.add_argument("--stages", default="rfd,mpnn,af2,score_af2",
                    help="comma list of: rfd,mpnn,af2,score_af2  |  optional AlphaFold Server route: afs_export,afs_import,score_afs")
    ap.add_argument("--afserver_zips", help="folder with fold_*.zip downloaded from alphafoldserver.com")
    ap.add_argument("--output_dir", help="use this folder instead of the one in the settings file")
    args = ap.parse_args()

    S = runpy.run_path(args.settings)["SETTINGS"]           # load the settings dictionary
    out = Path(args.output_dir or S["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")

    if "rfd" in stages:
        print("\n[1/4] RFdiffusion: designing backbones")
        backbones = step1_rfdiffusion(S, out)
    if "mpnn" in stages:
        print("\n[2/4] ProteinMPNN: designing sequences")
        step2_mpnn(S, sorted((out / "1_rfdiffusion").glob("design_*.pdb")), out)
    jobs = read_csv(out / "2_mpnn" / "sequences.csv") if {"af2", "score_af2", "afs_export", "afs_import", "score_afs"} & set(stages) else []
    if "afs_export" in stages:
        print("\n[3/4] AlphaFold Server: export jobs")
        step3_afserver_export(jobs, out)
    if "afs_import" in stages:
        print("\n[3/4] AlphaFold Server: import results")
        step3_afserver_import(args.afserver_zips, jobs, out)
    if "af2" in stages:
        print("\n[3/4] AlphaFold2 initial guess (Bennett): predicting binder + target")
        step3_af2(S, jobs, out)
    if "score_af2" in stages:
        print("\n[4/4] Scoring (AF2 initial guess)")
        step4_score_af2(S, jobs, out)
    if "score_afs" in stages:
        print("\n[4/4] Scoring (AlphaFold Server results)")
        step4_score_afs(S, jobs, out)


if __name__ == "__main__":
    main()
