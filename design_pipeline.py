"""
Protein binder design pipeline:   RFdiffusion  ->  ProteinMPNN  ->  AlphaFold2 "initial guess" (Bennett lab)

What each tool does (one line each):
  1. RFdiffusion : draws NEW protein backbones (just the "skeleton", no sequence) that stick to your target.
                   (Optional: RFD4-Proteina, "backbone_tool": "rfd4", also designs the sequence, so step 2 is skipped.)
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
import gzip
import json
import os
import re
import runpy
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
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
            expanded.append({"cmd": w["cmd"], "env": env, "weight": w.get("weight", 1),
                             "batch": w.get("batch")})            # optional: designs this worker makes at once (RFD3 / RFD4)
    return expanded


def split_by_weight(items, workers):
    """Cut a list into one chunk per worker, each chunk's size proportional to the worker's weight."""
    items, total, shares, start = list(items), sum(w["weight"] for w in workers), [], 0
    exact = [len(items) * w["weight"] / total for w in workers]
    counts = [int(x) for x in exact]                       # whole items first ...
    leftover = len(items) - sum(counts)                    # ... then the leftovers go to the biggest fractional parts
    for k in sorted(range(len(workers)), key=lambda k: exact[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    for n in counts:
        shares.append(items[start:start + n])
        start += n
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


def find_binder_and_target(chains, pdb=None):
    """Which chain is the binder and which is the target?
    - RFD3 backbones come with a small side file (design_N.json, written by convert_rfd3_output) that says so.
    - RFdiffusion 1 writes the NEW residues as Glycine (no sequence yet), so the chain with the most
      Glycine is the binder and the other chain is the target."""
    if pdb is not None and Path(pdb).with_suffix(".json").exists():
        side = json.loads(Path(pdb).with_suffix(".json").read_text())
        return side["binder_chain"], side["target_chain"]
    if len(chains) != 2:
        raise ValueError(f"Expected 2 chains (binder + target), found {list(chains)}")
    gly_fraction = {c: np.mean([name == "GLY" for name, _ in res]) for c, res in chains.items()}
    binder = max(gly_fraction, key=gly_fraction.get)
    target = [c for c in chains if c != binder][0]
    return binder, target


def sequence_of(residues):
    return "".join(seq1(name) for name, _ in residues)


# ----------------------------------------------------------------------------
# Step 0 (stage "glyco"): where could the TARGET carry N-glycans?   Needs no GPU; takes seconds.
#   1. Sequon scan: Asn-X-Ser/Thr (X is not Pro) in the target chain = where an N-glycan CAN be attached.
#   2. Structure check: is that Asn on the surface? (Shrake-Rupley solvent exposure, about 150 A^2 = fully exposed Asn)
#   3. Optional predictors that say whether the site is really occupied:
#        - ReGlyco "scan" (structure based; runs automatically when "reglyco_cmd" is in the settings)
#        - NetNGlyc (sequence based, human-trained): the DTU website needs a sign-in, so the pipeline writes the FASTA
#          file, you run NetNGlyc there (or with the standalone program) and give the output back
#          with  --netnglyc_output <file>  or  "netnglyc_cmd" in the settings.
#   4. Warnings: sequons near your hotspots, or just outside the kept target range.
#   N-glycosylation only happens to proteins that pass through the secretory pathway (secreted / membrane proteins).
# ----------------------------------------------------------------------------
SEQUON = re.compile(r"(?=(N[^P][ST]))")            # Asn - X (not Pro) - Ser/Thr; the lookahead also finds overlapping motifs
ASN_FULL_SASA = 150.0                               # roughly the solvent-exposed area of a fully exposed Asn (A^2)


def contig_pieces(target_range):
    """'A16-108,A113-151' (or 'A16-108/A113-151') -> [('A', 16, 108), ('A', 113, 151)]; bare numbers are ignored."""
    return [(c, int(a), int(b)) for c, a, b in re.findall(r"([A-Za-z]+)(\d+)-(\d+)", target_range)]


def glyco_scan_baseline(S):
    """Sequon scan + solvent exposure + distance to the hotspots, for the target chain of the input PDB."""
    from Bio.PDB.SASA import ShrakeRupley
    pieces = contig_pieces(S["target_range"])
    if not pieces:
        raise SystemExit("glyco: cannot read target_range (expected something like A16-108,A113-151)")
    chain_id = pieces[0][0]
    structure = PDBParser(QUIET=True).get_structure("target", str(Path(S["target_pdb"]).resolve()))
    ShrakeRupley().compute(structure[0], level="R")                       # sets residue.sasa
    residues = [r for r in structure[0][chain_id] if r.id[0] == " " and "CA" in r]
    by_number = {r.id[1]: r for r in residues}
    kept = lambda n: any(c == chain_id and a <= n <= b for c, a, b in pieces)
    hotspots = []
    for h in S["hotspots"].split(","):
        m = re.fullmatch(r"\s*([A-Za-z]+)(\d+)\s*", h)
        if m and m.group(1) == chain_id and int(m.group(2)) in by_number:
            hotspots.append((h.strip(), by_number[int(m.group(2))]["CA"].coord))
    sequence = "".join(seq1(r.get_resname()) for r in residues)
    rows = []
    for m in SEQUON.finditer(sequence):
        trio = residues[m.start():m.start() + 3]
        numbers = [r.id[1] for r in trio]
        if numbers[1] != numbers[0] + 1 or numbers[2] != numbers[1] + 1:
            continue                                                       # a gap in the file: not a real motif
        asn = trio[0]
        near = min(((float(np.linalg.norm(asn["CA"].coord - xyz)), name) for name, xyz in hotspots), default=(None, ""))
        rows.append({"chain": chain_id, "residue": numbers[0], "motif": m.group(1), "in_target_range": kept(numbers[0]),
                     "sasa": round(float(asn.sasa), 1), "rsa": round(float(asn.sasa) / ASN_FULL_SASA, 2),
                     "exposed": float(asn.sasa) / ASN_FULL_SASA >= 0.2,
                     "nearest_hotspot": near[1], "hotspot_dist_A": None if near[0] is None else round(near[0], 1)})
    index_to_number = {i + 1: r.id[1] for i, r in enumerate(residues)}     # FASTA position -> PDB residue number
    return rows, chain_id, sequence, index_to_number


def parse_netnglyc(text, index_to_number):
    """Read NetNGlyc's text output. Lines look like:  <name> <position> <motif> <potential> (<n>/<m>) <result>
    This is written from the public description of the output, NOT verified against a real output file; it prints
    a warning when it recognises no line so that a different format is noticed."""
    found = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\S+)\s+(\d+)\s+([A-Z]+)\s+([0-9]*\.[0-9]+)\s+\((\d+)/(\d+)\)\s+(\S+)", line)
        if m and int(m.group(2)) in index_to_number:
            found[index_to_number[int(m.group(2))]] = {"netnglyc_potential": float(m.group(4)),
                                                        "netnglyc_jury": f"{m.group(5)}/{m.group(6)}",
                                                        "netnglyc_call": m.group(7)}
    if not found:
        print("  ! no NetNGlyc result lines recognised in that file (is it the text output of NetNGlyc 1.0?)")
    return found


def reglyco_scan(S, folder):
    """Run `reglyco scan` on the target PDB (CPU only, seconds). It attaches a GlcNAc sugar to every N-X-S/T sequon and
    checks which sites can carry a glycan without clashing. Its own caveat: "structural accessibility only; not proof of
    biological glycosylation". Returns {(chain, residue number): {...columns...}}.
    The GlcNAc conformers come from the GlycoShape website the first time and are cached: set "reglyco_cache" to a folder that
    is kept between runs, and "reglyco_offline": True on machines without internet (after one online run)."""
    out = folder / "reglyco_scan"
    cache = Path(S.get("reglyco_cache", folder / "reglyco_cache"))
    cmd = [str(c) for c in S["reglyco_cmd"]] + ["scan", "--protein", str(Path(S["target_pdb"]).resolve()), "--output", str(out),
                                                 "--overwrite", "--cache", str(cache), "--protein-cache", str(cache / "proteins")]
    if S.get("reglyco_offline"):
        cmd += ["--offline", "--protein-offline"]
    run(cmd, env={"CUDA_VISIBLE_DEVICES": "", **(S.get("reglyco_env") or {})})
    scan = json.loads((out / "scan.json").read_text())
    sites = {}
    for site in scan.get("sites", []):
        asn = site["sequon"]["asparagine"]
        sites[(asn["chain"], asn["number"])] = {"reglyco_accessible": bool(site["accessible"]),
                                                "reglyco_joint": bool(site.get("joint_subset")),
                                                "reglyco_steric_score": round(float(site.get("steric_score", 0.0)), 2)}
    return sites


def step0_glyco(S, out, netnglyc_output=None):
    folder = out / "0_glyco"
    folder.mkdir(parents=True, exist_ok=True)
    rows, chain_id, sequence, index_to_number = glyco_scan_baseline(S)
    (folder / "target.fasta").write_text(f">target_chain_{chain_id}\n{sequence}\n")          # for NetNGlyc (web or standalone)
    write_csv(folder / "target_numbering.csv", [{"fasta_position": i, "pdb_residue": n} for i, n in index_to_number.items()])

    predictors = {"structure (solvent exposure)": True}
    # --- optional: ReGlyco scan (automatic) ---
    reglyco = None
    if S.get("reglyco_cmd"):
        try:
            reglyco = reglyco_scan(S, folder)
        except (subprocess.CalledProcessError, OSError, ValueError) as error:       # e.g. no internet on a compute node
            print(f"  ! ReGlyco failed ({error}); continuing with the built-in scan only. It downloads glycan data from "
                  f"glycoshape.org the first time: run `--stages glyco` once on a machine with internet, then set "
                  f"\"reglyco_offline\" (REGLYCO_OFFLINE=1) for jobs on machines without it.")
    if reglyco is not None:
        for r in rows:
            r.update(reglyco.get((r["chain"], r["residue"]), {"reglyco_accessible": "", "reglyco_joint": "", "reglyco_steric_score": ""}))
        extra = sorted(n for c, n in reglyco if c == chain_id and n not in {r["residue"] for r in rows})
        if extra:
            print(f"  ! ReGlyco also lists sequons that the built-in scan did not: {extra} (check the PDB numbering)")
        predictors["ReGlyco scan"] = True
    # --- optional: NetNGlyc (standalone program, or a result file from the website) ---
    text = None
    if S.get("netnglyc_cmd"):
        text = subprocess.run([str(c) for c in S["netnglyc_cmd"]] + [str(folder / "target.fasta")], capture_output=True, text=True, check=True).stdout
    elif netnglyc_output or S.get("netnglyc_output"):
        text = Path(netnglyc_output or S["netnglyc_output"]).read_text()
    if text is not None:
        (folder / "netnglyc_output.txt").write_text(text)
        scores = parse_netnglyc(text, index_to_number)
        for r in rows:
            r.update(scores.get(r["residue"], {"netnglyc_potential": "", "netnglyc_jury": "", "netnglyc_call": ""}))
            r["netnglyc_glycosylated"] = bool(r["netnglyc_potential"] != "" and r["netnglyc_potential"] > 0.5)   # NetNGlyc default cut-off
        predictors["NetNGlyc"] = True

    for r in rows:
        # Exposure alone cannot say a glycan is attached, so without a predictor the site is only a "candidate".
        # With predictors: how many of those that ran (plus exposure) support "this site is glycosylated"?
        if len(predictors) == 1:
            r["support"] = "-"
            r["verdict"] = "candidate (exposed)" if r["exposed"] else "candidate (buried)"
            continue
        votes = [r["exposed"]]
        if "ReGlyco scan" in predictors:
            votes.append(r.get("reglyco_accessible") is True)
        if "NetNGlyc" in predictors:
            votes.append(r.get("netnglyc_glycosylated") is True)
        r["support"] = f"{sum(votes)}/{len(votes)}"
        r["verdict"] = "likely" if all(votes) else ("possible" if any(votes) else "unlikely")
    warn_dist = float(S.get("glyco_warn_A", 15.0))
    for r in rows:
        r["warning"] = "; ".join(w for w in (
            "outside the kept target range (this glycan attachment point is not in the design model)" if not r["in_target_range"] else "",
            f"within {warn_dist:g} A of hotspot {r['nearest_hotspot']}: a glycan here could block the binder"
            if r["verdict"] not in ("unlikely", "candidate (buried)") and r["hotspot_dist_A"] is not None
            and r["hotspot_dist_A"] <= warn_dist else "") if w)
    if rows:
        write_csv(folder / "glyco_sites.csv", rows)

    print(f"  target chain {chain_id}: {len(sequence)} residues in the file; predictors used: {', '.join(predictors)}")
    if not rows:
        print("  -> no N-X-S/T sequon in the target: no N-glycan attachment site.")
    for r in rows:
        print(f"  Asn{r['residue']} ({r['motif']})  exposure {r['rsa']:.2f}  verdict: {r['verdict']} ({r['support']})"
              f"  nearest hotspot {r['nearest_hotspot'] or '-'} at {r['hotspot_dist_A'] if r['hotspot_dist_A'] is not None else '-'} A"
              f"{'  <-- ' + r['warning'] if r['warning'] else ''}")
    if rows:
        print(f"  Saved: {folder / 'glyco_sites.csv'}   (the FASTA for NetNGlyc is {folder / 'target.fasta'})")
        print("  Note: these are predictions (ReGlyco itself says: structural accessibility only, not proof of glycosylation), and "
              "N-glycosylation only happens to proteins that enter the secretory pathway: for an intracellular target, ignore this table.")
    if "NetNGlyc" not in predictors:
        print("  NetNGlyc not used: run it on 0_glyco/target.fasta and re-run with  --netnglyc_output <its text output>")
    if "ReGlyco scan" not in predictors:
        print("  ReGlyco not used: add 'reglyco_cmd' to the settings (see README) to add its structure-based scan")
    return rows


# ----------------------------------------------------------------------------
# Step 1: backbones  (RFdiffusion 3 is the default; RFdiffusion 1 is still available)
#   Both tools end up writing  1_rfdiffusion/design_0.pdb, design_1.pdb ...  so the later steps do not care which was used.
#   Choose with the setting  "backbone_tool": "rfd3" | "rfdiffusion"  (default: rfd3 if the settings have rfd3_cmd / rfd3_workers).
# ----------------------------------------------------------------------------
def fresh_backbone_folder(out):
    """1_rfdiffusion/ must not hold designs from an earlier run: both tools SKIP designs that already exist, and
    the later steps would then mix old and new backbones."""
    folder = out / "1_rfdiffusion"
    folder.mkdir(parents=True, exist_ok=True)
    old = sorted(folder.glob("design_*.pdb"))
    if old:
        raise SystemExit(f"{folder} already holds {len(old)} backbones from an earlier run.\n"
                         f"Use a NEW --output_dir: mixing runs gives wrong results.")
    return folder


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(path))]


def rfd3_spec(S):
    """The design recipe RFD3 reads (JSON). The binder is designed first, then a chain break (/0), then the target.
    Hotspots are given as residues; 'TIP' tells RFD3 to use each residue's tip atom (or write 'ALL')."""
    contig = f"{S['binder_length']},/0,{S['target_range'].replace('/', ',')}"
    atoms = S.get("rfd3_hotspot_atoms", "TIP")
    hotspots = {r.strip(): atoms for r in S["hotspots"].split(",") if r.strip()}
    spec = {"input": str(Path(S["target_pdb"]).resolve()), "contig": contig, "select_hotspots": hotspots,
            "infer_ori_strategy": "hotspots", "is_non_loopy": True}
    spec.update(S.get("rfd3_spec_extra", {}))       # advanced: any other RFD3 input field
    return {"design": spec}


def convert_rfd3_output(cif_gz, pdb_out):
    """RFD3 writes designs as .cif.gz + .json (binder = first chain, target renumbered from 1). Convert one to the
    layout used here: target = chain A with its ORIGINAL residue numbers, binder = chain B, plus a small
    design_N.json side file saying which chain is which (RFD3 binders carry a real sequence, so the 'all glycine'
    trick used for RFdiffusion 1 does not work)."""
    cif_gz = Path(cif_gz)
    meta = json.loads(cif_gz.with_name(cif_gz.name.replace(".cif.gz", ".json")).read_text())
    with gzip.open(cif_gz, "rt") as handle:
        model = MMCIFParser(QUIET=True).get_structure("x", handle)[0]
    split = lambda s: (re.fullmatch(r"([A-Za-z]+)(-?\d+)", s).group(1), int(re.fullmatch(r"([A-Za-z]+)(-?\d+)", s).group(2)))
    new_to_orig = {split(new): split(orig) for orig, new in meta["diffused_index_map"].items()}   # e.g. B1 <- A16
    target_chains = {c for c, _ in new_to_orig}
    chain_ids = [c.id for c in model]
    if len(target_chains) != 1 or len(chain_ids) != 2:
        raise ValueError(f"{cif_gz.name}: expected one binder chain + one target chain, found chains {chain_ids}")
    target_out = target_chains.pop()
    binder_out = [c for c in chain_ids if c != target_out][0]

    residues = [r for r in model[target_out] if r.id[0] == " "]
    restore = all((target_out, r.id[1]) in new_to_orig for r in residues)       # put the original numbers back
    structure = Structure("design")
    structure.add(Model(0))
    for new_id, old_id in (("A", target_out), ("B", binder_out)):
        chain = Chain(new_id)
        for r in [r for r in model[old_id] if r.id[0] == " "]:
            copy = r.copy()
            number = new_to_orig[(old_id, r.id[1])][1] if (old_id == target_out and restore) else r.id[1]
            copy.id = (" ", number, " ")
            chain.add(copy)
        structure[0].add(chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_out))
    Path(pdb_out).with_suffix(".json").write_text(json.dumps(
        {"binder_chain": "B", "target_chain": "A", "source": cif_gz.name, "rfd3_metrics": meta.get("metrics", {})}, indent=1))


def step1_rfd3(S, out):
    folder = fresh_backbone_folder(out)
    raw = out / "1_rfd3_raw"                                    # RFD3's own files, one sub-folder per worker
    raw.mkdir(parents=True, exist_ok=True)
    spec_path = raw / "spec.json"
    spec_path.write_text(json.dumps(rfd3_spec(S), indent=1))
    workers = workers_for(S, "rfd3")
    counts = [len(c) for c in split_by_weight(range(S["num_backbones"]), workers)]      # designs per worker
    batch = max(1, int(S.get("rfd3_batch_size", 1)))   # designs RFD3 makes at once (they share one binder length)
    commands, plan = [], []
    for k, (w, count) in enumerate(zip(workers, counts)):
        if not count:
            continue
        size = min(int(w.get("batch") or batch), count)         # a small card can have its own, smaller batch
        wdir = raw / f"w{k}"
        cmd = w["cmd"] + [f"out_dir={wdir}", f"inputs={spec_path}", f"diffusion_batch_size={size}",
                          f"n_batches={-(-count // size)}", "skip_existing=False", "prevalidate_inputs=True"]
        if S.get("rfd3_ckpt"):
            cmd.append(f"ckpt_path={S['rfd3_ckpt']}")
        cmd += [str(x) for x in S.get("rfd3_overrides", [])]        # advanced: any other RFD3 setting, e.g. inference_sampler.step_scale=1.5
        commands.append((cmd, w.get("env")))
        plan.append((wdir, count))
    run_parallel(commands)
    made = 0
    for wdir, count in plan:
        files = sorted(wdir.glob("*.cif.gz"), key=natural_key)[:count]
        if len(files) < count:
            raise SystemExit(f"RFD3 wrote {len(files)} designs in {wdir}, expected {count}: see the messages above")
        for f in files:
            convert_rfd3_output(f, folder / f"design_{made}.pdb")
            made += 1
    print(f"  -> {made} backbones made (RFD3)")
    return sorted(folder.glob("design_*.pdb"))


def backbone_tool(S):
    """Which tool makes the backbones: the setting "backbone_tool", else the first tool the settings give a command for
    (RFD4, then RFD3, then RFdiffusion 1)."""
    return S.get("backbone_tool") or ("rfd4" if (S.get("rfd4_workers") or S.get("rfd4_cmd")) else
                                      "rfd3" if (S.get("rfd3_workers") or S.get("rfd3_cmd")) else "rfdiffusion")


# ---- RFD4-Proteina: designs the binder's SEQUENCE and structure together, so ProteinMPNN is not needed ---------------
def rfd4_target(S, folder):
    """RFD4 reads CIF (not PDB) and refuses to bond across a gap in the target, so every kept piece of the target becomes
    its own chain (A, B, C ...) with its ORIGINAL residue numbers. Writes target.cif and returns
    [(new_chain, original_chain, [original residue numbers])]."""
    pieces = contig_pieces(S["target_range"])
    if not pieces or re.search(r"(?:^|[,/])\s*\d+\s*(?:$|[,/])", S["target_range"]):
        raise SystemExit("rfd4: target_range must be only kept pieces such as A16-108,A113-151 "
                         "(a bare number that designs new residues in a gap is not supported with RFD4)")
    model = PDBParser(QUIET=True).get_structure("target", str(Path(S["target_pdb"]).resolve()))[0]
    # The CIF is written by hand: RFD4 takes the residue numbers from label_seq_id, and Biopython's writer renumbers that
    # column from 1 in every chain. Here label_seq_id and auth_seq_id both hold the ORIGINAL numbers.
    rows, layout, serial = [], [], 0
    for entity, (letter, (chain_id, first, last)) in enumerate(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", pieces), start=1):
        have = {r.id[1]: r for r in model[chain_id] if r.id[0] == " " and "CA" in r}
        missing = [n for n in range(first, last + 1) if n not in have]
        if missing:
            raise SystemExit(f"rfd4: target residues {chain_id}{missing[0]}... are not in the PDB file "
                             f"(piece {chain_id}{first}-{last}); leave them out of target_range, e.g. A16-108,A113-151")
        for n in range(first, last + 1):
            res = have[n].get_resname()
            for atom in have[n]:
                element = (atom.element or atom.get_id()[0]).strip().upper()
                if element in ("H", "D"):
                    continue
                serial += 1
                x, y, z = atom.coord
                rows.append(f"ATOM {serial} {element} {atom.get_id()} . {res} {letter} {entity} {n} ? {x:.3f} {y:.3f} {z:.3f} "
                            f"1.00 {atom.get_bfactor():.2f} {n} {res} {letter} {atom.get_id()} 1")
        layout.append((letter, chain_id, list(range(first, last + 1))))
    head = ["data_target", "loop_"] + ["_atom_site." + c for c in
            "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id "
            "pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id auth_comp_id auth_asym_id "
            "auth_atom_id pdbx_PDB_model_num".split()]
    (folder / "target.cif").write_text("\n".join(head + rows) + "\n#\n")
    return layout


def rfd4_spec(S, layout, target_cif):
    """The recipe RFD4 reads, as YAML text. Binder first, then each target piece after a chain break (/0).
    conditions: the target's backbone and sequence are given (C_CRDGRP, C_SEQ) and the hotspot residues are marked (C_HOT).
    The keys 1 and true are written bare on purpose: that is how RFD4's own examples spell them."""
    q = json.dumps                                            # a JSON string is also a valid YAML string
    letters = [letter for letter, _, _ in layout]
    contig = ", /0, ".join([str(S["binder_length"])] + [f"{l}{numbers[0]}-{numbers[-1]}" for l, _, numbers in layout])
    hot = {}
    for h in S["hotspots"].split(","):
        m = re.fullmatch(r"\s*([A-Za-z]+)(\d+)\s*", h)
        if not m:
            continue
        where = [l for l, c, numbers in layout if c == m.group(1) and int(m.group(2)) in numbers]
        if not where:
            raise SystemExit(f"rfd4: hotspot {h.strip()} is not inside target_range {S['target_range']}")
        hot.setdefault(where[0], []).append(m.group(2))
    select = lambda text: "{select: " + q(text) + "}"
    lines = ["design:",
             f"  input: {q(str(Path(target_cif).resolve()))}",
             f"  contig: {q(contig)}",
             "  conditions:",
             "    C_CRDGRP: {1: [" + select(f"sel('[{','.join(letters)}]/*/*/[N,CA,C,O]')") + "]}",
             "    C_SEQ: {true: [" + ", ".join(select(f"sel('{l}')") for l in letters) + "]}",
             "    C_HOT: {true: [" + ", ".join(select(f"sel('{l}/*/[{','.join(n)}]')") for l, n in hot.items()) + "]}"]
    lines += [f"  {key}: {q(value)}" for key, value in S.get("rfd4_spec_extra", {}).items()]   # advanced: other input fields
    return "\n".join(lines) + "\n"


def convert_rfd4_output(generated_cif, pdb_out, layout):
    """RFD4 writes <name>_generated.cif: the new binder chain plus the target pieces, each numbered from 1. Convert it to
    the layout used here: target = chain A (pieces joined, ORIGINAL residue numbers back), binder = chain B (with the sequence
    RFD4 designed), plus the small design_N.json side file saying which chain is which."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(str(generated_cif))             # read the raw table: RFD4's file has no B-factor column, so MMCIFParser fails
    rows = list(zip(d["_atom_site.auth_asym_id"], d["_atom_site.auth_seq_id"], d["_atom_site.label_comp_id"],
                    d["_atom_site.label_atom_id"], d["_atom_site.Cartn_x"], d["_atom_site.Cartn_y"], d["_atom_site.Cartn_z"],
                    d["_atom_site.type_symbol"]))
    pieces = {letter: numbers for letter, _, numbers in layout}
    binder = sorted({r[0] for r in rows} - set(pieces))
    if len(binder) != 1 or set(pieces) - {r[0] for r in rows}:
        raise ValueError(f"{Path(generated_cif).name}: expected target chains {sorted(pieces)} + one binder chain, "
                         f"found {sorted({r[0] for r in rows})}")
    lines, serial = [], 0
    for new_chain, old_chains in (("A", list(pieces)), ("B", binder)):
        for old in old_chains:
            for chain, number, resname, atom, x, y, z, element in rows:
                if chain != old:
                    continue
                if old in pieces:
                    number = pieces[old][int(number) - 1]
                serial += 1
                name = f" {atom:<3}" if len(atom) < 4 and len(element) == 1 else f"{atom:<4}"
                lines.append(f"ATOM  {serial:5d} {name} {resname:>3} {new_chain}{int(number):4d}    "
                             f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00          {element:>2}")
        lines.append("TER")
    Path(pdb_out).write_text("\n".join(lines) + "\nEND\n")
    Path(pdb_out).with_suffix(".json").write_text(json.dumps(
        {"binder_chain": "B", "target_chain": "A", "source": Path(generated_cif).parent.name}, indent=1))


def step1_rfd4(S, out):
    folder = fresh_backbone_folder(out)
    raw = out / "1_rfd4_raw"                                   # RFD4's own files, one sub-folder per worker
    raw.mkdir(parents=True, exist_ok=True)
    layout = rfd4_target(S, raw)
    spec_path = raw / "spec.yaml"
    spec_path.write_text(rfd4_spec(S, layout, raw / "target.cif"))
    workers = workers_for(S, "rfd4")
    counts = [len(c) for c in split_by_weight(range(S["num_backbones"]), workers)]      # designs per worker
    batch = max(1, int(S.get("rfd4_batch_size", 8)))           # designs made at once
    commands, plan = [], []
    for k, (w, count) in enumerate(zip(workers, counts)):
        if not count:
            continue
        wdir = raw / f"w{k}"
        # RFD4's default seed is fixed, so two workers would make the SAME designs: give each worker its own seed
        cmd = w["cmd"] + [f"inputs={spec_path}", f"out_dir={wdir}", f"num_replicates={count}",
                          f"batch_size={min(int(w.get('batch') or batch), count)}",
                          f"seed={int(S.get('rfd4_seed', 1)) + 1000 * k}"]
        if not any(str(a).startswith("inference_experiment=") for a in w["cmd"]):
            cmd.append("~metrics")                  # (the "quickstart" experiment already skips the metrics)
        if S.get("rfd4_ckpt"):
            cmd.append(f"hf_ckpt_path={S['rfd4_ckpt']}")
        if S.get("rfd4_revision"):
            cmd.append(f"hf_ckpt_revision={S['rfd4_revision']}")        # pin the exact model version (the model card asks for it)
        cmd += [str(x) for x in S.get("rfd4_overrides", [])]
        commands.append((cmd, w.get("env")))
        plan.append((wdir, count))
    run_parallel(commands)
    made = 0
    for wdir, count in plan:
        files = sorted((wdir / "inference_samples").glob("*/*_generated.cif"), key=natural_key)[:count]
        if len(files) < count:
            raise SystemExit(f"RFD4 wrote {len(files)} designs in {wdir}, expected {count}: see the messages above")
        for f in files:
            convert_rfd4_output(f, folder / f"design_{made}.pdb", layout)
            made += 1
    print(f"  -> {made} designs made (RFD4: backbone AND sequence)")
    return sorted(folder.glob("design_*.pdb"))


def step1_backbones(S, out):
    tool = backbone_tool(S)
    print(f"  backbone tool: {tool}")
    return {"rfd3": step1_rfd3, "rfd4": step1_rfd4}.get(tool, step1_rfdiffusion)(S, out)


def step1_rfdiffusion(S, out):
    folder = fresh_backbone_folder(out)
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
        # (the target part never changes)
        candidates = [p for i, p in enumerate(parts) if len(p) == binder_length and p != native_parts[i]]
        if not candidates:   # ProteinMPNN happened to reproduce the input sequence exactly
            candidates = [p for p in parts if len(p) == binder_length]
        score = float(re.search(r"(?<!\w)score=([\d.]+)", header).group(1))  # ProteinMPNN score: lower = better
        found.append((candidates[0], score))
    return found


def design_sequences(S, worker, pdbs, folder):
    """Run ProteinMPNN on a list of backbones, one after another, on ONE worker (= one GPU)."""
    rows = []
    for pdb in pdbs:
        chains = read_chains(pdb)
        binder, target = find_binder_and_target(chains, pdb)
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


def native_sequences(backbones, out):
    """RFD4 designs the binder's sequence together with its structure: use that sequence as it is (no ProteinMPNN)."""
    rows = []
    for pdb in backbones:
        chains = read_chains(pdb)
        binder, target = find_binder_and_target(chains, pdb)
        rows.append({"job": f"{pdb.stem}_native", "backbone": str(pdb.resolve()), "binder_chain": binder,
                     "target_chain": target, "binder_seq": sequence_of(chains[binder]),
                     "target_seq": sequence_of(chains[target]), "mpnn_score": ""})
    rows.sort(key=lambda r: natural_key(r["job"]))
    (out / "2_mpnn").mkdir(parents=True, exist_ok=True)
    write_csv(out / "2_mpnn" / "sequences.csv", rows)
    print(f"  -> {len(rows)} sequences taken from RFD4 (ProteinMPNN skipped; set \"rfd4_mpnn\": True to redesign them with it)")
    return rows


def step2_mpnn(S, backbones, out):
    if backbone_tool(S) == "rfd4" and not S.get("rfd4_mpnn"):
        return native_sequences(backbones, out)
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
            # the binder gets a NEW sequence from ProteinMPNN, so keep only its backbone (RFD3 binders also carry side
            # chains of the residues RFD3 chose, which would no longer match the new residue names)
            if new_chain == "A" and line[12:16].strip() not in ("N", "CA", "C", "O"):
                continue
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
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")   # every column that appears
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="RFdiffusion -> ProteinMPNN -> AlphaFold2 (Bennett initial guess) binder design")
    ap.add_argument("--settings", default=str(Path(__file__).with_name("settings.py")))
    ap.add_argument("--stages", default="glyco,rfd,mpnn,af2,score_af2",
                    help="comma list of: glyco (scan the target for N-glycosylation sites), rfd,mpnn,af2,score_af2  |  "
                         "optional AlphaFold Server route: afs_export,afs_import,score_afs")
    ap.add_argument("--afserver_zips", help="folder with fold_*.zip downloaded from alphafoldserver.com")
    ap.add_argument("--netnglyc_output", help="text output of NetNGlyc 1.0 for 0_glyco/target.fasta (stage glyco)")
    ap.add_argument("--output_dir", help="use this folder instead of the one in the settings file")
    args = ap.parse_args()

    S = runpy.run_path(args.settings)["SETTINGS"]           # load the settings dictionary
    out = Path(args.output_dir or S["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")

    if "glyco" in stages:
        print("\n[0/4] Target glycosylation scan (N-X-S/T sequons)")
        step0_glyco(S, out, args.netnglyc_output)
    if "rfd" in stages:
        print("\n[1/4] Backbones (RFdiffusion 3 by default; RFD4 also designs the sequence): designing backbones")
        backbones = step1_backbones(S, out)
    if "mpnn" in stages:
        print("\n[2/4] ProteinMPNN: designing sequences (with RFD4 the sequences already exist)")
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
