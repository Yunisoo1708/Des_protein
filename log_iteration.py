"""
log_iteration.py -- turn one finished design run into notes for an Obsidian vault (folder binder-design/).

    python log_iteration.py --run <output_dir> --title "RFD4 + AF3 reward search, 50 designs" --goal "does the search beat plain RFD4?" \
        --parent 2026-10-27-rfd4-plain-50 --meta generator="RFD4 + beam search" --vault /path/to/vault

Each run becomes ONE iteration note plus a small note per passing design (or the top 3 if nothing passes), the results table, and a refreshed
iterations.md overview:

    binder-design/iterations/<slug>.md       what was tried, setup, generated numbers, YOUR/THE AGENT'S findings, next step, provenance
    binder-design/designs/<slug>--<job>.md   sequence + metrics of a notable design, linked to its iteration
    binder-design/data/<slug>-results.csv    the run's results table, so numbers can be re-checked
    binder-design/iterations.md              one-line-per-iteration overview (rebuilt from the notes' properties; no plugin needed)

Rules this script keeps (the vault records past fabrication, so provenance matters):
  * Everything under "Results" is generated from files in the run folder; "Findings" and "Next step" are left as TODO for a person or agent
    to write, and the note stays `status: draft` until someone sets `status: reviewed`.
  * It never overwrites an existing note (your edits are safe); pass --overwrite to replace one on purpose.
  * It writes only inside <vault>/binder-design/, never to index.md, log.md or any other note.
  * Without --vault the notes go to <run>/iteration_note/binder-design/ (copy that folder into the vault); --reindex <vault> rebuilds iterations.md.
  * With --outbox <dir> the notes are queued in <dir>/<slug>/ and a DONE marker is written last; vault_pull.sh (on the laptop) copies queued
    iterations into the vault and marks them PULLED. Only runs started with --log are ever queued.

Reads: results_af3.csv | results_af2.csv | results_afserver.csv, run_info.json (written by design_pipeline.py), run_meta.json (written by
setup.sh run-rfd4). Standard library only.
"""
import argparse
import csv
import datetime
import hashlib
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

FOLDER = "binder-design"
RESULT_FILES = ("results_af3.csv", "results_af2.csv", "results_afserver.csv")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def stats(rows):
    col = lambda k: [v for v in (fnum(r.get(k)) for r in rows) if v is not None]
    iptm, pae, pl, rm = col("iptm"), col("pae_interaction"), col("plddt_binder"), col("rmsd_binder")
    s = {"n_designs": len(rows), "n_pass": sum(r.get("pass") == "PASS" for r in rows)}
    if iptm:
        s.update(iptm_ge_0_8=sum(x >= 0.8 for x in iptm), iptm_ge_0_7=sum(x >= 0.7 for x in iptm), iptm_ge_0_5=sum(x >= 0.5 for x in iptm),
                 median_iptm=round(statistics.median(iptm), 2), max_iptm=round(max(iptm), 2))
    if pae:
        s.update(median_pae=round(statistics.median(pae), 1), best_pae=round(min(pae), 1))
    if pl:
        s.update(median_plddt=round(statistics.median(pl), 1))
    if rm:
        s.update(best_rmsd=round(min(rm), 2))
    return s


def frontmatter(d):
    lines = ["---"]
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, (list, tuple)):
            v = "[" + ", ".join(str(x) for x in v) + "]"
        elif isinstance(v, str) and (v.startswith("[[") or ":" in v or "#" in v or v.strip() != v or v == ""):
            v = json.dumps(v)
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip()
                if v.startswith('"') and v.endswith('"'):
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                out[k.strip()] = v
    return out


def write_new(path, text, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"  exists, left untouched: {path}")
        return False
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path}")
    return True


def reindex(root):
    notes = []
    for p in sorted((root / "iterations").glob("*.md")):
        fm = parse_frontmatter(p.read_text(encoding="utf-8"))
        if fm.get("type") == "design-iteration":
            notes.append((fm.get("date", ""), p.stem, fm))
    notes.sort()
    rows = ["| date | iteration | route | designs | pass | iPTM>=0.8 | max iPTM | status | parent |", "|---|---|---|---|---|---|---|---|---|"]
    for date, slug, fm in notes:
        rows.append(f"| {date} | [[{slug}]] | {fm.get('route', '')} | {fm.get('n_designs', '')} | {fm.get('n_pass', '')} | "
                    f"{fm.get('iptm_ge_0_8', '')} | {fm.get('max_iptm', '')} | {fm.get('status', '')} | {fm.get('parent', '')} |")
    text = (frontmatter({"type": "index", "updated": datetime.date.today().isoformat()}) +
            "\n# Binder design iterations\n\nRebuilt by `log_iteration.py --reindex` from each iteration note's properties. "
            "Open an iteration for setup, findings and provenance. `draft` = numbers generated, interpretation not yet reviewed.\n\n" +
            "\n".join(rows) + "\n\nProcess and conventions: [[README]].\n")
    (root / "iterations.md").write_text(text, encoding="utf-8")
    print(f"  rebuilt {root / 'iterations.md'} ({len(notes)} iterations)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="output folder of a finished run (holds results_af3.csv ...)")
    ap.add_argument("--title", help="short title of the iteration")
    ap.add_argument("--goal", default="TODO: what question did this iteration try to answer?")
    ap.add_argument("--slug", help="note name; default <date>-<route>-<n>")
    ap.add_argument("--parent", help="slug of the iteration this one follows from")
    ap.add_argument("--route", help="af2 | af3 | rfd4_af3 | ...; default from run_info.json")
    ap.add_argument("--target", default="5YXH", help="target name for the note")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra facts for the Setup table (repeatable)")
    ap.add_argument("--vault", help="Obsidian vault folder; notes go to <vault>/binder-design/")
    ap.add_argument("--out", help="folder to write binder-design/ into when no --vault is given")
    ap.add_argument("--outbox", help="queue folder for the vault puller: notes go to <outbox>/<slug>/binder-design/ and a DONE marker is written last")
    ap.add_argument("--top", type=int, default=10, help="rows in the top-designs table")
    ap.add_argument("--overwrite", action="store_true", help="replace existing notes (default: never)")
    ap.add_argument("--reindex", metavar="VAULT_OR_FOLDER", help="only rebuild iterations.md of that vault (or binder-design folder)")
    a = ap.parse_args()

    if a.reindex:
        base = Path(a.reindex)
        reindex(base if base.name == FOLDER else base / FOLDER)
        return
    if not (a.run and a.title):
        ap.error("--run and --title are required")

    run = Path(a.run).expanduser().resolve()
    res = next((run / f for f in RESULT_FILES if (run / f).exists()), None)
    if res is None:
        sys.exit(f"no results file ({', '.join(RESULT_FILES)}) in {run}: finish the scoring stage first")
    rows = list(csv.DictReader(open(res, encoding="utf-8")))
    if not rows:
        sys.exit(f"{res} is empty")
    info, meta = load_json(run / "run_info.json"), load_json(run / "run_meta.json")
    inv = (info.get("invocations") or [{}])
    last = inv[-1]
    cfg = last.get("settings", {})
    route = a.route or meta.get("route") or ("rfd4_af3" if "rfd4" in run.name.lower() else "af3" if res.name == "results_af3.csv" else "af2")
    date = datetime.date.fromtimestamp(res.stat().st_mtime).isoformat()
    st = stats(rows)
    slug = a.slug or f"{date}-{route}-{st['n_designs']}"
    outbox = Path(a.outbox).expanduser() / slug if a.outbox else None
    root = (Path(a.vault).expanduser() / FOLDER) if a.vault else (outbox / FOLDER) if outbox else (Path(a.out).expanduser() if a.out else run / "iteration_note") / FOLDER
    sha = hashlib.sha256(res.read_bytes()).hexdigest()[:16]
    versions = sorted({i.get("code_version") for i in inv if i.get("code_version")})

    setup = {"route": route, "target": a.target}
    setup.update({k: meta[k] for k in meta if k not in ("route",)})
    rfd4_run = route.startswith("rfd4") or "RFD4" in str(meta.get("generator", ""))
    for k in ("binder_length", "af3_diffusion_samples", "af3_target_template", "num_backbones", "seqs_per_backbone", "hotspots", "target_range", "af3_flash_attention"):
        if k in cfg and not (rfd4_run and k in ("num_backbones", "seqs_per_backbone")):     # RFdiffusion-only settings mean nothing for RFD4
            setup[k] = cfg[k]
    for kv in a.meta:
        k, _, v = kv.partition("=")
        setup[k.strip()] = v.strip()
    setup_rows = "\n".join(f"| {k} | {v} |" for k, v in setup.items())

    def rank_key(r):   # best first: highest iPTM; without iPTM (AF2) the lowest interface pAE
        i = fnum(r.get("iptm"))
        return -i if i is not None else (fnum(r.get("pae_interaction")) if fnum(r.get("pae_interaction")) is not None else 99.0)

    ranked = sorted(rows, key=rank_key)
    notable = [r for r in ranked if r.get("pass") == "PASS"] or ranked[:3]
    label = "Passing designs" if any(r.get("pass") == "PASS" for r in rows) else "Best designs (none passed)"

    def drow(r):
        link = f"[[{slug}--{r['job']}]]" if r in notable else r["job"]
        return (f"| {link} | {len(r.get('binder_seq') or '')} | {r.get('iptm', '')} | {r.get('pae_interaction', '')} | "
                f"{r.get('plddt_binder', '')} | {r.get('rmsd_binder', '')} | {r.get('pass', '')} |")

    fm = {"type": "design-iteration", "slug": slug, "date": date, "status": "draft", "target": a.target, "route": route,
          "parent": f"[[{a.parent}]]" if a.parent else None, **st, "source_run": run.name, "results_sha256": sha,
          "code_version": ", ".join(versions) or None, "tags": ["binder-design", f"route/{route}"]}
    counts = [("designs scored", st["n_designs"]), ("pass (pipeline cutoffs)", st["n_pass"])]
    for k, lab in (("iptm_ge_0_8", "iPTM >= 0.8"), ("iptm_ge_0_7", "iPTM >= 0.7"), ("iptm_ge_0_5", "iPTM >= 0.5"), ("median_iptm", "median iPTM"),
                   ("max_iptm", "max iPTM"), ("median_pae", "median interface pAE"), ("best_pae", "best interface pAE"),
                   ("median_plddt", "median binder pLDDT"), ("best_rmsd", "best binder RMSD (A)")):
        if k in st:
            counts.append((lab, st[k]))
    note = frontmatter(fm) + f"""
# {a.title}

> **Status: draft.** Numbers under *Setup*, *Results* and *Provenance* are generated from the run files. *Findings*, *Caveats* and *Next step* are
> interpretation: unreviewed until `status: reviewed` is set in the properties. Do not cite them as fact before that.

## Goal
{a.goal}
{f'Follows from [[{a.parent}]].' if a.parent else ''}

## Setup
| item | value |
|---|---|
{setup_rows}

## Results (generated)
| measure | value |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in counts) + f"""

Pass = the pipeline's cutoffs (iPTM >= 0.8, binder pLDDT >= 80, binder RMSD <= 2.0 A, interface pAE <= 10) unless the Setup says otherwise.
Full table: `data/{slug}-results.csv`.

### Top {min(a.top, len(rows))} by iPTM
| design | binder len | iPTM | pAE | pLDDT | RMSD | verdict |
|---|---|---|---|---|---|---|
""" + "\n".join(drow(r) for r in ranked[:a.top]) + f"""

### {label}
""" + "\n".join(f"- [[{slug}--{r['job']}]]: iPTM {r.get('iptm', '')}, RMSD {r.get('rmsd_binder', '')} A" for r in notable) + """

## Findings (agent, unreviewed)
TODO: what do the numbers show, compared with the parent iteration? Quote numbers from *Results*; say what is NOT shown.

## Caveats
TODO: what limits these results? (same predictor guides and judges, near-copy designs, single run, in silico only ...)

## Next step
TODO: the next iteration and why.

## Provenance
""" + f"""- source run folder: `{run.name}` (on the machine that ran it); results file `{res.name}`, sha256 `{sha}`
- code version: {', '.join(versions) or 'unknown'}; invocations recorded: {len(inv) if info else 0} (`run_info.json`)
- note generated {datetime.datetime.now().isoformat(timespec='seconds')} by `log_iteration.py`
"""
    print(f"iteration {slug}  ->  {root}")
    wrote = write_new(root / "iterations" / f"{slug}.md", note, a.overwrite)
    if wrote:
        (root / "data").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res, root / "data" / f"{slug}-results.csv")
    for r in notable:
        dfm = {"type": "design", "iteration": f"[[{slug}]]", "job": r["job"], "date": date, "target": a.target, "route": route,
               "iptm": r.get("iptm"), "pae_interaction": r.get("pae_interaction"), "plddt_binder": r.get("plddt_binder"),
               "rmsd_binder": r.get("rmsd_binder"), "verdict": r.get("pass"), "binder_length": len(r.get("binder_seq") or ""),
               "lab_status": "not tested", "tags": ["binder-design", "design"]}
        body = frontmatter(dfm) + f"""
# {r['job']} ({slug})

From [[{slug}]]. Predicted in silico only: **not tested in the lab**.

| iPTM | pAE | binder pLDDT | binder RMSD (A) | verdict |
|---|---|---|---|---|
| {r.get('iptm', '')} | {r.get('pae_interaction', '')} | {r.get('plddt_binder', '')} | {r.get('rmsd_binder', '')} | {r.get('pass', '')} |

## Binder sequence ({len(r.get('binder_seq') or '')} aa)
```
{r.get('binder_seq', '')}
```

## Lineage
TODO: parent design, if this one was made from another (e.g. partial re-noising of [[...]]).
"""
        write_new(root / "designs" / f"{slug}--{r['job']}.md", body, a.overwrite)
    if a.vault:
        reindex(root)
    elif outbox:
        if not (outbox / "PULLED").exists():                    # DONE last: the puller must never see a half-written bundle
            (outbox / "DONE").write_text(f"{slug}\n{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        print(f"queued in {outbox}: the scheduled vault puller copies it into the vault")
    else:
        print(f"copy the folder into your vault, then run:  python log_iteration.py --reindex <vault>")


if __name__ == "__main__":
    main()
