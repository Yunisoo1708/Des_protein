# Instructions for agents working in this repo

## Rule 1: the user's design iterations reach the Obsidian vault by themselves; agents do not interpret them

The vault folder `binder-design/` (on the user's laptop, `C:\Research\llm-wiki`) is the user's record of what was tried. The user reads the results and writes
the interpretation (*Findings*, *Caveats*, *Next step*) **themselves**. Do not fill those sections unless the user asks you to.

### How a run gets there
```
run started with --log  ->  result notes queued in ~/vault_outbox/<slug>/ (DONE marker written last)
                        ->  vault_pull.sh on the laptop (Windows scheduled task "BinderVaultPull", every 5 min) copies it into the vault
                        ->  PULLED marker on the remote, iterations.md rebuilt
```
- **Only runs started with `--log` are ever pulled.** Start a run with `--log` when the user starts it or asks for it to be logged:
  `./setup.sh run-rfd4 --n 50 --log --title "..." --goal "..." --parent <slug of the previous iteration>`,
  or `python design_pipeline.py --stages ... --log --title "..." --goal "..." --parent <slug>`.
- **Do not use `--log` for your own tests, smoke runs or diagnostics.** Without the flag the draft notes stay on the remote (`<output_dir>/iteration_note/`) and nothing is pulled.
- Notes appear in the vault within about 5 minutes of the run ending, if the GPU machine is reachable (otherwise the next cycle retries). They carry generated numbers only; the
  interpretation sections say TODO and `status: draft`. Existing notes in the vault are never overwritten.
- Slug = `<date>-<route>-<n designs>` unless you pass `--slug` to `log_iteration.py`; link iterations with `--parent`.

### If the user asks you to write the interpretation
Edit the note in the vault directly (never through the pull). Quote numbers from *Results* or `data/<slug>-results.csv`; say what the numbers do **not** show; compare with the
parent iteration like for like (say if the comparison is unfair); name the standing caveats (AF3 guiding and judging the same designs, near-copy designs, in silico only);
never claim binding or call a design "validated"; leave `status: draft` (only the user sets `reviewed`; `invalid` marks a wrong run, kept with its reason).

### Checking and troubleshooting the pull (on the laptop)
- Log: `wsl.exe -- tail ~/.binder-pull.log`. Run now: `Start-ScheduledTask -TaskName BinderVaultPull`. Pause: `Disable-ScheduledTask -TaskName BinderVaultPull`.
- Queue on the remote: `ls ~/vault_outbox/*/` (`DONE` = waiting, `PULLED` = copied). To pull an iteration again, delete its `PULLED` marker.
- Install on a laptop: copy `vault_pull.sh` (LF line endings) to `%USERPROFILE%\.binder-pull\`, check `REMOTE` and `VAULT` at the top, and register
  `powershell.exe -NoProfile -WindowStyle Hidden -Command "wsl.exe -- bash /mnt/c/Users/<you>/.binder-pull/vault_pull.sh"` with a 5-minute repetition (Task Scheduler).
  The pull needs key-based `ssh`/`rsync` from WSL to the GPU machine.

### Do not
- edit the vault's `index.md`, `log.md`, `hermes/`, `concepts/` or `sources/`; the user decides whether the hub is linked from `index.md`.
- put credentials, home-directory paths, or weights in a note.
- invent a result, a citation or a comparison that is not in the run files. If a number is missing, write "not recorded".
- delete a run folder before the user has had the chance to log it; deleting anything on the remote machine needs the user's explicit approval.

## Rule 2: how this repo is used
- `./setup.sh` shows the state of the machine and the next step; `settings.py` is the only settings file (`ROUTE=af2|af3|rfd4_af3`).
- Use a fresh `output_dir` for every run (`OUTPUT_DIR=...`). `design_pipeline.py` records each invocation in `<output_dir>/run_info.json`;
  `./setup.sh run-rfd4` records the generator facts in `run_meta.json`. Those two files are the provenance of a note.
- Before logging, check the run is valid: `results_*.csv` has the number of rows you meant to score and job names are unique (an earlier run collapsed 50 designs
  onto 4 names and reported nonsense). A wrong run can still be logged; mark it `status: invalid` with the reason.
- Run `python tests/test_workflow.py` after code changes.
