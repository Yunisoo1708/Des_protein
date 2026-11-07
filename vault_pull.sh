#!/usr/bin/env bash
# vault_pull.sh -- runs on the LAPTOP (inside WSL), not on the GPU machine.
# Copies finished iterations from the remote outbox into the Obsidian vault, then rebuilds iterations.md.
#
# Only runs that were started with --log are in the outbox (design_pipeline.py --log / ./setup.sh run-rfd4 --log); nothing else is ever pulled.
# A run is pulled once its DONE marker exists (written last, so a half-written note is never copied); afterwards the remote folder gets a
# PULLED marker. Existing notes in the vault are never overwritten (rsync --ignore-existing), so your own edits are safe.
# If the GPU machine is unreachable the script exits quietly and tries again at the next scheduled run.
#
# Settings (environment variables, defaults below). Scheduled by Windows Task Scheduler; see AGENTS.md ("Scheduled pull").
set -u
REMOTE="${REMOTE:-yunisoo@100.122.31.47}"
OUTBOX="${OUTBOX:-vault_outbox}"                                        # relative to the remote home folder
VAULT="${VAULT:-/mnt/c/Research/llm-wiki}"
REPO="${REPO:-workflows/Protein_design_workflow}"                       # remote repo, only used to fetch log_iteration.py for the reindex
LOGFILE="${LOGFILE:-$HOME/.binder-pull.log}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

exec 9>"$HOME/.binder-pull.lock"; flock -n 9 || exit 0                   # never run two pulls at once
[ -d "$VAULT/.obsidian" ] || { log "vault not found at $VAULT"; exit 0; }

# slugs whose DONE marker exists and that were not pulled yet
queue="$("${SSH[@]}" "$REMOTE" "cd '$OUTBOX' 2>/dev/null && for d in */; do d=\${d%/}; [ -f \"\$d/DONE\" ] && [ ! -f \"\$d/PULLED\" ] && echo \"\$d\"; done; true" 2>/dev/null)" || { log "remote unreachable, will retry"; exit 0; }
[ -n "$queue" ] || exit 0

pulled=0
while IFS= read -r slug; do
  [ -n "$slug" ] || continue
  mkdir -p "$VAULT/binder-design"
  if rsync -rt --no-perms --ignore-existing -e "${SSH[*]}" "$REMOTE:$OUTBOX/$slug/binder-design/" "$VAULT/binder-design/" 2>>"$LOGFILE"; then
    "${SSH[@]}" "$REMOTE" "touch '$OUTBOX/$slug/PULLED'" 2>/dev/null
    log "pulled $slug into $VAULT/binder-design"
    pulled=$((pulled + 1))
  else
    log "copy of $slug failed, will retry"
  fi
done <<< "$queue"

if [ "$pulled" -gt 0 ]; then
  [ -f "$VAULT/binder-design/README.md" ] || log "note: $VAULT/binder-design/README.md is missing"
  "${SSH[@]}" "$REMOTE" "cat '$REPO/log_iteration.py'" 2>/dev/null | python3 - --reindex "$VAULT" >> "$LOGFILE" 2>&1 || log "reindex failed"
fi
exit 0
