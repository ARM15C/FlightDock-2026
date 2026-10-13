#!/usr/bin/env bash
# ============================================================================
# RESPONSIBILITY: fan the 600 ARMBLAD-035 legs across the SIGNED width of 24 slots, one wave.
#   COPY of baseline025_wave_launcher.sh (L2: the parent flew 600/600 as-is). Pattern
#   carried verbatim from the VALIDATED armbsig_wave_launcher.sh (all slots concurrent
#   under setsid, PGID banked under the CONTRACT filename, wait, DONE marker -- no stagger; the
#   w24 pilot measured Delta-sigma exactly 0 at widths 8/16/24). Slot identity lives ONLY in
#   SLOT_MANIFEST_ARMBLAD035.json via armblad035_driver.sh; this file NEVER reads a seed and
#   never assigns a slot.
#
#   ON RB-68 AND "--parallel", STATED PLAINLY SO NOBODY LOOKS FOR A FLAG THAT ISN'T HERE.
#   RB-68 -- "--width without --parallel runs SEQUENTIALLY and is a defect" -- is a property of
#   armbsig_exam_wave.py, a different tool with those two flags. THIS family has no such flags:
#   parallelism is STRUCTURAL, `setsid ... &` per slot followed by one `wait`. The invariant
#   RB-68 protects is that legs must not run one-at-a-time, so it is checked POSITIVELY here:
#   G-e counts the per-slot PGID files the drivers write and REFUSES if fewer than WIDTH slots
#   actually started. A structural guarantee still deserves a witness (rc 0 is not proof).
#
#   600 legs / 24 slots = 25 legs per slot, run SEQUENTIALLY WITHIN a slot. One wave, not 25.
#   Wall clock is therefore one slot's depth: 25 legs at the measured 66.5-82.4 legs/h across 24
#   lanes puts the wave at roughly 7.3-9.0 h.
#
# PUBLIC API: armblad035_wave_launcher.sh <dryrun|fly> [waveidx] (run INSIDE tmux for fly)
# SIDE-EFFECTS: spawns WIDTH drivers (each writes leg dirs, slot logs, PGIDs); writes
#   WAVE_PGID.txt, armblad035_wave.log, armblad035_wave_slot_e<N>.out,
#   ARMBLAD035_WAVE_DONE_w<waveidx>_<mode>.txt. Kills nothing, signals nothing.
# CAPS: file soft 75 / hard 95 code-lines; function cap 35.
# COVERAGE: armblad035_selfcheck.py -- every guard driven BOTH directions.
# ============================================================================
set -o pipefail
MODE="${1:?usage: armblad035_wave_launcher.sh <dryrun|fly> [waveidx]}"
case "$MODE" in dryrun|fly) ;; *) echo "bad mode $MODE"; exit 2;; esac
WAVEIDX="${2:-0}"
case "$WAVEIDX" in ''|*[!0-9]*) echo "bad waveidx $WAVEIDX"; exit 2;; esac
FD=/home/ubuntu/FlightDock
D=$FD/analysis/armblad035_20260830T1600Z
FD_AT=$FD/analysis/armbtop_20260817T1240Z         # helpers COMPOSED, never copied
PREREG=$D/ARMBLAD035_PREREG.md
MAN=$D/SLOT_MANIFEST_ARMBLAD035.json
TABLE=$D/SEED_CELL_TABLE_ARMBLAD035.json
[ -f "$MAN" ] || { echo "manifest absent: $MAN"; exit 2; }
# WIDTH: ARMBLAD-035 carries no width signature block, so width is the MANIFEST's --
# recorded there at build time, defaulting to 24. Read from the GATED manifest, never a
# literal here, so the value that flies is the value the gate checked.
WIDTH=$(python "$D/armblad035_manifest.py" --gate --manifest "$MAN" --table "$TABLE" \
        --doc "$PREREG" >/dev/null 2>&1 && python - "$MAN" <<'PY'
import json,sys; print(json.load(open(sys.argv[1]))["width"])
PY
) || { echo "REFUSE: manifest gate refused -- see armblad035_manifest.py --gate"; exit 9; }
case "$WIDTH" in ''|*[!0-9]*) echo "REFUSE: width unreadable from the gated manifest"; exit 9;; esac
export ARMBLAD035_MANIFEST="$MAN" WAVEIDX
export ARMB_WAVE="armblad035_w${WAVEIDX}"
log(){ echo "[$(date -u +%FT%TZ)] [b035wave] $*" | tee -a "$D/armblad035_wave.log"; }

# CONTRACT FILENAME: the driver reads "$D/WAVE_PGID.txt" and M-1P admits a process only if its
# ANCESTRY reaches this pgid. Write it BEFORE any driver is spawned.
ps -o pgid= -p $$ | tr -d " " > "$D/WAVE_PGID.txt"

# --- refusal guards, fail-closed; each must be ABLE to refuse ---------------
# G-a: never re-fly over a completed/partial wave at THIS wave index.
if [ "$MODE" = "fly" ] && ls "$D"/SLOT_DONE_w${WAVEIDX}_e*_fly.txt >/dev/null 2>&1; then
  log "REFUSE (G-a): prior fly SLOT_DONE markers at w${WAVEIDX} -- bank or bump waveidx"; exit 9
fi
# G-c: A WAVE MAY NOT FLY UNLESS THE TRACKER OF RECORD NAMES IT. On 2026-08-18 a wave put 508
# legs and 5 h of 24xl through the parent launcher with NO the gap surfaced two days
# later, and only because a later dispatch happened to census the box. Discipline did not catch
# it and would not have.
if [ "$MODE" = "fly" ]; then
  LG=$FD/docs/TASK_LEDGER.md
  [ -n "${ARMBLAD035_LEDGER_ROW:-}" ] \
    || { log "REFUSE (G-c): ARMBLAD035_LEDGER_ROW unset -- name the ledger row authorising this wave"; exit 9; }
  grep -q "^| ${ARMBLAD035_LEDGER_ROW} |" "$LG" \
    || { log "REFUSE (G-c): no row '${ARMBLAD035_LEDGER_ROW}' in docs/TASK_LEDGER.md"; exit 9; }
  log "G-c: authorised by ledger row ${ARMBLAD035_LEDGER_ROW}"
fi
# G-b: no SITL already on this box. Censused by /proc/<pid>/exe, NOT pgrep -f: the cmdline form
# cannot self-match-proof itself, and R3's guard read `ArduPlane` while the binary is lowercase
# `arduplane`, so it never fired.
if [ "$MODE" = "fly" ]; then
  N=0
  for p in /proc/[0-9]*; do
    case "$(readlink "$p/exe" 2>/dev/null)" in *arduplane|*ardupilot) N=$((N+1));; esac
  done
  [ "$N" -eq 0 ] || { log "REFUSE (G-b): $N arduplane already running (census non-zero)"; exit 9; }
fi
# G-d: BETWEEN-WAVES ATTRITION GATE, pooled and outcome-blind. An EMPTY corpus is the first
# wave, not a failure -- said out loud because the gate itself calls no-legs rc 2 by design.
if [ "$MODE" = "fly" ]; then
  GD_ROOTS=""
  for R in "$D"/legs/*/; do
    ls "$R"*/leg_meta.json >/dev/null 2>&1 && GD_ROOTS="$GD_ROOTS --legs ${R%/}"
  done
  if [ -z "$GD_ROOTS" ]; then
    log "G-d: no prior legs -- first wave of this corpus, nothing to pool"
  else
    log "G-d roots: $GD_ROOTS"
    python "$FD_AT/wave_attrition_gate.py" $GD_ROOTS \
      --max-attrition "${ARMBLAD035_MAX_ATTRITION:-0.145}" >> "$D/armblad035_wave.log" 2>&1
    GD=$?
    [ "$GD" -eq 0 ] || { log "REFUSE (G-d): attrition gate rc $GD -- verdict in armblad035_wave.log"; exit 9; }
    log "G-d: pooled attrition within bar ${ARMBLAD035_MAX_ATTRITION:-0.145}"
  fi
fi

log "wave start mode=$MODE width=$WIDTH waveidx=$WAVEIDX head=$(git -C "$FD" rev-parse --short HEAD 2>/dev/null)"
T0=$(date +%s)
rm -f "$D"/SLOT_e*_PGID.txt
for E in $(seq 1 "$WIDTH"); do
  setsid bash "$D/armblad035_driver.sh" "$MODE" "$E" "$WIDTH" \
    >> "$D/armblad035_wave_slot_e${E}.out" 2>&1 &
done
wait
# G-e: THE PARALLELISM WITNESS (see the RB-68 note in the header). Every slot that actually ran
# wrote its own PGID file; fewer than WIDTH means the fan-out did not fan out.
STARTED=$(ls "$D"/SLOT_e*_PGID.txt 2>/dev/null | wc -l)
[ "$STARTED" -eq "$WIDTH" ] \
  || log "REFUSE (G-e): $STARTED of $WIDTH slots started -- the wave did not fan out"
RCS=""
for E in $(seq 1 "$WIDTH"); do
  [ -f "$D/SLOT_DONE_w${WAVEIDX}_e${E}_${MODE}.txt" ] && RCS="$RCS e$E:done" || RCS="$RCS e$E:MISSING"
done
T1=$(date +%s)
log "WAVE COMPLETE wall=$((T1-T0))s slots_started=$STARTED/$WIDTH rcs:$RCS"
echo "mode=$MODE width=$WIDTH waveidx=$WAVEIDX wall_s=$((T1-T0)) slots_started=$STARTED rcs:$RCS ts=$(date -u +%FT%TZ)" \
  > "$D/ARMBLAD035_WAVE_DONE_w${WAVEIDX}_${MODE}.txt"
[ "$STARTED" -eq "$WIDTH" ] || exit 5
case "$RCS" in *MISSING*) exit 4;; esac
exit 0
