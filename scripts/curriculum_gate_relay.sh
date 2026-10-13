#!/usr/bin/env bash
# =============================================================================
# Curriculum-v3 GATE RELAY DRIVER (T-74). Polls logs/gate_requests/ and hands
# each unprocessed request to gate_relay_core.py (which spawns sealed eval
# lanes, adjudicates, and atomically publishes logs/gate_verdicts/<label>.json
# — the only input that can promote training). Loop stays dumb by design: all
# decisions, seed draws, and fail-closed rules live in the unit-tested core.
# Responsibility: polling + heartbeat + logging ONLY. Side effects: heartbeat
# file, relay log; everything else is the core's declared surface.
# Caps: file soft 85 / hard 100 lines. RAISED 60/80 -> 85/100 by T-275
#   (E-5), DECLARED: the one-shot PANEL_OVERDUE watchdog around the blocking
#   core call (+18 lines) — a stalled panel pages in minutes, not 5.5 h.
# Usage: setsid bash scripts/curriculum_gate_relay.sh [poll_seconds]
#   env: RELAY_LANES (default 12) · RELAY_LEGS_ROOT (default
#   logs/gate_relay_legs) · RELAY_DRYRUN=1 (single pass, dry spawner)
# =============================================================================
set -uo pipefail
FD_ROOT="/home/ubuntu/FlightDock"
cd "$FD_ROOT" || exit 3
# T-62 guard: activate:69 dereferences $PYTHONPATH unguarded; under set -u an
# unset PYTHONPATH kills the shell rc=127 with ZERO output. Guard BEFORE source.
export PYTHONPATH="${PYTHONPATH:-}"
source flightdock_native_venv/bin/activate
POLL="${1:-60}"
LANES="${RELAY_LANES:-12}"
LEGS_ROOT="${RELAY_LEGS_ROOT:-$FD_ROOT/logs/gate_relay_legs}"
REQ_DIR="${RELAY_REQ_DIR:-$FD_ROOT/logs/gate_requests}"
VERDICTS_DIR="${RELAY_VERDICTS_DIR:-$FD_ROOT/logs/gate_verdicts}"
LEDGER="${RELAY_LEDGER:-$FD_ROOT/logs/gate_relay_ledger.jsonl}"
HB="$FD_ROOT/logs/gate_relay_heartbeat.txt"
LOG="$FD_ROOT/logs/gate_relay.log"
mkdir -p "$LEGS_ROOT" "$(dirname "$HB")"
ps -o pgid= -p $$ | tr -d ' ' > "$FD_ROOT/logs/RELAY_PGID.txt"
echo "$$" > "$FD_ROOT/logs/RELAY_PID.txt"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
log "relay up: poll=${POLL}s lanes=$LANES legs_root=$LEGS_ROOT dryrun=${RELAY_DRYRUN:-0}"
while :; do
  date -u +%FT%TZ > "$HB"
  if [ -d "$REQ_DIR" ]; then
    for req in "$REQ_DIR"/*.json; do
      [ -e "$req" ] || continue
      [ -e "$req.processed" ] && continue
      # R-1: ANY terminal state blocks re-dispatch, not just success. Failure
      # rcs used to leave no marker, so a HOLD re-dispatched every poll into
      # the same leg dir — 36 legs in one gate_W2/, 84 seeds burned (07-30).
      [ -e "$req.terminal" ] && continue
      label="$(basename "$req" .json)"
      [ -e "$VERDICTS_DIR/$label.json" ] && continue
      # T-452 G-GATE-GO fail-CLOSED: dispatch needs "$req.GO" (CONSUMED: one GO = one panel; 08-12 self-dispatch). Disable: RELAY_GATE_GO_DISABLE=1. Detector: tests/unit/test_gate_relay_go_guard.py
      if [ "${RELAY_GATE_GO_DISABLE:-0}" != "1" ] && [ ! -e "$req.GO" ]; then
        [ -e "$req.awaiting_go" ] || { : > "$req.awaiting_go"; log "AWAITING OPERATOR GO: $label (touch $req.GO)"; }; continue
      fi
      [ -e "$req.GO" ] && { mv -f "$req.GO" "$req.GO.consumed_$(date -u +%Y%m%dT%H%M%SZ)"; rm -f "$req.awaiting_go"; log "GO consumed: $label"; }
      log "processing $label"
      extra=()
      [ "${RELAY_DRYRUN:-0}" = "1" ] && extra+=(--dry-run)
      # T-275 duration watchdog: the core call BLOCKS this loop, so a stalled
      # panel was invisible for hours (W12: 5.5 h). A one-shot subshell pages
      # PANEL_OVERDUE past the core-declared budget. Advisory: no abort.
      budget="$(python -c "import json,sys; sys.path.insert(0,'scripts'); \
import gate_relay_core as c; print(c.panel_budget_s(json.load(open('$req'))))" \
        2>/dev/null || echo 0)"
      wpid=""
      if [ "$budget" -gt 0 ] 2>/dev/null; then
        (
          sleep "$budget"
          echo "[$(date -u +%FT%TZ)] PANEL OVERDUE: $label exceeded ${budget}s budget — panel stalled or unhealthy (T-275)" | tee -a "$LOG" >&2
          printf '{"label":"%s","budget_s":%s,"stamped_utc":"%s"}\n' \
            "$label" "$budget" "$(date -u +%FT%TZ)" \
            > "$(dirname "$VERDICTS_DIR")/PANEL_OVERDUE_$label.json"
        ) & wpid=$!
      fi
      python scripts/gate_relay_core.py --request "$req" \
        --legs-root "$LEGS_ROOT" --lanes "$LANES" \
        --verdicts-dir "$VERDICTS_DIR" --ledger "$LEDGER" "${extra[@]}" \
        >> "$LOG" 2>&1
      log "$label rc=$?"
      if [ -n "$wpid" ]; then
        pkill -P "$wpid" 2>/dev/null; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
      fi
    done
  fi
  [ "${RELAY_DRYRUN:-0}" = "1" ] && { log "dryrun single pass done"; exit 0; }
  sleep "$POLL"
done
