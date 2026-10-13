#!/usr/bin/env bash
# =============================================================================
# RESPONSIBILITY: fly the ARMBLAD-035 legs belonging to ONE slot (env_id), sequentially within
#   the slot, from SLOT_MANIFEST_ARMBLAD035.json. DERIVED from armbsig_driver.sh: the env block
#   and the spawner call are carried VERBATIM so this corpus is instrument-comparable to ARM-B's
#   own bank. That parent is frozen-as-flown and is NOT modified.
#
# DECLARED DELTAS vs the parent -- these are the whole change set:
#   1. AUTHORITY. Gates ARMBLAD-035's own signatures (S15 prereg, S16 table sha, S18 width),
#      column-0 anchored, fail-closed. S17 (abort-warrant) is REPORTED by the manifest and
#      gates no leg.
#   2. SIGMA PER LEG comes from the manifest row, as in the parent. The rungs are 0.25 except
#      S0, which is commanded 0.0 with turbulence OFF (SAP S6 item 3: "0.0 + turbulence off
#      for S0"). Exporting a constant at slot scope would silently fly every rung at one sigma.
#   3. THE S0 BRANCH -- prereg S6.1, and the reason this driver exists at all.
#      leg_sigma_gate.py:137-141 enters the CALM_CONFIRMED tier ONLY when --paired-csv is
#      supplied AND that leg's resolved_wind_max_ms == 0. The parent supplies none, so at S0
#      the leg would be band-compared against band(0.00)=0.0000 and left UNGATED. Supplying it
#      on EVERY rung is worse: the calm predicate keys on MEAN WIND, not commanded sigma, so
#      the W0 rung (mean wind 0, sigma 0.25, turbulence correctly present) would be misrouted
#      into the calm tier and REFUSED for carrying wind it is supposed to carry.
#      So: --paired-csv on the S0 rung and NO other, read from the ROW's paired_csv field --
#      never re-derived here, because the ruling has ONE writer (the seed table).
#   4. THE S0 POSITIVE WITNESS. resolved_wind_max() returns None on ANY read failure and the
#      gate then silently STAYS ON TIER 1 -- so a bad --paired-csv path reproduces Defect 1
#      with an rc that looks unrelated. armblad035_calm_witness.py asserts the gate's own
#      calm_commanded stamp; its absence is a HALT, because a rung that cannot be witnessed is
#      INCONCLUSIVE (prereg S13 item 3), never quietly kept.
#   5. CAMPAIGN NAME + provenance strings.
# PUBLIC API: armblad035_driver.sh <dryrun|fly> <envid 1..WIDTH> [width]
# SIDE-EFFECTS: leg dir + lane log + PGID; slot log; SLOT_DONE marker. Never writes the
#   manifest, the table or the prereg. Kills nothing, signals nothing, draws no seed.
# CAPS: file soft 120 / hard 155 code-lines; function cap 30.
# COVERAGE: dryrun mode (plan only, zero SITL) + armblad035_selfcheck.py, which drives every
#   authority gate and the S0 branch BOTH directions.
# =============================================================================
set -uo pipefail
MODE="${1:?usage: armblad035_driver.sh <dryrun|fly> <envid> [width]}"
ENVID="${2:?envid required: 1..WIDTH}"
WIDTH="${3:-24}"
case "$MODE" in dryrun|fly) ;; *) echo "bad mode $MODE"; exit 2;; esac
[ "$ENVID" -ge 1 ] 2>/dev/null && [ "$ENVID" -le "$WIDTH" ] || { echo "bad envid"; exit 2; }

FD=/home/ubuntu/FlightDock
D=$FD/analysis/armblad035_20260830T1600Z
P=$FD/analysis/armb_refly_20260807T1830Z          # parent helpers, COMPOSED never copied
A=$FD/analysis/armbsig_20260824T0030Z             # ARM-B lineage helpers
B=$FD/analysis/benchmark_n100_20260718T190000Z
PREREG=$D/ARMBLAD035_PREREG.md
MAN="${ARMBLAD035_MANIFEST:-$D/SLOT_MANIFEST_ARMBLAD035.json}"
TABLE=$D/SEED_CELL_TABLE_ARMBLAD035.json
CKPT=$FD/analysis/c006_campaign_20260728T110000Z/ckpt_ws1/model_final.zip
OUT=$D/legs
mkdir -p "$OUT"; cd "$FD" || exit 3
export PYTHONPATH="${PYTHONPATH:-}"               # T-62 guard, BEFORE source
source flightdock_native_venv/bin/activate || exit 3
log(){ echo "[$(date -u +%FT%TZ)] [e$ENVID] $*" | tee -a "$D/slot_e${ENVID}.log"; }
ps -o pgid= -p $$ | tr -d " " > "$D/SLOT_e${ENVID}_PGID.txt"

[ -f "$MAN" ]   || { log "GATE FAIL: manifest absent"; exit 11; }
[ -f "$CKPT" ]  || { log "GATE FAIL: ckpt absent"; exit 11; }
# AUTHORITY GATES, column-0 anchored and fail-closed. The prereg carries INDENTED templates of
# these same lines by design; a grep without ^ matches a template and passes on an UNSIGNED doc.
grep -q "^OPERATOR-SIGNATURE: ARMBLAD035-[0-9]\{8\}" "$PREREG" \
  || { log "GATE FAIL: prereg block A unsigned"; exit 11; }
grep -q "^OPERATOR-SIGNATURE: ARMBLAD035-SEEDS-" "$PREREG" \
  || { log "GATE FAIL: seed table block B not countersigned"; exit 11; }
grep -q "^OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-" "$PREREG" \
  || { log "GATE FAIL: block C realized-sigma band NOT ruled -- at sigma 0.35 the scaled band 0.1118 is LOOSER than A-10's signed 0.10 and this campaign may not pick one silently"; exit 11; }
python "$D/armblad035_manifest.py" --gate --manifest "$MAN" --table "$TABLE" --doc "$PREREG" \
  >> "$D/slot_e${ENVID}.log" 2>&1 \
  || { log "GATE FAIL: manifest gate refused"; exit 11; }

# ---- venue: ONE sourced file, never retyped (see armblad035_venue_preamble.sh) ----------
# The first B4 attempt inlined this block and omitted `source scripts/eval_venue_env.sh` plus
# four warmup/safety knobs; all 24 slots HALTED on the EVAL contract before leg 1 (zero SITL,
# zero seeds, zero steps). Selfcheck limb S9 now EXECUTES the contract against this preamble.
source "$D/armblad035_venue_preamble.sh" || { log "GATE FAIL: venue preamble missing"; exit 11; }

WAVE_PGID="$(cat "$D/WAVE_PGID.txt" 2>/dev/null || echo 0)"
log "config-echo mode=$MODE slot=$ENVID width=$WIDTH wave_pgid=$WAVE_PGID head=$(git rev-parse --short HEAD) man_sha=$(sha256sum "$MAN" | cut -c1-12)"
if [ "$MODE" = "fly" ]; then
  python $FD/analysis/armbdesc_20260820T1300Z/preflight_contract.py >> "$D/slot_e${ENVID}.log" 2>&1 \
    || { log "HALT: EVAL env contract violated before leg 1"; exit 12; }
fi

N=0; FLOWN=0; SKIP=0
while read -r SEED TRAJ WIND LABEL SIGMA CELL PCSV; do
  N=$((N+1))
  LEGROOT="$OUT/$CELL"; LEG="$LEGROOT/${LABEL}_${SEED}_${TRAJ}"
  mkdir -p "$LEGROOT"
  if ls "$LEG"/results_eid*.json >/dev/null 2>&1; then
    SKIP=$((SKIP+1)); continue                    # LEG_DONE-idempotent
  fi
  SIGMA_FMT=$(awk -v s="$SIGMA" 'BEGIN{printf "%.2f", s}')
  export FD_V3_SIGMA_OVERRIDE_MS="$SIGMA_FMT,$SIGMA_FMT,$SIGMA_FMT"
  unset FD_V3_SIGMA_OVERRIDE_MS_ACK
  if [ "$SIGMA_FMT" = "0.35" ]; then
    export FD_V3_SIGMA_OVERRIDE_MS_ACK="ARMBLAD-035 (prereg SIGNED): sigma 0.350 is OFF the A7 taxonomy by construction. It is the H3 moderate-turbulence basis the operator elected 2026-08-24, and the PID reference must fly the SAME cell to describe it. This campaign is the sigma-0.35 twin of ARMBLAD-025 (fresh block 10775-10804, harness of 2026-08-30): same 30 seeds, same nine wind rungs, same trajectories, same still-air control -- sigma is the ONE variable that moves. Declared, not widened, never rounded."
  elif [ "$SIGMA_FMT" = "0.00" ]; then
    export FD_V3_SIGMA_OVERRIDE_MS_ACK="ARMBLAD-035 (prereg SIGNED, block D): the S0 rung is the zero-disturbance control -- wind 0 AND turbulence OFF, identical to ARMBLAD-025's S0 rung by construction (operator election 2026-08-30) so the two ladders stay symmetric at 600 legs and the pair is the harness-replica control. Commanded sigma 0.0 is asserted POSITIVELY by the CALM_CONFIRMED tier, never by a band."
  else
    log "HALT leg $N: sigma '$SIGMA' is neither 0.35 nor 0.00 -- no ack is registered for it"; exit 15
  fi
  if [ "$MODE" = "dryrun" ]; then
    echo "plan[e$ENVID:$N] seed=$SEED env_id=$ENVID traj=$TRAJ wind=$WIND cell=$CELL sigma=$SIGMA_FMT paired_csv=$PCSV"
    continue
  fi
  python "$P/armb_refly_preflight.py" --m1p --envid "$ENVID" \
      --wave-pgid "$WAVE_PGID" --allowlist "$B/runner/M1_ALLOWLIST.json" \
      >> "$D/slot_e${ENVID}.log" 2>&1 \
    || { log "HALT leg $N: M-1P refuse (slot-scoped census)"; exit 13; }
  log "leg $N seed=$SEED env_id=$ENVID traj=$TRAJ wind=$WIND cell=$CELL sigma=$SIGMA_FMT pcsv=$PCSV"
  python scripts/curriculum_eval_spawner.py --checkpoint "$CKPT" \
    --stage-label "$LABEL" --out-dir "$LEGROOT" --seeds "$SEED" --env-id "$ENVID" \
    --traj-map "$SEED:$TRAJ" --force-zero-residual \
    > "$LEGROOT/lane_${SEED}_${TRAJ}.log" 2>&1
  RC=$?
  python "$A/../armbtop_20260817T1240Z/armbtop_spawn_guard.py" --lane-log "$LEGROOT/lane_${SEED}_${TRAJ}.log" \
    >> "$D/slot_e${ENVID}.log" 2>&1 \
    || { log "HALT leg $N seed=$SEED: spawn guard REFUSED"; exit 14; }
  python "$P/leg_bridge_gate.py" --leg "$LEG" >> "$D/slot_e${ENVID}.log" 2>&1
  BRRC=$?
  [ $BRRC -eq 0 ] || { log "REFUSE leg seed=$SEED bridge rc=$BRRC (2=NO_PAIRED_CSV, 3=UNRESOLVABLE)"; RC=$BRRC; }
  python "$P/armb_refly_preflight.py" --cell-witness --leg "$LEG" \
      --expect-wind "$WIND" >> "$D/slot_e${ENVID}.log" 2>&1 \
    || log "WARN leg seed=$SEED cell witness unreadable (leg stays in denominator)"
  # ===== THE S0 BRANCH (prereg S6.1). paired_csv is READ from the row, never re-derived. =====
  # Block C ruled SCALED: A-10's RULE, not A-10's NUMBER. 0.30 is the sigma sd was MEASURED at
  # (R3: 210 legs / 630 axis draws, sd 0.04015), so it is the reference the ratio is taken
  # against -- never the campaign's own sigma. This line is BYTE-IDENTICAL to the parent's, and
  # that is the point: same rule, so +/-32% relative at 0.35 exactly as at 0.25.
  # DISCLOSED: at sigma 0.35 this yields 0.1118, which is LOOSER than A-10's signed 0.10. The
  # comments at armbsig_driver.sh:382 and wave_sigma_hook.py:27 claim it is "STRICTLY TIGHTER
  # ... at every sigma we fly" -- true for grid 1, FALSE here. Recorded, not repaired (L2).
  BAND=$(awk -v s="$SIGMA" 'BEGIN{printf "%.4f", 2.3877*0.04015*(s/0.30)}')
  if [ "$PCSV" = "True" ]; then
    CSV="$LEG/paired/env0/paired_channels_ep0001.csv"
    [ -r "$CSV" ] || { log "HALT leg $N seed=$SEED: S0 rung but paired CSV unreadable at $CSV -- without it the calm tier cannot engage and the rung would be UNGATED"; exit 16; }
    python "$P/leg_sigma_gate.py" --leg "$LEG" --env-id "$ENVID" \
        --commanded "$SIGMA" --band "$BAND" --paired-csv "$CSV" \
        >> "$D/slot_e${ENVID}.log" 2>&1
    SIGRC=$?
    python "$D/armblad035_calm_witness.py" --leg "$LEG" >> "$D/slot_e${ENVID}.log" 2>&1
    CWRC=$?
    [ $CWRC -ne 5 ] || { log "HALT leg $N seed=$SEED: calm tier NEVER RAN on an S0 leg -- silent tier-1 fallback, rung UNGATED (prereg S13 item 3)"; exit 17; }
    [ $SIGRC -eq 0 ] || log "REFUSE leg seed=$SEED S0 calm rc=$SIGRC (2=UNSAMPLED/INCONCLUSIVE, 3=DELIVERED WIND)"
  else
    python "$P/leg_sigma_gate.py" --leg "$LEG" --env-id "$ENVID" \
        --commanded "$SIGMA" --band "$BAND" \
        >> "$D/slot_e${ENVID}.log" 2>&1
    SIGRC=$?
    [ $SIGRC -eq 0 ] || log "REFUSE leg seed=$SEED realized-sigma rc=$SIGRC (2=UNSAMPLED/INCONCLUSIVE, 3=OUT-OF-BAND)"
  fi
  BRIDGE_RC=$RC
  [ $SIGRC -eq 0 ] || RC=$SIGRC
  python "$P/armb_leg_capture.py" --leg "$LEG" --seed "$SEED" --traj "$TRAJ" --wind "$WIND" \
      --env-id "$ENVID" --rc "$BRIDGE_RC" --sigma-rc "$SIGRC" --wave "${ARMB_WAVE:-unspecified}" \
      >> "$D/slot_e${ENVID}.log" 2>&1 || log "WARN capture failed seed=$SEED"
  [ $RC -eq 0 ] && FLOWN=$((FLOWN+1)) || log "leg seed=$SEED rc=$RC (stays in denominator)"
done < <(python - "$MAN" "$ENVID" <<'PYEOF'
import json, os, sys
m = json.load(open(sys.argv[1])); e = int(sys.argv[2])
cells = [c for c in os.environ.get("ARMBLAD035_CELLS", "").split(",") if c]
for r in m["legs"]:
    if r["env_id"] == e and (not cells or r["cell"] in cells):
        print(r["seed"], r["traj"], r["wind_ms"], r["label"], r["sigma_ms"], r["cell"],
              r["paired_csv"])
PYEOF
)
log "slot $ENVID $MODE complete: planned=$N flown=$FLOWN skipped=$SKIP"
echo "slot=$ENVID mode=$MODE planned=$N flown=$FLOWN skipped=$SKIP ts=$(date -u +%FT%TZ)" \
  > "$D/SLOT_DONE_w${WAVEIDX:-0}_e${ENVID}_${MODE}.txt"
exit 0
