#!/bin/bash
# RESPONSIBILITY: fire ONE 600-leg BENCH600 run (design L: the flown ARMBLAD ladder table as-is, 24-wide,
#   25 slots) for the vaulted parent-A2 pair at its OWN floor 0.75 with NO exam knob (E1), on the box this
#   runs on, only when every gate holds. TWIN of analysis/b25r_20260830T0040Z/b25r_exam_fire.sh (the
#   no-knob parent) with SEVEN declared deltas and nothing else:
#     D1 RUN selects the table: H1H2 -> ARMBLAD025 (sigma 0.25 wind rungs + S0 0.00), H3 -> ARMBLAD035
#        (sigma 0.35 + S0 0.00); each sha-gated against its own countersignature line in ITS prereg.
#     D2 MANIFEST GATE (rc 14): bench600_manifest_check.py asserts the signed (policy, table, dose, no-knob)
#        quadruple for RUN, so a sigma-0.35 corpus can never be filed as the H1/H2 run or the reverse.
#     D3 DRIVER = bench600_overlay_exam.py (the overlay parent re-pointed at the S0 calm-tier wave twin, E3).
#     D4 out prefix bench600_, markers *_BENCH600, tmux bench600<label>; witness records the RUN + table sha.
#     D5 the +75 s witness expects 24 SITL and 24 slot logs exactly as the parent (600 rows deal 25 deep).
#     D7 SIGNATURE GATE (rc 16 / 17), REBUILT 2026-09-02 19:2xZ: delegated to bench600_sig_gate.py, which
#        requires a NON-SPACE after the run token AND a 64-hex sha equal to this run's banked PRE-signature
#        blob sha in PREREG_SHAS.json. The first form of this gate was a bare grep for the run prefix, and
#        the UNSIGNED placeholder ("BENCH600-H1H2- ____") matched it -- measured PASS on both unsigned
#        preregs, so nothing stopped an unsigned fire. Its fixture accepted rc in (16, 0) and so could not
#        catch it. BENCH600_PREREG / BENCH600_SHAS override the paths for fixtures only.
#     D8 RESUME (BENCH600_RESUME=1): a 10 h run that dies mid-flight could not be restarted -- the FIRED
#        marker short-circuits to rc 0. Resume re-runs EVERY gate, reuses the RECORDED out-dir (never a new
#        one), and relaunches; the wave itself SKIPs any leg already holding results_eid*.json, and its own
#        ledger-cursor refusal proves the resumed wave still spends nothing. Draws no seed, re-flies nothing.
#     D6 BENCH600_FIXTURE=1 stops AFTER every gate and BEFORE the launch line (rc 0, prints GATES PASS), so the
#        selfcheck can drive the all-gates-pass path on dev without spawning; unset on every real fire.
#   Census (proc-exe + pattern file), pin, PROT SHAS.txt, buffer-reachable refusal: BYTE-IDENTICAL to the
#   parent. The parent is NOT edited. Idempotent (FIRED_BENCH600.txt); one dispatch dir per run.
# PUBLIC API: env LABEL PROT DOSE PIN DD RUN [BENCH600_FIXTURE=1] [BENCH600_RESUME=1] bash bench600_fire.sh
#   rc 0 launched-or-already-fired. 11 census not clean. 12 wrong pin. 13 sha mismatch / buffer reachable /
#   unsigned table. 14 manifest refusal. 15 unknown RUN. 16 prereg not validly signed. 17 sha bank absent
# SIDE-EFFECTS: one tmux `bench600<label>`, <DD>/bench600/bench600_<LABEL>_<ts>, OUT_DIR_BENCH600.txt,
#   FIRED_BENCH600.txt (or RESUMED_BENCH600.txt, appended, on D8). Draws no seed, trains nothing, kills
#   nothing, no AWS. A resume re-uses the RECORDED out-dir, so no leg directory is ever duplicated.
# SIZE: soft 55 / hard 70 lines. COVERAGE: bench600_selfcheck.py -- unknown RUN 15, wrong pin 12, tampered
#   SHAS 13, census pattern file absent 11, manifest sibling-table 14; bench600_selfcheck_fire.py drives D7
#   both directions (unsigned MUST be 16, correctly signed MUST be 0, wrong sha 16, absent bank 17) and D8
#   resume (out-dir reused, FIRED not clobbered); and the parent's fixtures inherited.
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}" "${RUN:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; B=$FD/analysis/bench600_20260902T0310Z; mkdir -p "$DD/bench600"
RESUME=${BENCH600_RESUME:-0}
[ -f "$DD/FIRED_BENCH600.txt" ] && [ "$RESUME" != 1 ] && { echo "already fired BENCH600 at $(cat "$DD/FIRED_BENCH600.txt") -- set BENCH600_RESUME=1 to re-run every gate and relaunch into the SAME out-dir"; exit 0; }
case "$RUN" in
  H1H2) TB=$FD/analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json; PR=$FD/analysis/armblad025_20260830T1600Z/ARMBLAD025_PREREG.md; SIG="ARMBLAD025-SEEDS-";;
  H3)   TB=$FD/analysis/armblad035_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD035.json; PR=$FD/analysis/armblad035_20260830T1600Z/ARMBLAD035_PREREG.md; SIG="ARMBLAD035-SEEDS-";;
  *) echo "GATE D1: unknown RUN $RUN (H1H2|H3)"; exit 15;;
esac
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt -- a missing file would read 0 matches by FAILURE (fail-open); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|bench600_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
PREREG=${BENCH600_PREREG:-$B/PREREG_BENCH600_$RUN.md}; SHAS=${BENCH600_SHAS:-$B/PREREG_SHAS.json}
python3 "$B/bench600_sig_gate.py" --prereg "$PREREG" --run "$RUN" --shas "$SHAS" || exit $?
TSHA=$(sha256sum "$TB" | cut -d' ' -f1)
grep -q "^OPERATOR-SIGNATURE: ${SIG}.*$TSHA" "$PR" || { echo "GATE: table $TB sha $TSHA carries no ${SIG} countersignature in $PR"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
MF=$B/WAVE_MANIFEST_BENCH600.json
python3 "$B/bench600_manifest_check.py" --manifest "$MF" --run "$RUN" --ckpt "$CK" --table "$TB" --dose "$DOSE" || { echo "GATE D2: manifest refused run=$RUN"; exit 14; }
[ "${BENCH600_FIXTURE:-0}" = 1 ] && { echo "GATES PASS (fixture stop before launch: D6 -- the 19:00Z dev fixture reached the launch line because a 444 model refused the corruption meant to trip the gate; the twin's preflight lacked +x and nothing flew)"; exit 0; }
if [ "$RESUME" = 1 ] && [ -s "$DD/OUT_DIR_BENCH600.txt" ]; then OUT=$(cat "$DD/OUT_DIR_BENCH600.txt")
else OUT=$DD/bench600/bench600_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); fi
mkdir -p "$OUT"
tmux new-session -d -s "bench600${LABEL,,}" "ARMB_WAVE=bench600_${RUN,,} flightdock_native_venv/bin/python3 $B/bench600_overlay_exam.py --dose $DOSE --ckpt $CK --out-dir $OUT --width 24 --ledger $LG --table $TB > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_BENCH600.txt"
if [ "$RESUME" = 1 ]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$DD/RESUMED_BENCH600.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_BENCH600.txt"; fi
echo "LAUNCHED BENCH600 $RUN $LABEL $(cat "$DD/FIRED_BENCH600.txt") OUT=$OUT ckpt=$CK dose=$DOSE table=$TSHA"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_BENCH600.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" "$RUN" "$TSHA" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot,run,tsha=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"run":run,"table_sha":tsha,"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
