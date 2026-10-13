#!/bin/bash
# PIDO TWIN of analysis/s1abl_20260915T0400Z/s1abl_fire.sh (the parent is NOT edited) with FOUR declared deltas:
#   D1 RUN selects the TABLE only; the policy is ALWAYS FlightDock-S1's frozen pair (34c7f116) at floor 1.00, loaded so the
#      harness runs and multiplied by ZERO: PIDO25 = ARMBLAD025, PIDO35 = ARMBLAD035, each sha-gated against ITS
#      countersignature in ITS prereg. What flies is the PID-baseline + 0.25 m down-offset.
#   D2 manifest gate = pido_manifest_check.py (knob PAIR {voff 0.25, scales 0.0}; binds the TABLE sha from the row); its
#      KNOBFLAG line ("--vert-offset-down-m 0.25 --action-scales 0.00") is exported to the driver as PIDO_KNOBFLAG and
#      appended to the argv.
#   D3 driver = pido_overlay_exam.py (declared knob SET rule); signature gate = pido_sig_gate.py (token prefix PIDO-, run
#      token ALL); markers *_PIDO, tmux pido<label>, out prefix pido_, ARMB_WAVE=pido_<run>; env names PIDO_FIXTURE /
#      PIDO_RESUME / PIDO_PREREG / PIDO_SHAS / PIDO_DIR.
#   D4 the LAUNCH witness carries action_scales from the OVERLAY witness beside the offset.
#   Everything else (census by proc-exe + pattern file, pin, PROT SHAS, buffer refusal, +75 s witness, RESUME, FIXTURE) is the
#   parent's, byte-for-byte in behaviour.
# PUBLIC API: env LABEL PROT DOSE PIN DD RUN [PIDO_FIXTURE=1] [PIDO_RESUME=1] [PIDO_PREREG] [PIDO_SHAS] [PIDO_DIR] bash pido_fire.sh
#   rc 0 launched-or-already-fired. 11 census. 12 pin. 13 sha/buffer/table. 14 manifest. 15 unknown RUN. 16 unsigned. 17 bank absent
# SIDE-EFFECTS: one tmux, <DD>/pido/pido_<LABEL>_<ts>, OUT_DIR_PIDO.txt, FIRED_PIDO.txt|RESUMED_PIDO.txt, LAUNCH_WITNESS_PIDO.json
# SIZE: soft 55 / hard 70 lines. COVERAGE: pido_selfcheck.py (F1-F8 + sig F7a-d + resume F10/F11 + knob/table forwarding F9a-d + G1).
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}" "${RUN:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; B=${PIDO_DIR:-$FD/analysis/pido_20260916}; mkdir -p "$DD/pido"
RESUME=${PIDO_RESUME:-0}
[ -f "$DD/FIRED_PIDO.txt" ] && [ "$RESUME" != 1 ] && { echo "already fired PIDO at $(cat "$DD/FIRED_PIDO.txt") -- set PIDO_RESUME=1 to re-run every gate and relaunch into the SAME out-dir"; exit 0; }
case "$RUN" in PIDO25) LAD=025;; PIDO35) LAD=035;; *) echo "GATE D1: unknown RUN $RUN (PIDO25|PIDO35)"; exit 15;; esac
TB=$FD/analysis/armblad${LAD}_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD${LAD}.json; PR=$FD/analysis/armblad${LAD}_20260830T1600Z/ARMBLAD${LAD}_PREREG.md; SIG="ARMBLAD${LAD}-SEEDS-"
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt (fail-open otherwise); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|pido_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
PREREG=${PIDO_PREREG:-$B/PREREG_PIDO.md}; SHAS=${PIDO_SHAS:-$B/PREREG_SHAS.json}
python3 "$B/pido_sig_gate.py" --prereg "$PREREG" --run ALL --shas "$SHAS" || exit $?
TSHA=$(sha256sum "$TB" | cut -d' ' -f1)
grep -q "^OPERATOR-SIGNATURE: ${SIG}.*$TSHA" "$PR" || { echo "GATE: table sha $TSHA carries no ${SIG} countersignature in $PR"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
MOUT=$(python3 "$B/pido_manifest_check.py" --manifest "$B/WAVE_MANIFEST_PIDO.json" --run "$RUN" --ckpt "$CK" --table "$TB" --dose "$DOSE") || { echo "$MOUT"; echo "GATE D2: manifest refused run=$RUN"; exit 14; }
echo "$MOUT" | head -1; KNOBFLAG=$(echo "$MOUT" | sed -n 's/^KNOBFLAG=//p')
[ -n "$KNOBFLAG" ] || { echo "GATE D2: empty KNOBFLAG -- a PIDO run without both knobs is not this dispatch"; exit 14; }
ARGV="--dose $DOSE --ckpt $CK --width 24 --ledger $LG --table $TB $KNOBFLAG"
[ "${PIDO_FIXTURE:-0}" = 1 ] && { echo "GATES PASS (fixture stop before launch) KNOBFLAG=[$KNOBFLAG] ARGV=$ARGV"; exit 0; }
if [ "$RESUME" = 1 ] && [ -s "$DD/OUT_DIR_PIDO.txt" ]; then OUT=$(cat "$DD/OUT_DIR_PIDO.txt"); else OUT=$DD/pido/pido_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); fi
mkdir -p "$OUT"
tmux new-session -d -s "pido${LABEL,,}" "ARMB_WAVE=pido_${RUN,,} PIDO_KNOBFLAG='$KNOBFLAG' flightdock_native_venv/bin/python3 $B/pido_overlay_exam.py --out-dir $OUT $ARGV > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_PIDO.txt"
if [ "$RESUME" = 1 ]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$DD/RESUMED_PIDO.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_PIDO.txt"; fi
echo "LAUNCHED PIDO $RUN $LABEL $(cat "$DD/FIRED_PIDO.txt") OUT=$OUT ckpt=$CK dose=$DOSE knob=[$KNOBFLAG] table=$TSHA"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin','vert_offset_down_m','action_scales')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_PIDO.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" "$RUN" "$TSHA" "$KNOBFLAG" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot,run,tsha,kf=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"run":run,"table_sha":tsha,"knobflag":kf,"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
