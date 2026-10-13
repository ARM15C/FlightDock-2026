#!/bin/bash
# S1ABL TWIN of analysis/cs600_20260903T0230Z/cs600_fire.sh (the parent is NOT edited) with FOUR declared deltas:
#   D1 RUN selects the POLICY AND the TABLE: S1A25 = CANDSSW-leg1 (34c7f116, FlightDock-S1's checkpoint) @1.00 NO knob on
#      ARMBLAD025; S1A35 = the same on ARMBLAD035; A2V25 = parent A2 (3d1cf6ca, FlightDock-A2) @0.75 + vert-offset 0.25 on
#      ARMBLAD025; A2V35 = the same on ARMBLAD035. Each table is sha-gated against ITS countersignature in ITS prereg.
#   D2 manifest gate = the parent's cs600_manifest_check.py CALLED BY PATH (knob null or {voff 0.25}; it binds the TABLE sha
#      from the row, so a 025 table offered to a 035 run refuses 33 -> 14); its KNOBFLAG line is exported to the driver as
#      CS600_KNOBFLAG (the env name the parent driver reads) and appended to the argv.
#   D3 driver = the parent's cs600_overlay_exam.py BY PATH (declared-knob rule); signature gate = s1abl_sig_gate.py (token
#      prefix S1ABL-, run token ALL: one prereg, four runs, one signature); markers *_S1ABL, tmux s1abl<label>, out prefix
#      s1abl_, ARMB_WAVE=s1abl_<run>; env names S1ABL_FIXTURE / S1ABL_RESUME / S1ABL_PREREG / S1ABL_SHAS.
#   D4 S1ABL_DIR overrides the dispatch dir so s1abl_selfcheck.py can drive the gates from a working copy in FIXTURE mode.
#   Everything else (census by proc-exe + pattern file, pin, PROT SHAS, buffer refusal, +75 s witness, RESUME, FIXTURE) is the
#   parent's, byte-for-byte in behaviour.
# PUBLIC API: env LABEL PROT DOSE PIN DD RUN [S1ABL_FIXTURE=1] [S1ABL_RESUME=1] [S1ABL_PREREG] [S1ABL_SHAS] [S1ABL_DIR] bash s1abl_fire.sh
#   rc 0 launched-or-already-fired. 11 census. 12 pin. 13 sha/buffer/table. 14 manifest. 15 unknown RUN. 16 unsigned. 17 bank absent
# SIDE-EFFECTS: one tmux, <DD>/s1abl/s1abl_<LABEL>_<ts>, OUT_DIR_S1ABL.txt, FIRED_S1ABL.txt|RESUMED_S1ABL.txt, LAUNCH_WITNESS_S1ABL.json
# SIZE: soft 55 / hard 70 lines. COVERAGE: s1abl_selfcheck.py (F1-F8 + sig F7a-d + resume F10/F11 + knob/table forwarding F9a-f + G1).
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}" "${RUN:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; B=${S1ABL_DIR:-$FD/analysis/s1abl_20260915T0400Z}; CS=$FD/analysis/cs600_20260903T0230Z; mkdir -p "$DD/s1abl"
RESUME=${S1ABL_RESUME:-0}
[ -f "$DD/FIRED_S1ABL.txt" ] && [ "$RESUME" != 1 ] && { echo "already fired S1ABL at $(cat "$DD/FIRED_S1ABL.txt") -- set S1ABL_RESUME=1 to re-run every gate and relaunch into the SAME out-dir"; exit 0; }
case "$RUN" in S1A25|A2V25) LAD=025;; S1A35|A2V35) LAD=035;; *) echo "GATE D1: unknown RUN $RUN (S1A25|S1A35|A2V25|A2V35)"; exit 15;; esac
TB=$FD/analysis/armblad${LAD}_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD${LAD}.json; PR=$FD/analysis/armblad${LAD}_20260830T1600Z/ARMBLAD${LAD}_PREREG.md; SIG="ARMBLAD${LAD}-SEEDS-"
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt (fail-open otherwise); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|s1abl_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
PREREG=${S1ABL_PREREG:-$B/PREREG_S1ABL.md}; SHAS=${S1ABL_SHAS:-$B/PREREG_SHAS.json}
python3 "$B/s1abl_sig_gate.py" --prereg "$PREREG" --run ALL --shas "$SHAS" || exit $?
TSHA=$(sha256sum "$TB" | cut -d' ' -f1)
grep -q "^OPERATOR-SIGNATURE: ${SIG}.*$TSHA" "$PR" || { echo "GATE: table sha $TSHA carries no ${SIG} countersignature in $PR"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
MOUT=$(python3 "$CS/cs600_manifest_check.py" --manifest "$B/WAVE_MANIFEST_S1ABL.json" --run "$RUN" --ckpt "$CK" --table "$TB" --dose "$DOSE") || { echo "$MOUT"; echo "GATE D2: manifest refused run=$RUN"; exit 14; }
echo "$MOUT" | head -1; KNOBFLAG=$(echo "$MOUT" | sed -n 's/^KNOBFLAG=//p')
ARGV="--dose $DOSE --ckpt $CK --width 24 --ledger $LG --table $TB $KNOBFLAG"
[ "${S1ABL_FIXTURE:-0}" = 1 ] && { echo "GATES PASS (fixture stop before launch) KNOBFLAG=[$KNOBFLAG] ARGV=$ARGV"; exit 0; }
if [ "$RESUME" = 1 ] && [ -s "$DD/OUT_DIR_S1ABL.txt" ]; then OUT=$(cat "$DD/OUT_DIR_S1ABL.txt"); else OUT=$DD/s1abl/s1abl_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); fi
mkdir -p "$OUT"
tmux new-session -d -s "s1abl${LABEL,,}" "ARMB_WAVE=s1abl_${RUN,,} CS600_KNOBFLAG='$KNOBFLAG' flightdock_native_venv/bin/python3 $CS/cs600_overlay_exam.py --out-dir $OUT $ARGV > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_S1ABL.txt"
if [ "$RESUME" = 1 ]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$DD/RESUMED_S1ABL.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_S1ABL.txt"; fi
echo "LAUNCHED S1ABL $RUN $LABEL $(cat "$DD/FIRED_S1ABL.txt") OUT=$OUT ckpt=$CK dose=$DOSE knob=[$KNOBFLAG] table=$TSHA"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin','clamp_behind_positive_ms','forward_bias_ms','vert_offset_down_m')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_S1ABL.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" "$RUN" "$TSHA" "$KNOBFLAG" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot,run,tsha,kf=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"run":run,"table_sha":tsha,"knobflag":kf,"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
