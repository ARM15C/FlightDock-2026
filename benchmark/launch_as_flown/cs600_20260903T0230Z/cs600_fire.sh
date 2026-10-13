#!/bin/bash
# CS600 TWIN of analysis/bench600_20260902T0310Z/bench600_fire.sh (the parent is NOT edited) with FOUR declared deltas:
#   D1 RUN selects the POLICY, not the table: CSP = parent CAND-S @1.00 no knob; CSV = CANDSSW-leg1 @1.00 + vert-offset 0.25;
#      CSC = A2A-leg2 @0.75 + clamp 0.50. All three fly the SAME table, so the three configurations are within-seed comparable.
#      Both fly ARMBLAD025 (sigma 0.25 wind rungs + S0 0.00), sha-gated against its countersignature in ITS prereg.
#   D2 manifest gate = cs600_manifest_check.py (knob null or {voff 0.25}); its KNOBFLAG line is exported to the driver as
#      CS600_KNOBFLAG and appended to the argv, so the flag that flies is the flag that was signed and nothing else.
#   D3 driver = cs600_overlay_exam.py (declared-knob rule); signature gate = cs600_sig_gate.py with run token BOTH
#      (one prereg, two runs, one signature); markers *_CS600, tmux cs600<label>, out prefix cs600_, ARMB_WAVE=cs600_<run>.
#   D4 bench600_preflight.sh / bench600_exam_wave.py / bench600_exam_table.py / bench600_sigma_hook.py are REUSED by
#      absolute path (policy-agnostic S0 calm-tier twins); PREREG_SHAS.json is this dispatch's own bank.
#   Everything else (census by proc-exe + pattern file, pin, PROT SHAS, buffer refusal, +75 s witness, RESUME, FIXTURE)
#   is the parent's, byte-for-byte in behaviour.
# PUBLIC API: env LABEL PROT DOSE PIN DD RUN [CS600_FIXTURE=1] [CS600_RESUME=1] [CS600_PREREG] [CS600_SHAS] bash cs600_fire.sh
#   rc 0 launched-or-already-fired. 11 census. 12 pin. 13 sha/buffer/table. 14 manifest. 15 unknown RUN (CSP|CSV|CSC). 16 unsigned. 17 bank absent
# SIDE-EFFECTS: one tmux, <DD>/cs600/cs600_<LABEL>_<ts>, OUT_DIR_CS600.txt, FIRED_CS600.txt|RESUMED_CS600.txt, LAUNCH_WITNESS_CS600.json
# SIZE: soft 55 / hard 70 lines. COVERAGE: cs600_selfcheck.py (F1-F8 + sig F7a-d + resume F10/F11 + knob-forwarding F9a/b).
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}" "${RUN:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; B=$FD/analysis/cs600_20260903T0230Z; BB=$FD/analysis/bench600_20260902T0310Z; mkdir -p "$DD/cs600"
RESUME=${CS600_RESUME:-0}
[ -f "$DD/FIRED_CS600.txt" ] && [ "$RESUME" != 1 ] && { echo "already fired CS600 at $(cat "$DD/FIRED_CS600.txt") -- set CS600_RESUME=1 to re-run every gate and relaunch into the SAME out-dir"; exit 0; }
case "$RUN" in CSP|CSV|CSC) ;; *) echo "GATE D1: unknown RUN $RUN (CSP|CSV|CSC)"; exit 15;; esac
TB=$FD/analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json; PR=$FD/analysis/armblad025_20260830T1600Z/ARMBLAD025_PREREG.md; SIG="ARMBLAD025-SEEDS-"
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt (fail-open otherwise); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|cs600_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
PREREG=${CS600_PREREG:-$B/PREREG_CS600.md}; SHAS=${CS600_SHAS:-$B/PREREG_SHAS.json}
python3 "$B/cs600_sig_gate.py" --prereg "$PREREG" --run BOTH --shas "$SHAS" || exit $?
TSHA=$(sha256sum "$TB" | cut -d' ' -f1)
grep -q "^OPERATOR-SIGNATURE: ${SIG}.*$TSHA" "$PR" || { echo "GATE: table sha $TSHA carries no ${SIG} countersignature in $PR"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
MOUT=$(python3 "$B/cs600_manifest_check.py" --manifest "$B/WAVE_MANIFEST_CS600.json" --run "$RUN" --ckpt "$CK" --table "$TB" --dose "$DOSE") || { echo "$MOUT"; echo "GATE D2: manifest refused run=$RUN"; exit 14; }
echo "$MOUT" | head -1; KNOBFLAG=$(echo "$MOUT" | sed -n 's/^KNOBFLAG=//p')
ARGV="--dose $DOSE --ckpt $CK --width 24 --ledger $LG --table $TB $KNOBFLAG"
[ "${CS600_FIXTURE:-0}" = 1 ] && { echo "GATES PASS (fixture stop before launch) KNOBFLAG=[$KNOBFLAG] ARGV=$ARGV"; exit 0; }
if [ "$RESUME" = 1 ] && [ -s "$DD/OUT_DIR_CS600.txt" ]; then OUT=$(cat "$DD/OUT_DIR_CS600.txt"); else OUT=$DD/cs600/cs600_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); fi
mkdir -p "$OUT"
tmux new-session -d -s "cs600${LABEL,,}" "ARMB_WAVE=cs600_${RUN,,} CS600_KNOBFLAG='$KNOBFLAG' flightdock_native_venv/bin/python3 $B/cs600_overlay_exam.py --out-dir $OUT $ARGV > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_CS600.txt"
if [ "$RESUME" = 1 ]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$DD/RESUMED_CS600.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_CS600.txt"; fi
echo "LAUNCHED CS600 $RUN $LABEL $(cat "$DD/FIRED_CS600.txt") OUT=$OUT ckpt=$CK dose=$DOSE knob=[$KNOBFLAG] table=$TSHA"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin','clamp_behind_positive_ms','forward_bias_ms','vert_offset_down_m')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_CS600.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" "$RUN" "$TSHA" "$KNOBFLAG" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot,run,tsha,kf=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"run":run,"table_sha":tsha,"knobflag":kf,"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
