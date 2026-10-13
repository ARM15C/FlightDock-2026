#!/bin/bash
# CS600H3 TWIN of analysis/cs600_20260903T0230Z/cs600_fire.sh (parent NOT edited) with THREE declared deltas and nothing else:
#   D1 the table is ARMBLAD035 (600 legs: nine wind rungs sigma 0.35 + S0 sigma 0.00, the SAME seeds as ARMBLAD025), sha-gated
#      against ITS countersignature (ARMBLAD035-SEEDS-) in ITS prereg; RUN selects the policy: H3C = A2A-leg2 @0.75 + clamp 0.50,
#      H3V = CANDSSW-leg1 @1.00 + vert-offset 0.25 -- the sigma-0.35 twins of CS600's CSC and CSV; H3P (AMENDMENT 1) = parent
#      CAND-S @1.00 no knob on eval3 after CS600 CSP lands, signed separately (token H3P; H3C/H3V use H3BOTH).
#   D2 manifest = WAVE_MANIFEST_CS600H3.json (gated by the parent's cs600_manifest_check.py, CALLED by path -- it binds the
#      TABLE sha from the row, so a sigma-0.25 pair offered here refuses 33 -> 14); driver = the parent's cs600_overlay_exam.py
#      by path (declared-knob rule); signature gate = the parent's cs600_sig_gate.py by path with run token H3BOTH.
#   D3 markers *_CS600H3, out prefix cs600h3_, tmux cs600h3<label>, ARMB_WAVE=cs600h3_<run>.
#   Everything else is the parent's, byte-for-byte in behaviour. PUBLIC API / rc / SIDE-EFFECTS as the parent with the
#   CS600H3_* env names. SIZE: soft 55 / hard 70 lines. COVERAGE: cs600h3_selfcheck.py.
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}" "${RUN:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; B=$FD/analysis/cs600h3_20260903T0330Z; CS=$FD/analysis/cs600_20260903T0230Z; BB=$FD/analysis/bench600_20260902T0310Z; mkdir -p "$DD/cs600h3"
RESUME=${CS600H3_RESUME:-0}
[ -f "$DD/FIRED_CS600H3.txt" ] && [ "$RESUME" != 1 ] && { echo "already fired CS600H3 at $(cat "$DD/FIRED_CS600H3.txt") -- set CS600H3_RESUME=1 to re-run every gate and relaunch into the SAME out-dir"; exit 0; }
case "$RUN" in H3C|H3V) SIGTOK=H3BOTH;; H3P) SIGTOK=H3P;; *) echo "GATE D1: unknown RUN $RUN (H3C|H3V|H3P)"; exit 15;; esac
TB=$FD/analysis/armblad035_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD035.json; PR=$FD/analysis/armblad035_20260830T1600Z/ARMBLAD035_PREREG.md; SIG="ARMBLAD035-SEEDS-"
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt (fail-open otherwise); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|cs600h3_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
PREREG=${CS600H3_PREREG:-$B/PREREG_CS600H3.md}; SHAS=${CS600H3_SHAS:-$B/PREREG_SHAS.json}
python3 "$CS/cs600_sig_gate.py" --prereg "$PREREG" --run "$SIGTOK" --shas "$SHAS" || exit $?
TSHA=$(sha256sum "$TB" | cut -d' ' -f1)
grep -q "^OPERATOR-SIGNATURE: ${SIG}.*$TSHA" "$PR" || { echo "GATE: table sha $TSHA carries no ${SIG} countersignature in $PR"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
MOUT=$(python3 "$CS/cs600_manifest_check.py" --manifest "$B/WAVE_MANIFEST_CS600H3.json" --run "$RUN" --ckpt "$CK" --table "$TB" --dose "$DOSE") || { echo "$MOUT"; echo "GATE D2: manifest refused run=$RUN"; exit 14; }
echo "$MOUT" | head -1; KNOBFLAG=$(echo "$MOUT" | sed -n 's/^KNOBFLAG=//p')
ARGV="--dose $DOSE --ckpt $CK --width 24 --ledger $LG --table $TB $KNOBFLAG"
[ "${CS600H3_FIXTURE:-0}" = 1 ] && { echo "GATES PASS (fixture stop before launch) KNOBFLAG=[$KNOBFLAG] ARGV=$ARGV"; exit 0; }
if [ "$RESUME" = 1 ] && [ -s "$DD/OUT_DIR_CS600H3.txt" ]; then OUT=$(cat "$DD/OUT_DIR_CS600H3.txt"); else OUT=$DD/cs600h3/cs600h3_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); fi
mkdir -p "$OUT"
tmux new-session -d -s "cs600h3${LABEL,,}" "ARMB_WAVE=cs600h3_${RUN,,} CS600_KNOBFLAG='$KNOBFLAG' flightdock_native_venv/bin/python3 $CS/cs600_overlay_exam.py --out-dir $OUT $ARGV > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_CS600H3.txt"
if [ "$RESUME" = 1 ]; then date -u +%Y-%m-%dT%H:%M:%SZ >> "$DD/RESUMED_CS600H3.txt"; else date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_CS600H3.txt"; fi
echo "LAUNCHED CS600H3 $RUN $LABEL $(cat "$DD/FIRED_CS600H3.txt") OUT=$OUT ckpt=$CK dose=$DOSE knob=[$KNOBFLAG] table=$TSHA"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin','clamp_behind_positive_ms','forward_bias_ms','vert_offset_down_m')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_CS600H3.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" "$RUN" "$TSHA" "$KNOBFLAG" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot,run,tsha,kf=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"run":run,"table_sha":tsha,"knobflag":kf,"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
