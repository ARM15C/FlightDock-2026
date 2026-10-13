#!/bin/bash
# RESPONSIBILITY: fire ONE A4.2 table on GRID B25R (grid 5: 90 legs, ALL sigma 0.25 -- the sigma-0.20 half is PURGED,
#   T5; 24-wide) for a vaulted policy at its OWN floor, on the box this runs on, only when every gate holds.
#   Pre-authorized by the ~00:00Z (A2 leg 1 and CAND-2) re-based on grid B25R by the 00:38Z
#   confirmations. TWIN of c2mg1w_20260829T2230Z/g1w_exam_fire.sh with THREE declared deltas: table = grid B25R
#   (sha-gated), out prefix b25r_ + markers *_B25R, tmux b25r<label>. Gates byte-identical otherwise. Idempotent
#   (FIRED_B25R.txt). Scored ONLY against the ARM-B grid-B25R bank once it exists.
# PUBLIC API: env LABEL PROT DOSE PIN DD bash b25r_exam_fire.sh
#   rc 0 launched-or-already-fired · 11 census not clean · 12 wrong pin · 13 sha mismatch / buffer reachable
# SIDE-EFFECTS: one tmux `b25r<label>`, <DD>/b25r/b25r_<LABEL>_<ts>, OUT_DIR_B25R.txt, FIRED_B25R.txt, LAUNCH_WITNESS_B25R.json
# SIZE: soft 50 / hard 60 lines. COVERAGE: fixture on dev -- wrong pin rc 12, tampered SHAS rc 13, unsigned table rc 13,
#   census pattern file absent rc 11 (2026-08-30: the follow-on chain had not staged it; grep -f on a missing file counted 0).
set -u
: "${LABEL:?}" "${PROT:?}" "${DOSE:?}" "${PIN:?}" "${DD:?}"
FD=/home/ubuntu/FlightDock; cd "$FD" || exit 2; mkdir -p "$DD/b25r"
[ -f "$DD/FIRED_B25R.txt" ] && { echo "already fired B25R at $(cat "$DD/FIRED_B25R.txt")"; exit 0; }
sitl() { for p in /proc/[0-9]*; do readlink "$p/exe" 2>/dev/null; done | grep -c arduplane; }
[ -s "$DD/census_patterns.txt" ] || { echo "GATE: census pattern file absent at $DD/census_patterns.txt -- a missing file would read 0 matches by FAILURE (fail-open, caught 04:37Z on three fires); refusing"; exit 11; }
S=$(sitl); M=$(ps -eo pid,etimes,args | grep -F -f "$DD/census_patterns.txt" | grep -vE "grep -F|b25r_exam_fire" | wc -l)
T=$(tmux ls 2>/dev/null | wc -l); DIRTY=$(git diff --name-only HEAD -- src/ | wc -l); HEAD=$(git rev-parse --short HEAD)
[ "$S" = 0 ] && [ "$M" = 0 ] && [ "$T" = 0 ] && [ "$DIRTY" = 0 ] || { echo "GATE: census sitl=$S matches=$M tmux=$T dirty_src=$DIRTY"; exit 11; }
[ "$HEAD" = "$PIN" ] || { echo "GATE: HEAD $HEAD != $PIN"; exit 12; }
(cd "$PROT" && sha256sum -c --quiet SHAS.txt) || { echo "GATE: sha mismatch in $PROT"; exit 13; }
ls "$PROT"/replay_buffer* >/dev/null 2>&1 && { echo "GATE: replay buffer reachable in $PROT"; exit 13; }
TB=$FD/analysis/armbsig_20260824T0030Z/SEED_CELL_TABLE_ARMBSIG_B25R.json
B25R_SHA=b8ed7b5fa13a643825f1d5346fef6069b70813f2ab2f12b24e9f023cf5f4031b
echo "$B25R_SHA  $TB" | sha256sum -c --quiet || { echo "GATE: grid-B25R table sha mismatch"; exit 13; }
grep -q "^OPERATOR-SIGNATURE: ARMBSIG-SEEDS-B25R-.*$B25R_SHA" "$FD/analysis/armbsig_20260824T0030Z/ARMBSIG_PREREG.md" || { echo "GATE: no B25R countersignature"; exit 13; }
CK=$(ls "$PROT"/model_*.zip | head -1); LG=$FD/analysis/c006_campaign_20260728T110000Z/logs_w1418/gate_relay_ledger.CPRIME_COPY.jsonl
OUT=$DD/b25r/b25r_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
tmux new-session -d -s "b25r${LABEL,,}" "flightdock_native_venv/bin/python3 analysis/guardspike818_20260825T1700Z/crrl_overlay_exam.py --dose $DOSE --ckpt $CK --out-dir $OUT --width 24 --ledger $LG --table $TB > $OUT/launcher.log 2>&1"
echo "$OUT" > "$DD/OUT_DIR_B25R.txt"; date -u +%Y-%m-%dT%H:%M:%SZ > "$DD/FIRED_B25R.txt"
echo "LAUNCHED B25R $LABEL $(cat "$DD/FIRED_B25R.txt") OUT=$OUT ckpt=$CK dose=$DOSE"; sleep 75
S=$(sitl); SL=$(ls "$OUT/slots" 2>/dev/null | wc -l); LN=$(ls "$OUT"/B*/*/lane.log 2>/dev/null | wc -l)
VN=$(grep -m1 -h "VecNormalize:" "$OUT"/B*/*/lane.log 2>/dev/null | head -1 | sed 's/.*VecNormalize: //')
W=$(python3 -c "import json;d=json.load(open('$OUT/OVERLAY_WITNESS.json'));print(json.dumps({k:d.get(k) for k in ('dose','ckpt_sha','table_sha','pin')}))" 2>/dev/null || echo '{"witness":"UNREADABLE"}')
python3 - "$DD/LAUNCH_WITNESS_B25R.json" "$S" "$SL" "$LN" "$VN" "$W" "$OUT" "$HEAD" "$PROT" <<'PY'
import json,sys,datetime
p,s,sl,ln,vn,w,out,head,prot=sys.argv[1:]
d={"utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"out_dir":out,"pin":head,
   "at_plus_75s":{"sitl_by_proc_exe":int(s),"slot_logs":int(sl),"lane_logs":int(ln),"vecnormalize_path":vn},
   "overlay_witness":json.loads(w),"fanout_24_wide":int(s)==24 and int(sl)==24,"vecnormalize_in_protected_dir":vn.startswith(prot)}
json.dump(d,open(p,"w"),indent=1); print(json.dumps(d))
PY
