#!/bin/bash
# RESPONSIBILITY: the ONE sanctioned way to launch an 84-row exam (GUARDSPIKE-818 E2). Refuses if
#   the live venue (VENUE_PREAMBLE.sh + scripts/eval_venue_env.sh) is off-pin, sources the
#   preamble, writes <out-dir>/VENUE_WITNESS.json, then execs armbsig_exam_wave.py with the
#   caller's argv UNCHANGED. The frozen wave is never edited; this sits in front of it.
# PUBLIC API: exam_preflight.sh <every armbsig_exam_wave.py argument> (needs --out-dir, --ckpt)
# SIDE-EFFECTS: writes one JSON witness into --out-dir; exports the venue env; execs the wave.
# SIZE: soft 30 / hard 40 lines. COVERAGE: venue_check_selftest.py (wrapper-vs-direct dry-run diff)
set -euo pipefail
FD="${FD_ROOT:-/home/ubuntu/FlightDock}"
D="$FD/analysis/armbsig_20260824T0030Z"
OUT=""; CKPT=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out-dir" ] && OUT="$a"; [ "$prev" = "--ckpt" ] && CKPT="$a"; prev="$a"
done
[ -n "$OUT" ] && [ -n "$CKPT" ] || { echo "exam_preflight: --ckpt and --out-dir are required" >&2; exit 2; }
case "$OUT" in /*) ;; *) echo "exam_preflight: --out-dir must be ABSOLUTE (T-final's exam left 6 GB in ~)" >&2; exit 2;; esac
python3 "$D/venue_check.py" --fd "$FD" --check || exit 3
# shellcheck disable=SC1091
source "$D/VENUE_PREAMBLE.sh"
python3 "$D/venue_check.py" --fd "$FD" --witness "$OUT" --ckpt "$CKPT" || exit 3
exec python3 "$D/armbsig_exam_wave.py" "$@"
