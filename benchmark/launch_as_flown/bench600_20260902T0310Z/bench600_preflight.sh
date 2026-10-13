#!/bin/bash
# RESPONSIBILITY: TWIN of analysis/armbsig_20260824T0030Z/exam_preflight.sh with ONE declared delta:
#   it execs bench600_exam_wave.py (the wave twin carrying the S0 calm-tier hook) instead of the
#   frozen armbsig_exam_wave.py. Venue check, preamble and VENUE_WITNESS.json are the PARENT's own
#   files, called by absolute path, byte-identical behaviour. Neither parent is edited.
#   WHY A FILE AND NOT A MONKEYPATCH: the wave fans out slot children with subprocess.Popen on
#   os.path.abspath(__file__) (wave_fanout.py:50); an attribute patched in the parent process never
#   reaches the child that actually runs the per-lane gate. The twin has to be a file on disk.
# PUBLIC API: bench600_preflight.sh <every armbsig_exam_wave.py argument> (needs --out-dir, --ckpt)
# SIDE-EFFECTS: writes one JSON witness into --out-dir; exports the venue env; execs the wave twin.
# SIZE: soft 30 / hard 40 lines. COVERAGE: bench600_selfcheck.py (argv parse + exec target pinned).
set -euo pipefail
FD="${FD_ROOT:-/home/ubuntu/FlightDock}"
D="$FD/analysis/armbsig_20260824T0030Z"
B="$FD/analysis/bench600_20260902T0310Z"
OUT=""; CKPT=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out-dir" ] && OUT="$a"; [ "$prev" = "--ckpt" ] && CKPT="$a"; prev="$a"
done
[ -n "$OUT" ] && [ -n "$CKPT" ] || { echo "bench600_preflight: --ckpt and --out-dir are required" >&2; exit 2; }
case "$OUT" in /*) ;; *) echo "bench600_preflight: --out-dir must be ABSOLUTE" >&2; exit 2;; esac
python3 "$D/venue_check.py" --fd "$FD" --check || exit 3
# shellcheck disable=SC1091
source "$D/VENUE_PREAMBLE.sh"
python3 "$D/venue_check.py" --fd "$FD" --witness "$OUT" --ckpt "$CKPT" || exit 3
exec python3 "$B/bench600_exam_wave.py" "$@"
