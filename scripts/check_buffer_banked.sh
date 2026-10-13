#!/usr/bin/env bash
# =============================================================================
# BUFBANK-728 launch gate: refuse to launch while an UN-BANKED final replay
# buffer exists in the checkpoint dir.
#
# RESPONSIBILITY: train.py always writes replay_buffer_final.pkl to the SAME
#   fixed path, so the next run silently OVERWRITES the previous run's buffer.
#   On 2026-08-19 this destroyed D2a attempt-1's buffer (sha a407170f) and the
#   R1b baseline-reproduction detector could not run. This gate makes the
#   banking discipline mechanical: a final buffer may only be overwritten
#   after the bank step (run_drl_training.sh cleanup) has copied it into
#   banked_buffers/ and stamped a .BANKED marker carrying its sha256.
#
# USAGE: check_buffer_banked.sh <checkpoint_dir>
# EXIT: 0 = safe to launch (no buffer, or buffer banked and sha matches)
#           44 = REFUSE: buffer exists with no marker, or marker sha is stale
#           3 = usage error
# ESCAPE: FD_ALLOW_UNBANKED_BUFFER=1 skips the check (logged loudly).
# SIDE EFFECTS: none — read-only. The BANK step lives in the wrapper, not here.
# COVERAGE: tests/unit/test_check_buffer_banked.py (all four verdict paths).
# CAPS: file soft 60 / hard 80 code lines.
# =============================================================================
set -u

CKPT="${1:-}"
[ -z "$CKPT" ] && { echo "[bufbank] usage: $0 <checkpoint_dir>" >&2; exit 3; }

if [ "${FD_ALLOW_UNBANKED_BUFFER:-0}" = "1" ]; then
    echo "[bufbank] WARNING: FD_ALLOW_UNBANKED_BUFFER=1 — gate SKIPPED by" \
         "explicit escape; an existing final buffer may be overwritten." >&2
    exit 0
fi

BUF="$CKPT/replay_buffer_final.pkl"
MARKER="$BUF.BANKED"

# No incumbent buffer -> nothing to protect.
[ -f "$BUF" ] || exit 0

if [ ! -f "$MARKER" ]; then
    echo "[bufbank] REFUSED (rc 44): $BUF exists with NO .BANKED marker." >&2
    echo "[bufbank] Bank it first, e.g.:" >&2
    echo "[bufbank]   d=\"$CKPT/banked_buffers/manual_\$(date -u +%Y%m%dT%H%M%SZ)\"" >&2
    echo "[bufbank]   mkdir -p \"\$d\" && cp -p \"$BUF\" \"\$d/\" \\" >&2
    echo "[bufbank]     && sha256sum \"$BUF\" | cut -d' ' -f1 > \"$MARKER\"" >&2
    exit 44
fi

want="$(cut -d' ' -f1 "$MARKER" | head -1)"
have="$(sha256sum "$BUF" | cut -d' ' -f1)"
if [ "$want" != "$have" ]; then
    echo "[bufbank] REFUSED (rc 44): $BUF sha ($have) does not match its" \
         ".BANKED marker ($want) — the buffer changed AFTER banking." >&2
    exit 44
fi

echo "[bufbank] OK: final buffer banked (sha ${have:0:8}), safe to launch."
exit 0
