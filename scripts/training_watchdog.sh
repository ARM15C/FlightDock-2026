#!/usr/bin/env bash
# scripts/training_watchdog.sh — PID + heartbeat watchdog for DRL training.
#
# Launched by run_drl_training.sh in its own process group (setsid).
# Monitors the training process via:
#   1. PID file — if the PID is gone, training exited (wrapper handles cleanup)
#   2. Heartbeat file — if mtime goes stale, training is hung → kill + sweep
#
# Usage (called by wrapper, not directly):
#   training_watchdog.sh <pid_file> <heartbeat_file>
#
# Environment variables:
#   FLIGHTDOCK_WATCHDOG_POLL — poll interval in seconds (default: 5)
#   FLIGHTDOCK_WATCHDOG_STALE_MIN — heartbeat stale threshold in minutes (default: 10)
#   FLIGHTDOCK_MAX_ENVS — max environments for port sweep (default: 8)
#   FLIGHTDOCK_WATCHDOG_SHUTDOWN_GRACE_S — WDFIX-593. Seconds a SHUTDOWN may
#       take before it is treated as wedged (default: 900). A shutdown is
#       recognised by per-worker heartbeats DISAPPEARING after having existed;
#       train.py unlinks them in env close() and then writes the final model,
#       vecnormalize and replay buffer, which takes tens of seconds. Only the
#       run that never had per-worker heartbeats at all can reach the legacy
#       branch. Detector: analysis/wdfix_20260815T2330Z/watchdog_smoke.sh

set -uo pipefail

PID_FILE="${1:?Usage: training_watchdog.sh <pid_file> <heartbeat_file>}"
HEARTBEAT_FILE="${2:?Usage: training_watchdog.sh <pid_file> <heartbeat_file>}"

POLL_INTERVAL="${FLIGHTDOCK_WATCHDOG_POLL:-5}"
HEARTBEAT_STALE_MINUTES="${FLIGHTDOCK_WATCHDOG_STALE_MIN:-15}"
MAX_ENVS="${FLIGHTDOCK_MAX_ENVS:-8}"
PORT_STRIDE=100
SHUTDOWN_GRACE_S="${FLIGHTDOCK_WATCHDOG_SHUTDOWN_GRACE_S:-900}"
# WDFIX-593 latches. seen_per_worker: sticky once per-worker heartbeats appear.
# shutdown_first_seen: epoch when they vanished, 0 while they are present.
seen_per_worker=0
shutdown_first_seen=0

_is_flightdock_training() {
    # Verify /proc/PID/cmdline contains our training script.
    # Guards against PID reuse: if the original training process dies and the
    # kernel assigns the same PID to an unrelated process, we must not kill it.
    local pid="$1"
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || echo "")"
    [[ "$cmdline" == *"train.py"* || "$cmdline" == *"flightdock"* ]]
}

_sweep_ports() {
    # Kill all SITL/MAVProxy/leader processes in our port range.
    local killed=0
    for offset in $(seq 0 "$PORT_STRIDE" $(( (MAX_ENVS - 1) * PORT_STRIDE )) ); do
        local port=$((5760 + offset))
        local leader_port=$((14550 + offset))
        pkill -9 -u "$USER" -f "serial0=tcp:${port}" 2>/dev/null && ((killed++)) || true
        pkill -9 -u "$USER" -f "tcp:127.0.0.1:${port}" 2>/dev/null && ((killed++)) || true
        pkill -9 -u "$USER" -f "leader_broadcast_fixed.py.*${leader_port}" 2>/dev/null && ((killed++)) || true
    done
    # Kill stale forkserver processes
    pkill -9 -u "$USER" -f "from multiprocessing.forkserver import main" 2>/dev/null && ((killed++)) || true
    echo "[watchdog] Port sweep killed ${killed} process(es)"
}

echo "[watchdog] Started — PID_FILE=${PID_FILE} HEARTBEAT_FILE=${HEARTBEAT_FILE}"
echo "[watchdog] Poll=${POLL_INTERVAL}s Stale=${HEARTBEAT_STALE_MINUTES}min MaxEnvs=${MAX_ENVS}"

# Wait for PID file to appear (training may still be starting up)
for i in $(seq 1 60); do
    [ -f "$PID_FILE" ] && break
    sleep 1
done
if [ ! -f "$PID_FILE" ]; then
    echo "[watchdog] PID file never appeared after 60s — exiting"
    exit 1
fi

TRAINING_PID="$(cat "$PID_FILE" 2>/dev/null || echo "")"
if [ -z "$TRAINING_PID" ]; then
    echo "[watchdog] PID file empty — exiting"
    exit 1
fi
echo "[watchdog] Monitoring PID ${TRAINING_PID}"

# v19 Fix 2: Track last poll epoch for clock-jump detection (WSL2 host suspension)
last_poll_epoch=""
# v19 Fix 3: Verbose logging counter
poll_count=0

while true; do
    sleep "$POLL_INTERVAL"

    # v19 Fix 2: Detect WSL2 host suspension (Windows sleep/hibernate).
    # If epoch gap since last poll exceeds 10× POLL_INTERVAL, the host was
    # likely suspended. Skip this cycle to let processes resume gracefully.
    now="$(date +%s)"
    if [ -n "$last_poll_epoch" ]; then
        poll_gap=$(( now - last_poll_epoch ))
        if [ "$poll_gap" -gt $(( POLL_INTERVAL * 10 )) ]; then
            echo "[watchdog] Clock jump detected: ${poll_gap}s since last poll (expected ~${POLL_INTERVAL}s)"
            echo "[watchdog] Possible host suspension — skipping this cycle to allow recovery"
            # Touch all heartbeat files to prevent stale detection next cycle
            for j in $(seq 0 $((${FLIGHTDOCK_N_ENVS:-4} - 1))); do
                touch "/tmp/flightdock_hb_env${j}" 2>/dev/null || true
            done
            last_poll_epoch="$now"
            continue
        fi
    fi
    last_poll_epoch="$now"

    # v19 Fix 3: Periodic verbose status logging (every ~5 min = 60 × 5s polls)
    poll_count=$((poll_count + 1))
    if [ $((poll_count % 60)) -eq 0 ]; then
        ages=""
        for j in $(seq 0 $((${FLIGHTDOCK_N_ENVS:-4} - 1))); do
            hb_file="/tmp/flightdock_hb_env${j}"
            if [ -f "$hb_file" ]; then
                hb_mtime="$(stat -c %Y "$hb_file" 2>/dev/null || echo 0)"
                age_s=$(( now - hb_mtime ))
                ages="${ages} env${j}=${age_s}s"
            else
                ages="${ages} env${j}=MISSING"
            fi
        done
        echo "[watchdog] Status: poll=${poll_count} PID=${TRAINING_PID} alive=$(kill -0 $TRAINING_PID 2>/dev/null && echo Y || echo N)${ages}"
    fi

    # Check 1: Is the training process still alive?
    if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
        # Verify PID file still matches — if atexit or wrapper already
        # deleted/replaced it, another run may have started on the same ports.
        current_pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
        if [ "$current_pid" != "$TRAINING_PID" ]; then
            echo "[watchdog] PID file changed or deleted — wrapper handled cleanup, exiting"
            exit 0
        fi
        echo "[watchdog] Training PID ${TRAINING_PID} is gone — sweeping ports"
        _sweep_ports
        rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*
        echo "[watchdog] Cleanup complete — exiting"
        exit 0
    fi

    # Check 1.5: Is the PID still our training process? (guards against PID reuse)
    if ! _is_flightdock_training "$TRAINING_PID"; then
        echo "[watchdog] PID ${TRAINING_PID} is no longer flightdock training (PID reuse?) — sweeping ports"
        _sweep_ports
        rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*
        echo "[watchdog] Cleanup complete — exiting"
        exit 0
    fi

    # Check 2: Per-worker heartbeat files (v13 Commit 12)
    N_ENVS="${FLIGHTDOCK_N_ENVS:-4}"
    checked=0
    stale_s=$(( HEARTBEAT_STALE_MINUTES * 60 ))
    for i in $(seq 0 $((N_ENVS - 1))); do
        hb_file="/tmp/flightdock_hb_env${i}"
        if [ -f "$hb_file" ]; then
            checked=$((checked + 1))
            hb_mtime="$(stat -c %Y "$hb_file" 2>/dev/null || echo 0)"
            now="$(date +%s)"
            age_s=$(( now - hb_mtime ))
            if [ "$age_s" -gt "$stale_s" ]; then
                # C3t: Skip stale-kill during warmup phases (DAW + bridge seeding).
                # The warmup flag is created before warmup and removed before model.learn().
                if [ -f "/tmp/flightdock_warmup_active" ]; then
                    echo "[watchdog] Env $i stale (${age_s}s) but warmup active — skipping kill"
                    continue
                fi
                echo "[watchdog] Env $i heartbeat stale: ${age_s}s (threshold: ${stale_s}s)"
                echo "[watchdog] Training loop appears hung — killing PID ${TRAINING_PID}"
                kill -9 "$TRAINING_PID" 2>/dev/null || true
                sleep 2
                _sweep_ports
                rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*
                echo "[watchdog] Hung training killed and cleaned up — exiting"
                exit 0
            fi
        fi
    done
    # WDFIX-593. Latch the moment per-worker heartbeats are seen. Rationale:
    # the legacy branch below is "pre-v13 compat" — it is for a run that never
    # had per-worker heartbeats AT ALL. It was written as a per-poll test of
    # "checked -eq 0", which is a DIFFERENT question, because the per-worker
    # files are unlinked during NORMAL SHUTDOWN (env close) while train.py is
    # still writing model_final/vecnormalize/replay_buffer OUTSIDE its
    # try/finally. So a graceful finish read as a pre-v13 run and the legacy
    # branch kill -9'd the trainer mid-save. That is what killed s18b at
    # 2026-08-15T23:18:40Z, truncating replay_buffer_final.pkl and losing the
    # W15 gate request. The wrapper cannot defend against it: it reaps this
    # watchdog in its EXIT trap, strictly AFTER train.py returns.
    if [ "$checked" -gt 0 ]; then
        seen_per_worker=1
        shutdown_first_seen=0
    fi

    # Per-worker heartbeats existed and are now gone => SHUTDOWN, not a hang.
    # Grant a bounded grace so the final saves can finish, but still kill if
    # the process is genuinely wedged past it (hang protection preserved).
    if [ "$checked" -eq 0 ] && [ "${seen_per_worker:-0}" -eq 1 ]; then
        now="$(date +%s)"
        if [ "${shutdown_first_seen:-0}" -eq 0 ]; then
            shutdown_first_seen="$now"
            echo "[watchdog] Per-worker heartbeats gone — treating as SHUTDOWN;"\
                 "grace ${SHUTDOWN_GRACE_S}s before any kill"
        fi
        grace_age_s=$(( now - shutdown_first_seen ))
        if [ "$grace_age_s" -gt "$SHUTDOWN_GRACE_S" ]; then
            echo "[watchdog] Shutdown exceeded grace (${grace_age_s}s > ${SHUTDOWN_GRACE_S}s)"
            echo "[watchdog] Training appears wedged in shutdown — killing PID ${TRAINING_PID}"
            kill -9 "$TRAINING_PID" 2>/dev/null || true
            sleep 2
            _sweep_ports
            rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*
            echo "[watchdog] Wedged shutdown killed and cleaned up — exiting"
            exit 0
        fi
        # No sleep here: the loop head already sleeps POLL_INTERVAL.
        continue
    fi

    # Legacy fallback — ONLY for a run that never had per-worker heartbeats
    # (pre-v13 compat). Gated on the latch so a shutdown can never reach it.
    if [ "$checked" -eq 0 ] && [ "${seen_per_worker:-0}" -eq 0 ] \
            && [ -f "$HEARTBEAT_FILE" ]; then
        hb_mtime="$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || echo 0)"
        now="$(date +%s)"
        age_s=$(( now - hb_mtime ))
        if [ "$age_s" -gt "$stale_s" ]; then
            echo "[watchdog] Legacy heartbeat stale for ${age_s}s (threshold: ${stale_s}s)"
            echo "[watchdog] Training loop appears hung — killing PID ${TRAINING_PID}"
            kill -9 "$TRAINING_PID" 2>/dev/null || true
            sleep 2
            _sweep_ports
            rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*
            echo "[watchdog] Hung training killed and cleaned up — exiting"
            exit 0
        fi
    fi
done
