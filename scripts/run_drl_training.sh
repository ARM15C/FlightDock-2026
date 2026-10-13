#!/usr/bin/env bash
# scripts/run_drl_training.sh — Wrapper for unattended DRL training.
#
# Usage:
#   ./scripts/run_drl_training.sh [train.py arguments...]
#
# Features:
#   - EXIT trap: kills entire process group + port sweep on exit/crash
#   - Timestamped log capture
#   - Exit-code notification via send_training_notification.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${REPO_ROOT}/logs/training_runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# Parameterize port range from FLIGHTDOCK_MAX_ENVS (default 8)
MAX_ENVS="${FLIGHTDOCK_MAX_ENVS:-8}"
PORT_STRIDE=100

# G-TMUX (T-457, 2026-08-12) — fail-closed tmux precondition.
# Standing since 2026-07-12: campaigns ALWAYS run in tmux.
# Outside tmux this run dies with its ssh session — the setsid'd python
# survives but its tee writers are wrapper children (EPIPE) and the exit code
# is lost, so the campaign ends with no verdict and no log. Until now the only
# enforcement lived in one hand-copied block in an analysis-dir script, so
# every new wrapper had to remember it. train.py carries the same guard, so a
# wrapper that bypasses this script is still refused.
# --unsafe is train.py's existing development escape and is honoured here too.
_fd_unsafe=0
for _a in "$@"; do [ "$_a" = "--unsafe" ] && _fd_unsafe=1; done
if [ -z "${TMUX:-}" ] && [ "$_fd_unsafe" = "0" ]; then
    echo "[FATAL][TMUX] Training must run inside a tmux session." >&2
    echo "  Refusing before any SITL spawn. Launch as:" >&2
    echo "    tmux new-session -d -s <name> '$0 $*'" >&2
    echo "  Development-only override: --unsafe" >&2
    exit 9
fi

# G4 (T-457/T-458, wrapper limb added 2026-08-16) — fail-closed truth-channel
# precondition. train.py:1034 already CLAIMED this guard lived "here as well as
# in train.py so that a hand-written wrapper calling train.py directly cannot
# bypass them"; for the truth-channel half that claim was false (0 occurrences
# in this file) until now. It matters from 2026-08-16: the signed amendment
# BENCHMARK-T434EVAL-20260816 makes the gate/eval RELAY export
# FD_COLLISION_TRUTH_CHANNEL=1, so a training launch started from a relay shell
# inherits it in normal operation. With it armed, truth decides an episode
# boundary whose terminal_type the reward prices -> truth reaches the learner.
# Semantics MIRROR train.py exactly (not stricter): only "1" arms the channel,
# and only a ledger-id ack of the form T-<digits> waives it. A bare inherited
# "1" in the ack var is rejected by design.
if [ "${FD_COLLISION_TRUTH_CHANNEL:-}" = "1" ]; then
    case "${FD_COLLISION_TRUTH_CHANNEL_TRAIN_ACK:-}" in
        T-[0-9]*) : ;;   # explicit ledger-id acknowledgement; train.py re-checks
        *)
            echo "[FATAL][G4] FD_COLLISION_TRUTH_CHANNEL=1 in a TRAINING launch." >&2
            echo "  Truth-conditioned episode terminals are inadmissible for" >&2
            echo "  training (T-458). Refusing before any SITL spawn." >&2
            echo "  The gate/eval RELAY exports this flag; unset it here." >&2
            exit 43
            ;;
    esac
fi

# Tell train.py it is running under the wrapper (v10 Change 2)
export FLIGHTDOCK_WRAPPER=1

# v15 D6: Export watchdog-relevant env vars
# P=4 DummyVecEnv serial warmup takes ~20 min at 2x speedup (5 min/env x 4 envs).
# Default 15 min is insufficient. 30 min provides safe margin.
export FLIGHTDOCK_WATCHDOG_STALE_MIN="${FLIGHTDOCK_WATCHDOG_STALE_MIN:-30}"

# Parse --n-envs from CLI args for per-worker heartbeat enumeration
_n_envs=4  # default
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--n-envs" ]; then
        _n_envs="$_arg"
        break
    fi
    _prev="$_arg"
done
export FLIGHTDOCK_N_ENVS="$_n_envs"

# Parse --start-env-id the same way (T-271: cleanup()'s exit sweep is scoped
# to the env slots THIS run launched; train.py default is 0).
_start_env_id=0
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--start-env-id" ]; then
        _start_env_id="$_arg"
        break
    fi
    _prev="$_arg"
done
export FLIGHTDOCK_START_ENV_ID="$_start_env_id"

# Parse --log-dir for cleanup() post-campaign hooks (v15-commit Item B)
_log_dir=""
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--log-dir" ]; then
        _log_dir="$_arg"
        break
    fi
    _prev="$_arg"
done
export FLIGHTDOCK_LOG_DIR="${_log_dir:-${REPO_ROOT}/logs/training_runs}"

# Parse --checkpoint-dir (v15-commit Item B)
_checkpoint_dir=""
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--checkpoint-dir" ]; then
        _checkpoint_dir="$_arg"
        break
    fi
    _prev="$_arg"
done
export FLIGHTDOCK_CHECKPOINT_DIR="${_checkpoint_dir:-${REPO_ROOT}/checkpoints/default}"

# BUFBANK-728: refuse to launch while an UN-BANKED final replay buffer sits in
# the checkpoint dir — train.py writes replay_buffer_final.pkl to a fixed path
# and the next run overwrites it (destroyed D2a attempt-1's buffer 2026-08-19).
# Placed BEFORE the lockfile and port sweep so a refusal changes nothing.
if ! "${SCRIPT_DIR}/check_buffer_banked.sh" "$FLIGHTDOCK_CHECKPOINT_DIR"; then
    echo "[wrapper] ABORT: un-banked final replay buffer (see [bufbank] above)."
    exit 44
fi

echo "[wrapper] Watchdog: stale_min=${FLIGHTDOCK_WATCHDOG_STALE_MIN}, n_envs=${FLIGHTDOCK_N_ENVS}, log_dir=${FLIGHTDOCK_LOG_DIR}"

PID_FILE="/tmp/flightdock_training.pid"
HEARTBEAT_FILE="/tmp/flightdock_heartbeat"
WATCHDOG_PID=""

mkdir -p "$LOG_DIR"

# Define cleanup FIRST so the EXIT trap fires even if venv activation or
# lockfile creation fails under set -e. All variables referenced by cleanup
# are already defined above (WATCHDOG_PID="", PID_FILE, etc.) so early trap
# fire is safe — the guards handle empty/missing values.
cleanup() {
    local exit_code=$?
    echo "[wrapper] Training exited with code ${exit_code}" | tee -a "$LOG_FILE"

    # (Rev 3 Item #1) Emit wrapper_exit.txt as best-effort provenance.
    # Atomic (write temp + mv), idempotent, subshell-guarded so any
    # failure here never masks exit_code or subsequent cleanup.
    if [ -n "${FLIGHTDOCK_LOG_DIR:-}" ]; then
        (
            mkdir -p "${FLIGHTDOCK_LOG_DIR}" 2>/dev/null || exit 0
            _wx_tmp="${FLIGHTDOCK_LOG_DIR}/.wrapper_exit.txt.$$"
            _wx_final="${FLIGHTDOCK_LOG_DIR}/wrapper_exit.txt"
            {
                echo "# wrapper_exit"
                echo "rc=${exit_code}"
                echo "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            } > "${_wx_tmp}" 2>/dev/null \
                && mv -f "${_wx_tmp}" "${_wx_final}" 2>/dev/null
        ) || true
    fi

    # Kill the watchdog (v10 Change 4)
    if [ -n "$WATCHDOG_PID" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi

    # Read PID BEFORE deleting the file (v11 fix: review finding #8)
    training_pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
    rm -f "$PID_FILE" "$HEARTBEAT_FILE" /tmp/flightdock_hb_env*

    # Kill Python's process group (v11 Change 2b).
    # Python runs in its own session via setsid, so kill -PGID targets
    # Python and any in-session children. SITL processes launched with
    # start_new_session=True are in their own sessions — the port sweep
    # below is the mechanism that kills those.
    if [ -n "$training_pid" ]; then
        kill -TERM -- -"$training_pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- -"$training_pid" 2>/dev/null || true
    fi

    # Port-based sweep for SITL children in their own sessions.
    # T-271: scoped to the env slots THIS run launched (env_ids
    # start..start+n-1, from the --n-envs/--start-env-id already parsed
    # above). The old whole-box sweep (all MAX_ENVS slots regardless of
    # --n-envs) pkill-9'd the W12 gate panel's LIVE eval lanes (env2/env3,
    # ports 5960/6060) the second the learner halted on 2026-08-05, and the
    # relay adjudicated the truncated captures as failures (T-270). An
    # EXITING run may only kill what it started; the START-of-run preflight
    # sweep is deliberately unchanged (a starting run owns the box).
    # FD_WRAPPER_SWEEP_ALL=1 restores the whole-box sweep for manual use.
    local sweep_n="${FLIGHTDOCK_N_ENVS:-$MAX_ENVS}"
    local sweep_start="${FLIGHTDOCK_START_ENV_ID:-0}"
    if [ "${FD_WRAPPER_SWEEP_ALL:-0}" = "1" ]; then
        sweep_n="$MAX_ENVS"
        sweep_start=0
    fi
    local sweep_ports="" port _slot
    for _slot in $(seq 0 $(( sweep_n - 1 )) ); do
        sweep_ports="${sweep_ports} $(( 5760 + (sweep_start + _slot) * PORT_STRIDE ))"
    done
    echo "[wrapper] cleanup port sweep (scoped):${sweep_ports}" | tee -a "$LOG_FILE"
    for port in $sweep_ports; do
        local leader_port=$(( 14550 + port - 5760 ))
        pkill -9 -u "$USER" -f "serial0=tcp:${port}" 2>/dev/null || true
        pkill -9 -u "$USER" -f "tcp:127.0.0.1:${port}" 2>/dev/null || true
        pkill -9 -u "$USER" -f "leader_broadcast_fixed.py.*${leader_port}" 2>/dev/null || true
    done
    # Kill stale forkserver processes (v11 Change 1d)
    pkill -9 -u "$USER" -f "from multiprocessing.forkserver import main" 2>/dev/null || true

    # Notify
    if [ -f "${SCRIPT_DIR}/send_training_notification.py" ]; then
        python "${SCRIPT_DIR}/send_training_notification.py" \
            --exit-code "$exit_code" \
            --log-file "$LOG_FILE" \
            --timestamp "$TIMESTAMP" \
            2>/dev/null || true
    fi

    # Post-campaign analysis + artifact packing (v15-commit Items B/E)
    # Guarded with || true — set -euo pipefail cannot abort on failure.
    if [ -n "${FLIGHTDOCK_LOG_DIR:-}" ] && [ -f "${SCRIPT_DIR}/drl_campaign_analysis.py" ]; then
        echo "[wrapper] Running campaign analysis..." | tee -a "$LOG_FILE"
        python "${SCRIPT_DIR}/drl_campaign_analysis.py" \
            --log-dir "$FLIGHTDOCK_LOG_DIR" 2>/dev/null || true
    fi
    if [ -n "${FLIGHTDOCK_LOG_DIR:-}" ] && [ -f "${SCRIPT_DIR}/pack_artifacts.sh" ]; then
        echo "[wrapper] Packing artifacts..." | tee -a "$LOG_FILE"
        "${SCRIPT_DIR}/pack_artifacts.sh" \
            "$FLIGHTDOCK_LOG_DIR" \
            "${FLIGHTDOCK_CAMPAIGN_NAME:-unnamed}" 2>/dev/null || true
    fi

    # BUFBANK-728: bank the freshly written final buffer + model into a
    # run-stamped dir and stamp the .BANKED sha marker the launch gate
    # (check_buffer_banked.sh) requires. Failure is LOUD but never masks
    # the run's exit code or the remaining cleanup.
    _bb_buf="${FLIGHTDOCK_CHECKPOINT_DIR:-}/replay_buffer_final.pkl"
    if [ -n "${FLIGHTDOCK_CHECKPOINT_DIR:-}" ] && [ -f "$_bb_buf" ]; then
        _bb_dest="${FLIGHTDOCK_CHECKPOINT_DIR}/banked_buffers/final_$(date -u +%Y%m%dT%H%M%SZ)_${FLIGHTDOCK_CAMPAIGN_NAME:-unnamed}"
        if mkdir -p "$_bb_dest" \
           && cp -p "$_bb_buf" "$_bb_dest/" \
           && { [ ! -f "${FLIGHTDOCK_CHECKPOINT_DIR}/model_final.zip" ] \
                || cp -p "${FLIGHTDOCK_CHECKPOINT_DIR}/model_final.zip" "$_bb_dest/"; } \
           && sha256sum "$_bb_buf" | cut -d' ' -f1 > "${_bb_buf}.BANKED"; then
            echo "[wrapper] BUFBANK: final buffer banked to ${_bb_dest}" | tee -a "$LOG_FILE"
        else
            echo "[wrapper] BUFBANK FAILED: final buffer NOT banked — the next" \
                 "launch will be REFUSED by check_buffer_banked.sh (rc 44)" \
                 "until it is banked by hand." | tee -a "$LOG_FILE"
        fi
    fi

    # Release lockfile (v11 Change 1e)
    rm -f "${LOCK_FILE:-}"
}
trap cleanup EXIT

# Concurrent-run lockfile (v11 Change 1e) — prevents two wrappers from
# stomping each other's port range. Uses noclobber (set -C) for atomic
# file creation; if the file already exists, the redirect fails.
# IMPORTANT: trap is registered ABOVE so lockfile is cleaned up if anything
# below fails under set -e.
LOCK_FILE="/tmp/flightdock_training.lock"
if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
    existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")"
    echo "[wrapper] ABORT: lockfile ${LOCK_FILE} exists (PID ${existing_pid})."
    echo "[wrapper] Another training session may be running. Remove lockfile if stale:"
    echo "          rm -f ${LOCK_FILE}"
    exit 1
fi

# Activate venv
export PYTHONPATH="${PYTHONPATH:-}"   # T-62: venv activate:69 derefs $PYTHONPATH unguarded; set -u would abort here
source "${REPO_ROOT}/flightdock_native_venv/bin/activate"

echo "[wrapper] Starting training at ${TIMESTAMP}" | tee "$LOG_FILE"
echo "[wrapper] Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "[wrapper] Args: $*" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

# Defensive heartbeat / lock / pid sweep (added 2026-04-07).
#
# *** ADDED IN RESPONSE TO PHASE 8 STEP 1 LAUNCH FAILURE ***
# The first C3w full-run launch at 2026-04-07T03:20:42Z failed at
# 03:20:58Z with exit code 137 (SIGKILL) after 16 seconds. Root cause:
# stale per-worker heartbeat files at /tmp/flightdock_hb_env<N> left
# behind by the prior C3w audit run caused the training watchdog to
# compute age_s ≈ 12273 s, exceed the 1800 s stale threshold, and
# SIGKILL the new training PID before Python could create the
# /tmp/flightdock_warmup_active guard flag. The lockfile creation
# below (set -C noclobber on $LOCK_FILE) already protects against
# concurrent runs, so removing /tmp/flightdock_training.lock here is
# safe — if a concurrent run were active, the lockfile creation would
# have failed earlier and we never would have reached this line.
#
# This sweep is unconditional rather than conditional because the
# cost of removing files that should not exist is zero, while the
# cost of leaving stale files is a SIGKILL of the training process
# 16 seconds after launch. Reference: status_20260407_032048.json,
# watchdog_20260407_032048.log.
rm -f /tmp/flightdock_hb_env* \
      /tmp/flightdock_warmup_active \
      /tmp/flightdock_heartbeat
echo "[wrapper] Defensive heartbeat sweep complete (hb_env*, warmup_active, heartbeat)" \
    | tee -a "$LOG_FILE"

# (Rev 3 Item #2) Launch-time provenance.
#
# FLIGHTDOCK_WRAPPER_ARGV_JSON is INTERNAL one-shot transport for
# wrapper -> train.py argv passthrough. It is consumed once at
# config_snapshot emission time and is NOT a user-facing
# configuration surface. Do not document in --help or user docs.
#
# launch_command.txt is a human-readable convenience artifact.
# Canonical provenance is config_snapshot.json wrapper_argv/train_argv.
mkdir -p "${FLIGHTDOCK_LOG_DIR}" 2>/dev/null || true
FLIGHTDOCK_WRAPPER_ARGV_JSON="$(
    printf '%s\n' "$0" "$@" \
    | python -c "import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin.read().splitlines()]))" \
    2>/dev/null || echo "[]"
)"
export FLIGHTDOCK_WRAPPER_ARGV_JSON
{
    echo "# FlightDock wrapper launch provenance (human-readable)"
    echo "# Canonical provenance is config_snapshot.json wrapper_argv / train_argv."
    echo "# Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "## wrapper_argv"
    printf '%s\n' "$0" "$@"
    echo ""
    echo "## env (diff from defaults, best-effort)"
    for v in FLIGHTDOCK_WATCHDOG_STALE_MIN FLIGHTDOCK_N_ENVS \
             FLIGHTDOCK_LOG_DIR FLIGHTDOCK_CHECKPOINT_DIR \
             CUDA_VISIBLE_DEVICES; do
        echo "${v}=${!v:-<unset>}"
    done
    echo ""
    echo "## git HEAD"
    git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown"
    echo ""
    echo "## train.py SHA-256"
    sha256sum "${REPO_ROOT}/src/drl/train.py" 2>/dev/null \
        | awk '{print $1}' || echo "unknown"
} > "${FLIGHTDOCK_LOG_DIR}/launch_command.txt" 2>/dev/null || true

# Normalize deprecated FD_FW_VARIANT alias before SITL launch.
# tests/start_formation_sitl_fixed.sh accepts only {legacy, modern} and exits
# non-zero on any other value, which causes silent SITL non-spawn (Popen
# returns 0.01s without observing the bash exit code) and 30s heartbeat
# timeouts in the controller's preflight. Several historical campaign
# scripts set FD_FW_VARIANT=baseline; mirror the eval-side normalization
# in scripts/eval_baseline_zero_residual.py::_normalize_fd_fw_variant().
if [ "${FD_FW_VARIANT:-}" = "baseline" ]; then
    echo "[wrapper] Normalizing deprecated FD_FW_VARIANT='baseline' -> 'legacy'"
    export FD_FW_VARIANT=legacy
fi

# Launch watchdog in its own process group (v10 Change 4).
setsid "${SCRIPT_DIR}/training_watchdog.sh" \
    "$PID_FILE" "$HEARTBEAT_FILE" \
    >> "${LOG_DIR}/watchdog_${TIMESTAMP}.log" 2>&1 &
WATCHDOG_PID=$!
disown "$WATCHDOG_PID" 2>/dev/null || true

# v11 Change 2a: setsid gives Python its own session and process group.
# cleanup() can then kill -PGID to terminate Python + SITL children
# without killing the wrapper shell (which caused exit code 143).
#
# (Rev 3 Item #1) FD-split tee: stdout and stderr captured to
# separate per-invocation logs AND merged into $LOG_FILE for legacy
# reader compat. Process substitution (not a pipeline) preserves the
# child's exit status directly under set -e without pipefail masking.
mkdir -p "${FLIGHTDOCK_LOG_DIR}" 2>/dev/null || true
setsid python "${REPO_ROOT}/src/drl/train.py" "$@" \
    > >(tee -a "$LOG_FILE" "${FLIGHTDOCK_LOG_DIR}/wrapper_stdout.log") \
    2> >(tee -a "$LOG_FILE" "${FLIGHTDOCK_LOG_DIR}/wrapper_stderr.log" >&2)

# v19 P8: Detect SAC fallback sentinel from ZeroResidualFallbackCallback.
# If TD3 training stopped due to zero-residual trap, relaunch with SAC.
SENTINEL="${FLIGHTDOCK_LOG_DIR}/FALLBACK_REQUEST.json"
if [ -f "$SENTINEL" ]; then
    echo "[wrapper] FALLBACK_REQUEST detected — switching TD3 → SAC" | tee -a "$LOG_FILE"
    # Extract checkpoint path from sentinel JSON
    CHECKPOINT=$(python3 -c "import json; print(json.load(open('${SENTINEL}'))['checkpoint'])" 2>/dev/null || echo "")
    # Remove sentinel BEFORE relaunch to prevent infinite loop
    rm -f "$SENTINEL"
    if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
        echo "[wrapper] Resuming from: ${CHECKPOINT}" | tee -a "$LOG_FILE"
        # Build SAC args: replace --algorithm td3 with sac, add --resume-from
        SAC_ARGS=()
        SKIP_NEXT=false
        for arg in "$@"; do
            if $SKIP_NEXT; then
                SKIP_NEXT=false
                continue
            fi
            if [ "$arg" = "--algorithm" ]; then
                SAC_ARGS+=("--algorithm" "sac")
                SKIP_NEXT=true
                continue
            fi
            SAC_ARGS+=("$arg")
        done
        SAC_ARGS+=("--resume-from" "$CHECKPOINT")
        echo "[wrapper] SAC args: ${SAC_ARGS[*]}" | tee -a "$LOG_FILE"
        # (Rev 3 Item #1) FD-split tee — same rationale as TD3 launch.
        setsid python "${REPO_ROOT}/src/drl/train.py" "${SAC_ARGS[@]}" \
            > >(tee -a "$LOG_FILE" "${FLIGHTDOCK_LOG_DIR}/wrapper_stdout.log") \
            2> >(tee -a "$LOG_FILE" "${FLIGHTDOCK_LOG_DIR}/wrapper_stderr.log" >&2)
    else
        echo "[wrapper] WARNING: Sentinel found but checkpoint missing: ${CHECKPOINT}" | tee -a "$LOG_FILE"
    fi
fi

# --- GATEFIRE-627 L2 (2026-08-17): never report success on a missing gate ----
# s19 consumed its full 48,000-step budget, exited rc 0 with
# termination_reason "normal_budget_exhaustion", and never filed its gate
# request — the block existed to earn an exam, finished without asking for one,
# and REPORTED SUCCESS. This limb converts that silent failure into a loud one:
# if the stage consumed its nominal and no gate_requests/<label>.json exists,
# the wrapper writes a named sentinel and exits NON-ZERO.
#
# It is deliberately independent of the runtime fix (L1) — a stale deploy that
# lacks L1 is exactly the case this has to keep catching. The audit itself
# never raises and never guesses: the ONE definition of "nominal" it uses is
# the one the runtime uses, imported, so the two cannot drift.
if [ "${FD_CURRICULUM_V3:-0}" = "1" ]; then
    GATE_AUDIT_RC="$(REPO_ROOT="$REPO_ROOT" python - "$FLIGHTDOCK_LOG_DIR" <<'PYGATEAUDIT' 2>>"$LOG_FILE" || echo 0
import json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
try:
    from src.drl.callbacks import gate_request_audit, GATE_AUDIT_SENTINEL_NAME
    rc, report = gate_request_audit(sys.argv[1])
    if rc:
        with open(os.path.join(sys.argv[1], GATE_AUDIT_SENTINEL_NAME), "w") as fh:
            json.dump(report, fh, indent=1)
    sys.stderr.write("[wrapper][GATEFIRE-627] audit: %s\n" % json.dumps(report))
    print(rc)
except Exception as exc:                       # never abort on the audit itself
    sys.stderr.write("[wrapper][GATEFIRE-627] audit unavailable: %s\n" % exc)
    print(0)
PYGATEAUDIT
)"
    if [ "${GATE_AUDIT_RC:-0}" != "0" ]; then
        echo "[FATAL][GATEFIRE-627] This block consumed its stage nominal but" \
             "filed NO gate request. Refusing to report success." \
             | tee -a "$LOG_FILE" >&2
        echo "  sentinel: ${FLIGHTDOCK_LOG_DIR}/GATE_REQUEST_MISSING.json" \
             | tee -a "$LOG_FILE" >&2
        exit 17
    fi
fi
