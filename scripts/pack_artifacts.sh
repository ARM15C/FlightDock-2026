#!/usr/bin/env bash
# scripts/pack_artifacts.sh — Archive training artifacts for a campaign run.
#
# Usage:
#   ./scripts/pack_artifacts.sh <log-dir> <campaign-name>
#
# Creates: artifacts/<campaign-name>_<timestamp>/
#   - metrics.csv, policy_diagnostics.csv, campaign_analysis.md
#   - training + watchdog logs
#   - git state (branch, commit, diff --stat)
#   - config snapshot (PPO_CONFIG, LOG_STD_INIT, CURRICULUM_STAGES)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <log-dir> <campaign-name>"
    echo "  e.g. $0 logs/c005a C005a"
    exit 1
fi

LOG_DIR="$1"
CAMPAIGN_NAME="$2"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/${CAMPAIGN_NAME}_${TIMESTAMP}"

mkdir -p "$ARTIFACT_DIR"

# Copy training outputs
for f in metrics.csv policy_diagnostics.csv campaign_analysis.md reward_monitor.csv; do
    [ -f "${LOG_DIR}/${f}" ] && cp "${LOG_DIR}/${f}" "$ARTIFACT_DIR/" || true
done

# Patch C2 2026-05-06: copy paired_channels.csv per-env if produced.
mkdir -p "$ARTIFACT_DIR/paired_channels"
for envdir in "${LOG_DIR}"/env*/; do
    [ -d "$envdir" ] || continue
    envname="$(basename "$envdir")"
    paired_files=$(ls "${envdir}"paired_channels_ep*.csv 2>/dev/null || true)
    if [ -n "$paired_files" ]; then
        mkdir -p "$ARTIFACT_DIR/paired_channels/${envname}"
        cp "${envdir}"paired_channels_ep*.csv "$ARTIFACT_DIR/paired_channels/${envname}/" 2>/dev/null || true
    fi
done

# Copy latest training + watchdog logs
TRAIN_LOG_DIR="${REPO_ROOT}/logs/training_runs"
if [ -d "$TRAIN_LOG_DIR" ]; then
    # Copy most recent training log (|| true absorbs ls exit 2 under pipefail)
    latest_train="$(ls -t "$TRAIN_LOG_DIR"/train_*.log 2>/dev/null || true | head -1)"
    [ -n "$latest_train" ] && cp "$latest_train" "$ARTIFACT_DIR/training.log" || true
    # Copy most recent watchdog log
    latest_wd="$(ls -t "$TRAIN_LOG_DIR"/watchdog_*.log 2>/dev/null || true | head -1)"
    [ -n "$latest_wd" ] && cp "$latest_wd" "$ARTIFACT_DIR/watchdog.log" || true
fi

# Git state
{
    echo "Branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "--- diff --stat (vs HEAD) ---"
    git -C "$REPO_ROOT" diff --stat HEAD 2>/dev/null || echo "(clean)"
} > "$ARTIFACT_DIR/git_state.txt"

# Config snapshot via Python one-liner
# Patch C1 2026-05-06: include CONDITION_PROFILES + harness env vars for traceability.
python3 -c "
import os, sys; sys.path.insert(0, '${REPO_ROOT}/src')
from drl import config
print('PPO_CONFIG:', vars(config.PPO_CONFIG))
print('LOG_STD_INIT:', config.LOG_STD_INIT)
print('ACTION_NET_GAIN:', config.ACTION_NET_GAIN)
print('CURRICULUM_STAGES:')
for k, v in sorted(config.CURRICULUM_STAGES.items()):
    print(f'  Stage {k}: {vars(v)}')
print('CONDITION_PROFILES:')
for k, v in sorted(config.CONDITION_PROFILES.items()):
    print(f'  {k}: {v}')
print('DEFAULT_VALIDATION_PROFILE:', config.DEFAULT_VALIDATION_PROFILE)
print('RUNTIME_ENV:')
for env_key in ('HARNESS_UWB_ENABLE', 'HARNESS_LEADER_PROFILE',
                'FLIGHTDOCK_REWARD_SOURCE_MODE'):
    print(f'  {env_key}: {os.environ.get(env_key, \"<unset>\")!r}')
" > "$ARTIFACT_DIR/config_snapshot.txt" 2>/dev/null || echo "[pack] Config snapshot failed (non-fatal)"

echo "[pack] Artifacts saved to ${ARTIFACT_DIR}"
ls -la "$ARTIFACT_DIR"
