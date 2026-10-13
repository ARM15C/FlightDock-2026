# RESPONSIBILITY: the EXACT venue ARM-B's own driver exports before every leg of the signed
#   84-row bank, replicated so ARMBLAD-035's PID reference is measured on the SAME venue as the
#   bank it will be compared against. Not invented here: every line is carried from
#   armbsig_20260824T0030Z/VENUE_PREAMBLE.sh, which is itself carried from armbsig_driver.sh.
#   A venue difference would make the comparison meaningless.
#
#   WHY THIS FILE EXISTS AT ALL, stated plainly: the first B4 launch attempt (2026-08-26
#   19:32:29Z) INLINED the env block into the driver and omitted the sourced venue file plus
#   four warmup/safety knobs. All 24 slots HALTED on `EVAL env contract violated before leg 1`
#   with 6 violations. Nothing burned -- zero SITL, zero seeds, zero steps, wall 0 s -- because
#   preflight_contract.py is fail-closed and runs BEFORE leg 1. The venue now lives in ONE
#   sourced file with an executable detector (selfcheck limb S9) instead of being retyped.
# SIDE-EFFECTS: exports environment variables only. Spawns nothing, writes nothing.
# NOT SET HERE (the driver sets them per row): FD_V3_SIGMA_OVERRIDE_MS[_ACK].
# NOT SET HERE (curriculum_eval_spawner sets them per leg): HARNESS_LEADER_PROFILE,
#   HARNESS_UWB_ENABLE, FLIGHTDOCK_PAIRED_CSV_DIR.
FD=/home/ubuntu/FlightDock
B035D=$FD/analysis/armblad035_20260830T1600Z
export FD_LEADER_TX_DELAY_S=0 FD_JSON_PAIR_FREEZE=1 FD_FDM_RECV_TIMEOUT_S=0
export FD_UWB_TRUTH_SOURCED=1 HARNESS_UWB_RATE_HZ=10
export FD_CURRICULUM_V3=1 FD_TRAIN_BACKEND=json FD_DR_SEED_V2=1
export FD_ALLOW_WARMUP_OVERRIDE=1 FD_WARMUP_MAX_SIM_S=900
export FD_LEADER_WIND=1
export FD_WARMUP_WALL_TIMEOUT_S=2100 FLIGHTDOCK_WATCHDOG_STALE_MIN=40
export FD_V3_REHEARSAL_DISABLE=1
export FD_COLLISION_TRUTH_CHANNEL=1     # T434: judge terminals on TRUTH
export FD_GUARD_CEILING_CHANNEL=env     # GUARD-ELECT-588 elected channel
export ARMB_WAVE="${ARMB_WAVE:-armblad035}"   # corpus provenance
# RELAYDEMOTE-768: the EVAL contract knobs come from ONE generated file, SOURCED not retyped.
# This is the line whose absence halted all 24 slots on the first attempt.
source "$FD/scripts/eval_venue_env.sh" || { echo "FATAL: scripts/eval_venue_env.sh missing (RELAYDEMOTE-768); regenerate with python tools/gen_eval_venue_env.py --write"; exit 2; }
export FD_WARMUP_ACCEPT_GATE=1
export FD_WARMUP_SETTLE_DWELL_N=60
export FD_WARMUP_GATE_BELOW_TOL_M=2.0 FD_SAFETY_TERMINALS=SAFETY_CONTACT,SAFETY_NOPASS
export FD_FDM_BANK_DIR="${FD_FDM_BANK_DIR:-$B035D/fdm_bank}"
# sigma is PER LEG, never at slot scope -- a constant here would fly every rung at one sigma
unset FD_V3_SIGMA_OVERRIDE_MS
unset FD_ENV_SIMSTATE_RATE_MATCH
unset FD_ALIGN_MODE_CENSUS FD_TRAIN_BOOT_SCREEN FD_INJECT_CONST_ACTION
unset FD_RESPID_BYPASS_CRRL FD_RESPID2_INJECT FD_CRRL_BPC_BETA_HDG_NEG
unset FD_RAW_EPOCH_COMP FD_EPOCH_PROBE FD_LEADER_CONT_CLOCK
unset FD_LEADER_BOOT_PROBE FD_LEADER_PROBE_DIR
export FLIGHTDOCK_CAMPAIGN_NAME=BASELINE_035_PID_REFERENCE
