#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Curriculum-v3.1 SEALED-GATE live eval spawner + evidence
#   WRITERS (P1-b slice). Spawns eval_drl_checkpoint.py one leg at a time at
#   a v3 STAGE CELL (fixed-W wind via the FD_CURRICULUM_V3 sampler; trajectory
#   FORCED via HARNESS_LEADER_PROFILE for gate balance; rehearsal DISABLED via
#   FD_V3_REHEARSAL_DISABLE=1; evals ALWAYS --speedup 1.0), then writes the
#   two evidence files the P1-a adjudication core (curriculum_gate_runner.py)
#   demands: leg_meta.json + safety_summary.json. Sibling module by design —
#   the P1-a runner stays READ-only over evidence and under its 300-LoC cap.
# EVIDENCE WRITERS:
#   leg_meta.json: {seed, trajectory, stage_label, checkpoint_sha256,
#     abort_pre_control (spawn rc != 0)}.
#   bin_meta.json (T-60, ONLY when --bank-bin): the leg's OWN dataflash BIN
#     {banked, bin_path, bin_bytes, bin_sha256, work_dir, error}. Unflagged,
#     every leg runs from the SHARED ardupilot_formation/ArduPlane, so its log
#     lands on the same logs/00000001.BIN and SITL's unconditional --wipe makes
#     each leg DESTROY its predecessor's (T-60, MEASURED 2026-07-23). The flag
#     points the launcher's SITL_WORK_DIR hook inside the leg => paired by
#     construction, no copy, no mtime inference. ~79 MB/leg (MEASURED 78 kB/s
#     over the 44-leg #514 corpus) — hence opt-in.
#   safety_summary.json: hard_fails = breakaway_events + collision_events
#     counted from TRANSITION_SNAPSHOT.log (tracker-of-record: lines with
#     new=EMERGENCY_BREAKAWAY / SAFETY_COLLISION). Log ABSENT => hard_fails
#     null + instrument_gap true => the runner FAILs the leg (fail-closed,
#     evidence-missing rule) — never fabricated, never defaulted to 0.
# NO-RETRY RULE: a leg dir already holding leg_meta.json is NEVER respawned
#   (idempotent resume; FAILs stand — gates never resample).
# PUBLIC API: spawn_leg, write_leg_meta, write_safety_summary, write_bin_meta,
#   count_hard_fails, bridge_leg, build_legs, main. CLI: --checkpoint
#   --stage-label --out-dir --seeds a,b,.. [--vecnormalize P] [--env-id N]
#   [--dry-run] [--traj-map s:t,..] [--force-zero-residual] [--bank-bin]
#   (last three: T-58/T-60, Part A; ALL default-off => byte-identical
#   behaviour for every pre-T-58 caller).
# SIDE-EFFECTS: creates leg dirs under --out-dir; spawns eval subprocesses
#   (SITL legs — campaign dispatches own the launch hygiene: tmux/setsid/PGID).
#   With --bank-bin ONLY: also creates <leg>/sitl_work/ as ArduPlane's working
#   directory (eeprom.bin + logs/ land there instead of the shared tree).
#   T-277: at finalization also creates <leg>/paired/env0 -> env<K> (absolute
#   symlink, the relay bridge's exact semantic) whenever a paired CSV exists —
#   the T-131 orphaning fix at the source.
# CAPS: file soft 310 / hard 330 LoC; function cap 60. RAISED 2026-08-05
#   (T-277) from 260/300 with the T-131 source fix (was 282 with a T-60
#   soft-cap disclosure). The T-60 guidance to split evidence-writing into a
#   sibling module STANDS for the next substantive writer; bridge_leg is
#   finalization plumbing pinned to the relay's rule and belongs beside the
#   finalization call it serves.
# COVERAGE: tests/unit/test_curriculum_v3_machinery.py (writers + no-retry +
#   balance + label cross-pin) + test_t58_spawner_parta_capabilities.py (T-58
#   flags; AST) + test_t60_spawner_bin_banking.py (T-60 work dir + banking,
#   incl. the shared-path decoy control) + test_curriculum_eval_spawner.py
#   (T-277 bridge: both directions + relay-equivalence pin).
#   Live spawn path: LANECON CERTIFIED_N16 2026-07-23 (18 real legs).
# =============================================================================
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

FD = "/home/ubuntu/FlightDock"
sys.path.insert(0, FD)
from src.drl import config  # noqa: E402 (needs FD on sys.path first)

EVAL_PY = os.path.join(FD, "scripts", "eval_drl_checkpoint.py")
TRAJS = ("straight", "racetrack")
# T-322/A-1: exit-code contract. 0 all legs clean; RC_PARTIAL_LEGS = some legs
# failed but EVERY leg's evidence was written (lane callers accept it and let
# the faultscreen replace); 1/2 stay crash/usage = total lane failure.
RC_PARTIAL_LEGS = 20
# SPAWNFRESH-783. A leg whose dir already holds
# leg_meta.json is SKIPPED above ("NO-RETRY: evidence exists"), which is right
# for a resumed wave and WRONG for a campaign that must fly fresh legs -- and
# rc could not tell the two apart, because a fully skipped run still returned 0.
# MEASURED TWICE on 2026-08-21: a control leg that never flew was banked as
# flown, and ARMBDESC wave 1 would have reused 35 legs flown under a superseded
# episode-termination rule with every rc green. --require-fresh makes the
# caller's intent explicit and gives the skip its own sentinel.
RC_SKIPPED_EXISTING = 21


def freshness_rc(failed, skipped, require_fresh, dry_run):
    """The exit code, given what the run actually did.

    Precedence deliberately preserves the T-322/A-1 contract: RC_PARTIAL_LEGS
    still wins, because gate_relay_core treats an UNEXPECTED rc as total lane
    failure and re-ordering these would turn a replaceable fault into a gate
    abort. The JSON accounting line carries both facts regardless.
    A dry run spawns nothing BY DESIGN, so it can never be 'stale'.
    """
    if failed:
        return RC_PARTIAL_LEGS
    if require_fresh and not dry_run and skipped:
        return RC_SKIPPED_EXISTING
    return 0
# T-60: per-leg ArduPlane working directory, relative to the leg dir. Both the
# writer (spawn_leg, via SITL_WORK_DIR) and the reader (write_bin_meta) derive
# the path from THIS constant, so they cannot drift apart.
BIN_WORK_SUBDIR = "sitl_work"
# Stage-label -> v3 stage index (mirror of config.STAGE_LABELS_V3; the
# cross-pin unit test asserts this mapping against src/drl/config.py).
# A5 diagonal (R1 signed 2026-08-05): derived FROM config so this cross-pin
# can never silently diverge again — test_label_map_cross_pin asserts the
# round-trip. (Was a hardcoded formula map; conforming edit disclosed at A5.)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:      # importlib-by-path loads have no pkg context
    sys.path.insert(0, _SCRIPTS)
import leg_terminal_reader as _ltr  # noqa: E402 (S-12: terminal channel)
import leg_demoted_reader as _ldr  # noqa: E402 (TERMDEMOTE-767)
LABEL_TO_STAGE = {v: k for k, v in config.STAGE_LABELS_V3.items()}
_BREAKAWAY_RE = re.compile(r"new=EMERGENCY_BREAKAWAY")
_COLLISION_RE = re.compile(r"SAFETY_COLLISION")   # LEGACY-NEVER-FIRED (S-12):
# the translog vocabulary is FSM states — this token cannot occur there.
# Kept tagged pending a DELETION dispatch; no code consumes it.


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_hard_fails(leg_dir):
    """Breakaways from the TRANSITION_SNAPSHOT.log tracker; collisions from
    the ENV'S OWN episode terminal (leg_terminal_reader, S-12 — the old
    translog token count was a guard that could not fire). Missing EITHER
    source => (None, None, None) — the caller writes hard_fails null
    (fail-closed)."""
    path = None
    for root, _dirs, files in os.walk(leg_dir):
        if "TRANSITION_SNAPSHOT.log" in files:
            path = os.path.join(root, "TRANSITION_SNAPSHOT.log")
            break
    terms = _ltr.terminal_types(leg_dir)
    if path is None or terms is None:
        return None, None, None
    brk = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if _BREAKAWAY_RE.search(line):
                brk += 1
    # TERMDEMOTE-767: a demoted collision is still a collision EVENT. Count
    # the demoted intervals too -- one terminal used to mean exactly one
    # event, and `intervals` is that same count once the leg flies on.
    # Under-counting here would be a fail-OPEN in hard_fails.
    coll = terms.count("SAFETY_COLLISION") + (
        _ldr.demoted_events(leg_dir, "SAFETY_COLLISION") or 0)
    return brk + coll, brk, coll


def write_safety_summary(leg_dir):
    hard, brk, coll = count_hard_fails(leg_dir)
    payload = {"hard_fails": hard, "breakaway_events": brk,
               "collision_events": coll,
               "tracker": "TRANSITION_SNAPSHOT.log",
               "instrument_gap": hard is None}
    with open(os.path.join(leg_dir, "safety_summary.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def write_bin_meta(leg_dir):
    """T-60: record the leg's OWN dataflash BIN. Only called under --bank-bin,
    where spawn_leg pointed SITL_WORK_DIR at <leg_dir>/sitl_work, so the log is
    INSIDE the leg — pairing by construction, never by mtime (parallel legs
    collide on mtime). Runs AFTER the subprocess exits, so the log is flushed.
    ALWAYS writes the file so a reader can tell 'ran but not banked' from
    'never ran': failure => banked=false + error, never a missing file and
    never a silent fallback to the shared tree."""
    wd = os.path.join(leg_dir, BIN_WORK_SUBDIR)
    payload = {"banked": False, "bin_path": None, "bin_bytes": None,
               "bin_sha256": None, "work_dir": wd, "error": None}
    try:
        cands = sorted(glob.glob(os.path.join(wd, "logs", "*.BIN")),
                       key=os.path.getmtime)
        if not cands:
            payload["error"] = "no BIN under %s/logs" % wd
        else:
            src = cands[-1]
            payload.update(banked=True, bin_path=src,
                           bin_bytes=os.path.getsize(src),
                           bin_sha256=sha256_file(src))
    except Exception as e:
        payload["error"] = "%s: %s" % (type(e).__name__, e)
    with open(os.path.join(leg_dir, "bin_meta.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def write_leg_meta(leg_dir, seed, trajectory, stage_label, checkpoint,
                   abort_pre_control):
    payload = {"seed": int(seed), "trajectory": trajectory,
               "stage_label": stage_label,
               "checkpoint_sha256": (sha256_file(checkpoint)
                                     if os.path.isfile(checkpoint) else None),
               "abort_pre_control": bool(abort_pre_control)}
    with open(os.path.join(leg_dir, "leg_meta.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def bridge_leg(leg_dir):
    """T-131 SOURCE FIX (T-277): birth the leg with the pinned instrument
    path. Same rule and link semantic as the panel-level instrument of record
    (gate_relay_core.bridge_paired_layout, leg body) — the equivalence unit
    test pins the two together so they cannot drift. Idempotent; a leg
    without a paired CSV is left alone (downstream liveness guard fails it,
    correctly). Returns True iff a link was created.

    THE LINK IS RELATIVE (T-317) — see bridge_paired_layout for the four
    measured incidents the absolute form caused. `../env<K>` resolves
    identically in place and survives quarantine, a panel move and a mirror.
    """
    link = os.path.join(leg_dir, "paired", "env0")
    if os.path.exists(link):
        return False
    # DANGLING is not ABSENT — os.path.exists() is False for both, so this
    # used to raise FileExistsError instead of repairing.
    if os.path.lexists(link):
        os.remove(link)
    src = None
    for d in sorted(os.listdir(leg_dir)):
        if d.startswith("env") and os.path.isfile(
                os.path.join(leg_dir, d, "paired_channels_ep0001.csv")):
            src = os.path.join(leg_dir, d)
            break
    if src is None:
        return False
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(os.path.relpath(src, os.path.dirname(link)), link)
    return True


def build_legs(seeds, out_dir, stage_label, traj_map=None):
    """Balanced leg list: seeds alternate straight/racetrack in order —
    len(seeds) legs total; the caller supplies a bar-conformant count
    (mid12: 12 seeds -> 6/6; anchor30: 30 -> 15/15). T-58: an explicit
    traj_map overrides the k%2 default PER SEED — scenario equality against
    a flown corpus whose trajectory set is not k%2-reproducible (the Part A
    #514 block is 10 straight / 2 racetrack). Fail-closed: a traj_map seed
    absent from seeds is a hard error, never silently ignored."""
    tmap = dict(traj_map or {})
    unknown = set(tmap) - {int(s) for s in seeds}
    if unknown:
        raise SystemExit(
            f"--traj-map seeds not in --seeds: {sorted(unknown)}")
    legs = []
    for k, seed in enumerate(seeds):
        traj = tmap.get(int(seed), TRAJS[k % 2])
        legs.append({"seed": int(seed), "trajectory": traj,
                     "dir": os.path.join(
                         out_dir, f"{stage_label}_{seed}_{traj}")})
    return legs


def spawn_leg(leg, stage_label, checkpoint, vecnormalize, env_id, dry_run,
              force_zero=False, bank_bin=False):
    """One sealed eval leg at the stage cell. Env contract: FD_CURRICULUM_V3
    sampler supplies the fixed-W wind + sigma to the FDM (#465 channel);
    rehearsal disabled; trajectory forced; eval ALWAYS 1.0x."""
    stage_idx = LABEL_TO_STAGE[stage_label]
    cmd = [sys.executable, EVAL_PY,
           "--checkpoint", checkpoint,
           "--stage", str(stage_idx),
           "--n-episodes", "1",
           "--speedup", "1.0",
           "--env-id", str(env_id),
           "--seed", str(leg["seed"]),
           "--output-dir", leg["dir"]]
    if vecnormalize:
        cmd += ["--vecnormalize", vecnormalize]
    if force_zero:
        # T-58 passthrough of the T-57 eval flag (Part A arm P0: forced-zero
        # residual IS the baseline FSM at env:1614). Default-off => every
        # pre-T-58 caller's command line is byte-identical.
        cmd += ["--force-zero-residual"]
    env = dict(os.environ,
               FD_CURRICULUM_V3="1",
               FD_TRAIN_BACKEND="json",
               FD_DR_SEED_V2="1",
               FD_V3_REHEARSAL_DISABLE="1",
               HARNESS_LEADER_PROFILE=leg["trajectory"],
               HARNESS_UWB_ENABLE="1",
               # T-55 companion: route the paired-truth CSV (the corridor
               # instrument's DECISIVE input) into THIS leg's dir instead of
               # the global logs/env<k>/ (env:4735-4740 fallback), where two
               # waves at the same env_id would be indistinguishable.
               FLIGHTDOCK_PAIRED_CSV_DIR=leg["dir"])
    env.pop("FD_CURRICULUM_V2", None)   # mutual exclusion (config raises)
    if bank_bin:
        # T-60: give THIS leg its own ArduPlane working directory. The launcher
        # pushd's into SITL_WORK_DIR just before exec (start_formation_sitl_
        # fixed.sh:659-663); otherwise it cd's to the SHARED $ARDUPILOT_DIR/
        # ArduPlane (:244) and every leg's dataflash collides on
        # logs/00000001.BIN. Set LAST so per-leg beats any inherited value.
        env["SITL_WORK_DIR"] = os.path.join(leg["dir"], BIN_WORK_SUBDIR)
    if dry_run:
        print("DRY:", " ".join(cmd),
              f"[traj={leg['trajectory']} rehearsal=OFF speedup=1.0]"
              + (f" [SITL_WORK_DIR={env['SITL_WORK_DIR']}]" if bank_bin else ""))
        return None
    os.makedirs(leg["dir"], exist_ok=True)
    rc = subprocess.run(cmd, env=env).returncode
    return rc


def parse_traj_map(spec):
    """Parse --traj-map 'seed:traj,...'. Unknown trajectory => refuse.

    Extracted from main() unchanged (R-1, T-313) so the reporting path stays
    inside the 60-line function cap. Behaviour is byte-for-byte the same,
    INCLUDING the SystemExit: a silently-dropped trajectory would break
    scenario equality against a flown corpus (T-58).
    """
    traj_map = {}
    for tok in (spec.split(",") if spec else []):
        if not tok.strip():
            continue
        seed_s, _, traj = tok.partition(":")
        if traj not in TRAJS:
            raise SystemExit(f"--traj-map trajectory {traj!r} not in {TRAJS}")
        traj_map[int(seed_s)] = traj
    return traj_map


def main():
    ap = argparse.ArgumentParser(
        description="Curriculum v3.1 sealed-gate eval spawner (P1-b)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vecnormalize", default=None)
    ap.add_argument("--stage-label", required=True,
                    choices=sorted(LABEL_TO_STAGE))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", required=True,
                    help="comma-separated seed list (bar-conformant count)")
    ap.add_argument("--env-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--traj-map", default=None,
                    help="comma list seed:trajectory overriding the k%%2 "
                         "default per seed (T-58; scenario equality vs a "
                         "flown corpus). Unknown trajectory or seed => error.")
    ap.add_argument("--force-zero-residual", action="store_true",
                    help="pass --force-zero-residual through to "
                         "eval_drl_checkpoint.py (T-57/T-58; Part A arm P0)")
    ap.add_argument("--require-fresh", action="store_true",
                    help="SPAWNFRESH-783: exit 21 if any leg was SKIPPED for "
                         "an existing leg_meta.json. Default OFF (see the "
                         "RC_SKIPPED_EXISTING comment for the full rationale).")
    ap.add_argument("--bank-bin", action="store_true",
                    help="give each leg its own SITL_WORK_DIR so its dataflash "
                         "BIN lands inside the leg dir instead of clobbering "
                         "the shared ArduPlane log, and write bin_meta.json "
                         "(T-60). Costs ~79 MB/leg. Required to score a leg "
                         "with the BIN-based corrected corridor instrument.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    traj_map = parse_traj_map(args.traj_map)
    legs = build_legs(seeds, args.out_dir, args.stage_label, traj_map)
    spawned = skipped = 0
    failed = []          # R-1 (T-313): every leg whose CHILD returned non-zero
    for leg in legs:
        if os.path.exists(os.path.join(leg["dir"], "leg_meta.json")):
            skipped += 1        # NO-RETRY: evidence exists, never respawn
            continue
        rc = spawn_leg(leg, args.stage_label, args.checkpoint,
                       args.vecnormalize, args.env_id, args.dry_run,
                       force_zero=args.force_zero_residual,
                       bank_bin=args.bank_bin)
        if args.dry_run:
            continue
        if rc != 0:
            failed.append({"seed": leg["seed"], "rc": rc})
        write_leg_meta(leg["dir"], leg["seed"], leg["trajectory"],
                       args.stage_label, args.checkpoint,
                       abort_pre_control=(rc != 0))
        write_safety_summary(leg["dir"])
        bridge_leg(leg["dir"])              # T-277; T-131 fix at the source
        if args.bank_bin:
            write_bin_meta(leg["dir"])      # T-60; subprocess has EXITED => flushed
        spawned += 1
    print(json.dumps({"legs": len(legs), "spawned": spawned,
                      "skipped_existing": skipped,
                      "legs_failed": len(failed), "failed": failed,
                      "dry_run": bool(args.dry_run)}))
    # R-1 (T-313): this module used to return None and be called WITHOUT
    # sys.exit, so it exited 0 on every flown leg no matter how the child died.
    # Both eval drivers do `RC=$?` immediately after this call, so their rc was
    # structurally blind: armb_s030_driver.sh:95 counted a crashed leg as FLOWN.
    #
    # COUNT failures; do NOT max() the return codes. subprocess returncode is
    # NEGATIVE for a signalled child (-9 = SIGKILL), so max(0, -9) == 0 would
    # report a KILLED leg as success -- the exact silent-pass class this fixes.
    #
    # T-322/A-1: partial failure returns the
    # SENTINEL 20, not 1. gate_relay_core._spawn_lanes and
    # gate_relay_retention._spawn_stratum treat any unexpected rc as total
    # lane failure ("caller aborts WITHOUT writing a verdict"), so rc=1 for
    # one dead leg out of N turned a replaceable fault into a gate abort that
    # stalls the training pause. 20 = "some legs failed, every leg's evidence
    # (leg_meta.json/abort_pre_control) was written"; 1/2 remain crash/usage.
    # Single-leg drivers keep their semantics: [ $RC -eq 0 ] is still false.
    return freshness_rc(failed, skipped, args.require_fresh, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
