#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: N6 HALF-OPEN AUTHORITY PROBE EXECUTOR (N6AUTH-806, class VALIDATION).
#   Re-fly the frozen gate_W16 policy on A's EXACT 24 seeds (W16 08-17 panel +
#   A's W1 draw), today's venue, with ONE declared, reverted source delta:
#     src/drl/config.py:308 CRRL_BPC_FLOOR_NEG (0.25, 0.0, 0.25)
#                                             -> (0.25, 0.0, 0.50)
#   i.e. the policy's altitude-axis DESCENT floor in DOCKING_CREEP /
#   HOLD_CONTACT opens from 0.25 m to 0.50 m (N5 priced 1.00). Speed brake untouched
#   (ELECT-788 stays), POS floors stay zero (cannot push the follower up).
#   Hypothesis (prereg §3): a half-open floor keeps N5's bust cut WITHOUT N5's dwell cost.
#   Paired leg-for-leg against A (AREGATE-803) and N5 (N5AUTH-805).
#
# DERIVED FROM a_regate_launch.py (AREGATE-803), reused not rewritten:
#   signature gate, venue derivation from the relay's own env block, custody
#   assert over EVERY block, spawn, adjudicate, boot_pass dwell via
#   curriculum_gate_runner._frozen(), score_leg via leg_verdict, G3 table.
#   ADDED: assert_patch (N4b-1 pattern), g4_static (closes A's D1 for this
#   dispatch), paired delta vs A, applied-vs-commanded altitude residual in
#   the pre-trip window (the falsifier that the delta reached the plant),
#   post-flight patch REVERT with a witness in the readout.
#
# CUSTODY: ZERO spend. Both cells go through pinned_seeds (row n=0). EVERY
#   cursor must move by exactly 0 and the gate-verdicts dir file count must be
#   unchanged; anything else is rc 6 and the panel is NOT read.
#
# PUBLIC API: none -- CLI only.
#   python3 n6_auth_launch.py [--dry-run] [--out-root DIR] [--cells W16,W1]
#                             [--keep-patch]
#   rc 0 ok · 2 precondition refused · 3 spawn failure · 4 adjudication
#   failure · 5 protocol invalid · 6 CUSTODY BREACH · 7 REVERT FAILED.
#
# SIDE-EFFECTS: creates <out-root>/{legs/{W16,W1},n6_auth_adjudication*.json,
#   N6_AUTH_READOUT.md,PANEL_ETA.json,SEED_DRAW.json}; appends TWO n=0 reuse
#   rows to the gate relay seed ledger; runs tools/leakage_static_check.py
#   (read-only); on a real (non-dry) run REVERTS src/drl/config.py via
#   `git checkout --` at the end and writes the witness. Nothing else.
# CAPS: soft 800 / hard 850 LoC. DISCLOSED: the plan declared 700/760 and the
#   derived file landed at 809 -- my LoC estimates run LOW (A: 350-400 est,
#   634 actual); raised here rather than compressed. Function cap 70.
#   Coverage: n6_auth_selfcheck.py (RUN, never just compiled).
# =============================================================================
import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

FD = "/home/ubuntu/FlightDock"
CAMP = os.path.join(FD, "analysis/c006_campaign_20260728T110000Z")
HERE = os.path.join(FD, "analysis/n6auth_20260822T2130Z")
PREREG = os.path.join(HERE, "N6_AUTH_PREREG.md")
SIG_RE = re.compile(r"^OPERATOR-SIGNATURE: N6AUTH-", re.M)
A_LEGS = os.path.join(CAMP, "logs_w1418/aregate803_20260822T164903Z/legs")
G4_STATIC = os.path.join(FD, "tools/leakage_static_check.py")
PIN = "8d19c38c9a"
CKPT = os.path.join(CAMP, "ckpt_w1418/v3_gate_snapshots/gate_W16")
CKPT_SHA16 = "3f6638d7974ab8f6"
VECNORM_SHA16 = "07cd09636af9fb05"
SPAWNER = os.path.join(FD, "scripts/curriculum_eval_spawner.py")
RUNNER = os.path.join(FD, "scripts/curriculum_gate_runner.py")
SEED_BLOCKS = os.path.join(FD, "analysis/curriculumv3_20260717T025122Z"
                               "/SEED_BLOCKS.json")
LEDGER = os.path.join(CAMP, "logs_w1418/gate_relay_ledger.jsonl")
VERDICTS_DIR = os.path.join(CAMP, "logs_w1418/gate_verdicts")
RELAY = os.path.join(CAMP, "w1424_relay.sh")

BAR_M = 2.00          # H1, criterion-frozen, decoupled from the guard
EPS = 5e-4            # BOUNDARY class half-width (N4b12.D1)
DWELL_S = 3.0         # G1 bar
GUARD_EXPECT = -2.0   # today's venue; asserted, NEVER altered by A
W1_SPEND = 0          # N6 spends NOTHING: both cells are pinned re-flies (n=0)

# ---- THE DECLARED PATCH. Exactly one file, exactly one line (N4b-1 pattern).
PATCH_FILE = "src/drl/config.py"
DECLARED_OLD = ("CRRL_BPC_FLOOR_NEG: tuple = (0.25, 0.0, 0.75) "
                "# FLOORELECT-822 (signed 2026-08-28): speed-NEG brake 0.25, "
                "alt-NEG descent 0.75, terminal window only")   # F1-REPIN 2026-08-29
DECLARED_NEW = ("CRRL_BPC_FLOOR_NEG: tuple = (0.25, 0.0, 0.50) "
                "# N6AUTH-806 (SIGNED N6AUTH-20260822): alt-NEG floor HALF-open "
                "(0.50) for ONE validation panel; REVERTED after the run")
# A's per-leg truth peaks (AREGATE-803 out/aregate_verd_*.json): the pairing.
N5_PEAKS = {"W16_47026_racetrack": 2.03, "W16_47040_straight": 0.92,
            "W16_47093_straight": 2.52, "W16_47117_racetrack": 1.56,
            "W16_47152_racetrack": 0.02, "W16_47159_straight": 1.52,
            "W16_47197_straight": -8.41, "W16_47201_racetrack": 1.71,
            "W16_47231_straight": 2.40, "W16_47320_straight": 1.96,
            "W16_47440_racetrack": 1.34, "W16_47480_racetrack": 1.89,
            "W1_47009_straight": 0.79, "W1_47094_racetrack": 0.60,
            "W1_47113_racetrack": 1.73, "W1_47132_racetrack": 0.40,
            "W1_47150_racetrack": 1.88, "W1_47163_racetrack": 1.34,
            "W1_47174_straight": 1.06, "W1_47235_straight": 0.83,
            "W1_47238_straight": 1.10, "W1_47317_straight": 1.32,
            "W1_47372_racetrack": 0.72, "W1_47393_straight": 2.46}
# N5 (floor 1.00) per-leg truth peaks, n5auth805 readout: reported BESIDE A,
# decides nothing. The §6 quantity stays the paired delta vs A.
A_PEAKS = {"W16_47026_racetrack": 2.17, "W16_47040_straight": 1.86,
           "W16_47093_straight": 1.84, "W16_47117_racetrack": 1.40,
           "W16_47152_racetrack": 2.67, "W16_47159_straight": 2.65,
           "W16_47197_straight": 2.16, "W16_47201_racetrack": 1.92,
           "W16_47231_straight": 2.21, "W16_47320_straight": 1.43,
           "W16_47440_racetrack": 1.85, "W16_47480_racetrack": 2.40,
           "W1_47009_straight": 2.32, "W1_47094_racetrack": 1.56,
           "W1_47113_racetrack": 1.81, "W1_47132_racetrack": 1.79,
           "W1_47150_racetrack": 1.65, "W1_47163_racetrack": 1.76,
           "W1_47174_straight": 2.49, "W1_47235_straight": 1.50,
           "W1_47238_straight": 1.67, "W1_47317_straight": 1.90,
           "W1_47372_racetrack": 0.87, "W1_47393_straight": 2.23}
# Prereg §6 reading, in code.
RULE = {"lever_max_busts": 4, "lever_min_drop_m": 0.30, "lever_min_dwell": 16,
        "inert_min_busts": 8, "inert_max_abs_delta_m": 0.10, "inconclusive_reach": 0.10}

# W16: the 08-17 gate panel's own seeds. Order r,s,s,r,r,s,s,r,s,s,r,r --
# NOT index parity. The default balanced map would silently re-pair (N3/N4a).
W16_PANEL = [(47026, "racetrack"), (47040, "straight"), (47093, "straight"),
             (47117, "racetrack"), (47152, "racetrack"), (47159, "straight"),
             (47197, "straight"), (47201, "racetrack"), (47231, "straight"),
             (47320, "straight"), (47440, "racetrack"), (47480, "racetrack")]
# W1 has NO prior panel to pair against, so its map is DECLARED here rather
# than inherited: alternating, 6 of each, recorded in SEED_DRAW.json so the
# panel is reproducible from the artifact alone.
W1_PANEL = [(47132, "racetrack"), (47317, "straight"), (47372, "racetrack"),
            (47393, "straight"), (47163, "racetrack"), (47009, "straight"),
            (47150, "racetrack"), (47238, "straight"), (47094, "racetrack"),
            (47174, "straight"), (47113, "racetrack"), (47235, "straight")]
# ^ A's W1 draw verbatim (SEED_DRAW.json): order IS the pairing.

VENUE = {"FD_COLLISION_TRUTH_CHANNEL": "1", "FD_GUARD_CEILING_CHANNEL": "env",
         "FD_DEMOTE_TERMINALS": "SAFETY_AIRSPEED,SAFETY_COLLISION",
         "FD_SAFETY_TERMINALS": "SAFETY_CONTACT,SAFETY_NOPASS",
         "FD_V3_REHEARSAL_DISABLE": "1"}
UNSET = ("FD_CEILING_SHAPE_ENABLE", "FD_FSM_NESTING_REPAIR", "ARMB_WAVE",
         "FD_TELEM_FRESHNESS", "FD_FORMATION_LOST_TERM",
         "FD_NOPASS_TERM_AHEAD_M", "FD_NOPASS_TERM_PERSIST_S",
         "FD_NOPASS_TERM_OFFAXIS_M")
AXES = ("behind", "below", "cross")


def die(rc, msg):
    print(f"N6-AUTH REFUSED ({rc}): {msg}", file=sys.stderr)
    raise SystemExit(rc)


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def assert_signature():
    if not os.path.isfile(PREREG):
        die(2, f"prereg missing: {PREREG}")
    if not SIG_RE.search(open(PREREG).read()):
        die(2, "prereg carries NO '^OPERATOR-SIGNATURE: N6AUTH-' line at "
               "column 0 — an operator act, not mine to author.")


def assert_patch():
    """THE patch invariant (N4b-1 pattern). HEAD stays on the pin; the working
    tree may differ from it ONLY by the one declared line. Returns the diff."""
    head = sh("git", "-C", FD, "rev-parse", "HEAD")
    if not head.startswith(PIN):
        die(2, f"git pin {head[:12]} != {PIN}; N6 varies the WORKING TREE, not HEAD")
    paths = sorted(q for q in sh("git", "-C", FD, "diff", "--name-only",
                                 "HEAD").splitlines() if q.strip())
    if PATCH_FILE not in paths:
        die(2, f"the declared patch is NOT applied: {PATCH_FILE} is unmodified "
               f"(dirty: {paths or 'nothing'})")
    others = [q for q in paths if q != PATCH_FILE]
    nonoutput = [q for q in others if not q.startswith("analysis/")]
    if nonoutput:
        die(2, f"undeclared SOURCE modifications present, refusing: {nonoutput}")
    rel = os.path.relpath(RELAY, FD)
    if rel in others:
        die(2, f"{rel} is modified -- it is the VENUE INPUT this launcher "
               "derives the leg environment from; it must be pristine")
    if others:
        print(f"NOTE: {len(others)} campaign output file(s) dirty, allowed and "
              f"recorded: {others}", file=sys.stderr)
    numstat = sh("git", "-C", FD, "diff", "--numstat", "--", PATCH_FILE)
    if numstat.split("\t")[:2] != ["1", "1"]:
        die(2, f"expected exactly 1 insertion and 1 deletion, got '{numstat}'")
    diff = sh("git", "-C", FD, "diff", "--", PATCH_FILE)
    minus = [l[1:] for l in diff.splitlines()
             if l.startswith("-") and not l.startswith("---")]
    plus = [l[1:] for l in diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")]
    if minus != [DECLARED_OLD] or plus != [DECLARED_NEW]:
        die(2, "the working-tree edit is NOT the declared patch.\n"
               f"  expected -: {DECLARED_OLD!r}\n  actual   -: {minus!r}\n"
               f"  expected +: {DECLARED_NEW!r}\n  actual   +: {plus!r}")
    return diff


def revert_patch():
    """Rollback path of record (prereg §5.7). Witness = the tree is clean of
    src/ edits afterwards; rc 7 if it is not."""
    subprocess.run(["git", "-C", FD, "checkout", "--", PATCH_FILE])
    paths = sorted(q for q in sh("git", "-C", FD, "diff", "--name-only",
                                 "HEAD").splitlines() if q.strip())
    src_dirty = [q for q in paths if not q.startswith("analysis/")]
    if src_dirty:
        die(7, f"REVERT FAILED: source still dirty: {src_dirty}")
    return ("REVERT WITNESS: `git diff --name-only HEAD` has no src/ entry; "
            f"dirty={paths or 'nothing'}")


def g4_static():
    """G4(a), EXECUTED (closes A's D1 for this dispatch). G4(b)/(c) are NOT
    run here and the readout says so; they are INHERITED from the pinned tree."""
    r = subprocess.run([sys.executable, G4_STATIC], capture_output=True, text=True)
    if r.returncode != 0:
        die(2, f"G4(a) static leakage check rc={r.returncode}: "
               f"{r.stdout[-300:]}{r.stderr[-300:]}")
    verdict = next((l for l in r.stdout.splitlines() if l.startswith("VERDICT")),
                   "VERDICT line missing")
    if "no truth-reachability found" not in verdict:
        die(2, f"G4(a) verdict not clean: {verdict}")
    return verdict


def assert_policy_and_box():
    for f, want in (("model.zip", CKPT_SHA16),
                    ("vecnormalize.pkl", VECNORM_SHA16)):
        p = os.path.join(CKPT, f)
        if not os.path.isfile(p):
            die(2, f"gate_W16 snapshot missing {f}")
        got = sh("sha256sum", p)[:16]
        if got != want:
            die(2, f"gate_W16 {f} sha {got} != {want}")
    live = []
    for p in glob.glob("/proc/[0-9]*/exe"):
        try:
            exe = os.readlink(p)
        except OSError:
            continue
        if "arduplane" in exe.lower():
            live.append(p)
    if live:
        die(2, f"{len(live)} SITL process(es) already running — refusing to "
               "add legs to a box that is flying something else")


def leg_env():
    """Today's venue, DERIVED by executing the relay's own env block (never
    transcribed -- the N3 D6 lesson) and asserted against the §2 allowlist.
    A applies NO override on top: what comes back is what flies."""
    lines = open(RELAY).read().splitlines()
    try:
        lo = next(i for i, l in enumerate(lines)
                  if l.startswith("# --- env line"))
        hi = next(i for i, l in enumerate(lines)
                  if l.startswith('source "$FD_ROOT/scripts/eval_venue_env.sh"'))
    except StopIteration:
        die(2, f"{RELAY}: env-block markers not found; cannot derive venue")
    block = "\n".join(lines[lo:hi + 1])
    pre = (f'set -u; FD_ROOT="{FD}"; ART="{CAMP}"; LOGS="$ART/logs_w1418"; '
           f'cd "$FD_ROOT" || exit 3; export PYTHONPATH="${{PYTHONPATH:-}}"\n')
    r = subprocess.run(["bash", "-c", pre + block + "\nenv -0"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        die(2, f"relay env block rc={r.returncode}: {r.stderr.strip()[:300]}")
    env = dict(kv.split("=", 1) for kv in r.stdout.split("\0") if "=" in kv)
    bad = {k: (env.get(k), v) for k, v in VENUE.items() if env.get(k) != v}
    bad.update({k: (env[k], None) for k in UNSET if k in env})
    if bad:
        die(2, f"derived relay env disagrees with §2 allowlist: {bad}")
    got = env.get("FD_GUARD_CEILING_BELOW_M")
    if got != str(GUARD_EXPECT):
        die(2, f"derived venue FD_GUARD_CEILING_BELOW_M={got!r}, expected "
               f"'{GUARD_EXPECT}' (w1424_relay.sh:71) — A holds it UNTOUCHED")
    print(f"VENUE: guard {got} UNTOUCHED, speed brake LIVE, TELEM live; the ONE "
          "delta is the alt-NEG floor patch asserted by assert_patch().",
          file=sys.stderr)
    return env


# ------------------------------ custody --------------------------------------
def cursors():
    """Spent-per-block from the append-only ledger. The SAME arithmetic
    next_seeds/pinned_seeds use, so a drift here is a drift there."""
    out = {}
    if not os.path.isfile(LEDGER):
        return out
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out[r.get("block")] = out.get(r.get("block"), 0) + r.get("n", 0)
    return out


def assert_custody(before, after, expect_full_gate_delta):
    """The operator's bound, in code. Every block is checked, not just the one
    being spent -- a spend that lands in the WRONG block is the failure this
    catches, and a delta-only check on full_gate would never see it."""
    problems = []
    for blk in sorted(set(before) | set(after)):
        d = after.get(blk, 0) - before.get(blk, 0)
        want = expect_full_gate_delta if blk == "full_gate" else 0
        if d != want:
            problems.append(f"{blk}: moved {d:+d}, bound says {want:+d}")
    return problems


def prevalidate_seeds(cells):
    """Everything pinned_seeds/next_seeds would check, MINUS the write.

    WHY THIS EXISTS: --dry-run skips both write-APIs, so without this a signed
    rehearsal would prove the signature, tree, policy, venue and spawn paths
    and leave the SEED path entirely unexercised -- which is exactly where N3
    (D5: LEDGER pointed at logs/ instead of logs_w1418/, so pinned_seeds would
    have written to an empty file) and N4a (D2: block 'confirmation' instead of
    'full_gate', which would have failed custody CLOSED) both went wrong. Both
    were found by EXECUTION, neither was reachable by reading. Returns the
    exact seeds W1 WOULD draw so the rehearsal can print the real panel."""
    if not os.path.isfile(LEDGER):
        die(2, f"seed ledger not found at {LEDGER} — pinned_seeds would append "
               "to a file nobody reads (the N3 D5 shape). Refusing.")
    try:
        blocks = json.load(open(SEED_BLOCKS))["blocks"]
    except (OSError, ValueError, KeyError) as exc:
        die(2, f"cannot read SEED_BLOCKS {SEED_BLOCKS}: {exc!r}")
    if "full_gate" not in blocks:
        die(2, f"block 'full_gate' absent from {SEED_BLOCKS}; blocks present: "
               f"{sorted(blocks)} (the N4a D2 shape)")
    order = blocks["full_gate"]["frozen_order"]
    used = cursors().get("full_gate", 0)
    out = {"block": "full_gate", "block_size": len(order), "cursor": used,
           "remaining": len(order) - used}
    if "W16" in cells:
        seeds = [s for s, _ in W16_PANEL]
        outside = [x for x in seeds if x not in order]
        if outside:
            die(2, f"W16 pin: seeds outside the frozen order: {outside} — "
                   "pinned_seeds would fail CLOSED at the real call")
        if len(set(seeds)) != len(seeds):
            die(2, "W16 pin: duplicate seeds")
        out["W16_ok"] = True
    if "W1" in cells:
        seeds = [s for s, _ in W1_PANEL]
        outside = [x for x in seeds if x not in order]
        if outside:
            die(2, f"W1 pin: seeds outside the frozen order: {outside}")
        if len(set(seeds)) != len(seeds):
            die(2, "W1 pin: duplicate seeds")
        out["W1_ok"] = True
        out["W1_cursor_after"] = used          # pinned: cursor does not move
    return out


def draw_seeds(cells, dry):
    """W16 AND W1 = pinned re-flies of A's panels (n=0 each). ZERO spend."""
    sys.path.insert(0, os.path.join(FD, "scripts"))
    from gate_relay_seeds import pinned_seeds
    before = cursors()
    panel, draw = {}, {"cursor_before": dict(before)}
    if "W16" in cells:
        seeds = [s for s, _ in W16_PANEL]
        if not dry:
            pinned_seeds(LEDGER, seeds, len(seeds), "refly:N6AUTH-806:W16",
                         SEED_BLOCKS, block="full_gate")
        panel["W16"] = list(W16_PANEL)
        draw["W16"] = {"seeds": seeds, "api": "pinned_seeds", "n_ledger": 0,
                       "traj_map": {str(s): t for s, t in W16_PANEL},
                       "note": "08-17 gate panel; order NOT index parity"}
        mid = cursors()
        if not dry:
            bad = assert_custody(before, mid, 0)
            if bad:
                die(6, f"CUSTODY BREACH on the W16 pin (must move NOTHING): {bad}")
    if "W1" in cells:
        seeds = [s for s, _ in W1_PANEL]
        if not dry:
            pinned_seeds(LEDGER, seeds, len(seeds), "refly:N6AUTH-806:W1",
                         SEED_BLOCKS, block="full_gate")
        panel["W1"] = list(W1_PANEL)
        draw["W1"] = {"seeds": seeds, "api": "pinned_seeds", "n_ledger": 0,
                      "traj_map": {str(s): t for s, t in W1_PANEL},
                      "note": "A's W1 draw re-flown verbatim; order IS the pairing"}
        if not dry:
            bad = assert_custody(before, cursors(), 0)
            if bad:
                die(6, f"CUSTODY BREACH on the W1 pin (must move NOTHING): {bad}")
    draw["cursor_after"] = cursors()
    return panel, draw


# ------------------------------ flight ---------------------------------------
def spawn_stratum(stratum, rows, out_dir, env, dry):
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
    env_ids = {s: i + 1 for i, (s, _) in enumerate(sorted(rows))}
    if len(set(env_ids.values())) != len(rows):
        die(2, f"{stratum}: env-id lanes are not distinct — parallel spawn "
               "would collide legs onto one SITL instance (N4a D1)")
    procs = []
    for seed, traj in rows:
        cmd = [sys.executable, SPAWNER, "--checkpoint",
               os.path.join(CKPT, "model.zip"),
               "--vecnormalize", os.path.join(CKPT, "vecnormalize.pkl"),
               "--stage-label", stratum, "--out-dir", out_dir,
               "--seeds", str(seed), "--env-id", str(env_ids[seed]),
               "--traj-map", f"{seed}:{traj}"]
        if dry:
            print("DRY " + " ".join(cmd))
            continue
        procs.append(subprocess.Popen(cmd, env=env))
    bad = [r for r in (p.wait() for p in procs) if r not in (0, 20)]
    if bad:
        die(3, f"{stratum}: lane rcs {bad} — NO readout written")


def adjudicate(cells, legs_root, out_path):
    spec = out_path.replace(".json", "_spec.json")
    json.dump({"strata": sorted(cells), "phase": "confirm", "prior_strikes": 0,
               "safety_criteria": "v2_real_events"}, open(spec, "w"))
    rc = subprocess.run([sys.executable, RUNNER, "--mode", "retention",
                         "--legs-dir", legs_root, "--spec", spec,
                         "--out", out_path]).returncode
    if rc != 0 or not os.path.isfile(out_path):
        die(4, f"adjudicator rc={rc} — NO readout written")
    return json.load(open(out_path))


# ------------------------------ readouts -------------------------------------
def dwell_triple(legs_root, cells, frozen):
    """G1. ALL THREE dwells from the ONE WRITER, not re-derived. `frozen` is
    the adjudicator instance curriculum_gate_runner._frozen() already loaded
    from its own T0_ADJ pin -- passed in rather than re-imported from a
    hardcoded path, so this can never read a DIFFERENT copy of the frozen
    instrument than the one that scored the panel. Only the intersection
    decides (GOALS.md §G1); the others are reported beside it and decide
    nothing."""
    out = {}
    for cell in cells:
        rows = []
        for d in sorted(glob.glob(os.path.join(legs_root, cell, f"{cell}_*"))):
            try:
                bp = frozen.boot_pass(d)
            except Exception as exc:                       # NON-ESTIMABLE, named
                rows.append({"leg": os.path.basename(d),
                             "non_estimable": repr(exc)[:120]})
                continue
            rows.append({k: bp.get(k) for k in
                         ("leg", "perceived_m3b_s", "truth_m3b_s",
                          "simultaneous_f9_s", "intersection_v3_s",
                          "intersection_live", "intersection")})
        out[cell] = rows
    return out


def peak_stats(rows):
    """G7/H1 leg-level. p95/p99 are NOT computed: on n=12 the empirical p95 is
    the max under another name and p99 does not exist. Operator-elected (a)."""
    peaks = [r["truth_peak_up_m"] for r in rows
             if r.get("truth_peak_up_m") is not None]
    if not peaks:
        return {"n": 0, "p95": "NON-ESTIMABLE", "p99": "NON-ESTIMABLE"}
    return {"n": len(peaks), "p50": round(statistics.median(peaks), 4),
            "max": round(max(peaks), 4), "min": round(min(peaks), 4),
            "p95": f"NON-ESTIMABLE at n={len(peaks)}",
            "p99": f"NON-ESTIMABLE at n={len(peaks)}"}


def bust_read(rows):
    """Authoritative limb ONLY (gate_safety_events.py:242, UNROUNDED). The
    rounded field is reported but is NOT a second instrument (N4b12.D1)."""
    auth = [r for r in rows
            if any(str(x).startswith("truth_ceiling_bust=")
                   for x in r.get("fail_reasons") or [])]
    field = [r for r in rows if (r.get("truth_peak_up_m") or 0) > BAR_M]
    boundary = [r for r in rows
                if abs((r.get("truth_peak_up_m") or 0) - BAR_M) <= EPS]
    return {"n_bust": len(auth),
            "n_bust_by_rounded_field": len(field),
            "reads_match": len(auth) == len(field),
            "boundary_legs": [r.get("leg") for r in boundary],
            "busts": sorted((r.get("leg"), r.get("truth_peak_up_m"))
                            for r in auth)}


def guard_thresholds(legs_root):
    seen = {}
    for p in glob.glob(os.path.join(legs_root, "**", "*"), recursive=True):
        if not os.path.isfile(p) or os.path.getsize(p) > 200_000_000:
            continue
        try:
            with open(p, errors="ignore") as fh:
                for m in re.finditer(r'"threshold_m":\s*(-?[0-9.]+)', fh.read()):
                    v = round(float(m.group(1)), 3)
                    seen[v] = seen.get(v, 0) + 1
        except OSError:
            continue
    return seen


def pvt_table(legs_root, cells):
    """G3, tick-level. NON-CANONICAL: Estimator B needs
    paired_tick_log_env0.jsonl, absent from every current leg, and emitting it
    would need the Method-C env path -- i.e. a venue change A forbids. This is
    the paired same-tick frame, writer env_formation_sitl.py, columns
    delta_env_raw_<axis>_m (perceived - truth), used writer-derived and never
    recomputed. BINDING: NEVER difference these against an Estimator-B or G6
    number; side-by-side under a banner only.

    NOTE the n here is TICKS, not legs -- thousands of them -- so p95/p99 ARE
    estimable at this level. That is a different quantity from H1's leg-level
    p95 and the two must never be conflated."""
    out = {}
    for cell in cells:
        vals = {a: [] for a in AXES}
        for p in glob.glob(os.path.join(legs_root, cell, f"{cell}_*", "env*",
                                        "paired_channels_ep*.csv")):
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    for a in AXES:
                        try:
                            vals[a].append(float(r[f"delta_env_raw_{a}_m"]))
                        except (TypeError, ValueError, KeyError):
                            continue
        per = {}
        for a in AXES:
            v = vals[a]
            if not v:
                per[a] = {"n_ticks": 0, "status": "NON-ESTIMABLE"}
                continue
            av = sorted(abs(x) for x in v)
            def q(f):
                return round(av[min(len(av) - 1, int(f * len(av)))], 4)
            per[a] = {"n_ticks": len(v),
                      "signed_median_m": round(statistics.median(v), 4),
                      "abs_p50_m": q(.50), "abs_p95_m": q(.95),
                      "abs_p99_m": q(.99)}
        out[cell] = per
    return out


def paired_delta(rows):
    """Per-leg truth peak vs A on the SAME seed. Median of paired deltas is the
    prereg §6 quantity. Legs A never scored are NON-ESTIMABLE."""
    out, deltas = [], []
    for r in rows:
        a, p = A_PEAKS.get(r.get("leg")), r.get("truth_peak_up_m")
        if a is None or p is None:
            out.append({"leg": r.get("leg"), "n5": p, "a": a, "delta": "NON-ESTIMABLE"})
            continue
        d = round(p - a, 4)
        deltas.append(d)
        out.append({"leg": r.get("leg"), "n5": p, "a": a, "delta": d})
    med = round(statistics.median(deltas), 4) if deltas else None
    return out, med, len(deltas)


def authority_window(leg_dir, win_s=5.0):
    """Applied vs commanded altitude residual in the win_s before the first
    truth tick >= BAR_M (or the peak). Writer columns res_policy_2 /
    res_applied_2 from flight_state, consumed as written. This is the
    falsifier that the patch reached the plant (prereg §5 F4)."""
    import gzip
    p = glob.glob(os.path.join(leg_dir, "env*", "flight_state_ep0001.csv.gz"))
    if not p:
        return None
    rows = []
    with gzip.open(p[0], "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((float(r["sim_time_s"]), -float(r["raw_below_m"]),
                             float(r["res_policy_2"]), float(r["res_applied_2"])))
            except (TypeError, ValueError, KeyError):
                continue
    if not rows:
        return None
    rows.sort()
    anchor = (next((x for x in rows if x[1] >= BAR_M), None)
              or max(rows, key=lambda x: x[1]))
    win = [x for x in rows if anchor[0] - win_s <= x[0] <= anchor[0] and x[2] < 0]
    if not win:
        return {"ticks_down": 0, "ratio": "NON-ESTIMABLE"}
    pol = statistics.mean(x[2] for x in win)
    app = statistics.mean(x[3] for x in win)
    return {"ticks_down": len(win), "policy_mean": round(pol, 3),
            "applied_mean": round(app, 3),
            "ratio": round(app / pol, 3) if pol else "NON-ESTIMABLE"}


def n5_reading(n_bust, med_delta, dwell, reach_med, a_reach_med):
    """Prereg §6, in code. Returns (verdict, why). Pure function."""
    if reach_med is None or a_reach_med is None:
        return "INCONCLUSIVE", "applied/commanded ratio not estimable on one side"
    if (abs(reach_med - a_reach_med) < RULE["inconclusive_reach"]
            and n_bust >= RULE["inert_min_busts"]):
        return "INCONCLUSIVE", (f"delta did not reach the plant: applied/commanded "
                               f"{reach_med} vs A {a_reach_med} (F4)")
    if med_delta is None:
        return "INCONCLUSIVE", "no paired legs"
    if (n_bust <= RULE["lever_max_busts"] and -med_delta >= RULE["lever_min_drop_m"]
            and dwell >= RULE["lever_min_dwell"]):
        return "AUTHORITY-IS-THE-LEVER", "busts, paired drop and dwell inside the signed bounds"
    if n_bust >= RULE["inert_min_busts"] or abs(med_delta) < RULE["inert_max_abs_delta_m"]:
        return "INERT", "busts or paired delta outside the lever bound"
    return "PARTIAL", "between the two signed bounds"


def write_readout(root, cells, adj, rows, dwells, pvt, thr, draw, custody_ok):
    L = ["# N6 AUTHORITY PROBE READOUT — A's 24 seeds, alt-NEG floor HALF-open (0.50)",
         "", "**N6AUTH-806.** Frozen policy `gate_W16` "
         f"(`{CKPT_SHA16}` / `{VECNORM_SHA16}`). Venue UNTOUCHED: guard "
         f"{GUARD_EXPECT}, CRRL brake LIVE, TELEM live (measured inert). "
         "ONE declared patch (verbatim below), REVERTED after the run.", "",
         f"Guard thresholds echoed by fire events: {thr or 'NONE'}", "",
         "**CUSTODY (prereg §4):** no strikes; EVERY block +0 (both cells "
         "pinned n=0). Verified: "
         f"**{'PASS' if custody_ok else 'BREACH'}**.",
         f"Cursor {draw.get('cursor_before', {}).get('full_gate')} -> "
         f"{draw.get('cursor_after', {}).get('full_gate')}.", "",
         "## G7 / H1 — vertical, leg-level", "",
         "| cell | k/n | n_bust (authoritative) | rounded-field read | "
         "boundary legs | p50 | max | p95 | p99 |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        st = adj["strata"].get(c, {})
        b, ps = bust_read(rows.get(c, [])), peak_stats(rows.get(c, []))
        L.append(f"| {c} | {st.get('k')}/{st.get('n')} | **{b['n_bust']}** | "
                 f"{b['n_bust_by_rounded_field']} | "
                 f"{b['boundary_legs'] or 'none'} | {ps.get('p50')} | "
                 f"{ps.get('max')} | {ps['p95']} | {ps['p99']} |")
    L += ["", "**p95/p99 are NOT reported and NOT approximated.** On n=12 legs "
          "the empirical p95 lies between the 11th and 12th order statistic — "
          "it is the maximum under another name — and p99 does not exist. "
          "(a), prereg §4. The H1 p95 is bought once, later, "
          "on the benchmark corpus.", "",
          "The rounded-field column is NOT a second instrument (N4b12.D1): the "
          "scorer busts on the UNROUNDED peak and reports it rounded. Legs "
          "within 5e-4 m of the bar are BOUNDARY, counted with the "
          "authoritative limb and named above.", "",
          "T1: every figure here is an EXCLUSION-ZONE reading, never contact. "
          "Clearance c = 3.42 − e.", "",
          "## G1 — M3b dwell, all three, intersection decisive", "",
          "| cell | leg | perception-only | truth-only | f9 simultaneous | "
          "**intersection** | live | >=3.0 s |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for r in dwells.get(c, []):
            if r.get("non_estimable"):
                L.append(f"| {c} | {r['leg']} | — | — | — | **NON-ESTIMABLE** "
                         f"| — | {r['non_estimable']} |")
                continue
            L.append(f"| {c} | {r['leg']} | {r['perceived_m3b_s']} | "
                     f"{r['truth_m3b_s']} | {r['simultaneous_f9_s']} | "
                     f"**{r['intersection_v3_s']}** | {r['intersection_live']} "
                     f"| {'YES' if r['intersection'] else 'no'} |")
    L += ["", "Only the intersection decides (GOALS.md §G1). Perception-only "
          "is estimator belief, not truth-confirmed. Truth is OFFLINE, "
          "post-hoc, and entered no in-loop path.", "",
          "## G3 — perception-vs-truth, per axis, tick-level", "",
          "> ⚠ **NON-CANONICAL ESTIMATOR.** Estimator B (operator-locked "
          "2026-06-12) requires `paired_tick_log_env0.jsonl`, which no current "
          "leg carries; emitting it needs the Method-C env path, i.e. a venue "
          "change A exists to forbid. This is the **paired same-tick frame** — "
          "writer-derived `delta_env_raw_<axis>_m` (perceived − truth), "
          "`env_formation_sitl.py`. **BINDING: never difference these against "
          "an Estimator-B or G6 figure — side-by-side under a banner only.**",
          "", "> `n` below is **TICKS**, not legs. Tick-level p95/p99 ARE "
          "estimable; they are a different quantity from H1's leg-level p95 "
          "above and must never be conflated.", "",
          "| cell | axis | n ticks | signed median (bias) | abs p50 | abs p95 "
          "| abs p99 |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        for a in AXES:
            d = pvt.get(c, {}).get(a, {})
            if d.get("status") == "NON-ESTIMABLE":
                L.append(f"| {c} | {a} | 0 | NON-ESTIMABLE | — | — | — |")
                continue
            L.append(f"| {c} | {a} | {d.get('n_ticks')} | "
                     f"{d.get('signed_median_m')} | {d.get('abs_p50_m')} | "
                     f"{d.get('abs_p95_m')} | {d.get('abs_p99_m')} |")
    L += ["", "All three axes, always (GOALS.md §G3).", "",
          "## Seed draw, verbatim", "", "```json",
          json.dumps(draw, indent=1), "```", "",
          "## Adjudication, verbatim (computed read-only; NO strike recorded)", "", "```json",
          json.dumps(adj, indent=1), "```", ""]
    open(os.path.join(root, "N6_AUTH_READOUT.md"), "w").write("\n".join(L))


def g1_pass_count(dwells):
    """Legs with truth-confirmed intersection dwell >= 3.0 s, from the ONE
    WRITER rows dwell_triple() returned. N5's readout fed the adjudication's
    per-stratum FAILURE count k into this slot (N5 sidecar D1); this reads
    the column the prereg rule actually names."""
    return sum(1 for rs in dwells.values() for r in rs
               if isinstance(r.get("intersection_v3_s"), (int, float))
               and r["intersection_v3_s"] >= 3.0)


def write_n6_sections(out, cells, rows, legs, diff, g4, adj, dwells):
    """Appends the N6-specific sections: patch, G4, paired delta vs A,
    applied-vs-commanded (N6 and A on the same seeds), and the §6 reading."""
    L = ["", "## The declared patch, verbatim", "", "```diff", diff, "```", "",
         "## G4", "", f"G4(a) static, EXECUTED at the pin: `{g4}`", "",
         "G4(b) runtime probe / G4(c) boundary check: **NOT RUN -- INHERITED** "
         "from the pinned tree; not quotable as 'N6 passed G4'.", "",
         "## Paired delta, truth peak vs A (same seed, policy, venue; ONE delta) — N5's 1.00 column beside it", "",
         "| cell | leg | N6 peak | A peak | delta (N6 - A) | N5 peak (1.00) |", "|---|---|---|---|---|---|"]
    all_d, nb, reach, a_reach = [], 0, [], []
    for c in cells:
        per, med, n = paired_delta(rows.get(c, []))
        for r in per:
            L.append(f"| {c} | {r['leg']} | {r['n5']} | {r['a']} | {r['delta']} | {N5_PEAKS.get(r['leg'])} |")
            if isinstance(r["delta"], float):
                all_d.append(r["delta"])
        nb += bust_read(rows.get(c, []))["n_bust"]
        L.append(f"| {c} | **median** | | | **{med}** (n={n}) | |")
    med_all = round(statistics.median(all_d), 4) if all_d else None
    L += ["", f"Pooled paired median delta: **{med_all}** (n={len(all_d)}).", "",
          "## Applied vs commanded altitude residual, 5 s before trip/peak (F4)", "",
          "| cell | leg | N6 policy mean | N6 applied mean | N6 ratio | A ratio |",
          "|---|---|---|---|---|---|"]
    for c in cells:
        for d in sorted(glob.glob(os.path.join(legs, c, f"{c}_*"))):
            leg = os.path.basename(d)
            w = authority_window(d) or {}
            a = authority_window(os.path.join(A_LEGS, c, leg)) or {}
            L.append(f"| {c} | {leg} | {w.get('policy_mean')} | {w.get('applied_mean')} "
                     f"| {w.get('ratio')} | {a.get('ratio')} |")
            if isinstance(w.get("ratio"), float):
                reach.append(w["ratio"])
            if isinstance(a.get("ratio"), float):
                a_reach.append(a["ratio"])
    rr = round(statistics.median(reach), 3) if reach else None
    ar = round(statistics.median(a_reach), 3) if a_reach else None
    dwell_total = g1_pass_count(dwells)   # D1 fix: G1 column, NEVER adjudication k
    verdict, why = n5_reading(nb, med_all, dwell_total, rr, ar)
    L += ["", f"Median applied/commanded: N6 **{rr}** vs A **{ar}**.", "",
          "## Reading (prereg §6, applied in code)", "",
          f"busts {nb}/24 · paired median delta {med_all} m · dwell {dwell_total}/24 "
          f"· reach N6 {rr} vs A {ar}", "", f"**{verdict}** -- {why}", "",
          "Dwell below 16/24 is a COST regardless of the verdict. "
          "T1: exclusion-zone readings, never contact."]
    open(out, "a").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="N6 half-open authority probe executor "
                                             "(see N6_AUTH_PREREG.md)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--cells", default="W16,W1")
    ap.add_argument("--keep-patch", action="store_true",
                    help="do NOT revert config.py at the end")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in ("W16", "W1"):
            die(2, f"unknown cell {c!r}; the signed design is W16 and W1")
    assert_signature()
    diff = assert_patch()
    dirty = []
    assert_policy_and_box()
    g4 = g4_static()
    print(f"G4(a): {g4}", file=sys.stderr)
    env = leg_env()
    sys.path.insert(0, FD)
    from src.drl import env_contract
    env_contract.assert_contract(env_contract.EVAL, env=env)
    verdicts_before = (len(os.listdir(VERDICTS_DIR))
                       if os.path.isdir(VERDICTS_DIR) else 0)
    root = args.out_root or os.path.join(
        CAMP, "logs_w1418",
        "n6auth806_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    legs = os.path.join(root, "legs")
    cur_before = cursors()
    pre = prevalidate_seeds(cells)
    print(f"SEED PRE-VALIDATION: {json.dumps(pre)}", file=sys.stderr)
    panel, draw = draw_seeds(cells, args.dry_run)
    if not args.dry_run:
        os.makedirs(legs, exist_ok=True)
        json.dump(draw, open(os.path.join(root, "SEED_DRAW.json"), "w"), indent=1)
        json.dump({"label": "N6-AUTH", "cells": cells,
                   "spawned_epoch": int(time.time()), "budget_s": 3600,
                   "note": "advisory only; nothing aborts"},
                  open(os.path.join(root, "PANEL_ETA.json"), "w"), indent=1)
    for cell in cells:
        spawn_stratum(cell, panel[cell], os.path.join(legs, cell), env,
                      args.dry_run)
    if args.dry_run:
        print("DRY RUN complete — no legs, no ledger rows, no readout.")
        print(f"cells={cells}  dirty_outputs={len(dirty)}")
        print("CUSTODY: dry run draws NOTHING; cursors untouched "
              f"({cur_before.get('full_gate')} full_gate). The real call spends 0 too.")
        print("PATCH (asserted; left in place for the real call):\n" + diff)
        print("SEED PATH pre-validated read-only (ledger present, block "
              "resolvable, W16 seeds inside the frozen order, W1 headroom "
              f"sufficient): {json.dumps(pre)}")
        return 0
    adj = adjudicate(cells, legs, os.path.join(root,
                                               "n6_auth_adjudication.json"))
    sys.path.insert(0, os.path.join(FD, "scripts"))
    import curriculum_gate_runner as cgr
    frozen, _ = cgr._frozen()
    rows = {c: [cgr.leg_verdict(os.path.join(legs, c, d), frozen,
                                "v2_real_events")
                for d in sorted(os.listdir(os.path.join(legs, c)))]
            for c in cells}
    breaches = assert_custody(cur_before, cursors(),
                              W1_SPEND if "W1" in cells else 0)
    verdicts_after = (len(os.listdir(VERDICTS_DIR))
                      if os.path.isdir(VERDICTS_DIR) else 0)
    if verdicts_after != verdicts_before:
        breaches.append(f"gate_verdicts grew {verdicts_before}->"
                        f"{verdicts_after}: A must record NO strike")
    dwells = dwell_triple(legs, cells, frozen)
    write_readout(root, cells, adj, rows,
                  dwells, pvt_table(legs, cells),
                  guard_thresholds(legs), draw, not breaches)
    out = os.path.join(root, "N6_AUTH_READOUT.md")
    write_n6_sections(out, cells, rows, legs, diff, g4, adj, dwells)
    if not args.keep_patch:
        open(out, "a").write("\n\n## Revert\n\n" + revert_patch() + "\n")
    if breaches:
        print(f"CUSTODY BREACH: {breaches}\n{out}", file=sys.stderr)
        return 6
    if adj.get("invalid_protocol"):
        print(f"INVALID_PROTOCOL — no reading.\n{out}", file=sys.stderr)
        return 5
    print(f"N6-AUTH PANEL COMPLETE  cells={cells}\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
