"""Fail-closed preflight assertion of the curriculum-v3 environment contract.

WHY THIS MODULE EXISTS. The D7 environment line is carried by shell drivers
(`ws1_launch.sh`, `ws1_relay.sh`) and by nothing else. Two fail-OPEN paths
follow, both observed or narrowly avoided in C006:

  1. `scripts/curriculum_gate_relay.sh` exports nothing but PYTHONPATH, and
     `gate_relay_core.py` spawns eval lanes with `subprocess.Popen(cmd)` and
     NO `env=` — the lanes inherit the RELAY's environment wholesale. A relay
     restarted from a bare shell therefore flies sealed gate legs on a
     different backend, warmup budget, UWB source and sigma than training,
     silently. (This already cost one restart: W-1, 2026-07-29.)
  2. `ws1_relay.sh` correctly sets `FD_V3_REHEARSAL_DISABLE=1` because sealed
     gates fly the current cell only. Copying that env block into a TRAINING
     driver would silently disable the A5 anti-forgetting rehearsal — the
     same variable, the opposite correct value, no error either way.

The asymmetry is the whole point: the contract is role-dependent, so this
module takes an explicit `role` and asserts opposite polarity per role.

DELIBERATELY NOT A VALUE SETTER. This module never exports anything. It reads
the environment, compares it to the declared contract, and raises. Drivers
remain the single source of truth for values; this is the detector.

PUBLIC API: EnvContractViolation, check, assert_contract, render, main
SIDE-EFFECTS: reads os.environ; writes stdout only from `render`/`main`.
  Never mutates the environment, never spawns, never signals.
ESCAPE HATCHES (both loud): `FD_ENV_CONTRACT_ENFORCE=0` downgrades raise to
  a printed warning; `FD_ENV_CONTRACT_DRYRUN=1` makes `main` resolve and
  report without any caller launching anything — so fail-closed behaviour is
  never first exercised on a live launch path.
CAPS: file soft 190 / hard 230 LoC; function cap 45.
  Soft overage disclosed in the comment block below (kept OUT of this
  docstring on purpose -- see there).
COVERAGE: tests/unit/test_env_contract.py
"""
# ---------------------------------------------------------------------------
# CAP DISCLOSURE (TERMDEMOTE-767). SOFT OVERAGE DISCLOSED, NOT RAISED.
#   This file exceeds its soft cap of 190 code-lines; hard 230 is clear and no
#   cap was moved. The residual is the R5b/R7 declarations and their rationale
#   -- declaration, not logic; `_check_train_forbidden` is 11 lines against a
#   function cap of 45.
#   WHY THIS SITS IN A COMMENT AND NOT THE DOCSTRING: the house counter
#   (tools/check_loc_caps.py) EXCLUDES whole-line comments but COUNTS docstring
#   lines. Written as a docstring, this disclosure grew the file it was
#   describing and its own stated figure went stale twice. Read the current
#   number from check_loc_caps, never from a number typed here.
# ---------------------------------------------------------------------------
import os
import sys

TRAIN = "train"
EVAL = "eval"
ROLES = (TRAIN, EVAL)

# R1 — the D7 line, at the exact values both shipped drivers export.
D7_REQUIRED = {
    "FD_TRAIN_BACKEND": "json",          # G5 JSON-lockstep, non-negotiable
    "FD_JSON_PAIR_FREEZE": "1",
    "FD_FDM_RECV_TIMEOUT_S": "0",
    "FD_UWB_TRUTH_SOURCED": "1",         # UWB=ON ALWAYS (standing directive)
    "HARNESS_UWB_RATE_HZ": "10",
    "FD_CURRICULUM_V3": "1",
    "FD_LEADER_WIND": "1",
    # ARMED-ELECT-737: the
    # T-391 warmup acceptance gate is armed for evals AND training. A launch
    # without it flies the pre-election warmup contract silently — refused.
    "FD_WARMUP_ACCEPT_GATE": "1",
    # GATETOL-746: the gate must enforce the M2
    # spec it sits at -- 3.0 s of continuous in-station dwell, not the 0.15 s
    # (3 samples) it accepted before, and M2's own +/-2.0 m vertical box.
    #
    # ★ WHY THE TOLERANCE IS A GATE-LOCAL KNOB AND NOT A formation_config EDIT:
    # PRECONTACT_BELOW_OUTER_TOL_M is consumed by the FROZEN baseline FSM via
    # EnvelopeChecker.is_in_precontact_outer_envelope
    # (tests/baseline_formation_controller.py:6938). Lowering it globally would
    # change R7.1 behaviour through the back door. The override therefore lives
    # on the gate, and IT IS LISTED HERE so it cannot be forgotten: a launch
    # missing it flies the wide 4.0 m box while the record claims 2.0.
    # Exact-string match -- "2.0", not "2".
    "FD_WARMUP_SETTLE_DWELL_N": "60",
    "FD_WARMUP_GATE_BELOW_TOL_M": "2.0",
    # NOPASSTERM-770. Role-AGNOSTIC on purpose: a terminal changes which events
    # END an episode, and the policy must train under the rule it is examined
    # under. Exact string, order fixed; config refuses an unknown name at
    # import. Legs flown without it are NOT poolable with legs flown with it.
    "FD_SAFETY_TERMINALS": "SAFETY_CONTACT,SAFETY_NOPASS",
}
# R2 — taxonomy. The override may legitimately carry a probe
# value, but it must be one we can name; an unrecognised triple is reported
# so a stray 0.75 is loud rather than silent.
A7_SIGMA = {"0.10,0.10,0.10": "light", "0.20,0.20,0.20": "moderate",
            "0.30,0.30,0.30": "severe"}
# R4 — truth-side observability knobs that must never be armed in training
# (G4: no truth-side field may reach actor/critic/reward/replay/observation).
# FD_COLLISION_TRUTH_CHANNEL added 2026-08-16 (T434AMEND hardening (a)): the
# signed BENCHMARK-T434EVAL-20260816 makes the gate/eval RELAY export it, so a
# training process launched from a relay shell can now inherit it in normal
# operation -- previously it could only arrive by hand. With it armed, truth
# decides an episode boundary whose terminal_type the reward prices
# (R_TERMINAL_SAFETY), i.e. truth reaching the learner: Tier-0.
# DELIBERATE STRICTNESS: this check trips on ANY non-empty value (the knob
# convention here), whereas train.py's rc-43 interlock keys on == "1". The
# contract is therefore the stricter of the two and refuses even an explicit
# "0" in a TRAIN role; unset is the clean state, and every launcher unsets it.
# Waivable the same way as any R4 knob, via FD_COLLISION_TRUTH_CHANNEL_ACK.
# EVAL is unaffected by construction: _check_truth returns [] unless TRAIN.
# FD_RAW_ALIGN_CENTER added 2026-08-18 (FIXAELECT-717): SIBLING of
# FD_RAW_EPOCH_COMP on the same align seam, governed identically. Defence in
# depth, NOT a claim it is independently Tier-0 (with the collision truth
# channel unset it feeds logging only); it matters because a training process
# launched from a relay shell INHERITS the relay's exports.
TRUTH_KNOBS = ("FD_RAW_EPOCH_COMP", "FD_COLLISION_TRUTH_CHANNEL",
               "FD_RAW_ALIGN_CENTER")
REHEARSAL = "FD_V3_REHEARSAL_DISABLE"
ACK_SUFFIX = "_ACK"
# R5 — FIXAELECT-717 (SIGNED, epoch 2026-08-18T21:04:46Z). An eval leg without
# the flag is silently measured on the PRE-epoch substrate. EVAL-only: TRAIN is
# barred from this knob by R4, so it CANNOT live in the role-agnostic D7 set.
EVAL_REQUIRED = {"FD_RAW_ALIGN_CENTER": "1",
                 # R5b -- TERMDEMOTE-767.
                 # Exact set. A leg carrying a DIFFERENT set is flying
                 # different rules and must refuse, not silently produce
                 # non-poolable data.
                 #
                 # !!! OWED, BLOCKING: RELAYDEMOTE-768 !!!
                 # If you are reading this because a GATE or PROMOTION-EXAM leg
                 # just REFUSED with "R5 FD_DEMOTE_TERMINALS=None": that is
                 # EXPECTED and it is NOT a bug. [MEASURED 2026-08-20] all 24
                 # relay scripts under analysis/c006_campaign_20260728T110000Z/
                 # export this knob ZERO times. The operator elected Option 2
                 # on 2026-08-20: land R5b now, update the relays in their own
                 # dispatch, and accept that the exam path is BLOCKED until
                 # then. DO NOT waive this with FD_DEMOTE_TERMINALS_ACK -- a
                 # waived exam flies the OLD termination rules and its legs are
                 # not poolable with anything. Do RELAYDEMOTE-768 instead.
                 # Tracker row: docs/RELAYDEMOTE-768.
                 "FD_DEMOTE_TERMINALS": "SAFETY_AIRSPEED,SAFETY_COLLISION"}
# R6 — GUARDRESTORE-756. The config DEFAULT for
# GUARD_ALT_CEILING_BELOW_M is the restored -2.0 (the eval system definition:
# guard actuation is an intervention, so evals fly the exposed system). The
# -1.75 TRAINING device survives ONLY through this env override; requiring it
# for TRAIN means a training launch without it REFUSES instead of silently
# training at -2.0. TRAIN-only: benchmark/eval drivers must NOT set the var
# (their venue is -2.0 and the resolved value is witnessed per leg — the
# evidence layer, not this contract, guards the eval venue). Promotion exams
# run from the training campaign env and inherit -1.75. Exact-string match — "-1.75".
#
# GUARDPARITY-794: the -1.75
# TRAIN-only device is RETIRED. Three attempts under it (8/12, 8/12, 7/12)
# never beat the 9/12 zero-residual substrate; it truncates every excursion
# 0.25 m before the 2.00 m bar, so the critic sees "breakaway" where it should
# see "you crossed 2.0 and here is what it cost"; and train/eval parity is
# house doctrine (train under the rule you are examined under). TRAIN now
# REQUIRES the eval definition, -2.0, by exact string — so a relay shell still
# exporting the old -1.75 REFUSES rather than silently training on the retired
# venue. The exam inherits -2.0 with it; never pool guard-fire counts across
# the change.
TRAIN_REQUIRED = {"FD_GUARD_CEILING_BELOW_M": "-2.0"}
# R7 -- TERMDEMOTE-767. FD_DEMOTE_TERMINALS changes which events END an
# episode. EVAL/exam legs must carry it (R5b above); TRAIN must NOT. A training
# process launched from a relay shell INHERITS the relay's exports -- the exact
# fail-open this module was written for.
# Demoted terminals in training would change the replay buffer's `done` labels
# silently, which the operator has explicitly deferred pending measured data.
TRAIN_FORBIDDEN = ("FD_DEMOTE_TERMINALS",)
# R8 -- TERMDEMOTE-767. Knobs an EVAL role must NOT carry because they would
# SILENTLY DEFEAT the demotion -- the episode still dies on a guard fire, just
# under a different terminal name.
#
# FD_FORMATION_LOST_TERM is the live one, and the chain is exact. The breakaway
# handler's own contract in tests/baseline_formation_controller.py is
# "4. Transitions to INTERCEPT for safe reacquire", and config's
# FORMATION_LOST_STATES == ("INTERCEPT",). So on an eval leg with that flag on:
# a demoted SAFETY_COLLISION fires the vertical guard -> the guard commands
# EMERGENCY_BREAKAWAY -> the FSM enters INTERCEPT -> the episode terminates as
# SAFETY_FORMATION_LOST. That is precisely the outcome demotion exists to
# prevent, wearing another name, and no collision-limb check would notice.
#
# [MEASURED 2026-08-20] SAFETY_FORMATION_LOST has fired in 0 of 5917 banked
# episodes, and the flag is set today only by a TRAINING launcher
# (analysis/c006_campaign_20260728T110000Z/w0a2_launch.sh). This rule keeps it
# that way rather than trusting that it stays that way.
#
# Keyed on the VALUE THAT ENABLES, mirroring config's own parse
# (os.environ.get(...) == "1"), so an explicit "0" is correctly read as OFF and
# does not refuse a leg that is already safe.
EVAL_FORBIDDEN = {"FD_FORMATION_LOST_TERM": "1"}
# Provenance PER KEY. _check_eval_required iterates EVAL_REQUIRED, so a single
# hardcoded citation in the message template was correct only while that table
# had one entry. A refusal banner naming the wrong dispatch sends the reader to
# the wrong which is worse than naming none.
EVAL_REQUIRED_CITE = {"FD_RAW_ALIGN_CENTER": "FIXAELECT-717",
                      "FD_DEMOTE_TERMINALS": "TERMDEMOTE-767"}


class EnvContractViolation(RuntimeError):
    """Raised when the resolved environment breaks the declared contract."""


def _acked(var, env):
    """A violation may be waived only by an explicit, non-empty ack var."""
    return bool(env.get(var + ACK_SUFFIX, "").strip())


def _check_d7(env):
    # Waivable by its own ack, the escape R2/R4 already grant: the T-430/TZOH
    # no-wind A/B drivers `unset FD_LEADER_WIND` deliberately. With no ack path
    # that experiment is unrepresentable, which is how a contract ends up
    # bypassed wholesale instead of waived on the record.
    out = []
    for key, want in sorted(D7_REQUIRED.items()):
        got = env.get(key)
        if got is not None and got.strip() == want:
            continue
        if _acked(key, env):
            continue
        if got is None:
            out.append("R1 %s UNSET (contract requires %r); set %s%s=<reason> "
                       "to waive" % (key, want, key, ACK_SUFFIX))
        else:
            out.append("R1 %s=%r, contract requires %r; set %s%s=<reason> "
                       "to waive" % (key, got, want, key, ACK_SUFFIX))
    return out


def _check_eval_required(role, env):
    """R5: EVAL-only required values. Mirrors `_check_truth`'s role gating."""
    if role != EVAL:
        return []
    out = []
    for key, want in sorted(EVAL_REQUIRED.items()):
        got = env.get(key)
        if got is not None and got.strip() == want:
            continue
        if _acked(key, env):
            continue
        out.append("R5 %s=%r, EVAL contract requires %r (%s); set "
                   "%s%s=<reason> to waive"
                   % (key, got, want, EVAL_REQUIRED_CITE.get(key, "EVAL"),
                      key, ACK_SUFFIX))
    return out


def _check_train_required(role, env):
    """R6: TRAIN-only required values (GUARDRESTORE-756). Mirrors R5's
    role gating with the opposite role."""
    if role != TRAIN:
        return []
    out = []
    for key, want in sorted(TRAIN_REQUIRED.items()):
        got = env.get(key)
        if got is not None and got.strip() == want:
            continue
        if _acked(key, env):
            continue
        out.append("R6 %s=%r, TRAIN contract requires %r (GUARDRESTORE-756: "
                   "the -1.75 training device is env-override-only); set "
                   "%s%s=<reason> to waive" % (key, got, want, key, ACK_SUFFIX))
    return out


def _check_sigma(env):
    raw = env.get("FD_V3_SIGMA_OVERRIDE_MS")
    if raw is None:
        return []                      # unset => config default (A7 light)
    key = raw.strip()
    if key in A7_SIGMA:
        return []
    if _acked("FD_V3_SIGMA_OVERRIDE_MS", env):
        return []
    return ["R2 FD_V3_SIGMA_OVERRIDE_MS=%r is not an A7 taxonomy triple "
            "(%s); set FD_V3_SIGMA_OVERRIDE_MS_ACK=<reason> for a probe"
            % (key, "/".join(sorted(A7_SIGMA)))]


def _check_rehearsal(role, env):
    """Same variable, opposite required polarity per role."""
    setv = env.get(REHEARSAL, "").strip()
    if role == TRAIN and setv == "1" and not _acked(REHEARSAL, env):
        return ["R3 %s=1 in a TRAINING process — this disables the A5 "
                "anti-forgetting rehearsal. Set %s%s=<reason> to declare it."
                % (REHEARSAL, REHEARSAL, ACK_SUFFIX)]
    if role == EVAL and setv != "1":
        return ["R3 %s is not 1 in an EVAL/relay process — sealed gate legs "
                "must fly the CURRENT cell only, never the rehearsal mixture."
                % REHEARSAL]
    return []


def _check_truth(role, env):
    if role != TRAIN:
        return []
    out = []
    for knob in TRUTH_KNOBS:
        if env.get(knob, "").strip() and not _acked(knob, env):
            out.append("R4 %s armed in a TRAINING process (G4 bars the truth "
                       "channel from every training surface); set %s%s=<reason>"
                       % (knob, knob, ACK_SUFFIX))
    return out


def _check_eval_forbidden(role, env):
    """R8: knobs an EVAL role must NOT carry (they defeat the demotion)."""
    if role != EVAL:
        return []
    out = []
    for key, bad in sorted(EVAL_FORBIDDEN.items()):
        if (env.get(key) or "").strip() != bad or _acked(key, env):
            continue
        out.append("R8 %s=%r would SILENTLY DEFEAT the demoted terminals "
                   "(TERMDEMOTE-767): the breakaway enters INTERCEPT and the "
                   "leg dies as SAFETY_FORMATION_LOST instead; set %s%s="
                   "<reason> to waive" % (key, bad, key, ACK_SUFFIX))
    return out


def _check_train_forbidden(role, env):
    """R7: knobs a TRAIN role must NOT carry. Opposite polarity of R5."""
    if role != TRAIN:
        return []
    out = []
    for key in TRAIN_FORBIDDEN:
        got = env.get(key)
        if not (got or "").strip() or _acked(key, env):
            continue
        out.append("R7 %s=%r, TRAIN must NOT set it (TERMDEMOTE-767: demoted "
                   "terminals are EVAL-only; an inherited eval env would "
                   "silently change training); set %s%s=<reason> to waive"
                   % (key, got, key, ACK_SUFFIX))
    return out


def check(role, env=None):
    """Return the list of violations. Empty list == contract satisfied."""
    if role not in ROLES:
        raise ValueError("role must be one of %r, got %r" % (ROLES, role))
    env = os.environ if env is None else env
    return (_check_d7(env) + _check_sigma(env)
            + _check_rehearsal(role, env) + _check_truth(role, env)
            + _check_eval_required(role, env)
            + _check_train_required(role, env)
            + _check_train_forbidden(role, env)
            + _check_eval_forbidden(role, env))


def assert_contract(role, env=None, stream=None):
    """Fail closed on violation. Call at process entry, BEFORE any spawn.

    `FD_ENV_CONTRACT_ENFORCE=0` downgrades to a printed warning — an
    intentional, loud escape hatch for reproduction and rollback.
    """
    env = os.environ if env is None else env
    problems = check(role, env)
    if not problems:
        return []
    banner = "[ENV_CONTRACT] %s role=%s: %d violation(s):\n  %s" % (
        "VIOLATION", role, len(problems), "\n  ".join(problems))
    if env.get("FD_ENV_CONTRACT_ENFORCE", "1").strip() == "0":
        print(banner + "\n[ENV_CONTRACT] ENFORCE=0 — continuing anyway.",
              file=stream or sys.stderr)
        return problems
    raise EnvContractViolation(banner)


def render(role, env=None):
    env = os.environ if env is None else env
    problems = check(role, env)
    lines = ["[ENV_CONTRACT] role=%s resolved:" % role]
    for key in sorted(D7_REQUIRED):
        lines.append("  %-24s = %r" % (key, env.get(key)))
    sig = env.get("FD_V3_SIGMA_OVERRIDE_MS")
    lines.append("  %-24s = %r (%s)" % ("FD_V3_SIGMA_OVERRIDE_MS", sig,
                                        A7_SIGMA.get((sig or "").strip(),
                                                     "config default")))
    lines.append("  %-24s = %r" % (REHEARSAL, env.get(REHEARSAL)))
    for key in sorted(EVAL_REQUIRED):        # R5; auditable in the leg log
        lines.append("  %-24s = %r" % (key, env.get(key)))
    lines.append("  verdict: %s" % ("OK" if not problems else
                                    "%d VIOLATION(S)" % len(problems)))
    lines.extend("  ! " + p for p in problems)
    return "\n".join(lines)


def main(argv=None):
    """Dry-run entry point: resolve and report WITHOUT launching anything."""
    argv = sys.argv[1:] if argv is None else argv
    role = argv[0] if argv else TRAIN
    print(render(role))
    return 0 if not check(role) else 1


if __name__ == "__main__":
    sys.exit(main())
