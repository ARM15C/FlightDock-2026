#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T-321 — stop replacement waves booting SOLO, without letting
#   the panel adjudicate more legs than it contracts for. Two functions, and
#   the second exists only because the first would otherwise break N.
#
# THE DEFECT [MEASURED 2026-08-08, gate_D1a attempt_001].
#   gate_relay_core.lane_split() does `lanes = max(1, min(lanes, len(seeds)))`,
#   so a ONE-seed replacement wave is pinned to exactly ONE lane -- env1 --
#   however many of the 24 are free. The observed consequence:
#
#     opening wave, 16 legs across env1..env12: 5 FAULT (31%)
#     round1..round5, 1 leg each, ALL on env1: 5 FAULT (100%)
#
#   All five carried the identical one-tick T-93 signature (0.936 m upwind /
#   1.536 m downwind at wind 6.0), and env1 produced a CLEAN leg when it booted
#   inside the 16-way wave. Under the wave-measured 31% rate, 5/5 is p ~= 0.3%.
#   [HYPOTHESIS, confound named] the SOLO-BOOT pattern -- not the lane identity
#   -- is what latches the stale ATTITUDE stamp behind T-93. It is consistent
#   with T-93's still-open trigger RCA, but the W12 retention screen also saw
#   4/6 faults on a MULTI-leg wave, so solo-boot cannot be the whole story.
#   What is measured and not in doubt: single-leg replacement waves fault far
#   above the opening-wave rate, and each one costs a full ~19-minute round.
#
# WHY TRIMMING IS NOT OPTIONAL.
#   _shortfall() reports positive shortfalls only. Draw 2 to replace 1 and the
#   panel can finish with 13 legs against a pre-registered N=12: the next
#   _shortfall() returns {} and the panel sails through as "clean_panel" while
#   the adjudicator scores an N nobody registered. So over-provisioning is
#   admissible ONLY with a fixed, outcome-blind trim -- which is the whole of
#   trim_surplus() below.
#
# PUBLIC API: plan_draw(need, trajs, minimum) -> dict ·
#   trim_surplus(panel_dir, want, traj_fn, relink_fn=None) -> list ·
#   leg_seed(leg_dir) -> int|None · selfcheck() -> int.
# SIDE-EFFECTS: trim_surplus RENAMES surplus leg dirs into a sibling
#   `<panel>_surplus/` -- moved intact, NEVER deleted, exactly the quarantine
#   precedent. plan_draw is pure. Nothing here reads a leg's pass/fail outcome.
# CAPS: file soft 200 / hard 260 code-LoC; function cap 40.
# COVERAGE: tests/unit/test_gate_relay_overprovision.py + the executed
#   --selfcheck, both directions on every rule.
# =============================================================================
import json
import os
import sys

MIN_WAVE = 2          # legs per replacement wave; see the trade-off below
SURPLUS_SUFFIX = "_surplus"


def plan_draw(need, trajs, minimum=MIN_WAVE):
    """Inflate a replacement draw so the wave is never a SOLO boot.

    Extra legs are spread across `trajs` in order, so a wave replacing one
    `straight` also flies a `racetrack` rather than doubling one stratum.

    TRADE-OFF, stated because it spends a governed resource: each extra leg
    burns a frozen replacement seed. MIN_WAVE=2 burns at most one spare per
    round and lifts P(round yields a clean leg) from ~0.52 to ~0.77 at the
    measured fault rate; 3 would reach ~0.89 and burn up to two. 2 is the
    elected value. Wall-clock is unchanged either way
    -- the extra legs fly on lanes that were idle.

    Returns a NEW dict; `need` is never mutated. An empty need stays empty,
    so a satisfied panel never draws.
    """
    out = {t: int(c) for t, c in dict(need).items() if int(c) > 0}
    if not out:
        return out                     # satisfied panel: never draw
    order = list(trajs) or sorted(out)
    while sum(out.values()) < int(minimum):
        # Fill the SPARSEST trajectory first (ties by `trajs` order). Walking
        # `order` from index 0 instead would hand the spare straight back to
        # the stratum that already needed one, doubling it -- the opposite of
        # spreading, and it would keep the wave inside one trajectory.
        t = min(order, key=lambda x: (out.get(x, 0), order.index(x)))
        out[t] = out.get(t, 0) + 1
    return out


def leg_seed(leg_dir):
    """Seed from leg_meta.json. None if unreadable -- such a leg sorts LAST
    and is trimmed first, because a leg we cannot identify is the one we
    least want to keep. Mirrors leg_trajectory()'s shape and failure mode."""
    try:
        with open(os.path.join(leg_dir, "leg_meta.json")) as fh:
            return int(json.load(fh)["seed"])
    except Exception:
        return None


def trim_surplus(panel_dir, want, traj_fn, relink_fn=None):
    """Move legs beyond `want` per trajectory into a sibling `_surplus/` dir.

    THE SELECTION RULE IS FIXED HERE, BEFORE ANY OUTCOME IS SEEN: keep the
    EARLIEST-DRAWN (lowest seed) legs per trajectory; trim the rest. Seeds
    come from the frozen ledger in draw order before a leg flies, so this
    cannot see a score. It mirrors the benchmark's "replacement draws
    ascending within stratum". Choosing WHICH clean leg to keep after seeing
    scores would be adaptive selection -- precisely what a screen must never
    do -- so the rule is mechanical, and this docstring is the pre-registration.

    Trajectories absent from `want` are left alone; a leg whose trajectory is
    unreadable is left for the UNSAMPLED path to quarantine, never trimmed
    here (two mechanisms must not both claim one leg).
    """
    by_traj = {}
    for leg in sorted(os.listdir(panel_dir)):
        d = os.path.join(panel_dir, leg)
        if not os.path.isdir(d):
            continue
        t = traj_fn(d)
        if t in want:
            by_traj.setdefault(t, []).append((leg_seed(d), leg))
    moved, sroot = [], panel_dir.rstrip("/") + SURPLUS_SUFFIX
    for t, legs in sorted(by_traj.items()):
        legs.sort(key=lambda p: (p[0] is None, p[0] if p[0] is not None else 0,
                                 p[1]))
        for seed, leg in legs[int(want[t]):]:
            os.makedirs(sroot, exist_ok=True)
            dest = os.path.join(sroot, leg)
            os.rename(os.path.join(panel_dir, leg), dest)
            if relink_fn:
                relink_fn(dest)        # legacy ABSOLUTE links die on rename
            moved.append({"leg": leg, "seed": seed, "traj": t,
                          "why": "surplus_over_provision"})
    return moved


def _mk(panel, name, traj, seed):
    d = os.path.join(panel, name)
    os.makedirs(d)
    with open(os.path.join(d, "leg_meta.json"), "w") as fh:
        json.dump({"trajectory": traj, "seed": seed}, fh)
    return d


def _traj_fn(d):
    try:
        with open(os.path.join(d, "leg_meta.json")) as fh:
            return json.load(fh).get("trajectory")
    except Exception:
        return None


def _cases_plan(trajs):
    """plan_draw, both directions."""
    return [("a 1-leg need is inflated to the minimum",
             sum(plan_draw({"straight": 1}, trajs).values()), 2),
            ("...and the spare goes to the OTHER trajectory",
             plan_draw({"straight": 1}, trajs), {"straight": 1, "racetrack": 1}),
            ("a wave already >= minimum is UNCHANGED",
             plan_draw({"straight": 2, "racetrack": 1}, trajs),
             {"straight": 2, "racetrack": 1}),
            ("a SATISFIED panel still draws NOTHING", plan_draw({}, trajs), {}),
            ("...and a zero-count need is treated as satisfied",
             plan_draw({"straight": 0}, trajs), {}),
            ("the input dict is not mutated",
             (lambda n: (plan_draw(n, trajs), n)[1])({"straight": 1}),
             {"straight": 1}),
            ("minimum is honoured above 2",
             sum(plan_draw({"straight": 1}, trajs, minimum=3).values()), 3)]


def _cases_trim(tmp, trajs):
    """trim_surplus, both directions, with the N-integrity property named."""
    panel = os.path.join(tmp, "S0")
    os.makedirs(panel)
    want = {"straight": 2, "racetrack": 2}
    for nm, tj, sd in (("L_a", "straight", 300), ("L_b", "straight", 100),
                       ("L_c", "straight", 200), ("L_d", "racetrack", 500),
                       ("L_e", "racetrack", 400)):
        _mk(panel, nm, tj, sd)
    moved = trim_surplus(panel, want, _traj_fn)
    kept = sorted(os.listdir(panel))
    sur = sorted(os.listdir(panel + SURPLUS_SUFFIX))
    # second pass on an exactly-sized panel must be a NO-OP
    again = trim_surplus(panel, want, _traj_fn)
    empty = os.path.join(tmp, "E")
    os.makedirs(empty)
    _mk(empty, "L_only", "straight", 7)
    noop = trim_surplus(empty, {"straight": 2, "racetrack": 2}, _traj_fn)
    return [("N INTEGRITY: the panel is trimmed to exactly `want`",
             len(kept), sum(want.values())),
            ("the EARLIEST-DRAWN straight legs are kept (100, 200)",
             kept, ["L_b", "L_c", "L_d", "L_e"]),
            ("the highest-seed straight leg is the one trimmed",
             sur, ["L_a"]),
            ("...and it is MOVED, not deleted",
             os.path.isfile(os.path.join(panel + SURPLUS_SUFFIX, "L_a",
                                         "leg_meta.json")), True),
            ("the trim is recorded with its seed and reason",
             [(m["leg"], m["seed"], m["why"]) for m in moved],
             [("L_a", 300, "surplus_over_provision")]),
            ("an exactly-sized panel is a NO-OP (idempotent)", again, []),
            ("an UNDER-sized panel is never trimmed", noop, [])]


def selfcheck():
    import tempfile
    trajs = ("straight", "racetrack")
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, got, want in _cases_plan(trajs) + _cases_trim(tmp, trajs):
            ok = got == want
            print("  %s %s: %s want=%s" % ("OK " if ok else "BAD", name, got,
                                           want))
            bad += 0 if ok else 1
    print("SELFCHECK", "PASS" if not bad else "FAIL (%d)" % bad)
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(selfcheck() if "--selfcheck" in sys.argv else 0)
