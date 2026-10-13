#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: R-3 — make every
#   sealed panel CLASSIFY its legs with the T-93 fault classifier and REPLACE
#   contaminated legs from the frozen seed blocks until the panel is N clean,
#   with trajectory balance preserved. The replacement rule is fixed HERE,
#   before any outcome is seen — the executor never knows whether a leg
#   passed or failed, only whether its truth channel is trustworthy.
#
# WHY THIS FILE EXISTS — WS1-r0's S0 "retention strike" was 100% fault-class.
#   The T-92 aligner defect (+1.24 m truth displacement) contaminated 16/42
#   sealed legs (~38%); every retention failure in every stratum was a FAULT
#   leg failing on exactly the truth-coupled conjunct, while clean legs ran
#   20/20 (T-126, WS1_R0_VERDICT.md). At that incidence every future panel
#   keeps tripping on phantom failures and burning frozen seeds. T-119 made
#   the classifier the acceptance gate of record with fault legs EXCLUDED,
#   never corrected — this module operationalises that election so panels
#   self-correct instead of holding on artifacts.
#
# POLICY (fixed before outcomes, fail-closed at every edge):
#   CLEAN -> keep.
#   FAULT -> quarantine (os.rename into a sibling *_fault_quarantine/
#                 dir — moved intact, NEVER deleted) and replace.
#   UNSAMPLED -> same as FAULT: a leg that cannot be classified cannot be
#                 certified clean. Systemic UNSAMPLED exhausts the round
#                 budget and HOLDs — it cannot silently pass.
#   BORDERLINE -> KEEP the leg and FLAG it in the manifest (SCREENPOINT-720,
#                 ). It used to HOLD the whole panel
#                 for a human. Under the v2 mean-bias rule the middle band is
#                 the ordinary post-Fix outcome — 11 of 14 cohort legs land
#                 there — so holding on it would stall every panel, and
#                 quarantining it would re-create the 55% over-quarantine that
#                 exhausted D2a one racetrack seat short. Still never silently
#                 binned: the manifest carries the leg, its mean_ticks and the
#                 rule_version, so the verdict is read WITH it in view.
#   Rounds are bounded (MAX_ROUNDS); a shortfall after the last round HOLDs.
#
# PUBLIC API: load_classifier, leg_trajectory, explicit_tmap,
#   ensure_clean_panel, MAX_ROUNDS, CLASSIFIER_PATH
# SIDE-EFFECTS: renames FAULT/UNSAMPLED leg dirs into the quarantine sibling
#   (data intact); writes ONE faultscreen_manifest.json per panel. Never
#   deletes, never signals a process, never edits a leg's contents, never
#   reads a leg's pass/fail outcome.
# CAPS: file soft 135 / hard 150 LoC (code lines); function cap 45. At 149 —
#   ONE line under the hard cap. SCREENPOINT-720 was held to a net-zero line
#   budget for exactly this reason (the repointed CLASSIFIER_PATH paid for the
#   BORDERLINE flag block). The NEXT change to this file must SPLIT it.
#   (Soft recalibrated 120 -> 135 by R-5, which added `_relink_paired` and the
#   seed ceiling. Stated, not silently exceeded; the HARD cap is unchanged —
#   raising a hard cap to fit code is the move that is not available, which is
#   why R-5's test bloat was fixed by splitting the file instead.)
# COVERAGE: tests/unit/test_gate_relay_faultscreen.py
# =============================================================================
import importlib.util
import json
import os

import gate_relay_overprovision as _op   # T-321: solo-boot fix + N integrity

FD = "/home/ubuntu/FlightDock"
#: SCREENPOINT-720: repointed from the v1 median rule to the v2 MEAN-bias rule.
#: Fix A (elected, epoch 2026-08-18T21:04:46Z) drives the median tick to 0.0 BY
#: CONSTRUCTION, so v1 read CLEAN on every leg and could no longer quarantine
#: anything. v1 is NOT edited and NOT deleted: it stays frozen for historical
#: provenance and keeps its SHA pin in tools/pvt_paired_census.py. Rollback is
#: this one constant (or FD_V3_FAULTSCREEN_DISABLE=1 to skip the screen).
CLASSIFIER_PATH = os.path.join(FD, "tools", "faultclass_bias_v2.py")
#: Replacement rounds per panel. R-5 (2026-07-31) raised this 2 -> 7 after
#: Phase A HELD one leg short. P(panel completes) = (1 - p^(R+1))^N; at the
#: MEASURED incidence p=0.40 (8 FAULT / 20 flown) and N=12 that is 46% at R=2
#: and 99.9% at R=7. EXPECTED COST DOES NOT GROW WITH R — it is N/(1-p) ~= 20
#: legs either way; R only truncates the tail. Pinned by
#: test_round_bound_arithmetic_is_sufficient_at_measured_incidence so changing
#: it forces re-deriving the arithmetic.
MAX_ROUNDS = 7
#: Hard ceiling on legs flown per panel, as a multiple of N. Raising MAX_ROUNDS
#: lengthens the tail, so this closes the failure mode that opens: a
#: pathological panel draining a frozen seed block. Refuses to DRAW past the
#: ceiling rather than spending first and complaining after.
#: CEILRAISE-20260814 (signed dispatch, analysis/ceilraise_20260814T0400Z/):
#: 3 -> 5 on the MEASURED W13 retention clean rate 18/42 = 42.9% (the 3x
#: ceiling was sized before any campaign measured its fault rate). At p_clean
#: 0.43, want-6 ceiling 30: P(trip) ~= 0.3% (vs 10-20% at 18); want-12
#: ceiling 60: ~0.01%. Still fail-closed: trips whenever clean rate falls
#: below ~20% — the pathological regime it was built for. Seed-block drain
#: worst case rises 18 -> 30 per retention panel against ~4,350 unused seeds.
#: NOTE (FM-2): a live gate_relay_core process keeps its imported value until
#: that request ends — this raise reaches the NEXT gate request only.
MAX_PANEL_LEGS_FACTOR = 5
TRAJS = ("straight", "racetrack")


def load_classifier(path=CLASSIFIER_PATH):
    """The T-93 classifier of record (T-119), loaded like the runner loads
    its frozen adjudicator. A missing classifier raises — a panel that cannot
    screen must not fly as if it had."""
    spec = importlib.util.spec_from_file_location("faultclass_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_leg


def leg_trajectory(leg_dir):
    """Trajectory from leg_meta.json; None if unreadable (counts as shortfall
    in NEITHER stratum — the leg will be quarantined as UNSAMPLED anyway)."""
    try:
        with open(os.path.join(leg_dir, "leg_meta.json")) as fh:
            t = json.load(fh).get("trajectory")
        return t if t in TRAJS else None
    except Exception:
        return None


def explicit_tmap(seeds, need):
    """Assign replacement seeds to trajectories to cover `need` exactly.

    The default balanced tmap is index-parity; replacements must instead
    mirror WHAT WAS LOST (quarantine 2 straight => fly 2 straight), or the
    panel drifts off its contracted per-trajectory floor.
    """
    order = [t for t in TRAJS for _ in range(int(need.get(t, 0)))]
    assert len(order) == len(seeds), (order, seeds)
    return {int(s): t for s, t in zip(seeds, order)}


def _classify_all(panel_dir, classify):
    legs = sorted(d for d in os.listdir(panel_dir)
                  if os.path.isdir(os.path.join(panel_dir, d)))
    return {leg: classify(os.path.join(panel_dir, leg)) for leg in legs}


def _relink_paired(leg_dir):
    """Re-point a MOVED leg's own `paired/env0` at its new location.

    WHY: gate_relay_core.bridge_paired_layout writes ABSOLUTE symlinks, so
    os.rename orphans them and the moved leg re-reads as
    UNSAMPLED/no_paired_csv — i.e. the evidence for an exclusion is destroyed
    AT THE MOMENT OF EXCLUSION (live-found in Phase A, 2026-07-31). Scoped to
    quarantined legs, which are already outside adjudication, so the shared
    bridge helper and the frozen adjudication path stay untouched.
    """
    link = os.path.join(leg_dir, "paired", "env0")
    if not os.path.islink(link) or os.path.exists(link):
        return False                      # absent, or still resolving
    envs = sorted(d for d in os.listdir(leg_dir)
                  if d.startswith("env")
                  and os.path.isdir(os.path.join(leg_dir, d)))
    if not envs:
        return False                      # nothing to point at; leave dangling
    os.remove(link)                       # the dangling link only, never data
    # RELATIVE (T-317): repairing with an absolute link merely re-arms the
    # same failure for the NEXT move. This path now only ever sees LEGACY
    # absolute links, since both writers emit relative ones at birth.
    os.symlink(os.path.relpath(os.path.join(leg_dir, envs[0]),
                               os.path.dirname(link)), link)
    return True


def _relink_paired_tree(root):
    """`_relink_paired` for a moved tree: repoint EVERY dangling paired/env0
    under `root`, not just root's own.

    WHY: when a NON-leg directory is quarantined wholesale (the W11 panel
    moved `retention/` itself), the legs NESTED inside it all keep absolute
    links to the pre-move path — W9/W10 retention evidence read 6/6
    UNSAMPLED in place (T-233). Same failure `_relink_paired` exists for,
    one directory level up. os.walk does not follow symlinks, so the
    paired/env0 targets are never descended into."""
    fixed = 0
    for dirpath, _dirs, _files in os.walk(root):
        if _relink_paired(dirpath):
            fixed += 1
    return fixed


def _quarantine_bad(panel_dir, qroot, labels, rnd, manifest):
    """Move FAULT/UNSAMPLED legs to the quarantine sibling — intact, never
    deleted — re-point their paired link, and record them in the manifest."""
    for leg in sorted(l for l, r in labels.items()
                      if r.get("label") in ("FAULT", "UNSAMPLED")):
        os.makedirs(qroot, exist_ok=True)
        dest = os.path.join(qroot, "round%d_%s" % (rnd, leg))
        os.rename(os.path.join(panel_dir, leg), dest)
        relinked = _relink_paired_tree(dest) > 0
        # Keys are the classifier's OWN (`disp_median_m`, not `disp`): reading
        # a key the classifier never emits recorded null for every leg and made
        # the exclusions unauditable (R-5).
        manifest.append({"leg": leg, "round": rnd,
                         "label": labels[leg].get("label"),
                         "disp_median_m": labels[leg].get("disp_median_m"),
                         "n_rows": labels[leg].get("n_rows"),
                         "ticks_est": labels[leg].get("ticks_est"),
                         "mean_ticks": labels[leg].get("mean_ticks"),
                         "rule_version": labels[leg].get("rule_version"),
                         "why": labels[leg].get("why"),
                         "paired_relinked": relinked})


def _shortfall(panel_dir, want):
    """{traj: missing_count} for the legs still present vs the target."""
    have = {}
    for leg in os.listdir(panel_dir):
        t = leg_trajectory(os.path.join(panel_dir, leg))
        if t:
            have[t] = have.get(t, 0) + 1
    return {t: want[t] - have.get(t, 0) for t in want
            if want[t] - have.get(t, 0) > 0}


def ensure_clean_panel(panel_dir, want, classify, spawn_replacements,
                       max_rounds=MAX_ROUNDS):
    """Drive the panel at `panel_dir` to `want` clean legs per trajectory.

    spawn_replacements(need: {traj: count}) -> bool flies the replacement
    legs (drawing its own seeds) and returns success. Returns
    (ok, why, manifest); manifest records every quarantined leg with its
    label, displacement and round — the audit trail that keeps the exclusion
    honest.
    """
    qroot = panel_dir.rstrip("/") + "_fault_quarantine"
    manifest = []
    why = "fault_replacement_exhausted"
    ceiling = MAX_PANEL_LEGS_FACTOR * sum(want.values())
    flown = len([d for d in os.listdir(panel_dir)
                 if os.path.isdir(os.path.join(panel_dir, d))])
    for rnd in range(max_rounds + 1):
        labels = _classify_all(panel_dir, classify)
        # SCREENPOINT-720: BORDERLINE KEEPS the
        # leg and flags it, where it used to HOLD the whole panel for a human.
        # Under the v2 bias rule the middle band is the ordinary post-Fix
        # outcome — 11 of 14 cohort legs land there — so holding on it would
        # stall every panel, and quarantining it would re-create the 55%
        # over-quarantine that exhausted D2a one racetrack seat short. The leg
        # stays in the panel and the manifest carries the flag, so the verdict
        # can be read with it in view rather than the evidence disappearing.
        manifest.extend({"leg": l, "round": rnd, "label": "BORDERLINE",
                         "kept": True, "mean_ticks": r.get("mean_ticks"),
                         "rule_version": r.get("rule_version")}
                        for l, r in sorted(labels.items())
                        if r.get("label") == "BORDERLINE")
        _quarantine_bad(panel_dir, qroot, labels, rnd, manifest)
        need = _shortfall(panel_dir, want)
        if not need:
            why = "clean_panel"
            break
        if rnd == max_rounds:
            break
        # T-321: never draw a SOLO wave — lane_split() pins a 1-seed draw to
        # env1 alone and all 5 such D1a waves faulted. Inflated BEFORE the
        # ceiling test, so the seed ceiling still binds on what is really flown.
        # Spares go ONLY to trajectories the panel actually contracts for: a
        # stratum with want=0 must not burn a frozen seed it can never keep.
        need = _op.plan_draw(need, [t for t in TRAJS if want.get(t)])
        if flown + sum(need.values()) > ceiling:
            why = "seed_ceiling_exceeded"     # refuse to DRAW, not just report
            break
        if not spawn_replacements(need):
            why = "replacement_spawn_fail"
            break
        flown += sum(need.values())
    # SIBLING path, never inside panel_dir: the retention runner os.listdir()s
    # the stratum dir and would score a stray FILE as a failed leg
    # ("leg_meta_missing_or_bad") — silently corrupting n.
    _finalize(panel_dir, want, why, manifest, rnd, qroot)
    return why == "clean_panel", why, manifest


# Trim any over-provision surplus, then write the ONE manifest. Extracted from
# ensure_clean_panel to stay inside the 45-line function cap after T-321 —
# nothing was trimmed to fit and no cap was raised. Written as comments, not a
# docstring, because docstrings COUNT as code lines and this file sits 3 lines
# under its HARD cap; a hard cap is an not mine to raise.
def _finalize(panel_dir, want, why, manifest, rnd, qroot):
    # T-321: an over-provisioned wave can leave MORE than `want`; _shortfall()
    # reports positives only, so an N=13 panel would pass as "clean_panel" and
    # be adjudicated against a registered N=12. The trim is outcome-blind.
    surplus = (_op.trim_surplus(panel_dir, want, leg_trajectory,
                                _relink_paired_tree)
               if why == "clean_panel" else [])
    with open(panel_dir.rstrip("/") + "_faultscreen_manifest.json", "w") as fh:
        json.dump({"result": why, "quarantined": manifest, "surplus": surplus,
                   "want": want, "rounds_used": rnd,
                   "quarantine_dir": qroot if manifest else None},
                  fh, indent=1)
