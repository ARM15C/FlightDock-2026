#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: the ONE definition of what a ARMBLAD-035 leg IS -- the binding of every leg
#   to its (rung, wind, sigma, trajectory, seed), and the invariants that binding must hold.
#   COPY, not import, of baseline025_seed_plan.py: BASELINE-025 flew as-is on eval2 (600/600)
#   and the two must be able to diverge without either editing the other. The sibling
#   armblad035_seed_cli.py writes SEED_CELL_TABLE_ARMBLAD035.json and emits the .sha256 the
#   operator countersigns at prereg block B. No leg flies before that signature.
#
#   ONE VARIABLE MOVES, AND ONLY ONE ("keep everything symmetric with
#   BASELINE-025"). The nine WIND rungs fly sigma 0.35 where the parent flew 0.25. Everything
#   else is carried byte-for-byte:
#     * the SAME 30 seeds, FRESH block 10775-10804 (never flown before 2026-08-30) -- so every leg here has a TWIN in
#       BASELINE-025 matched on (wind, seed, trajectory) and the contrast is PAIRED leg-for-leg,
#       not two independent samples. That pairing is the whole point of the campaign;
#     * the SAME nine wind rungs, 0-8 m/s at 1 m/s, wind_from_deg 180.0;
#     * the SAME still-air S0 rung (wind 0 AND turbulence OFF, sigma 0.00) -- kept for symmetry
#       under prereg block D, so both campaigns are 10 rungs x 2 trajectories x 30 seeds = 600;
#     * the SAME trajectory rule: trajectory is NOT bound to seed, every seed flies BOTH at
#       EVERY rung, so straight-vs-racetrack stays a WITHIN-SEED contrast;
#     * the SAME S6.1 paired_csv ruling: --paired-csv on the still-air rung and NOWHERE else.
#       Absent on S0 the rung is banded against band(0.00)=0.0000 and UNGATED; present on a
#       wind rung it misroutes that rung into the calm tier, which keys on MEAN WIND, and
#       REFUSES a correct leg. Each row CARRIES the field so the driver READS the ruling.
#     * NO RNG and no clock: two builds are byte-identical.
#     * The table may not outlive its signature -- the countersignature binds the SHA, never
#       the filename.
#
#   sigma is a plan-level constant in the rung tuple, NOT a stage-cell property, so this
#   campaign creates no stage cell and does not modify config.py. Cells key as B35-W<N>, with
#   the still-air rung keeping the parent's B00-S0.
#
#   NOT A TRUTH PATH: this module holds integers, floats, strings and booleans. It imports
#   nothing from src/ or drl/, reads no environment, and runs only BEFORE any leg is spawned.
#   It cannot reach an actor, critic, reward, replay buffer, observation or policy target.
#   The static leakage checker's surface is the six src/drl training-path files and does not
#   scan analysis/, so tests/unit/test_armblad_seed_plan.py asserts this directly, by AST.
# PUBLIC API: RUNGS, BLOCK, TRAJS, DISPATCH
#             rung_rows(block, rungs) -> list[dict]
#             build_table(block=BLOCK, rungs=RUNGS) -> dict
#             assert_no_new_seeds(table, die) -> None
#             assert_complete_cross(table, die) -> None
#             assert_paired_csv_on_s0_only(table, die) -> None
# SIDE-EFFECTS: **NONE.** This module builds a dict and refuses malformed ones. It opens no
#   file, spawns nothing, kills nothing, draws no seed, touches no ledger, contacts no box and
#   reads no environment variable. Writing the table and running the on-disk seed firewall are
#   the sibling armblad035_seed_cli.py's job.
# SIZE: soft 80 / hard 100 code-LoC; function cap 30.
# COVERAGE: armblad035_seed_limbs.py (every limb driven to its refusal as well as its pass)
#   and tests/unit/test_armblad_seed_plan.py, EXECUTED.
# =============================================================================
import sys

DISPATCH = "ARMBLAD-035"
TRAJS = ("straight", "racetrack")

#: (label, wind_ms, sigma_ms). Labels and winds are the LIVE v3 ladder cells --
#: src/drl/config._V3_LADDER_SPEC, corpus convention W<N> => wind = N/2 m/s. No stage cell is
#: created by this campaign; limb B2h asserts every row here against the live config.
#: S0 is the still-air control: wind 0 AND turbulence off, and it is the ONLY sigma-0 rung.
RUNGS = (
    ("S0",  0.0, 0.00),                       # carried from the parent, UNCHANGED (block D)
    ("W0",  0.0, 0.35), ("W2",  1.0, 0.35), ("W4",  2.0, 0.35),
    ("W6",  3.0, 0.35), ("W8",  4.0, 0.35), ("W10", 5.0, 0.35),
    ("W12", 6.0, 0.35), ("W14", 7.0, 0.35), ("W16", 8.0, 0.35),
)

#: FRESH block 10775-10804: 30 never-flown labels, contiguous after
#: G7's 10774, SHARED by both ARMBLAD ladders so sigma is a within-seed contrast. No junk ARM-B leg
#: dir (the purged 10475-10504 corpora) can share a name with any leg here.
#: (parent text, superseded:) The 30 ARMBSIG-810 sweep labels, RE-FLOWN -- these seeds were already registered
#: and already spent. Re-using them is what keeps this grid PAIRED with the candidate exams.
BLOCK = tuple(range(10775, 10805))
ORIGIN = "block-10775-10804 (FRESH, never flown; 30 labels drawn 2026-08-30 for the two ARMBLAD ladders; shared with ARMBLAD-025)"


def die(rc, msg):
    print("ARMBLAD035 SEEDPLAN REFUSED (%d): %s" % (rc, msg), file=sys.stderr)
    sys.exit(rc)


def _sigkey(sigma):
    """0.35 -> 'B35', 0.0 -> 'B00'. The cell key is sigma AND wind; neither alone is a cell."""
    return "B%02d" % int(round(sigma * 100))


def rung_rows(block, rungs):
    """Frozen ascending, outcome-blind, full cross. No RNG: two runs are byte-identical."""
    out = []
    for label, wind, sigma in rungs:
        for seed in block:
            for traj in TRAJS:
                out.append({
                    "seed": int(seed), "label": label, "wind_ms": float(wind),
                    "sigma_ms": float(sigma), "traj": traj, "role": "reference",
                    "cell": "%s-%s" % (_sigkey(sigma), label), "origin": ORIGIN,
                    # prereg S6.1 -- the ruling travels WITH the row, never re-derived
                    "paired_csv": bool(sigma == 0.0),
                })
    return out


def build_table(block=BLOCK, rungs=RUNGS):
    legs = rung_rows(block, rungs)
    seen = set()
    for lg in legs:
        k = (lg["cell"], lg["seed"], lg["traj"])
        if k in seen:
            die(2, "duplicate leg %s" % (k,))
        seen.add(k)
    return {
        "dispatch": DISPATCH, "grid": "wind ladder x trajectory, zero-residual ARM-B",
        "convention": "W<N> => wind = N/2 m/s (corpus); cell = sigma AND wind; S0 = still air",
        "controller": "ARM-B (--force-zero-residual)",
        "wind_from_deg": 180.0,
        "n_legs": len(legs), "n_distinct_seeds": len({l["seed"] for l in legs}),
        "n_rungs": len(rungs), "block": [int(block[0]), int(block[-1])],
        "new_seeds_drawn": 30, "reserve_spent": False, "legs": legs,
    }


def assert_no_new_seeds(table, die_fn):
    """Every seed must come from THE fresh block 10775-10804 and the header must declare exactly that
    draw (30 labels) with the 414-seed replacement reserve untouched. Anything else is refused."""
    stray = sorted({l["seed"] for l in table["legs"]} - set(BLOCK))
    if stray:
        die_fn(2, "table carries %d seed(s) outside the fresh block 10775-10804: %s. "
                  "ARMBLAD-035 flies that block and nothing else; the reserve stays unspent at 414"
               % (len(stray), stray[:5]))
    if table.get("new_seeds_drawn") != 30 or table.get("reserve_spent") is not False:
        die_fn(2, "table header does not declare the 30-label draw with the reserve unspent")


def assert_complete_cross(table, die_fn):
    """INVERTED from the lineage on purpose: every seed must fly EVERY rung on BOTH
    trajectories, so straight-vs-racetrack and rung-to-rung are WITHIN-seed contrasts."""
    want = {(lb, t) for lb, _, _ in RUNGS for t in TRAJS}
    for seed in {l["seed"] for l in table["legs"]}:
        got = {(l["label"], l["traj"]) for l in table["legs"] if l["seed"] == seed}
        if got != want:
            die_fn(2, "seed %d flies %d of %d (rung, trajectory) combinations; the cross must "
                      "be COMPLETE or the within-seed contrast is not paired"
                   % (seed, len(got), len(want)))


def assert_paired_csv_on_s0_only(table, die_fn):
    """Prereg S6.1. Both directions are load-bearing: absent on S0 leaves the still-air rung
    banded against band(0.00)=0.0000 and UNGATED; present on a wind rung misroutes it into the
    calm tier, which keys on MEAN WIND, and REFUSES a correct leg (the W0 case)."""
    on = {l["label"] for l in table["legs"] if l["paired_csv"]}
    calm = {lb for lb, _, s in RUNGS if s == 0.0}
    if on != calm:
        die_fn(2, "paired_csv is set on %s but the still-air rung(s) are %s. It belongs on the "
                  "calm rung and NOWHERE else (prereg S6.1)" % (sorted(on) or "nothing", sorted(calm)))


