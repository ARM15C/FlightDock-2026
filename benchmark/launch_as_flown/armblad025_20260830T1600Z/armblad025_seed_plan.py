#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: the ONE definition of what a ARMBLAD-025 leg IS -- the binding of every leg
#   to its (rung, wind, sigma, trajectory, seed), and the invariants that binding must hold.
#   The sibling armblad025_seed_cli.py writes it to SEED_CELL_TABLE_ARMBLAD025.json and emits
#   the .sha256 the operator countersigns at prereg S16. No leg flies before that signature.
#
#   FIVE INVARIANTS THIS MODULE EXISTS TO HOLD.
#   * EXACTLY THIRTY FRESH SEEDS, 10775-10804, AND NOTHING ELSE. the
#     junk ARM-B ladders (block 10475-10504, flown pre warmup-retry-fix) are PURGED, and the fresh
#     ladders fly a block no leg has ever carried, so no fresh leg dir can share a name with a junk
#     one. assert_no_new_seeds() refuses any seed outside the block AND a header that does not
#     declare the 30-label draw. The 414-seed replacement-only reserve is never touched.
#   * TRAJECTORY IS **NOT** BOUND TO SEED, AND THAT INVERTS THE LINEAGE. armbsig_seed_plan
#     deals one trajectory per seed and REFUSES a seed bound to two (build_table, "seed %d
#     bound to two trajectories"). ARMBLAD-025 flies EVERY seed on BOTH trajectories at
#     EVERY rung, deliberately: it makes straight-vs-racetrack a WITHIN-SEED contrast, which
#     is the whole reason the operator asked for the split. The guard is therefore inverted,
#     not dropped -- assert_complete_cross() refuses any seed that does NOT fly both.
#   * THE S0 GATE ARGUMENT IS DATA, NOT A DRIVER GUESS. Prereg S6.1 rules that --paired-csv
#     is supplied on the still-air rung and on NO other rung, because leg_sigma_gate.py:137-141
#     routes on MEAN WIND rather than commanded sigma: without it S0 is banded against
#     band(0.00)=0.0000 and is ungated; with it everywhere, the W0 rung (mean wind 0, sigma
#     0.25, turbulence correctly present) is misrouted into the calm tier and REFUSED. Each
#     row therefore CARRIES `paired_csv`, so the driver READS the ruling instead of
#     re-deriving it -- the one-writer rule (feedback_use_the_one_writer_not_a_guessed_field).
#   * NO RNG, ANYWHERE. Two runs are byte-identical; the limbs assert the absence by source
#     scan with ASSEMBLED needles so the detector cannot match its own text.
#   * THE TABLE MAY NOT OUTLIVE ITS SIGNATURE. The sha256 is of the exact bytes written;
#     a renamed or edited table cannot launch, because S16 binds the SHA, not the filename.
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
#   the sibling armblad025_seed_cli.py's job -- SPLIT out of this file, never merged back,
#   because this file reached 135 code-LoC against a hard 130 and caps are SPLIT, not raised
#   (the tremed_cont_limbs -> tremed_cont_static_limbs precedent, 2026-08-25).
# SIZE: soft 80 / hard 100 code-LoC; function cap 30.
# COVERAGE: armblad025_seed_limbs.py (B2a-B2i, EVERY limb driven to its refusal as well as
#   its pass) and tests/unit/test_armblad_seed_plan.py, EXECUTED -- a check that cannot
#   fail is not a gate (RB-20).
# =============================================================================
import sys

DISPATCH = "ARMBLAD-025"
TRAJS = ("straight", "racetrack")

#: (label, wind_ms, sigma_ms). Labels and winds are the LIVE v3 ladder cells --
#: src/drl/config._V3_LADDER_SPEC, corpus convention W<N> => wind = N/2 m/s. No stage cell is
#: created by this campaign; limb B2h asserts every row here against the live config.
#: S0 is the still-air control: wind 0 AND turbulence off, and it is the ONLY sigma-0 rung.
RUNGS = (
    ("S0",  0.0, 0.00),
    ("W0",  0.0, 0.25), ("W2",  1.0, 0.25), ("W4",  2.0, 0.25),
    ("W6",  3.0, 0.25), ("W8",  4.0, 0.25), ("W10", 5.0, 0.25),
    ("W12", 6.0, 0.25), ("W14", 7.0, 0.25), ("W16", 8.0, 0.25),
)

#: FRESH block 10775-10804: 30 never-flown labels, contiguous after
#: G7's 10774, SHARED by both ARMBLAD ladders so sigma is a within-seed contrast. No junk ARM-B leg
#: dir (the purged 10475-10504 corpora) can share a name with any leg here.
#: (parent text, superseded:) The 30 ARMBSIG-810 sweep labels, RE-FLOWN -- these seeds were already registered
#: and already spent. Re-using them is what keeps this grid PAIRED with the candidate exams.
BLOCK = tuple(range(10775, 10805))
ORIGIN = "block-10775-10804 (FRESH, never flown; 30 labels drawn 2026-08-30 for the two ARMBLAD ladders; shared with ARMBLAD-035)"


def die(rc, msg):
    print("ARMBLAD025 SEEDPLAN REFUSED (%d): %s" % (rc, msg), file=sys.stderr)
    sys.exit(rc)


def _sigkey(sigma):
    """0.25 -> 'B25', 0.0 -> 'B00'. The cell key is sigma AND wind; neither alone is a cell."""
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
                  "ARMBLAD-025 flies that block and nothing else; the reserve stays unspent at 414"
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


