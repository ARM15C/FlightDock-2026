#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: the two B2 limbs that read SOURCE and LIVE CONFIG rather than a built table.
#   Split out of armblad035_seed_limbs.py 2026-08-26 at 111 code-LoC against a hard 110 --
#   caps are SPLIT, never raised. B2h is the load-bearing one: it asserts the table's wind
#   ladder against the LIVE src/drl/config._V3_LADDER_SPEC, so a config drift cannot leave the
#   table quietly claiming a wind the venue no longer flies.
# PUBLIC API: b2h_ladder_matches_live_config() -> (id, ok, detail)
#             b2i_no_rng() -> (id, ok, detail)
# SIDE-EFFECTS: imports src/drl/config READ-ONLY and never calls into it; reads the planner and
#   CLI source as text. Writes nothing, spawns nothing, contacts no box.
# SIZE: soft 30 / hard 45 code-LoC; function cap 25.
# COVERAGE: registered by armblad035_seed_limbs.all_limbs(), which the unit test executes.
# =============================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import armblad035_seed_plan as sp                                          # noqa: E402
import armblad035_seed_cli as cli                                          # noqa: E402


def b2h_ladder_matches_live_config():
    """The table may not claim a wind the venue does not fly. Read the LIVE ladder."""
    sys.path.insert(0, os.path.join(FD, "src"))
    from drl import config as cfg
    stage = {lab: cfg.CURRICULUM_STAGES_V3[i].wind_max_ms
             for i, lab in cfg.STAGE_LABELS_V3.items()}
    frm = {lab: cfg.CURRICULUM_STAGES_V3[i].wind_from_deg
           for i, lab in cfg.STAGE_LABELS_V3.items()}
    bad = [(lb, w, stage.get(lb)) for lb, w, _ in sp.RUNGS
           if stage.get(lb) != w or frm.get(lb) != 180.0]
    return ("B2h_ladder_matches_live_config", not bad,
            "all 10 rungs match _V3_LADDER_SPEC wind and 180 deg" if not bad
            else "DRIFT: %s" % bad[:3])


def b2i_no_rng():
    # Assembled needles: a literal would appear in this file and the detector would match
    # ITSELF (the pgrep self-match lesson, in a new costume).
    src = open(sp.__file__).read() + open(cli.__file__).read()
    needles = ["import " + "random", "np." + "random", "from " + "random", "time" + "()"]
    return ("B2i_no_rng_no_clock", all(n not in src for n in needles)
            and ("import " + "random") not in src,
            "planner and CLI import no RNG and read no clock; assembled needles self-verified")


