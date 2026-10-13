#!/usr/bin/env python3
"""armblad025_calm_witness -- prove the still-air rung was actually GATED, not silently skipped.

RESPONSIBILITY
  Prereg S6.1 puts --paired-csv on the S0 rung so the CALM_CONFIRMED tier judges it. This module
  is the check that the tier ACTUALLY ENGAGED, and it exists because of one line in the gate:

      leg_sigma_calm.resolved_wind_max() returns None on ANY read failure, and
      leg_sigma_gate.py:137-141 then STAYS ON TIER 1.

  So a mistyped, missing or unbridged paired CSV does not raise. It falls back to the band
  comparison -- and at S0 the band is band(0.00) = 0.0000, which is exactly the ungated state
  the prereg's Defect 1 describes. rc would read 0 or 3 for reasons having nothing to do with
  still air, and nobody would know the rung was never witnessed. `rc 0` is not proof the block
  did its job (feedback_rc0_is_not_proof_the_block_did_its_job).

  The positive witness is the gate's OWN output: gate_calm() stamps calm_commanded=True and a
  state of CALM_CONFIRMED / ASSERT_FAIL / UNSAMPLED / INCONCLUSIVE. Tier 1 never writes
  calm_commanded at all. So its ABSENCE on an S0 leg is the tell, and it is a HALT: a rung that
  cannot be witnessed is INCONCLUSIVE (prereg S13 item 3), never quietly kept.

PUBLIC API
  assert_calm_engaged(leg_dir) -> (rc, why)
      rc 0 the calm tier ran AND confirmed (CALM_CONFIRMED)
      rc 3 the calm tier ran and REFUSED (ASSERT_FAIL: a calm cell delivered wind)
      rc 2 the calm tier ran but could not conclude (UNSAMPLED / INCONCLUSIVE)
      rc 5 the calm tier NEVER RAN -- the silent tier-1 fallback. The rung is UNGATED.
  main(argv) -> int

SIDE-EFFECTS
  Reads <leg_dir>/sigma_gate.json READ-ONLY. Writes nothing, spawns nothing, kills nothing.

CAPS: file soft 55 / hard 75 code-lines; function cap 25. The cap is declared from the
  MEASURED 61 code-lines of the finished file, not guessed before writing it -- three caps in
  this dispatch were first declared too tight and then met by splitting; declaring after
  measuring is the honest order and avoids a split that buys nothing.
COVERAGE: armblad025_manifest_limbs.py -- all four rc values driven on fixtures.
"""
import argparse
import json
import os
import sys

RC_PASS, RC_INCONCLUSIVE, RC_ASSERT, RC_NEVER_RAN = 0, 2, 3, 5
STATE_RC = {"CALM_CONFIRMED": RC_PASS, "ASSERT_FAIL": RC_ASSERT,
            "UNSAMPLED": RC_INCONCLUSIVE, "INCONCLUSIVE": RC_INCONCLUSIVE}


def assert_calm_engaged(leg_dir):
    p = os.path.join(leg_dir, "sigma_gate.json")
    if not os.path.isfile(p):
        return RC_NEVER_RAN, "no sigma_gate.json at %s -- the gate did not run at all" % p
    try:
        d = json.load(open(p))
    except (OSError, ValueError) as e:
        return RC_NEVER_RAN, "sigma_gate.json unreadable (%s)" % e
    if not d.get("calm_commanded"):
        return (RC_NEVER_RAN,
                "sigma_gate.json carries NO calm_commanded flag: the calm tier never ran and "
                "this S0 leg was judged on tier 1 against band(0.00)=0.0000 -- i.e. UNGATED. "
                "Cause is almost always an unresolvable --paired-csv, which resolved_wind_max() "
                "swallows by returning None. state=%r rc=%r" % (d.get("state"), d.get("rc")))
    state = d.get("state")
    return (STATE_RC.get(state, RC_INCONCLUSIVE),
            "calm tier ENGAGED: state=%s peak_wind_ned=%s why=%s"
            % (state, d.get("peak_wind_ned"), d.get("why")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARMBLAD-025 S0 calm-engagement witness")
    ap.add_argument("--leg", required=True)
    a = ap.parse_args(argv)
    rc, why = assert_calm_engaged(a.leg)
    print("CALM WITNESS rc=%d: %s" % (rc, why))
    return rc


if __name__ == "__main__":
    sys.exit(main())
