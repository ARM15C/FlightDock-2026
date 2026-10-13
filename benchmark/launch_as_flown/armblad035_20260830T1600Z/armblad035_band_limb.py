#!/usr/bin/env python3
"""armblad035_band_limb -- S10: the block-C realized-sigma band ruling, driven BOTH ways.

RESPONSIBILITY
  This limb has NO PARENT and exists because of a measured contradiction, not a hypothetical.
  The band rule is `2.3877 * 0.04015 * (sigma / 0.30)`. At the parent's sigma 0.25 that is
  0.0799 -- tighter than A-10's signed 0.10. At THIS campaign's sigma 0.35 it is 0.1118 --
  LOOSER. Two source comments (armbsig_driver.sh:382, wave_sigma_hook.py:27) assert it is
  "STRICTLY TIGHTER ... at every sigma we fly, so it can only refuse MORE legs, never
  manufacture a pass". True for grid 1; never updated when grid 2 flew above 0.30.

  So there is no defensible default and the campaign may not pick one silently. This limb
  proves the gate is a real choice: SCALED and CLAMPED must yield DIFFERENT bands, and a
  clause naming NEITHER must REFUSE. A gate that accepts "something is present" would let an
  unruled campaign fly under a band nobody chose.

  Split from armblad035_sig_limbs.py at 93 code-lines against a hard 90 -- caps are SPLIT,
  never raised (the parent's sig_limbs -> selfcheck precedent).
PUBLIC API: s10_band_ruling_is_a_real_choice(td) -> (id, ok, detail)
SIDE-EFFECTS: writes ONLY inside the caller-supplied tmp dir. Spawns nothing, flies nothing.
CAPS: soft 50 / hard 65 code-lines; function cap 30. DISCLOSED: the cap is CHOSEN here at
  file creation from the measured 49 code-lines -- a cap being SET, not an established cap
  being raised (the parent's baseline025_seed_cli precedent). No existing cap moved.
COVERAGE: registered by armblad035_selfcheck.all_limbs(), which the unit test executes.
NOT A TRUTH PATH: imports os/sys and this campaign's own gate module only.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import armblad035_sig_gates as sg                                          # noqa: E402

HEAD = "# doc\nOPERATOR-SIGNATURE: ARMBLAD035-20260827 - signed\n"
SCALED = "OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-20260827 - SCALED 0.1118\n"
CLAMPED = "OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-20260827 - CLAMPED 0.10\n"
VAGUE = "OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-20260827 - I rule the band\n"


def _doc(td, name, body):
    p = os.path.join(td, name)
    open(p, "w").write(body)
    return p


def s10_band_ruling_is_a_real_choice(td):
    """SCALED and CLAMPED must be DIFFERENT answers, and neither may be assumed."""
    def _refuses(fn):
        try:
            fn()
        except SystemExit:
            return True
        return False
    sc = sg.assert_sigma_band_ruled(_doc(td, "sc.md", HEAD + SCALED))
    cl = sg.assert_sigma_band_ruled(_doc(td, "cl.md", HEAD + CLAMPED))
    vague = _doc(td, "vague.md", HEAD + VAGUE)
    return ("S10_band_ruling_BOTH_ways_and_refuses_neither",
            sc["mode"] == "SCALED" and abs(sc["band"] - 0.1118) < 1e-9
            and cl["mode"] == "CLAMPED" and abs(cl["band"] - 0.10) < 1e-9
            and _refuses(lambda: sg.assert_sigma_band_ruled(vague)),
            "SCALED->0.1118, CLAMPED->0.1000, a clause naming NEITHER REFUSES")
