#!/usr/bin/env python3
# RESPONSIBILITY: run an 84-row exam with the CRRL altitude-descent floor opened to a declared
#   dose. The lever is priced: N5AUTH-805/N6AUTH-806 moved the MEDIAN TRUTH PEAK monotonically
#   1.85 -> 1.67 -> 1.34 m at floor 0.25 / 0.50 / 1.00 on a frozen policy, and peak height is what
#   makes a zone violation. Every candidate fails A4.2 limb 1, so this is the one measured lever of
#   the right sign. ONE config line changes, for the exam's duration, and is REVERTED with a
#   witness. The exam itself is the validated exam_preflight.sh + armbsig_exam_wave.py lineage,
#   argv unchanged -- no new launcher (RB-68: --width without --parallel runs SEQUENTIALLY).
# PUBLIC API: main(argv=None) -> int; exam_argv(...); write_witness(...). The declared-line patch
#   layer now lives in crrl_overlay_patch.py and is RE-EXPORTED here, so every existing caller and
#   selftest limb that reaches for ov.OLD / ov.apply_patch / ov.revert_patch still works.
# SIDE-EFFECTS: rewrites ONE line of src/drl/config.py on this box PER ELECTED LEVER (up to four:
#   --dose always, plus --beta / --clamp-behind-positive-ms / --forward-bias-ms), then reverts
#   it; writes OVERLAY_WITNESS.json into --out-dir; spawns the exam. Draws no seed, charges no
#   ledger, trains nothing, and never writes to the checkpoint it reads.
# SATREAD-821 (2026-08-28) added TWO CLI levers, both default-OFF and byte-identical when unset:
#   --table forwards the seed table exam_argv already accepted but main() never passed. Without it
#            an overlay exam could only fly the DEFAULT grid; the H3 re-fly needs grid 2.
#   --beta patches a SECOND declared line, CRRL_BPC_BETA. Where the floor is zero -- the climb
#            side, and both signs of heading -- beta is the ENTIRE bound, so it is an independent
#            lever. assert_only_declared demands len(_pairs) insertions/deletions, not a ternary.
# KNOBSWEEP-20260901 added the TWO LONGITUDINAL levers, both default-OFF and byte-identical when
#   unset. They exist because the shipped accelerating clamp (+0.10) was ELECTED as a bracket
#   MIDPOINT in 2026-04 and NEVER MEASURED -- the sweep that would have measured it was specified
#   (C3X_PRECOMMIT_SYNTHESIS.md) and cancelled on a 2-3 day compute cost that is now ~85 min/point.
#   --clamp-behind-positive-ms the accelerating ceiling; read per CALL, no gate, no smoothing.
# VERTOFF-20260902 added --vert-offset-down-m: the VERTICAL lever, gated tier >= 2 through the
#   SAME EMA as the bias; a wave that moves no held point did not fire it (INCONCLUSIVE).
#   --forward-bias-ms the forward bias; read per TICK, gated on milestone tier >= 2 and
#                               delivered through an EMA, so it is NOT instantaneous. A wave that
#                               moves no held point did not fire it: INCONCLUSIVE, never GREEN.
# rc: 2 checkpoint absent - 3 relay ledger absent - 4 seed table absent (all BEFORE any patch).
# SIZE: soft 70 / hard 95 code-lines; function cap 30. The hard cap was briefly moved to 110 for
#   the two-lever change and is RESTORED here by SPLITTING the patch layer out, not by raising it.
#   SOFT OVERAGE DISCLOSED, NOT RAISED: measures 76 against the soft 70. Hard cap intact. main()
#   likewise SPLIT (refuse_before_patch) when it crossed the function cap, not raised.
# COVERAGE: crrl_overlay_exam_selftest.py (O1-O7 unchanged, O8-O14 added)
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crrl_overlay_cli import build_parser                               # noqa: E402
from crrl_overlay_patch import (                                        # noqa: E402
    FD, PATCH_FILE, OLD, NEW, OLD_BETA, NEW_BETA, OLD_CLAMP, NEW_CLAMP, OLD_BIAS, NEW_BIAS,
    sh, declared_new, declared_new_beta, declared_new_clamp, declared_new_bias,
    declared_new_voff, apply_patch, assert_only_declared, revert_patch)

PREFLIGHT = os.path.join(FD, "analysis/armbsig_20260824T0030Z/exam_preflight.sh")

# The relay ledger the wave measures seed custody against. NOT the wave's default path, which
# does not exist on any box: this is the copy Candidate-S's own S-B exam used, so the overlay
# corpora and the S-B corpus they are compared with are custodied against the SAME ledger.
LEDGER = os.path.join(FD, "analysis/c006_campaign_20260728T110000Z/logs_w1418/"
                          "gate_relay_ledger.CPRIME_COPY.jsonl")


def exam_argv(ckpt, out_dir, width, ledger=None, table=None):
    """RB-68: --width WITHOUT --parallel runs the rows SEQUENTIALLY (28 h instead of 85 min).
    The ledger must be NAMED: the wave refuses a live run it cannot measure seed custody for."""
    a = [PREFLIGHT, "--ckpt", ckpt, "--out-dir", out_dir, "--width", str(width), "--parallel",
         "--ledger", ledger or LEDGER]
    return a + (["--table", table] if table else [])


def write_witness(a, diff, cmd):
    """The dose, the exact diff, the argv, the policy sha and the pin -- so a corpus can never be
    read without knowing which dose produced it."""
    json.dump({"dose": a.dose, "declared_new": declared_new(a.dose), "diff": diff,
               "beta": a.beta,
               "declared_new_beta": None if a.beta is None else declared_new_beta(a.beta),
               "clamp_behind_positive_ms": a.clamp_behind_positive_ms,
               "forward_bias_ms": a.forward_bias_ms,
               "declared_new_clamp": None if a.clamp_behind_positive_ms is None
               else declared_new_clamp(a.clamp_behind_positive_ms),
               "declared_new_bias": None if a.forward_bias_ms is None
               else declared_new_bias(a.forward_bias_ms),
               "vert_offset_down_m": a.vert_offset_down_m,
               "declared_new_voff": None if a.vert_offset_down_m is None
               else declared_new_voff(a.vert_offset_down_m),
               "table": a.table,
               "table_sha": None if not a.table else sh("sha256sum", a.table).split()[0],
               "argv": cmd, "ckpt_sha": sh("sha256sum", a.ckpt).split()[0],
               "ledger": a.ledger, "pin": sh("git", "-C", FD, "rev-parse", "HEAD"),
               "lever": "N5AUTH-805/N6AUTH-806 median truth peak 1.85/1.67/1.34 m at floor "
                        "0.25/0.50/1.00 on a frozen policy"},
              open(os.path.join(a.out_dir, "OVERLAY_WITNESS.json"), "w"), indent=1)


def refuse_before_patch(a):
    """Every gate that must fire BEFORE the tree is edited, in order, or 0. Measured on eval3,
    whose copy of Candidate-S's policy did not exist: the run died with an IndexError inside the
    witness, MID-PATCH. A refusal after the edit leaves a dirty tree behind."""
    if not os.path.isfile(a.ckpt):
        print("REFUSING: checkpoint absent at %s -- an exam cannot score a policy that is not "
              "on this box" % a.ckpt, file=sys.stderr)
        return 2
    os.makedirs(a.out_dir, exist_ok=True)
    if not os.path.isfile(a.ledger):
        print("REFUSING: relay ledger absent at %s -- seed custody cannot be MEASURED, so a live "
              "wave would be refused anyway" % a.ledger, file=sys.stderr)
        return 3
    if a.table is not None and not os.path.isfile(a.table):
        print("REFUSING: seed table absent at %s -- an exam that silently fell back to the "
              "default grid would be scored against the wrong venue" % a.table, file=sys.stderr)
        return 4
    return 0


def main(argv=None):
    a = build_parser(LEDGER).parse_args(argv)
    rc = refuse_before_patch(a)
    if rc:
        return rc
    kn = {"beta": a.beta, "clamp": a.clamp_behind_positive_ms, "bias": a.forward_bias_ms,
          "voff": a.vert_offset_down_m}
    apply_patch(a.dose, **kn)
    try:
        diff = assert_only_declared(a.dose, **kn)
        cmd = exam_argv(a.ckpt, a.out_dir, a.width, a.ledger, a.table)
        write_witness(a, diff, cmd)
        print("OVERLAY dose=%.2f beta=%s clamp=%s bias=%s voff=%s table=%s\n%s\nARGV: %s"
              % (a.dose, a.beta, a.clamp_behind_positive_ms, a.forward_bias_ms,
                 a.vert_offset_down_m, a.table, diff, " ".join(cmd)), flush=True)
        rc = 0 if a.dry_run else subprocess.run(cmd, cwd=FD).returncode
    finally:
        print(revert_patch(), flush=True)
    print("EXAM_RC=%d" % rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
