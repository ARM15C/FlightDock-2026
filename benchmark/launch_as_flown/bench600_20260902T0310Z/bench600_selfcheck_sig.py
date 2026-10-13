#!/usr/bin/env python3
# RESPONSIBILITY: the D7 SIGNATURE and D8 RESUME limbs of the BENCH600 selfcheck, SPLIT out of
#   bench600_selfcheck_fire.py when adding them pushed that file to 63 code-lines against its
#   declared hard 55. A cap is SPLIT, never raised -- the third time this rule has bound in this
#   dispatch. These limbs exist because the first D7 gate shipped as a NO-OP: it grepped for the
#   run prefix, the unsigned placeholder "BENCH600-H1H2- ____" matched, and its single fixture
#   accepted `rc in (16, 0)` so it could not tell. Every limb here asserts ONE exact rc.
# PUBLIC API: write_signed(path, real, bank, sha=None) -> None; sig_limbs(limb, fire, td, dd, real, bank)
#   limb = the parent's recorder; fire(prereg=, bank_path=) -> (rc, last_line), the sibling's driver
#   with pin and RUN already bound; td = a temp dir; dd = the fixture dispatch dir; real/bank = the
#   REAL prereg and the banked pre-signature shas.
# SIDE-EFFECTS: writes fixture preregs into td and one FIRED marker into dd, removed. No SITL.
# SIZE: soft 30 / hard 45 code-lines; function cap 30.
# COVERAGE: it IS the coverage; driven by bench600_selfcheck.py on every run.
import json
import os


def write_signed(path, real, bank, sha=None):
    """A copy of the REAL prereg plus one signature line. sha=None means the correctly banked one."""
    want = sha or json.load(open(bank))["runs"]["H1H2"]
    body = open(real, encoding="utf-8").read()
    with open(path, "w") as fh:
        fh.write(body + "\nOPERATOR-SIGNATURE: BENCH600-H1H2-20260902-fx -- signed, prereg sha %s\n" % want)


def sig_limbs(limb, fire, td, dd, real, bank):
    signed, wrong = os.path.join(td, "signed.md"), os.path.join(td, "wrongsha.md")
    write_signed(signed, real, bank)
    write_signed(wrong, real, bank, sha="0" * 64)               # signed, names a sha that is not this run's
    limb("F7a", fire(prereg=real)[0] == 16,
         "the REAL prereg, UNSIGNED placeholder -> MUST be 16 (the no-op grep returned 0 here)")
    limb("F7b", fire(prereg=signed)[0] == 0, "correctly signed, sha bound -> MUST be 0")
    limb("F7c", fire(prereg=wrong)[0] == 16, "signed but naming the WRONG sha -> MUST be 16")
    limb("F7d", fire(prereg=signed, bank_path=os.path.join(td, "no_bank.json"))[0] == 17,
         "sha bank absent -> MUST be 17 (an unverifiable signature is not a signature)")
    limb("F8", fire(prereg=os.path.join(td, "absent.md"))[0] == 16, "absent prereg -> 16")
    fired = os.path.join(dd, "FIRED_BENCH600.txt")
    open(fired, "w").write("2026-09-02T00:00:00Z\n")
    rc, last = fire(prereg=signed)
    limb("F10", rc == 0 and "already fired" in last, "FIRED marker present, no RESUME -> short-circuit rc 0")
    os.environ["BENCH600_RESUME"] = "1"
    rc, last = fire(prereg=signed)
    os.environ.pop("BENCH600_RESUME")
    limb("F11", rc == 0 and last.startswith("GATES PASS") and open(fired).read().startswith("2026-09-02"),
         "RESUME=1 -> every gate re-run, FIRED marker NOT clobbered, out-dir reused")
    os.remove(fired)
