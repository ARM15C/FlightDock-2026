#!/usr/bin/env python3
# RESPONSIBILITY: the fire-script limbs (F1-F8) of the BENCH600 selfcheck, SPLIT out of
#   bench600_selfcheck.py when that file measured 145 code-lines against its declared hard 140 --
#   a cap is split, never raised. Drives bench600_fire.sh with BENCH600_FIXTURE=1 (stops after every
#   gate, before the launch line) on a throw-away copy of the vault pair; spawns no SITL, no tmux.
#   2026-09-02 REBUILD: F7 used to accept `rc in (16, 0)` -- "either is the truth of the file". That is
#   not a detector, and it is exactly why the first D7 gate shipped as a no-op: the unsigned placeholder
#   matched its grep and the limb could not tell. F7 is now FOUR strict limbs (F7a unsigned MUST be 16,
#   F7b correctly signed MUST be 0, F7c wrong sha MUST be 16, F7d absent sha bank MUST be 17) and D8
#   resume gets F10/F11. No limb here may accept a range.
# PUBLIC API: fire_limbs(limb) -> None (limb = the parent's recorder: limb(tag, cond, msg))
# SIDE-EFFECTS: a temp dir, removed. Reads git HEAD. Nothing else.
# SIZE: soft 40 / hard 55 code-lines; function cap 40.
import importlib.util
import os
import shutil
import subprocess
import tempfile

FD = "/home/ubuntu/FlightDock"
B = os.path.join(FD, "analysis/bench600_20260902T0310Z")
PROT = os.path.join(FD, "analysis/b25r_20260830T0040Z/stage/A2/protected")


def _sig():
    spec = importlib.util.spec_from_file_location("sig", os.path.join(B, "bench600_selfcheck_sig.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fire_limbs(limb):
    td = tempfile.mkdtemp()
    prot, dd = os.path.join(td, "prot"), os.path.join(td, "dd")
    shutil.copytree(PROT, prot)
    os.makedirs(dd)
    open(os.path.join(dd, "census_patterns.txt"), "w").write("zzz_no_such_pattern_zzz\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FD, capture_output=True, text=True).stdout.strip()

    real = os.path.join(B, "PREREG_BENCH600_H1H2.md")        # the REAL one, unsigned until the operator signs
    bank = os.path.join(B, "PREREG_SHAS.json")
    signed = os.path.join(td, "signed.md")                   # written by the sibling before any limb runs
    _sig().write_signed(signed, real, bank)

    def fire(pin, run, prereg=signed, bank_path=bank):
        env = dict(os.environ, LABEL="fx", PROT=prot, DOSE="0.75", PIN=pin, DD=dd, RUN=run, BENCH600_FIXTURE="1",
                   BENCH600_PREREG=prereg, BENCH600_SHAS=bank_path)
        r = subprocess.run(["bash", os.path.join(B, "bench600_fire.sh")], cwd=FD, env=env, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        return r.returncode, out.splitlines()[-1] if out else ""
    limb("F1", fire(head, "H9")[0] == 15, "unknown RUN -> 15")
    limb("F2", fire("deadbeef1", "H1H2")[0] == 12, "wrong pin -> 12")
    os.rename(os.path.join(dd, "census_patterns.txt"), os.path.join(dd, "c.bak"))
    limb("F3", fire(head, "H1H2")[0] == 11, "census pattern file absent -> 11")
    os.rename(os.path.join(dd, "c.bak"), os.path.join(dd, "census_patterns.txt"))
    os.chmod(os.path.join(prot, "SHAS.txt"), 0o644)
    good = open(os.path.join(prot, "SHAS.txt")).read()
    open(os.path.join(prot, "SHAS.txt"), "w").write(good.replace("3d1cf6ca", "00000000"))
    limb("F4", fire(head, "H1H2")[0] == 13, "tampered SHAS -> 13")
    open(os.path.join(prot, "SHAS.txt"), "w").write(good)
    open(os.path.join(prot, "replay_buffer_final.pkl"), "w").close()
    limb("F5", fire(head, "H1H2")[0] == 13, "replay buffer reachable -> 13")
    os.remove(os.path.join(prot, "replay_buffer_final.pkl"))
    rc, last = fire(head, "H1H2")
    limb("F6", rc == 0 and last.startswith("GATES PASS") and not os.path.exists(os.path.join(dd, "FIRED_BENCH600.txt")),
         "all gates pass -> fixture stop before launch, no FIRED marker")
    _sig().sig_limbs(limb, lambda **kw: fire(head, "H1H2", **kw), td, dd, real, bank)   # F7a-d, F8, F10, F11
    shutil.rmtree(td, ignore_errors=True)
