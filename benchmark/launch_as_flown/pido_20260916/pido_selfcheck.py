#!/usr/bin/env python3
# PIDO TWIN of analysis/s1abl_20260915T0400Z/s1abl_selfcheck.py (parent NOT edited): coverage for pido_fire.sh,
#   pido_sig_gate.py, pido_manifest_check.py and pido_overlay_exam.py, driven with PIDO_FIXTURE=1 (stops after every gate,
#   before the launch line) on a throw-away copy of the staged S1 pair; spawns no SITL, no tmux, touches no box. Every limb
#   asserts ONE rc. Limbs:
#     F1 unknown RUN 15. F2 wrong pin 12. F3 census file absent 11. F4 tampered SHAS 13. F5 buffer reachable 13
#     F6 wrong dose 0.75 -> manifest 34 -> 14. F7a unsigned MUST be 16. F7b signed MUST be 0. F7c wrong sha 16
#     F7d bank absent 17. F8 absent prereg 16
#     F9a PIDO25: BOTH flags on the argv, ARMBLAD025. F9b PIDO35: BOTH flags, ARMBLAD035 (never 025)
#     F10/F11 resume. G1a-G1d manifest knob shapes {voff} / {scales} / {voff, scales 1.0} / null -> 36
#     G2 driver: undeclared knob 14. G3 one declared knob absent 14. G4 declared pair plus an extra 14
#     G5 driver: declared pair, clean argv, --dry-run through the parent -> rc 0, witness action_scales 0.0, both NEW lines
#        in the diff, REVERT WITNESS printed, config.py clean afterwards (this IS the overlay selftest for the new lever).
# PUBLIC API: [PIDO_DIR=<dispatch dir>] python pido_selfcheck.py (rc 0 = every limb passed)
# SIDE-EFFECTS: a temp dir, removed. Reads git HEAD. G5 patches and reverts src/drl/config.py through the parent's own
#   apply/revert (git checkout -- src/drl/config.py): config.py MUST be clean before running.
# SIZE: soft 80 / hard 100 code-lines; function cap 70.
import json, os, re, shutil, subprocess, sys, tempfile
FD = "/home/ubuntu/FlightDock"; B = os.environ.get("PIDO_DIR") or os.path.join(FD, "analysis/pido_20260916"); FAILS = []
T025 = "analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json"
BOTH = "--vert-offset-down-m 0.25 --action-scales 0.00"

def limb(tag, cond, msg):
    print(" %s %s: %s" % ("PASS" if cond else "FAIL", tag, msg)); FAILS.append(tag) if not cond else None

def main():
    td = tempfile.mkdtemp(); dd = os.path.join(td, "dd"); os.makedirs(dd)
    prot = os.path.join(td, "S1"); shutil.copytree(os.path.join(B, "stage/S1/protected"), prot)
    open(os.path.join(dd, "census_patterns.txt"), "w").write("zzz_none\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FD, capture_output=True, text=True).stdout.strip()
    bank = os.path.join(B, "PREREG_SHAS.json"); want = json.load(open(bank))["runs"]["ALL"]
    unsigned_text = re.sub(r"\nOPERATOR-SIGNATURE: PIDO-ALL-[^\n]*\n?$", "\nOPERATOR-SIGNATURE: PIDO-ALL-____ (unsigned)\n", open(os.path.join(B, "PREREG_PIDO.md")).read())
    real = os.path.join(td, "unsigned.md"); open(real, "w").write(unsigned_text)
    def signed_copy(path, sha):
        open(path, "w").write(unsigned_text + "\nOPERATOR-SIGNATURE: PIDO-ALL-20260917-fx -- signed, prereg sha %s\n" % sha)
    signed, wrong = os.path.join(td, "signed.md"), os.path.join(td, "wrong.md"); signed_copy(signed, want); signed_copy(wrong, "0" * 64)
    def fire(pin, run, prereg=signed, bank_path=bank, pr=None, dose="1.00"):
        env = dict(os.environ, LABEL="fx", PROT=pr or prot, DOSE=dose, PIN=pin, DD=dd, RUN=run,
                   PIDO_FIXTURE="1", PIDO_PREREG=prereg, PIDO_SHAS=bank_path, PIDO_DIR=B)
        r = subprocess.run(["bash", os.path.join(B, "pido_fire.sh")], cwd=FD, env=env, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip(); return r.returncode, out.splitlines()[-1] if out else ""
    limb("F1", fire(head, "S1A25")[0] == 15, "unknown RUN (an S1ABL token) -> 15")
    limb("F2", fire("deadbeef1", "PIDO25")[0] == 12, "wrong pin -> 12")
    os.rename(os.path.join(dd, "census_patterns.txt"), os.path.join(dd, "c.bak")); limb("F3", fire(head, "PIDO25")[0] == 11, "census file absent -> 11"); os.rename(os.path.join(dd, "c.bak"), os.path.join(dd, "census_patterns.txt"))
    sp = os.path.join(prot, "SHAS.txt"); os.chmod(sp, 0o644); good = open(sp).read(); open(sp, "w").write(good.replace("34c7f116", "00000000"))
    limb("F4", fire(head, "PIDO25")[0] == 13, "tampered SHAS -> 13"); open(sp, "w").write(good)
    open(os.path.join(prot, "replay_buffer_final.pkl"), "w").close(); limb("F5", fire(head, "PIDO25")[0] == 13, "replay buffer reachable -> 13"); os.remove(os.path.join(prot, "replay_buffer_final.pkl"))
    limb("F6", fire(head, "PIDO25", dose="0.75")[0] == 14, "PIDO25 at the WRONG dose 0.75 (signed 1.00) -> manifest 34 -> 14")
    limb("F7a", fire(head, "PIDO25", prereg=real)[0] == 16, "prereg WITHOUT its signature line (placeholder only) -> MUST be 16")
    limb("F7b", fire(head, "PIDO25")[0] == 0, "correctly signed, sha bound -> MUST be 0")
    limb("F7c", fire(head, "PIDO25", prereg=wrong)[0] == 16, "signed but WRONG sha -> MUST be 16")
    limb("F7d", fire(head, "PIDO25", bank_path=os.path.join(td, "nobank.json"))[0] == 17, "sha bank absent -> MUST be 17")
    limb("F8", fire(head, "PIDO25", prereg=os.path.join(td, "absent.md"))[0] == 16, "absent prereg -> 16")
    rc, last = fire(head, "PIDO25"); limb("F9a", rc == 0 and "KNOBFLAG=[%s]" % BOTH in last and last.count("--action-scales") == 2 and last.count("--vert-offset-down-m") == 2 and "--clamp" not in last and "ARMBLAD025.json" in last, "PIDO25: GATES PASS, BOTH flags, ARMBLAD025: %s" % last[-90:])
    rc, last = fire(head, "PIDO35"); limb("F9b", rc == 0 and "KNOBFLAG=[%s]" % BOTH in last and "ARMBLAD035.json" in last and "ARMBLAD025" not in last, "PIDO35: GATES PASS, BOTH flags, ARMBLAD035 only: %s" % last[-90:])
    fired = os.path.join(dd, "FIRED_PIDO.txt"); open(fired, "w").write("2026-09-17T00:00:00Z\n")
    rc, last = fire(head, "PIDO25"); limb("F10", rc == 0 and "already fired" in last, "FIRED present, no RESUME -> short-circuit rc 0")
    os.environ["PIDO_RESUME"] = "1"; rc, last = fire(head, "PIDO25"); os.environ.pop("PIDO_RESUME")
    limb("F11", rc == 0 and last.startswith("GATES PASS") and open(fired).read().startswith("2026-09-17"), "RESUME=1 -> gates re-run, FIRED not clobbered"); os.remove(fired)
    def mani(knob):
        m = json.load(open(os.path.join(B, "WAVE_MANIFEST_PIDO.json"))); m["runs"]["PIDO25"]["knob"] = knob; bad = os.path.join(td, "bad.json"); json.dump(m, open(bad, "w"))
        return subprocess.run(["python3", os.path.join(B, "pido_manifest_check.py"), "--manifest", bad, "--run", "PIDO25", "--ckpt", os.path.join(prot, "model_final.zip"), "--table", T025, "--dose", "1.0"], cwd=FD, capture_output=True, text=True).returncode
    for tag, knob in (("G1a", {"voff": 0.25}), ("G1b", {"scales": 0.0}), ("G1c", {"voff": 0.25, "scales": 1.0}), ("G1d", None)):
        limb(tag, mani(knob) == 36, "manifest knob %r (not the signed pair) -> 36" % (knob,))
    def ov(flag, argv):
        r = subprocess.run(["python3", os.path.join(B, "pido_overlay_exam.py")] + argv, cwd=FD, env=dict(os.environ, PIDO_KNOBFLAG=flag), capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()
    limb("G2", ov("", ["--dose", "1.0", "--action-scales", "0.0"])[0] == 14, "driver: undeclared knob on the argv -> 14")
    limb("G3", ov(BOTH, ["--dose", "1.0", "--vert-offset-down-m", "0.25"])[0] == 14, "driver: one declared knob ABSENT from the argv -> 14")
    limb("G4", ov(BOTH, ["--dose", "1.0", "--vert-offset-down-m", "0.25", "--action-scales", "0.0", "--beta", "0.5"])[0] == 14, "driver: declared pair plus an undeclared beta -> 14")
    out = os.path.join(td, "ovout"); ck = os.path.join(prot, "model_final.zip")
    rc, txt = ov(BOTH, ["--dose", "1.0", "--ckpt", ck, "--out-dir", out, "--table", T025, "--vert-offset-down-m", "0.25", "--action-scales", "0.0", "--dry-run"])
    w = json.load(open(os.path.join(out, "OVERLAY_WITNESS.json"))) if os.path.isfile(os.path.join(out, "OVERLAY_WITNESS.json")) else {}
    clean = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "src/"], cwd=FD, capture_output=True, text=True).stdout.strip() == ""
    limb("G5", rc == 0 and w.get("action_scales") == 0.0 and w.get("vert_offset_down_m") == 0.25 and "ACTION_SCALES: List[float] = [0.00, 0.00, 0.00]" in w.get("diff", "") and "VERT_OFFSET_DOWN_M: float = 0.25" in w.get("diff", "") and "REVERT WITNESS" in txt and clean,
         "driver: declared pair, dry-run through the parent -> rc %d, witness scales %s voff %s, both NEW lines in diff, reverted, config.py clean %s" % (rc, w.get("action_scales"), w.get("vert_offset_down_m"), clean))
    shutil.rmtree(td, ignore_errors=True)
    print("pido_selfcheck: %s (%d failing)" % ("PASS" if not FAILS else "FAIL " + ",".join(FAILS), len(FAILS))); return 0 if not FAILS else 1

if __name__ == "__main__":
    sys.exit(main())
