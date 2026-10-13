#!/usr/bin/env python3
# RESPONSIBILITY: coverage for cs600_fire.sh, cs600_manifest_check.py and cs600_overlay_exam.py, driven on dev with
#   CS600_FIXTURE=1 (stops after every gate, before the launch line) on throw-away copies of BOTH staged pairs; spawns
#   no SITL, no tmux, touches no box. Every limb asserts ONE rc. Limbs:
#     F1 unknown RUN 15. F2 wrong pin 12. F3 census file absent 11. F4 tampered SHAS 13. F5 buffer reachable 13
#     F6 CSV's pair under RUN=CSP -> manifest 32 -> 14. F7a unsigned MUST be 16. F7b signed MUST be 0. F7c wrong sha 16
#     F7d bank absent 17. F8 absent prereg 16. F9a CSP passes with an EMPTY knob flag and no knob token on the argv
#     F9b CSV passes with EXACTLY --vert-offset-down-m 0.25 and no clamp/bias token. F10/F11 resume (parent's D8)
#     F9c CSC forwards exactly clamp 0.50. F9d CSC at the wrong dose -> 14. F9e wrong pair for CSC -> 14
#     G1 manifest row with an UNSIGNED knob value -> 36. G2 overlay: undeclared knob on argv -> 14. G3 declared flag absent -> 14
# PUBLIC API: python cs600_selfcheck.py (rc 0 = every limb passed)
# SIDE-EFFECTS: a temp dir, removed. Reads git HEAD. Nothing else.
# SIZE: soft 70 / hard 90 code-lines; function cap 60.
import json, os, shutil, subprocess, sys, tempfile
FD = "/home/ubuntu/FlightDock"; B = os.path.join(FD, "analysis/cs600_20260903T0230Z"); FAILS = []

def limb(tag, cond, msg):
    print(" %s %s: %s" % ("PASS" if cond else "FAIL", tag, msg)); FAILS.append(tag) if not cond else None

def main():
    td = tempfile.mkdtemp(); dd = os.path.join(td, "dd"); os.makedirs(dd)
    prot = {r: os.path.join(td, r) for r in ("CSP", "CSV", "CSC")}
    for r, sub in (("CSP", "CANDS"), ("CSV", "CANDSSW1"), ("CSC", "A2AL2")):
        shutil.copytree(os.path.join(B, "stage/%s/protected" % sub), prot[r])
    open(os.path.join(dd, "census_patterns.txt"), "w").write("zzz_none\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FD, capture_output=True, text=True).stdout.strip()
    real, bank = os.path.join(B, "PREREG_CS600.md"), os.path.join(B, "PREREG_SHAS.json")
    want = json.load(open(bank))["runs"]["BOTH"]
    def signed_copy(path, sha):
        open(path, "w").write(open(real).read() + "\nOPERATOR-SIGNATURE: CS600-BOTH-20260903-fx -- signed, prereg sha %s\n" % sha)
    signed, wrong = os.path.join(td, "signed.md"), os.path.join(td, "wrong.md"); signed_copy(signed, want); signed_copy(wrong, "0" * 64)
    DOSE = {"CSP": "1.00", "CSV": "1.00", "CSC": "0.75"}
    def fire(pin, run, prereg=signed, bank_path=bank, pr=None, dose=None):
        env = dict(os.environ, LABEL="fx", PROT=pr or prot[run if run in prot else "CSP"], DOSE=dose or DOSE.get(run, "1.00"), PIN=pin, DD=dd, RUN=run,
                   CS600_FIXTURE="1", CS600_PREREG=prereg, CS600_SHAS=bank_path)
        r = subprocess.run(["bash", os.path.join(B, "cs600_fire.sh")], cwd=FD, env=env, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip(); return r.returncode, out.splitlines()[-1] if out else ""
    limb("F1", fire(head, "H1H2")[0] == 15, "unknown RUN -> 15")
    limb("F2", fire("deadbeef1", "CSP")[0] == 12, "wrong pin -> 12")
    os.rename(os.path.join(dd, "census_patterns.txt"), os.path.join(dd, "c.bak")); limb("F3", fire(head, "CSP")[0] == 11, "census file absent -> 11"); os.rename(os.path.join(dd, "c.bak"), os.path.join(dd, "census_patterns.txt"))
    sp = os.path.join(prot["CSP"], "SHAS.txt"); os.chmod(sp, 0o644); good = open(sp).read(); open(sp, "w").write(good.replace("1c10887a", "00000000"))
    limb("F4", fire(head, "CSP")[0] == 13, "tampered SHAS -> 13"); open(sp, "w").write(good)
    open(os.path.join(prot["CSP"], "replay_buffer_final.pkl"), "w").close(); limb("F5", fire(head, "CSP")[0] == 13, "replay buffer reachable -> 13"); os.remove(os.path.join(prot["CSP"], "replay_buffer_final.pkl"))
    limb("F6", fire(head, "CSP", pr=prot["CSV"])[0] == 14, "CSV's pair offered as RUN=CSP -> manifest 32 -> 14")
    limb("F7a", fire(head, "CSP", prereg=real)[0] == 16, "REAL prereg, unsigned placeholder -> MUST be 16")
    limb("F7b", fire(head, "CSP")[0] == 0, "correctly signed, sha bound -> MUST be 0")
    limb("F7c", fire(head, "CSP", prereg=wrong)[0] == 16, "signed but WRONG sha -> MUST be 16")
    limb("F7d", fire(head, "CSP", bank_path=os.path.join(td, "nobank.json"))[0] == 17, "sha bank absent -> MUST be 17")
    limb("F8", fire(head, "CSP", prereg=os.path.join(td, "absent.md"))[0] == 16, "absent prereg -> 16")
    rc, last = fire(head, "CSP"); limb("F9a", rc == 0 and "KNOBFLAG=[]" in last and "--vert-offset" not in last and "--clamp" not in last, "CSP: GATES PASS, EMPTY knob flag, no knob token: %s" % last[-60:])
    rc, last = fire(head, "CSV"); limb("F9b", rc == 0 and "KNOBFLAG=[--vert-offset-down-m 0.25]" in last and last.count("--vert-offset-down-m") == 2 and "--clamp" not in last and "--forward-bias" not in last, "CSV: GATES PASS, exactly voff 0.25 on the argv")
    rc, last = fire(head, "CSC")
    limb("F9c", rc == 0 and "KNOBFLAG=[--clamp-behind-positive-ms 0.50]" in last and last.count("--clamp-behind-positive-ms") == 2 and "--vert-offset" not in last, "CSC: GATES PASS, exactly clamp 0.50 on the argv, no offset token")
    limb("F9d", fire(head, "CSC", dose="1.00")[0] == 14, "CSC at the WRONG dose 1.00 (signed 0.75) -> manifest 34 -> 14")
    limb("F9e", fire(head, "CSC", pr=prot["CSV"])[0] == 14, "CSV's pair offered as RUN=CSC -> manifest 32 -> 14")
    fired = os.path.join(dd, "FIRED_CS600.txt"); open(fired, "w").write("2026-09-03T00:00:00Z\n")
    rc, last = fire(head, "CSP"); limb("F10", rc == 0 and "already fired" in last, "FIRED present, no RESUME -> short-circuit rc 0")
    os.environ["CS600_RESUME"] = "1"; rc, last = fire(head, "CSP"); os.environ.pop("CS600_RESUME")
    limb("F11", rc == 0 and last.startswith("GATES PASS") and open(fired).read().startswith("2026-09-03"), "RESUME=1 -> gates re-run, FIRED not clobbered"); os.remove(fired)
    m = json.load(open(os.path.join(B, "WAVE_MANIFEST_CS600.json"))); m["runs"]["CSV"]["knob"] = {"clamp": 0.25}; bad = os.path.join(td, "bad.json"); json.dump(m, open(bad, "w"))
    r = subprocess.run(["python3", os.path.join(B, "cs600_manifest_check.py"), "--manifest", bad, "--run", "CSV", "--ckpt", os.path.join(prot["CSV"], "model_final.zip"), "--table", "analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json", "--dose", "1.0"], cwd=FD, capture_output=True, text=True)
    limb("G1", r.returncode == 36, "manifest row carrying an UNSIGNED knob value (clamp 0.25) -> 36")
    def ov(flag, argv):
        r = subprocess.run(["python3", os.path.join(B, "cs600_overlay_exam.py")] + argv, cwd=FD, env=dict(os.environ, CS600_KNOBFLAG=flag), capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()[-90:]
    limb("G2", ov("", ["--dose", "1.0", "--vert-offset-down-m", "0.25"])[0] == 14, "overlay: undeclared knob on the argv -> 14")
    limb("G3", ov("--vert-offset-down-m 0.25", ["--dose", "1.0"])[0] == 14, "overlay: declared flag ABSENT from the argv -> 14")
    limb("G4", ov("--vert-offset-down-m 0.25", ["--dose", "1.0", "--clamp-behind-positive-ms", "0.50", "--vert-offset-down-m", "0.25"])[0] == 14, "overlay: declared voff plus an undeclared clamp -> 14")
    shutil.rmtree(td, ignore_errors=True)
    print("cs600_selfcheck: %s (%d failing)" % ("PASS" if not FAILS else "FAIL " + ",".join(FAILS), len(FAILS))); return 0 if not FAILS else 1

if __name__ == "__main__":
    sys.exit(main())
