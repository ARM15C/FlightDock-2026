#!/usr/bin/env python3
# S1ABL TWIN of analysis/cs600_20260903T0230Z/cs600_selfcheck.py (parent NOT edited): coverage for s1abl_fire.sh and
#   s1abl_sig_gate.py, driven with S1ABL_FIXTURE=1 (stops after every gate, before the launch line) on throw-away copies of
#   BOTH staged pairs; spawns no SITL, no tmux, touches no box. Every limb asserts ONE rc. Limbs:
#     F1 unknown RUN 15. F2 wrong pin 12. F3 census file absent 11. F4 tampered SHAS 13. F5 buffer reachable 13
#     F6 A2's pair under RUN=S1A25 -> manifest 32 -> 14. F7a unsigned MUST be 16. F7b signed MUST be 0. F7c wrong sha 16
#     F7d bank absent 17. F8 absent prereg 16
#     F9a S1A25: EMPTY knob flag, no knob token, ARMBLAD025 on the argv. F9b A2V25: EXACTLY --vert-offset-down-m 0.25, ARMBLAD025
#     F9c S1A35: empty flag, ARMBLAD035 (never 025). F9d A2V35: voff 0.25, ARMBLAD035. F9e A2V25 at the wrong dose 1.00 -> 34 -> 14
#     F9f S1's pair under RUN=A2V35 -> 32 -> 14. F10/F11 resume (parent's D8). G1 manifest row with an UNSIGNED knob -> 36
#   The overlay driver's own limbs (undeclared / absent / extra knob -> 14) belong to the parent, reused by path, not re-run here.
# PUBLIC API: [S1ABL_DIR=<dispatch dir>] python s1abl_selfcheck.py (rc 0 = every limb passed)
# SIDE-EFFECTS: a temp dir, removed. Reads git HEAD. Nothing else.
# SIZE: soft 70 / hard 90 code-lines; function cap 60.
import json, os, re, shutil, subprocess, sys, tempfile
FD = "/home/ubuntu/FlightDock"; B = os.environ.get("S1ABL_DIR") or os.path.join(FD, "analysis/s1abl_20260915T0400Z"); FAILS = []
CS = os.path.join(FD, "analysis/cs600_20260903T0230Z")
T025 = "analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json"

def limb(tag, cond, msg):
    print(" %s %s: %s" % ("PASS" if cond else "FAIL", tag, msg)); FAILS.append(tag) if not cond else None

def main():
    td = tempfile.mkdtemp(); dd = os.path.join(td, "dd"); os.makedirs(dd)
    prot = {st: os.path.join(td, st) for st in ("S1", "A2")}
    for st in prot:
        shutil.copytree(os.path.join(B, "stage/%s/protected" % st), prot[st])
    open(os.path.join(dd, "census_patterns.txt"), "w").write("zzz_none\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FD, capture_output=True, text=True).stdout.strip()
    bank = os.path.join(B, "PREREG_SHAS.json"); want = json.load(open(bank))["runs"]["ALL"]
    unsigned_text = re.sub(r"\n\nOPERATOR-SIGNATURE: S1ABL-ALL-[^\n]*\n?$", "\n", open(os.path.join(B, "PREREG_S1ABL.md")).read())
    real = os.path.join(td, "unsigned.md"); open(real, "w").write(unsigned_text)   # the prereg WITHOUT its signature line (valid before and after signing)
    def signed_copy(path, sha):
        open(path, "w").write(unsigned_text + "\nOPERATOR-SIGNATURE: S1ABL-ALL-20260915-fx -- signed, prereg sha %s\n" % sha)
    signed, wrong = os.path.join(td, "signed.md"), os.path.join(td, "wrong.md"); signed_copy(signed, want); signed_copy(wrong, "0" * 64)
    POL = {"S1A25": "S1", "S1A35": "S1", "A2V25": "A2", "A2V35": "A2"}; DOSE = {"S1": "1.00", "A2": "0.75"}
    def fire(pin, run, prereg=signed, bank_path=bank, pr=None, dose=None):
        st = POL.get(run, "S1")
        env = dict(os.environ, LABEL="fx", PROT=pr or prot[st], DOSE=dose or DOSE[st], PIN=pin, DD=dd, RUN=run,
                   S1ABL_FIXTURE="1", S1ABL_PREREG=prereg, S1ABL_SHAS=bank_path, S1ABL_DIR=B)
        r = subprocess.run(["bash", os.path.join(B, "s1abl_fire.sh")], cwd=FD, env=env, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip(); return r.returncode, out.splitlines()[-1] if out else ""
    limb("F1", fire(head, "CSV")[0] == 15, "unknown RUN (a CS600 token) -> 15")
    limb("F2", fire("deadbeef1", "S1A25")[0] == 12, "wrong pin -> 12")
    os.rename(os.path.join(dd, "census_patterns.txt"), os.path.join(dd, "c.bak")); limb("F3", fire(head, "S1A25")[0] == 11, "census file absent -> 11"); os.rename(os.path.join(dd, "c.bak"), os.path.join(dd, "census_patterns.txt"))
    sp = os.path.join(prot["S1"], "SHAS.txt"); os.chmod(sp, 0o644); good = open(sp).read(); open(sp, "w").write(good.replace("34c7f116", "00000000"))
    limb("F4", fire(head, "S1A25")[0] == 13, "tampered SHAS -> 13"); open(sp, "w").write(good)
    open(os.path.join(prot["S1"], "replay_buffer_final.pkl"), "w").close(); limb("F5", fire(head, "S1A25")[0] == 13, "replay buffer reachable -> 13"); os.remove(os.path.join(prot["S1"], "replay_buffer_final.pkl"))
    limb("F6", fire(head, "S1A25", pr=prot["A2"], dose="1.00")[0] == 14, "A2's pair offered as RUN=S1A25 -> manifest 32 -> 14")
    limb("F7a", fire(head, "S1A25", prereg=real)[0] == 16, "prereg WITHOUT its signature line (placeholder only) -> MUST be 16")
    limb("F7b", fire(head, "S1A25")[0] == 0, "correctly signed, sha bound -> MUST be 0")
    limb("F7c", fire(head, "S1A25", prereg=wrong)[0] == 16, "signed but WRONG sha -> MUST be 16")
    limb("F7d", fire(head, "S1A25", bank_path=os.path.join(td, "nobank.json"))[0] == 17, "sha bank absent -> MUST be 17")
    limb("F8", fire(head, "S1A25", prereg=os.path.join(td, "absent.md"))[0] == 16, "absent prereg -> 16")
    rc, last = fire(head, "S1A25"); limb("F9a", rc == 0 and "KNOBFLAG=[]" in last and "--vert-offset" not in last and "--clamp" not in last and "ARMBLAD025.json" in last, "S1A25: GATES PASS, EMPTY knob flag, ARMBLAD025: %s" % last[-70:])
    rc, last = fire(head, "A2V25"); limb("F9b", rc == 0 and "KNOBFLAG=[--vert-offset-down-m 0.25]" in last and last.count("--vert-offset-down-m") == 2 and "--clamp" not in last and "ARMBLAD025.json" in last, "A2V25: GATES PASS, exactly voff 0.25, ARMBLAD025: %s" % last[-70:])
    rc, last = fire(head, "S1A35"); limb("F9c", rc == 0 and "KNOBFLAG=[]" in last and "ARMBLAD035.json" in last and "ARMBLAD025" not in last, "S1A35: GATES PASS, empty flag, ARMBLAD035 only: %s" % last[-70:])
    rc, last = fire(head, "A2V35"); limb("F9d", rc == 0 and "KNOBFLAG=[--vert-offset-down-m 0.25]" in last and "ARMBLAD035.json" in last and "ARMBLAD025" not in last, "A2V35: GATES PASS, voff 0.25, ARMBLAD035 only: %s" % last[-70:])
    limb("F9e", fire(head, "A2V25", dose="1.00")[0] == 14, "A2V25 at the WRONG dose 1.00 (signed 0.75) -> manifest 34 -> 14")
    limb("F9f", fire(head, "A2V35", pr=prot["S1"], dose="0.75")[0] == 14, "S1's pair offered as RUN=A2V35 -> manifest 32 -> 14")
    fired = os.path.join(dd, "FIRED_S1ABL.txt"); open(fired, "w").write("2026-09-15T00:00:00Z\n")
    rc, last = fire(head, "S1A25"); limb("F10", rc == 0 and "already fired" in last, "FIRED present, no RESUME -> short-circuit rc 0")
    os.environ["S1ABL_RESUME"] = "1"; rc, last = fire(head, "S1A25"); os.environ.pop("S1ABL_RESUME")
    limb("F11", rc == 0 and last.startswith("GATES PASS") and open(fired).read().startswith("2026-09-15"), "RESUME=1 -> gates re-run, FIRED not clobbered"); os.remove(fired)
    m = json.load(open(os.path.join(B, "WAVE_MANIFEST_S1ABL.json"))); m["runs"]["A2V25"]["knob"] = {"clamp": 0.25}; bad = os.path.join(td, "bad.json"); json.dump(m, open(bad, "w"))
    r = subprocess.run(["python3", os.path.join(CS, "cs600_manifest_check.py"), "--manifest", bad, "--run", "A2V25", "--ckpt", os.path.join(prot["A2"], "model_final.zip"), "--table", T025, "--dose", "0.75"], cwd=FD, capture_output=True, text=True)
    limb("G1", r.returncode == 36, "manifest row carrying an UNSIGNED knob value (clamp 0.25) -> 36 (parent gate by path)")
    shutil.rmtree(td, ignore_errors=True)
    print("s1abl_selfcheck: %s (%d failing)" % ("PASS" if not FAILS else "FAIL " + ",".join(FAILS), len(FAILS))); return 0 if not FAILS else 1

if __name__ == "__main__":
    sys.exit(main())
