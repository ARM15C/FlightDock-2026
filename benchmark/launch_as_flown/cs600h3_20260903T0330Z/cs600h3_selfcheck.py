#!/usr/bin/env python3
# RESPONSIBILITY: coverage for cs600h3_fire.sh, cs600_manifest_check.py and cs600_overlay_exam.py, driven on dev with
#   CS600H3_FIXTURE=1 (stops after every gate, before the launch line) on throw-away copies of BOTH staged pairs; spawns
#   no SITL, no tmux, touches no box. Every limb asserts ONE rc. Limbs:
#     F1 unknown RUN 15. F2 wrong pin 12. F3 census file absent 11. F4 tampered SHAS 13. F5 buffer reachable 13
#     F6 CSV's pair under RUN=CSP -> manifest 32 -> 14. F7a unsigned MUST be 16. F7b signed MUST be 0. F7c wrong sha 16
#     F7d bank absent 17. F8 absent prereg 16. F9a CSP passes with an EMPTY knob flag and no knob token on the argv
#     F9b CSV passes with EXACTLY --vert-offset-down-m 0.25 and no clamp/bias token. F10/F11 resume (parent's D8)
#     F9c CSC forwards exactly clamp 0.50. F9d CSC at the wrong dose -> 14. F9e wrong pair for CSC -> 14
#     G1 manifest row with an UNSIGNED knob value -> 36. G2 overlay: undeclared knob on argv -> 14. G3 declared flag absent -> 14
# PUBLIC API: python cs600h3_selfcheck.py (rc 0 = every limb passed)
# SIDE-EFFECTS: a temp dir, removed. Reads git HEAD. Nothing else.
# SIZE: soft 80 / hard 100 code-lines; function cap 70.
import json, os, shutil, subprocess, sys, tempfile
FD = "/home/ubuntu/FlightDock"; B = os.path.join(FD, "analysis/cs600h3_20260903T0330Z"); FAILS = []

def limb(tag, cond, msg):
    print(" %s %s: %s" % ("PASS" if cond else "FAIL", tag, msg)); FAILS.append(tag) if not cond else None

def main():
    td = tempfile.mkdtemp(); dd = os.path.join(td, "dd"); os.makedirs(dd)
    prot = {r: os.path.join(td, r) for r in ("H3C", "H3V", "H3P")}
    for r, sub in (("H3C", "A2AL2"), ("H3V", "CANDSSW1"), ("H3P", "CANDS")):
        shutil.copytree(os.path.join(B, "stage/%s/protected" % sub), prot[r])
    open(os.path.join(dd, "census_patterns.txt"), "w").write("zzz_none\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FD, capture_output=True, text=True).stdout.strip()
    real, bank = os.path.join(B, "PREREG_CS600H3.md"), os.path.join(B, "PREREG_SHAS.json")
    bk = json.load(open(bank))["runs"]; want = bk["H3BOTH"]
    import re
    stripped = re.sub(r"^OPERATOR-SIGNATURE: CS600-(H3BOTH|H3P)-\S[^\n]*\n?", "", open(real).read(), flags=re.M)   # the REAL file carries BOTH real signatures now
    unsigned = os.path.join(td, "unsigned.md"); open(unsigned, "w").write(stripped)
    def signed_copy(path, sha, sha_p=None):
        body = stripped + "\nOPERATOR-SIGNATURE: CS600-H3BOTH-20260903-fx -- signed, prereg sha %s\n" % sha
        if sha_p: body += "OPERATOR-SIGNATURE: CS600-H3P-20260903-fx -- signed, prereg sha %s\n" % sha_p
        open(path, "w").write(body)
    signed, wrong = os.path.join(td, "signed.md"), os.path.join(td, "wrong.md"); signed_copy(signed, want); signed_copy(wrong, "0" * 64)
    signed_p = os.path.join(td, "signed_p.md"); signed_copy(signed_p, want, bk.get("H3P", "0" * 64))
    DOSE = {"H3C": "0.75", "H3V": "1.00", "H3P": "1.00"}
    def fire(pin, run, prereg=signed, bank_path=bank, pr=None, dose=None):
        env = dict(os.environ, LABEL="fx", PROT=pr or prot[run if run in prot else "H3C"], DOSE=dose or DOSE.get(run, "1.00"), PIN=pin, DD=dd, RUN=run,
                   CS600H3_FIXTURE="1", CS600H3_PREREG=prereg, CS600H3_SHAS=bank_path)
        r = subprocess.run(["bash", os.path.join(B, "cs600h3_fire.sh")], cwd=FD, env=env, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip(); return r.returncode, out.splitlines()[-1] if out else ""
    limb("F1", fire(head, "CSC")[0] == 15, "sigma-0.25 RUN token CSC offered here -> 15 (this twin knows only H3C|H3V)")
    limb("F2", fire("deadbeef1", "H3C")[0] == 12, "wrong pin -> 12")
    os.rename(os.path.join(dd, "census_patterns.txt"), os.path.join(dd, "c.bak")); limb("F3", fire(head, "H3C")[0] == 11, "census file absent -> 11"); os.rename(os.path.join(dd, "c.bak"), os.path.join(dd, "census_patterns.txt"))
    sp = os.path.join(prot["H3C"], "SHAS.txt"); os.chmod(sp, 0o644); good = open(sp).read(); open(sp, "w").write(good.replace("fd3a338a", "00000000"))
    limb("F4", fire(head, "H3C")[0] == 13, "tampered SHAS -> 13"); open(sp, "w").write(good)
    open(os.path.join(prot["H3C"], "replay_buffer_final.pkl"), "w").close(); limb("F5", fire(head, "H3C")[0] == 13, "replay buffer reachable -> 13"); os.remove(os.path.join(prot["H3C"], "replay_buffer_final.pkl"))
    limb("F6", fire(head, "H3C", pr=prot["H3V"])[0] == 14, "H3V's pair offered as RUN=H3C -> manifest 32 -> 14")
    limb("F7a", fire(head, "H3C", prereg=unsigned)[0] == 16, "the prereg with its §9 signature line STRIPPED (placeholder only) -> MUST be 16")
    limb("F7b", fire(head, "H3C")[0] == 0, "correctly signed, sha bound -> MUST be 0")
    limb("F7c", fire(head, "H3C", prereg=wrong)[0] == 16, "signed but WRONG sha -> MUST be 16")
    limb("F7d", fire(head, "H3C", bank_path=os.path.join(td, "nobank.json"))[0] == 17, "sha bank absent -> MUST be 17")
    limb("F8", fire(head, "H3C", prereg=os.path.join(td, "absent.md"))[0] == 16, "absent prereg -> 16")
    rc, last = fire(head, "H3C"); limb("F9a", rc == 0 and "KNOBFLAG=[--clamp-behind-positive-ms 0.50]" in last and "SEED_CELL_TABLE_ARMBLAD035.json" in last and "--vert-offset" not in last, "H3C: GATES PASS on the ARMBLAD035 table, exactly clamp 0.50")
    rc, last = fire(head, "H3V"); limb("F9b", rc == 0 and "KNOBFLAG=[--vert-offset-down-m 0.25]" in last and "SEED_CELL_TABLE_ARMBLAD035.json" in last and "--clamp" not in last, "H3V: GATES PASS on the ARMBLAD035 table, exactly voff 0.25")
    limb("F9c", fire(head, "H3C", dose="1.00")[0] == 14, "H3C at the WRONG dose 1.00 (signed 0.75) -> 14")
    m025 = json.load(open(os.path.join(B, "WAVE_MANIFEST_CS600H3.json"))); m025["runs"]["H3C"]["table_sha"] = "6be7b251698e4f776f336c4a6ac5522b021918640ff8fc0283d2288620fc3e18"
    b025 = os.path.join(td, "m025.json"); json.dump(m025, open(b025, "w"))
    r = subprocess.run(["python3", os.path.join(FD, "analysis/cs600_20260903T0230Z/cs600_manifest_check.py"), "--manifest", b025, "--run", "H3C", "--ckpt", os.path.join(prot["H3C"], "model_final.zip"), "--table", "analysis/armblad035_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD035.json", "--dose", "0.75"], cwd=FD, capture_output=True, text=True)
    limb("F9d", r.returncode == 33, "a row signed for the sigma-0.25 table, offered the sigma-0.35 table -> 33 (a sigma swap cannot pass the manifest)")
    rc, last = fire(head, "H3P", prereg=signed_p); limb("F9e", rc == 0 and "KNOBFLAG=[]" in last and "SEED_CELL_TABLE_ARMBLAD035.json" in last and "--vert-offset" not in last and "--clamp" not in last, "H3P (amendment): GATES PASS on the ARMBLAD035 table with an EMPTY knob flag")
    limb("F9f", fire(head, "H3P", prereg=signed)[0] == 16, "H3P against a file signed ONLY at §9 (no H3P line) -> MUST be 16")
    limb("F9g", fire(head, "H3P", prereg=signed_p, pr=prot["H3V"])[0] == 14, "H3V's pair offered as RUN=H3P -> manifest 32 -> 14")
    fired = os.path.join(dd, "FIRED_CS600H3.txt"); open(fired, "w").write("2026-09-03T00:00:00Z\n")
    rc, last = fire(head, "H3C"); limb("F10", rc == 0 and "already fired" in last, "FIRED present, no RESUME -> short-circuit rc 0")
    os.environ["CS600H3_RESUME"] = "1"; rc, last = fire(head, "H3C"); os.environ.pop("CS600H3_RESUME")
    limb("F11", rc == 0 and last.startswith("GATES PASS") and open(fired).read().startswith("2026-09-03"), "RESUME=1 -> gates re-run, FIRED not clobbered"); os.remove(fired)
    m = json.load(open(os.path.join(B, "WAVE_MANIFEST_CS600H3.json"))); m["runs"]["H3V"]["knob"] = {"clamp": 0.25}; bad = os.path.join(td, "bad.json"); json.dump(m, open(bad, "w"))
    r = subprocess.run(["python3", os.path.join(FD, "analysis/cs600_20260903T0230Z/cs600_manifest_check.py"), "--manifest", bad, "--run", "H3V", "--ckpt", os.path.join(prot["H3V"], "model_final.zip"), "--table", "analysis/armblad035_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD035.json", "--dose", "1.0"], cwd=FD, capture_output=True, text=True)
    limb("G1", r.returncode == 36, "manifest row carrying an UNSIGNED knob value (clamp 0.25) -> 36")
    def ov(flag, argv):
        r = subprocess.run(["python3", os.path.join(FD, "analysis/cs600_20260903T0230Z/cs600_overlay_exam.py")] + argv, cwd=FD, env=dict(os.environ, CS600_KNOBFLAG=flag), capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip()[-90:]
    limb("G2", ov("", ["--dose", "1.0", "--vert-offset-down-m", "0.25"])[0] == 14, "overlay: undeclared knob on the argv -> 14")
    limb("G3", ov("--vert-offset-down-m 0.25", ["--dose", "1.0"])[0] == 14, "overlay: declared flag ABSENT from the argv -> 14")
    limb("G4", ov("--vert-offset-down-m 0.25", ["--dose", "1.0", "--clamp-behind-positive-ms", "0.50", "--vert-offset-down-m", "0.25"])[0] == 14, "overlay: declared voff plus an undeclared clamp -> 14")
    shutil.rmtree(td, ignore_errors=True)
    print("cs600h3_selfcheck: %s (%d failing)" % ("PASS" if not FAILS else "FAIL " + ",".join(FAILS), len(FAILS))); return 0 if not FAILS else 1

if __name__ == "__main__":
    sys.exit(main())
