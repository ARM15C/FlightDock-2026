#!/usr/bin/env python3
# =============================================================================

#   instrument chain of record over all TWELVE benchmark corpora (7,200 legs) on the laptop, in parallel, into ONE
#   dated run directory, under the five SIGNED analysis plans. Six arms: FlightDock-A2, FlightDock-A2O,
#   FlightDock-S1, FlightDock-S1O, PID-baseline, PIDO (PID-baseline + 0.25 m down-offset), each at sigma 0.25 and
#   0.35. Every corpus is read from its FULL exam tree; the stripped `flightdock_stage/flight_data` copies are
#   QUARANTINED and are named nowhere in this file.
#   Per corpus: a2p600_shim.py (nested + flat symlink shims) -> armbref_readout.py (pinned)
#   -> windowed_rescore.py at 10.0 m (the aerial-docking terminal approach corridor, of record) and 3.0 m
#   (retained) -> voff_held_alt.py at 10.0 m. Nothing pinned is edited or re-implemented; every instrument is
#   invoked by absolute path and its sha256 is censused into INSTRUMENT_SHAS.json before anything runs.
#   PRECONDITIONS ENFORCED, NOT HOPED: (1) each plan carries a signature line whose sha equals the sha256 of the
#   file with that line removed, else REFUSE rc 2; (2) each corpus resolves exactly one tree and every one of its
#   600 legs binds by leg_meta.json `checkpoint_sha256` to the policy the plan names (PID legs carry ARM-B's checkpoint with the residual zeroed; PIDO
#   legs carry FlightDock-S1's checkpoint, which the exam loads and then multiplies by zero -- PREREG_PIDO.md
#   sections 2 and 5, witnessed per leg by TICK_CHECK_PIDO*.json), else REFUSE rc 3; (3) the run directory does
#   not already exist, else REFUSE rc 4.
#   Turbulence: sigma 0.25 and 0.35 only (T5). A peak above 2.00 m is a safety-EXCLUSION-ZONE violation (T1).
# PUBLIC API: CORPORA; plan_shas(plan_dir) -> dict; bind_corpus(tag) -> dict; chain(tag, run_dir) -> dict;
#   main(argv=None) -> int. CLI: --run-dir DIR [--workers 10] [--only TAG,...] [--selftest]
# SIDE-EFFECTS: creates <run-dir>/ (shims/, readouts/, logs/, *.json). Reads corpora READ-ONLY. Spawns the four
#   pinned instruments as subprocesses (no SITL, no training, no network, no S3, no kill of anything it did not
#   spawn). Imports nothing from src/drl.
# SIZE: soft 200 / hard 240 code-LoC; function cap 45.
# COVERAGE: --selftest: signature check accepts a correctly signed fixture and refuses a tampered one; the binding
#   check refuses a leg whose checkpoint sha differs from the plan's; the corpus table names twelve distinct trees;
#   find_tree() locates a nested exam tree, accepts a hyphenated cell directory and refuses an ambiguous base;
#   view_limb(): a PID-shaped (per-stage) fixture passes the PINNED shim, nested and flat, through bench_view(), an
#   exam-shaped tree comes back unchanged, and a leg name used twice refuses (added 2026-09-18 after both PID corpora
#   refused rc 3 in run_20260918T2129Z: the old limb proved the tree was FOUND, never that the shim could READ it).
# =============================================================================
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

FD = "/home/ubuntu/FlightDock"
PY = FD + "/flightdock_native_venv/bin/python"
AN = FD + "/analysis"
PLAN_DIR = "/home/ubuntu/paper_ch3_session/analysis/revision_20260915"
PLANS = ["ANALYSIS_PLAN_H1.md", "ANALYSIS_PLAN_H2.md", "ANALYSIS_PLAN_H3.md", "ANALYSIS_PLAN_OFFSET_ABLATION.md",
         "ANALYSIS_PLAN_COMPARATIVE.md"]
INSTR = {"shim": FD + "/analysis/a2p600_20260903T0545Z/a2p600_shim.py",
         "readout": FD + "/analysis/armbref_20260827T2100Z/armbref_readout.py",
         "windowed": FD + "/analysis/windowed_20260831T1900Z/windowed_rescore.py",
         "held": FD + "/analysis/vertoff_20260902T0500Z/voff_held_alt.py"}
MAN = {0.25: FD + "/analysis/armblad025_20260830T1600Z/SLOT_MANIFEST_ARMBLAD025.json",
       0.35: FD + "/analysis/armblad035_20260830T1600Z/SLOT_MANIFEST_ARMBLAD035.json"}
A2 = "3d1cf6cac098ef3e53fd5c0ed759d14be58448b242297bde18b68f1f7801d442"
S1 = "34c7f1161a84c0665000dd3960232fddba6690483cb93632e365e7b45449fb1c"
#: PID-baseline legs carry the ARM-B checkpoint (ARMBLAD025_PREREG.md: flown through the same exam spawner with the
#: learned residual forced to zero at the source). Value read from leg_meta.json;.
PIDB = "a70c4a88f8d176fac7ecad27d7458a871f48cf076dceda81763fdbc2d6d3b05c"
#: tag ->.
#: /home/ubuntu/flightdock_corpora/<arm>_s<sigma> is a symlink to the REAL run tree (MAP.md there
#: lists all twelve and their status). An entry that does not exist yet (the five PENDING the S3 pull) makes
#: find_tree() return nothing and resolve() REFUSE rc 3 by name -- nothing is ever scored from a stripped copy.
CORP = "/home/ubuntu/flightdock_corpora"
CORPORA = {"A2P025": (CORP + "/A2_s025", 0.25, A2), "A2P035": (CORP + "/A2_s035", 0.35, A2),
           "A2V25": (CORP + "/A2O_s025", 0.25, A2), "A2V35": (CORP + "/A2O_s035", 0.35, A2),
           "S1A25": (CORP + "/S1_s025", 0.25, S1), "S1A35": (CORP + "/S1_s035", 0.35, S1),
           "CSV025": (CORP + "/S1O_s025", 0.25, S1), "CSV035": (CORP + "/S1O_s035", 0.35, S1),
           "ARMB025": (CORP + "/PID_s025", 0.25, PIDB), "ARMB035": (CORP + "/PID_s035", 0.35, PIDB),
           "PIDO25": (CORP + "/PIDO_s025", 0.25, S1), "PIDO35": (CORP + "/PIDO_s035", 0.35, S1)}
SIG = re.compile(r"^OPERATOR-SIGNATURE: PLAN-[A-Z0-9]+-20260918 -- .*plan sha ([0-9a-f]{64})[ \t]*\n?", re.M)
PLACEHOLDER = "<sha256 of this file before signature>"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def plan_shas(plan_dir):
    """Every plan must be signed; the signed sha must equal the sha256 of the file before it was signed."""
    out = {}
    for p in PLANS:
        text = open(os.path.join(plan_dir, p), encoding="utf-8").read()
        m = SIG.search(text)
        if not m:
            raise SystemExit("REFUSE rc 2: %s carries no completed signature line" % p)
        # Two signing conventions are accepted: (a) the signed line was APPENDED below the template line, so the
        # file before signature is the text with that line removed; (b) the placeholder in the template line was
        # REPLACED by the sha, so the file before signature is the text with the placeholder restored.
        appended = text[:m.start()] + text[m.end():]
        replaced = text[:m.start(1)] + PLACEHOLDER + text[m.end(1):]
        digests = {hashlib.sha256(v.encode("utf-8")).hexdigest() for v in (appended, replaced)}
        if m.group(1) not in digests:
            raise SystemExit("REFUSE rc 2: %s signature sha does not match the file body" % p)
        out[p] = {"signed_sha": m.group(1), "file_sha": sha(os.path.join(plan_dir, p))}
    return out


def find_tree(base):
    """Every directory at, or up to two levels under, `base` that holds the manifest's cell directories (B00, B25,
    B35, and the hyphenated B00-S0 form the PID ladders use). Never guesses: the caller refuses unless there is one."""
    bases = sorted(glob.glob(base)) if any(c in base for c in "*?") else [base]
    out = []
    for b in bases:
        for d in [b] + sorted(glob.glob(os.path.join(b, "*"))) + sorted(glob.glob(os.path.join(b, "*", "*"))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "B[0-9][0-9]")) + glob.glob(os.path.join(d, "B[0-9][0-9]-*")):
                out.append(d)
    return sorted(set(out))


def resolve(tag):
    base = CORPORA[tag][0]
    hits = find_tree(base)
    if len(hits) != 1:
        raise SystemExit("REFUSE rc 3: corpus %s resolves to %d exam trees under %s" % (tag, len(hits), base))
    return hits[0]


def cell_dir(root, cell):
    """The leg's cell directory: the manifest's full cell (B25-W0, the PID-baseline layout), its bare form (B25, the
    exam layout) or the single hyphenated form beside the bare one (B00-S0)."""
    for d in (os.path.join(root, cell), os.path.join(root, cell.split("-")[0])):
        if os.path.isdir(d):
            return d
    hits = sorted(p for p in glob.glob(os.path.join(root, cell.split("-")[0]) + "-*") if os.path.isdir(p))
    if len(hits) != 1:
        raise SystemExit("REFUSE rc 3: cell %s under %s resolves to %d directories" % (cell, root, len(hits)))
    return hits[0]


def bind_corpus(tag):
    """Every leg's leg_meta.json must name the plan's checkpoint (the PID-baseline carries ARM-B's, residual zeroed).
    PIDO carries FlightDock-S1's checkpoint by design: the exam loads it and multiplies its output by zero."""
    root, sigma, want = resolve(tag), CORPORA[tag][1], CORPORA[tag][2]
    man = json.load(open(MAN[sigma]))
    seen, bad, missing = set(), [], []
    for leg in man["legs"]:
        d = os.path.join(cell_dir(root, leg["cell"]), "%s_%d_%s" % (leg["label"], leg["seed"], leg["traj"]))
        mp = os.path.join(d, "leg_meta.json")
        if not os.path.isdir(d) or not os.path.exists(mp):
            missing.append(d)
            continue
        m = json.load(open(mp))
        got = m.get("checkpoint_sha256")
        if (want is None and got) or (want is not None and got != want) or int(m.get("seed", -1)) != leg["seed"]:
            bad.append((d, got, m.get("seed")))
        seen.add(got)
    if missing or bad:
        raise SystemExit("REFUSE rc 3: %s binding failed: %d missing, %d wrong (first: %s)"
                         % (tag, len(missing), len(bad), (missing or bad)[:1]))
    return {"tag": tag, "root": root, "sigma": sigma, "checkpoint_sha256": want, "n_legs": len(man["legs"]),
            "manifest": MAN[sigma], "manifest_sha256": sha(MAN[sigma]), "table_sha256": man["source_table_sha256"]}


def bench_view(root, manifest, view):
    """The pinned shim reads the exam layout <root>/<B25>/<leg>. A PID-baseline tree is per stage (<root>/B25-W0/<leg>),
    so give the shim an exam-shaped view of it: one symlink per leg, view/<prefix>/<leg> -> the leg's real directory.
    An exam-layout tree is returned unchanged. Nothing is copied; a leg name used twice REFUSES rc 3 (measured
    2026-09-18: the shim refused both PID corpora on B00-S0 -> <root>/B00, which does not exist)."""
    if all(os.path.isdir(os.path.join(root, leg["cell"].split("-")[0])) for leg in manifest["legs"]):
        return root
    for leg in manifest["legs"]:
        name = "%s_%d_%s" % (leg["label"], leg["seed"], leg["traj"])
        link = os.path.join(view, leg["cell"].split("-")[0], name)
        if os.path.lexists(link):
            raise SystemExit("REFUSE rc 3: leg name %s appears twice in the per-stage tree %s" % (name, root))
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.join(cell_dir(root, leg["cell"]), name), link)
    return view


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=FD)
    if rc != 0:
        raise SystemExit("REFUSE: rc %d from %s (see %s)" % (rc, os.path.basename(cmd[1]), log))


def chain(tag, run_dir):
    b = bind_corpus(tag)
    shim, flat = os.path.join(run_dir, "shims", tag, "shim"), os.path.join(run_dir, "shims", tag, "flat")
    ro, log = os.path.join(run_dir, "readouts"), os.path.join(run_dir, "logs", tag + ".log")
    t0 = time.time()
    corpus = bench_view(b["root"], json.load(open(b["manifest"])), os.path.join(run_dir, "shims", tag, "view"))
    run([PY, INSTR["shim"], "--corpus", corpus, "--manifest", b["manifest"], "--shim", shim, "--flat", flat], log)
    out = os.path.join(ro, "%s_READOUT.json" % tag)
    run([PY, INSTR["readout"], "--legs-root", shim, "--manifest", b["manifest"], "--sigma", str(b["sigma"]),
         "--out", out], log)
    for w in (10.0, 3.0):
        run([PY, INSTR["windowed"], "--legs-root", flat, "--readout", out, "--window-m", str(w),
             "--out", os.path.join(ro, "WIN%d_%s.json" % (int(w), tag))], log)
    run([PY, INSTR["held"], "--legs-root", flat, "--window-m", "10.0",
         "--out", os.path.join(ro, "HELD10_%s.json" % tag)], log)
    b["seconds"] = round(time.time() - t0, 1)
    return b


def view_limb(td):
    """A PID-shaped fixture (per-stage cells) must pass the PINNED shim, nested and flat, through bench_view; an
    exam-shaped tree must come back unchanged; a leg name used twice must refuse."""
    man = {"legs": [{"cell": c, "label": l, "seed": 10775, "traj": "straight"} for c, l in (("B00-S0", "S0"),
                                                                                             ("B25-W0", "W0"))]}
    pid = os.path.join(td, "pid")
    for leg in man["legs"]:
        d = os.path.join(pid, leg["cell"], "%s_10775_straight" % leg["label"])
        os.makedirs(os.path.join(d, "env0"))
        open(os.path.join(d, "results_eid0.json"), "w").write("{}")
        open(os.path.join(d, "env0", "flight_state_ep0001.csv.gz"), "w").write("")
    mp = os.path.join(td, "man.json")
    json.dump(man, open(mp, "w"))
    v = bench_view(pid, man, os.path.join(td, "view"))
    rc = subprocess.call([PY, INSTR["shim"], "--corpus", v, "--manifest", mp, "--shim", os.path.join(td, "sh"),
                          "--flat", os.path.join(td, "fl")], stdout=subprocess.DEVNULL)
    os.makedirs(os.path.join(td, "exam", "B00"))
    os.makedirs(os.path.join(td, "exam", "B25"))
    same = bench_view(os.path.join(td, "exam"), man, os.path.join(td, "v2")) == os.path.join(td, "exam")
    try:
        bench_view(pid, man, os.path.join(td, "view"))           # every link exists already: a name used twice
        dup = False
    except SystemExit:
        dup = True
    return v != pid and rc == 0 and len(os.listdir(os.path.join(td, "fl"))) == 2 and same and dup


def selftest():
    with tempfile.TemporaryDirectory() as td:
        body = "# plan\n\nrule text\n\n## 7. Signature\n\n"
        line = "OPERATOR-SIGNATURE: PLAN-H1-20260918 -- I sign; plan sha %s\n"
        h = hashlib.sha256(body.encode()).hexdigest()
        for p in PLANS:
            open(os.path.join(td, p), "w").write(body + line % h)
        ok = plan_shas(td)[PLANS[0]]["signed_sha"] == h
        tmpl = body + line % PLACEHOLDER                      # convention (b): placeholder replaced in place
        h2 = hashlib.sha256(tmpl.encode()).hexdigest()
        open(os.path.join(td, PLANS[1]), "w").write(body + line % h2)
        ok = ok and plan_shas(td)[PLANS[1]]["signed_sha"] == h2
        open(os.path.join(td, PLANS[0]), "w").write(body + "tampered\n" + line % h)
        try:
            plan_shas(td)
            ok = False
        except SystemExit:
            pass
        ok = ok and len({v[0] for v in CORPORA.values()}) == 12 and len(CORPORA) == 12
        ok = ok and not any("flightdock_stage" in v[0] for v in CORPORA.values())   # the stripped copies are quarantined
        for cells, want in (("B00", 1), ("B00-S0", 1), ("", 0)):                    # nested tree, hyphenated cell, neither
            os.makedirs(os.path.join(td, "tree", "lvl", cells or "none"), exist_ok=True)
            ok = ok and len(find_tree(os.path.join(td, "tree"))) == want
            os.rename(os.path.join(td, "tree"), os.path.join(td, "tree_%s" % (cells or "none")))
        os.makedirs(os.path.join(td, "cells", "B25-W0"))
        ok = ok and os.path.basename(cell_dir(os.path.join(td, "cells"), "B25")) == "B25-W0"
        ok = ok and view_limb(td)
        print("SELFTEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.run_dir:
        ap.error("--run-dir required")
    if os.path.exists(a.run_dir):
        raise SystemExit("REFUSE rc 4: run dir exists: %s" % a.run_dir)
    for d in ("shims", "readouts", "logs"):
        os.makedirs(os.path.join(a.run_dir, d))
    json.dump(plan_shas(PLAN_DIR), open(os.path.join(a.run_dir, "PLAN_SHAS.json"), "w"), indent=1)
    json.dump({k: sha(v) for k, v in INSTR.items()}, open(os.path.join(a.run_dir, "INSTRUMENT_SHAS.json"), "w"), indent=1)
    tags = a.only.split(",") if a.only else list(CORPORA)
    bound = {t: bind_corpus(t) for t in tags}          # every corpus binds BEFORE any instrument runs
    json.dump(bound, open(os.path.join(a.run_dir, "CORPUS_BINDING.json"), "w"), indent=1)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        done = {t: f for t, f in ((t, ex.submit(chain, t, a.run_dir)) for t in tags)}
        res = {t: f.result() for t, f in done.items()}
    json.dump(res, open(os.path.join(a.run_dir, "RUN_MANIFEST.json"), "w"), indent=1)
    for t, r in res.items():
        print("%-8s %6.0f s %s" % (t, r["seconds"], r["root"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
