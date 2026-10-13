#!/usr/bin/env python3
# =============================================================================

#   safety chain of record on ALL TWELVE corpora inside this run directory: the AMPSAFETY-20260903 per-leg extractor
#   (safety_extract.py, invoked by absolute path, unedited) and then the ABORTREC-20260904 module set as BYTE
#   COPIES placed under <run-dir>/ticklevel/ with exactly two re-points, both recorded in TICKLEVEL_PROVENANCE.json:
#     (1) abort_common.py's corpus block names the twelve tags and binds each to THIS run's summary, THIS run's
#         readout and THIS run's flat shim (the original binds six banked corpora);
#     (2) abort_limbs.py's two literal 3600 denominators become 600 * len(A.TAGS).
#   Every threshold is still read from the producer at import (formation_config), never retyped. The modules
#   write their outputs beside themselves, so the copies live with their outputs and nothing under analysis/
#   is touched. Then: window pass and per-tick pass per corpus (parallel across corpora), build, recovery,
#   tables, GLM, impact and limbs (sequential). T1 vocabulary; sigma 0.25/0.35 only (T5).
# PUBLIC API: prepare(run_dir) -> dict; extract(run_dir, manifest, jobs); passes(run_dir, jobs); build(run_dir);
#   main(argv=None) -> int. CLI: --run-dir DIR [--prepare-only] [--jobs-per-corpus 2] [--selftest]
# SIDE-EFFECTS: creates <run-dir>/ticklevel/ (module copies, summaries/, WINDOW_TICKS_*, EVENT_GEOM_*,
#   RECOVERY_TRACE_*, ABORT_*.csv, RECOVERY_SUMMARY.csv). Spawns the extractor and the copied modules as
#   subprocesses; reads corpora read-only; no SITL, no network, no S3, kills nothing it did not spawn.
# SIZE: soft 140 / hard 170 code-LoC; function cap 40.
# COVERAGE: --selftest: prepare() into a temp dir yields an importable abort_common with twelve tags bound into that
#   dir, and exactly two re-based denominators in abort_limbs.
# =============================================================================
import argparse
import concurrent.futures as cf
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

FD = "/home/ubuntu/FlightDock"
PY = FD + "/flightdock_native_venv/bin/python"
EXTRACT = FD + "/analysis/ampsafety_20260903T1300Z/safety_extract.py"
ABORT = FD + "/analysis/abortrec_20260904T2200Z"
TAGS = ("A2P025", "A2P035", "CSV025", "CSV035", "ARMB025", "ARMB035", "S1A25", "S1A35", "A2V25", "A2V35",
        "PIDO25", "PIDO35")
POLICY = {"A2P": "FlightDock-A2", "CSV": "FlightDock-S1 (+0.25 m offset)", "ARMB": "PID-baseline",
          "S1A": "FlightDock-S1 (no offset)", "A2V": "FlightDock-A2 (+0.25 m offset)",
          "PIDO": "PID-baseline (+0.25 m offset)"}
BLOCK = re.compile(r"TAGS = \(.*?\nSIGMA = \{t: C4\.SIGMA\[t\] for t in TAGS\}\n", re.S)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def corpus_block(tl):
    lines = ["TAGS = %r" % (TAGS,), "TL = %r" % tl,
             "SUMMARY = {t: os.path.join(TL, 'summaries', t + '.jsonl.gz') for t in TAGS}",
             "READOUT = {t: os.path.join(TL, '..', 'readouts', t + '_READOUT.json') for t in TAGS}",
             "LEGS_ROOT = {t: os.path.join(TL, '..', 'shims', t, 'flat') for t in TAGS}",
             "POLICY = {t: %r[t.rstrip('0123456789')] for t in TAGS}" % POLICY,
             "SIGMA = {t: (0.25 if t.endswith('25') else 0.35) for t in TAGS}", ""]
    return "\n".join(lines)


def prepare(run_dir):
    """Byte-copy the ABORTREC modules into <run-dir>/ticklevel/ with the two recorded re-points."""
    tl = os.path.join(run_dir, "ticklevel")
    os.makedirs(os.path.join(tl, "summaries"), exist_ok=True)
    prov = {"source": ABORT, "files": {}, "repoints": []}
    for p in sorted(glob.glob(os.path.join(ABORT, "*.py"))):
        name, text = os.path.basename(p), open(p).read()
        prov["files"][name] = {"source_sha256": sha(p)}
        if name == "abort_common.py":
            m = BLOCK.search(text)
            if not m:
                raise SystemExit("REFUSE rc 5: abort_common.py corpus block not found; the re-point cannot be applied blind")
            text = text[:m.start()] + corpus_block(tl) + text[m.end():]
            prov["repoints"].append("abort_common.py: corpus block -> twelve tags bound into %s" % tl)
        if name == "abort_limbs.py":
            n = text.count("3600")
            text = text.replace("3600", "600 * len(A.TAGS)")
            prov["repoints"].append("abort_limbs.py: %d literal 3600 -> 600 * len(A.TAGS)" % n)
            prov["files"][name]["rebased_literals"] = n
        open(os.path.join(tl, name), "w").write(text)
        prov["files"][name]["copy_sha256"] = sha(os.path.join(tl, name))
    json.dump(prov, open(os.path.join(tl, "TICKLEVEL_PROVENANCE.json"), "w"), indent=1)
    return prov


def run(cmd, log, cwd):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd)
    if rc != 0:
        raise SystemExit("REFUSE: rc %d from %s (see %s)" % (rc, " ".join(cmd[1:3]), log))


def extract(run_dir, manifest, jobs):
    """AMPSAFETY extractor per corpus on the corpus ROOT (its own two-level leg discovery), then gzip."""
    tl = os.path.join(run_dir, "ticklevel")

    def one(tag):
        out, log = os.path.join(tl, "summaries", tag + ".jsonl"), os.path.join(run_dir, "logs", "ticklevel_%s.log" % tag)
        run([PY, EXTRACT, manifest[tag]["root"], out, "--arm", tag, "--sigma", str(manifest[tag]["sigma"]), "--jobs", str(jobs)], log, FD)
        with open(out, "rb") as f, gzip.open(out + ".gz", "wb") as g:
            shutil.copyfileobj(f, g)
        os.remove(out)
        return tag
    with cf.ThreadPoolExecutor(max_workers=len(TAGS)) as ex:
        list(ex.map(one, TAGS))


def passes(run_dir, jobs):
    tl = os.path.join(run_dir, "ticklevel")

    def one(tag):
        log = os.path.join(run_dir, "logs", "ticklevel_%s.log" % tag)
        run([PY, os.path.join(tl, "abort_window_run.py"), tag, tl, str(jobs)], log, tl)
        run([PY, os.path.join(tl, "abort_pass.py"), tag, tl, str(jobs)], log, tl)
        return tag
    with cf.ThreadPoolExecutor(max_workers=len(TAGS)) as ex:
        list(ex.map(one, TAGS))


def census(run_dir):
    """APPROACH_LEG_CENSUS.json for limb A10(ii), from THIS run: per corpus, legs with no corridor peak in the board
    (win null = never within 10.0 m, measured by windowed_rescore, an instrument independent of the per-tick
    extractor A10(i) reads). Replaces the Sep 4 banked six-corpus file."""
    import pandas as pd
    b = pd.read_csv(os.path.join(run_dir, "board", "BOARD_W10.csv"), usecols=["corpus", "win"])
    cen = {t: {"never_within_10m": int(b[b.corpus == t].win.isna().sum()), "legs": int((b.corpus == t).sum()),
               "source": "board/BOARD_W10.csv win null (windowed_rescore at 10.0 m)"} for t in TAGS}
    if any(v["legs"] != 600 for v in cen.values()):
        raise SystemExit("REFUSE rc 3: census needs 600 board legs per corpus: %s" % {t: v["legs"] for t, v in cen.items()})
    json.dump(cen, open(os.path.join(run_dir, "ticklevel", "APPROACH_LEG_CENSUS.json"), "w"), indent=1)
    return cen


def build(run_dir):
    census(run_dir)
    tl, log = os.path.join(run_dir, "ticklevel"), os.path.join(run_dir, "logs", "ticklevel_build.log")
    for mod in ("abort_build.py", "abort_recovery_run.py", "abort_tables.py", "abort_glm.py", "abort_impact.py", "abort_limbs.py"):
        run([PY, os.path.join(tl, mod), tl], log, tl)


def census_limb(td):
    """The file is absent before census() runs (the state that crashed limb A10 on 2026-09-18); census() then writes
    twelve corpora with the right never-within counts; a board short of 600 legs refuses."""
    import pandas as pd
    os.makedirs(os.path.join(td, "board"), exist_ok=True)
    rows = [{"corpus": t, "win": (None if i < TAGS.index(t) else 1.0)} for t in TAGS for i in range(600)]
    pd.DataFrame(rows).to_csv(os.path.join(td, "board", "BOARD_W10.csv"), index=False)
    p = os.path.join(td, "ticklevel", "APPROACH_LEG_CENSUS.json")
    had = os.path.exists(p)
    cen = census(td)
    ok = not had and len(cen) == 12 and all(cen[t]["never_within_10m"] == TAGS.index(t) for t in TAGS)
    ok = ok and json.load(open(p)) == cen
    pd.DataFrame(rows[:-1]).to_csv(os.path.join(td, "board", "BOARD_W10.csv"), index=False)
    try:
        census(td)
        ok = False
    except SystemExit:
        pass
    return ok


def selftest():
    with tempfile.TemporaryDirectory() as td:
        prov = prepare(td)
        spec = importlib.util.spec_from_file_location("ac", os.path.join(td, "ticklevel", "abort_common.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok = tuple(mod.TAGS) == TAGS and mod.LEGS_ROOT["S1A25"].startswith(td) and mod.SIGMA["A2V35"] == 0.35 \
            and prov["files"]["abort_limbs.py"]["rebased_literals"] == 2 and mod.POLICY["ARMB025"] == "PID-baseline" \
            and mod.POLICY["PIDO25"] == "PID-baseline (+0.25 m offset)" and mod.SIGMA["PIDO25"] == 0.25 \
            and mod.SIGMA["PIDO35"] == 0.35 and len(mod.TAGS) == 12
        ok = ok and census_limb(td)
        print("SELFTEST", "PASS" if ok else "FAIL", prov["repoints"])
        return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="skip extraction and passes; rerun build and limbs")
    ap.add_argument("--jobs-per-corpus", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    manifest = json.load(open(os.path.join(a.run_dir, "RUN_MANIFEST.json")))
    prov = prepare(a.run_dir)
    print("prepared:", prov["repoints"])
    if a.prepare_only:
        return 0
    if not a.build_only:
        extract(a.run_dir, manifest, a.jobs_per_corpus)
        passes(a.run_dir, a.jobs_per_corpus)
    build(a.run_dir)
    print("tick-level chain complete under", os.path.join(a.run_dir, "ticklevel"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
