#!/usr/bin/env python3
# RESPONSIBILITY: BENCH600 TWIN of analysis/armbsig_20260824T0030Z/armbsig_exam_table.py -- gate the
#   600-leg ARM-B LADDER table on ITS OWN countersignature, return its rows, deal them into lanes
#   through armbsig_manifest.assign (the ONE writer of slot identity, called by absolute path), and
#   name lane dirs sigma-scoped exactly as the parent. THREE declared deltas, all measured at the
#   wave's dry-run on 2026-09-02 19:0xZ (the overlay's --dry-run never reaches this gate -- a dry
#   run cannot exercise a live-only line -- the WAVE's dry-run does, and it REFUSED three ways):
#     D1 COUNTERSIGNATURE SOURCE: the parent reads grid 1/3/5/6/7 clauses in ARMBSIG_PREREG.md and
#        refused "sha drift: countersigned=968fc07.. on-disk=6be7b251..". The ladder tables are
#        countersigned in THEIR preregs (ARMBLAD025-SEEDS-08.30.2026 / ARMBLAD035-SEEDS-08.30.2026);
#        this reads that clause, keyed by the table's own on-disk sha, and refuses anything else.
#     D2 ROWS_BY_SHA: exactly the two ladder shas -> 600. An unlisted sha REFUSES outright (the
#        parent falls back to 84, which would also refuse; stated here so nobody reads a fallback).
#     D3 SIGMA ADMISSION: every
#        wind-rung row must carry the table's ONE venue sigma (0.25 for ARMBLAD025, 0.35 for
#        ARMBLAD035); sigma 0.00 is admitted ONLY on the S0 control rung -- label S0, wind 0.0,
#        paired_csv True -- and any other sigma (0.20 is PURGED) refuses with the parent's T5 text.
#   The parent is NOT edited; the four grids it serves keep their gate byte-for-byte.
# PUBLIC API: Refusal. table_rows(table_path). lanes(table_path, width=24). lane_dir(out_dir, lane)
#             . expected_by_sigma(table_path) -- the parent's API, unchanged, so the wave twin's
#             re-exports are the same names.
# SIDE-EFFECTS: none. Reads the table and one prereg; imports neither src/ nor torch.
# SIZE: soft 60 / hard 90 code-lines; function cap 40.
# COVERAGE: bench600_selfcheck.py -- both ladder tables pass (600 rows, 25 lanes deep at width 24);
#   grid B25R (signed elsewhere) refuses; a table whose sha has drifted refuses; an S0 row missing
#   paired_csv refuses; a wind-rung row at 0.20 refuses with the T5 text; a 0.00 row off S0 refuses.
import importlib.util
import json
import os
import re
import sys

FD = "/home/ubuntu/FlightDock"
PARENT = os.path.join(FD, "analysis/armbsig_20260824T0030Z")
LADDERS = {  # on-disk sha -> (venue sigma, prereg, countersignature regex)
    "6be7b251698e4f776f336c4a6ac5522b021918640ff8fc0283d2288620fc3e18":
        (0.25, os.path.join(FD, "analysis/armblad025_20260830T1600Z/ARMBLAD025_PREREG.md"),
         re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD025-SEEDS-\S+ [-—]+ ([0-9a-f]{64})", re.M)),
    "29c13944f7aaea07bc06bce7cf80b1b349787a613acbb880c4a56d217737e9ce":
        (0.35, os.path.join(FD, "analysis/armblad035_20260830T1600Z/ARMBLAD035_PREREG.md"),
         re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD035-SEEDS-\S+ [-—]+ ([0-9a-f]{64})", re.M)),
}
ROWS_BY_SHA = {s: 600 for s in LADDERS}
T5 = ("T5 SIGMA REFUSAL: this %s carries sigma %s; the only venues that exist are 0.25, 0.35 "
      " plus 0.00 on the S0 control rung ONLY "
      ". Refused at launch, not scored.")


class Refusal(Exception):
    """Fail-closed. Every guard here exists because its absence was measured once."""


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _sha(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def assert_countersigned(table_path):
    """D1+D2: the sha on disk must be one of the two ladders AND appear in that ladder's clause."""
    got = _sha(table_path)
    if got not in LADDERS:
        raise Refusal("table sha %s.. is not a BENCH600 ladder table (ARMBLAD025/035 only)" % got[:12])
    sigma, doc, rx = LADDERS[got]
    if not os.path.isfile(doc):
        raise Refusal("ladder prereg absent at %s" % doc)
    m = rx.search(open(doc, encoding="utf-8").read())
    if not m:
        raise Refusal("no column-0 SEEDS countersignature clause in %s" % os.path.basename(doc))
    if m.group(1) != got:
        raise Refusal("seed table sha drift: countersigned=%s on-disk=%s" % (m.group(1), got))
    return got, sigma


def sigma_refusal(rows, venue, what="countersigned table"):
    """D3: venue sigma on wind rungs; 0.00 only on a fully-declared S0 row; everything else refuses."""
    bad = set()
    for r in rows:
        s = float(r["sigma_ms"])
        calm = r.get("label") == "S0" and float(r.get("wind_ms", -1)) == 0.0 and r.get("paired_csv") is True
        if not (s == venue or (s == 0.0 and calm)):
            bad.add(s)
    return None if not bad else T5 % (what, ", ".join("%.2f" % b for b in sorted(bad)))


def table_rows(table_path):
    sha, venue = assert_countersigned(table_path)
    rows = json.load(open(table_path))["legs"]
    why = sigma_refusal(rows, venue)
    if why:
        raise Refusal(why)
    if len(rows) != ROWS_BY_SHA[sha]:
        raise Refusal("table has %d rows, expected %d for countersigned sha %s"
                      % (len(rows), ROWS_BY_SHA[sha], sha[:12]))
    return rows


def lanes(table_path, width=24):
    """Deal via armbsig_manifest.assign -- the ONE writer of slot identity, unchanged."""
    man = _load(os.path.join(PARENT, "armbsig_manifest.py"), "armbsig_manifest")
    return man.assign(table_rows(table_path), width)


def lane_dir(out_dir, lane):
    """SIGMA-SCOPED exactly as the parent: B00/S0_<seed>_<traj>, B25/W8_<seed>_<traj>, ..."""
    return os.path.join(out_dir, str(lane["cell"]).split("-")[0],
                        "%s_%s_%s" % (lane["label"], lane["seed"], lane["traj"]))


def expected_by_sigma(table_path):
    out = {}
    for r in table_rows(table_path):
        k = str(r["cell"]).split("-")[0]
        out[k] = out.get(k, 0) + 1
    return out


TABLE, N_ROWS = None, 600   # the parent's names, re-exported by the wave twin; no default table here
