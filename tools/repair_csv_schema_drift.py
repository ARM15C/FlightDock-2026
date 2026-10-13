#!/usr/bin/env python3
"""Offline repair of a mixed-width CSV produced by the T-466 stale-header defect.

RESPONSIBILITY: repair ONE CSV whose rows were appended under a stale,
    narrower header after the writer's column list grew additively (T-466).
    Rows are split by PHYSICAL width — old-header-width rows keyed to the old
    header, current-width rows to the live list — and re-emitted under one
    current header with "" backfill.
DECLARED LIMIT: pure ADDITIVE-GROWTH drift only. Any row matching neither
    width, a header not a strict subset of the current columns, or duplicate
    names => REFUSE (exit 1). This tool never guesses.
PUBLIC API: repair(csv_path, fieldnames, apply=False) -> dict (pure core;
    the dict IS the report) · main(argv=None) -> int (CLI entry) ·
    CLI --csv PATH --fieldnames-from MODULE:ATTR [--apply]
EXIT CODES (check_suite_baseline.py convention): 0 repaired/clean ·
    1 unrepairable (refused, nothing written) · 3 IO/usage.
SIDE-EFFECTS: reads always; writes ONLY under --apply, and then banks the
    original to <csv>.pre_repair_<UTC>.csv FIRST, writes a sibling .tmp,
    fsyncs, and os.replace()s it in (atomic, same fs). Never deletes.
CAPS: file soft 200 / hard 260 lines; function cap 60 lines.
COVERAGE: tests/unit/test_repair_csv_schema_drift.py (executed, incl. a
    deliberately-broken control that must exit nonzero).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import os
import sys
from typing import Dict, List, Optional
csv.field_size_limit(10 * 1024 * 1024)


def _load_fieldnames(spec: str) -> List[str]:
    """'pkg.mod:ATTR' -> list(ATTR). ATTR may be Class._FIELDNAMES."""
    mod_name, _, attr_path = spec.partition(":")
    if not mod_name or not attr_path:
        raise ValueError(f"--fieldnames-from must be MODULE:ATTR, got {spec!r}")
    obj = importlib.import_module(mod_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return list(obj)


def _classify(rows: List[List[str]], n_old: int, n_new: int) -> Dict:
    """Bucket rows by physical width; anything else is a hard refusal."""
    old_rows, new_rows, bad = [], [], []
    for lineno, row in enumerate(rows, start=2):
        if len(row) == n_new:
            new_rows.append(row)
        elif len(row) == n_old:
            old_rows.append(row)
        else:
            bad.append((lineno, len(row)))
    return {"old_rows": old_rows, "new_rows": new_rows, "bad": bad}


def repair(csv_path: str, fieldnames: List[str], apply: bool = False) -> Dict:
    """Analyse (and optionally repair) csv_path. Returns the report dict."""
    fieldnames = list(fieldnames)
    report: Dict = {"csv": csv_path, "current_cols": len(fieldnames),
                    "apply": bool(apply), "banked": None, "verdict": "UNKNOWN",
                    "rows_total": 0, "rows_old_width": 0, "rows_new_width": 0,
                    "problems": []}
    if len(set(fieldnames)) != len(fieldnames):
        report["verdict"] = "REFUSE"
        report["problems"].append("duplicate names in the current column list")
        return report
    with open(csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [r for r in reader]
    if header is None:
        report["verdict"] = "REFUSE"
        report["problems"].append("file has no header record")
        return report
    if header and header[0].startswith("﻿"):
        header[0] = header[0].lstrip("﻿")
    report["header_cols"] = len(header)
    report["rows_total"] = len(rows)
    if len(set(header)) != len(header):
        report["verdict"] = "REFUSE"
        report["problems"].append("duplicate names in the existing header")
        return report

    buckets = _classify(rows, len(header), len(fieldnames))
    report["rows_old_width"] = len(buckets["old_rows"])
    report["rows_new_width"] = len(buckets["new_rows"])
    if buckets["bad"]:
        report["verdict"] = "REFUSE"
        for lineno, width in buckets["bad"][:10]:
            report["problems"].append(
                f"line {lineno}: width {width} matches neither "
                f"{len(header)} (header) nor {len(fieldnames)} (current)")
        return report
    if header == fieldnames and not buckets["old_rows"]:
        report["verdict"] = "CLEAN"
        return report
    if not set(header) <= set(fieldnames):
        report["verdict"] = "REFUSE"
        report["problems"].append(
            "header columns absent from the current list (renamed/removed?): "
            f"{sorted(set(header) - set(fieldnames))[:8]}")
        return report
    report["verdict"] = "REPAIRABLE"
    if not apply:
        return report
    return _apply(csv_path, header, fieldnames, buckets, report)


def _apply(csv_path: str, header: List[str], fieldnames: List[str],
           buckets: Dict, report: Dict) -> Dict:
    """Bank the original, rewrite atomically, verify. Mutates/returns report."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    banked = f"{csv_path}.pre_repair_{stamp}.csv"
    if os.path.exists(banked):
        report["verdict"] = "REFUSE"
        report["problems"].append(f"bank path already exists: {banked}")
        return report
    with open(csv_path, "rb") as src, open(banked, "wb") as dst:
        dst.write(src.read())
        dst.flush()
        os.fsync(dst.fileno())
    report["banked"] = banked

    tmp = csv_path + ".tmp"
    written = 0
    with open(tmp, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, restval="",
                                extrasaction="raise")
        writer.writeheader()
        with open(csv_path, "r", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                keys = fieldnames if len(row) == len(fieldnames) else header
                writer.writerow(dict(zip(keys, row)))
                written += 1
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, csv_path)

    with open(csv_path, "r", newline="") as fh:
        rd = csv.reader(fh)
        new_header = next(rd, None)
        widths = {len(r) for r in rd}
    report["rows_written"] = written
    report["verify_header_cols"] = len(new_header or [])
    report["verify_row_widths"] = sorted(widths)
    ok = (new_header == fieldnames and written == report["rows_total"]
          and widths <= {len(fieldnames)})
    report["verdict"] = "REPAIRED" if ok else "REFUSE"
    if not ok:
        report["problems"].append("post-repair verification failed")
    return report


def _print(report: Dict) -> None:
    keys = ("verdict", "header_cols", "current_cols", "rows_total",
            "rows_old_width", "rows_new_width", "banked", "rows_written",
            "verify_row_widths")
    print(f"[repair_csv_schema_drift] {report['csv']}")
    for k in keys:
        if report.get(k) is not None:
            print(f"  {k:<18}: {report[k]}")
    for p in report["problems"]:
        print(f"  {'PROBLEM':<18}: {p}")
    if report["verdict"] == "REPAIRABLE":
        print("  (dry run — re-run with --apply to repair)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fieldnames-from", required=True,
                    help="MODULE:ATTR, e.g. "
                         "src.drl.callbacks:MetricsCallback._FIELDNAMES")
    ap.add_argument("--apply", action="store_true",
                    help="perform the repair (default: dry-run report only)")
    args = ap.parse_args(argv)
    if not os.path.exists(args.csv):
        print(f"[repair_csv_schema_drift] IO: no such file: {args.csv}")
        return 3
    try:
        fieldnames = _load_fieldnames(args.fieldnames_from)
    except Exception as exc:
        print(f"[repair_csv_schema_drift] USAGE: {exc}")
        return 3
    try:
        report = repair(args.csv, fieldnames, apply=args.apply)
    except OSError as exc:
        print(f"[repair_csv_schema_drift] IO: {exc}")
        return 3
    _print(report)
    return 0 if report["verdict"] in ("CLEAN", "REPAIRED", "REPAIRABLE") else 1


if __name__ == "__main__":
    sys.exit(main())
