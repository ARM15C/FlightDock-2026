#!/usr/bin/env python3
# wind_table.py -- ANALYSIS_PLAN_COMPARATIVE.md section 7: the wind-and-turbulence table in m/s and mph, from every
#   leg's simulator wind record (fdm_witness.jsonl, 1200 Hz wind_ned) on all twelve corpora, every one of them read
#   from its FULL local exam tree (2026-09-17). The streaming path is retained but unused; when a corpus IS listed in
#   S3PREFIX it is streamed (aws s3 cp <key> -) and never stored. REPRODUCTION RULE (refuses, never warns): the SD
#   of the gust must equal the leg's banked realized sigma (sigma_gate.json sigma_ned) to six decimals on all three
#   axes (the 2026-09-04 gust instrument's limb W1). S0 legs have an empty witness by design and are reported as
#   0 m/s with turbulence off. Every wind figure is a MEAN wind speed; "stage" never "rung".
# PUBLIC API: S3PREFIX; leg_stats(samples) -> dict; s3_index(tag, out) -> DataFrame; table(per_leg) -> DataFrame;
#   main(argv=None) -> int. CLI: --run-dir DIR [--out DIR] [--workers 8] [--only TAG,...] [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/BOARD_W10.csv and each leg's local sigma_gate.json; lists S3 prefixes
#   (read-only) and streams witness objects; writes S3_INDEX_<tag>.csv, WIND_PER_LEG.csv (append, resumable) and
#   WIND_TABLE.csv/.md under <out> (default <run-dir>/comparative/). Spawns `aws s3 ls` / `aws s3 cp`. Never writes
#   to S3, never deletes, never modifies a source tree.
# SIZE: soft 210 / hard 240 code-lines; function cap 30.
# COVERAGE: --selftest: a synthetic witness with a planted gust SD reproduces its own sigma to 6 dp, a mismatched
#   gate REFUSES, the mph conversion is exact, the table holds the S0, nine-stage, pooled and strong-wind rows, and a
#   per-leg file whose FIRST row is a calm leg reads back (fixed layout and the old 6-column-header layout).
import argparse
import ast
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore_run as R  # noqa: E402

BUCKET = os.environ.get("FLIGHTDOCK_BUCKET", "s3://flightdock-reproduction-2026/")
#: Corpora whose wind records must be STREAMED from the evidence store. EMPTY since 2026-09-17: the five corpora the
#: laptop held only as stripped copies are being pulled in full, so every one of the twelve is read LOCALLY and this
#: tool touches S3 not at all. The streaming path below is kept, unused, for any corpus that is ever remote again;
#: a corpus that is neither here nor complete on disk REFUSES rc 7 rather than being scored on a partial tree.
S3PREFIX = {}
ENV = dict(os.environ)
MPH = 3600.0 / 1609.344
DP, SW = 6, ("W12", "W14", "W16")
#: the ONE column list of WIND_PER_LEG.csv. A calm leg carries only the first six, so the header may never be taken
#: from the first row written (2026-09-18: a calm leg wrote a 6-column header and the read-back refused).
PER_COLS = ["corpus", "leg", "source", "ok", "note", "n_samples", "steady_mag", "gust_p95", "gust_max", "total_p95",
            "total_max", "sd_N", "sd_E", "sd_D"]


def leg_stats(a):
    """Steady wind = per-axis mean; gust = deviation from it; totals = the delivered samples. Magnitudes in m/s."""
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return {"n_samples": 0}
    g = a - a.mean(axis=0)
    gm, tm = np.linalg.norm(g, axis=1), np.linalg.norm(a, axis=1)
    r = {"n_samples": int(len(a)), "steady_mag": float(np.linalg.norm(a.mean(axis=0))),
         "gust_p95": float(np.percentile(gm, 95)), "gust_max": float(gm.max()),
         "total_p95": float(np.percentile(tm, 95)), "total_max": float(tm.max())}
    for i, ax in enumerate("NED"):
        r["sd_" + ax] = float(g[:, i].std(ddof=0))
    return r


def read_samples(src):
    """src is a local path or an s3:// key; a streamed object never touches the disk."""
    if src.startswith("s3://"):
        p = subprocess.Popen(["aws", "s3", "cp", src, "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
        out = [json.loads(l)["wind_ned"] for l in p.stdout if l.strip()]
        err = p.stderr.read().decode(errors="ignore")
        if p.wait() != 0:
            raise RuntimeError("aws s3 cp failed for %s: %s" % (src, err[-300:]))
        return out
    with open(src) as fh:
        return [json.loads(l)["wind_ned"] for l in fh if l.strip()]


def check_gate(gate_path, st):
    g = json.load(open(gate_path))
    if g.get("state") == "CALM_CONFIRMED":
        return st["n_samples"] == 0 or "calm leg carries samples"
    ned = g.get("sigma_ned")
    if not (isinstance(ned, list) and len(ned) == 3):
        return "no sigma_ned in gate"
    off = max(abs(st["sd_" + ax] - float(ned[i])) for i, ax in enumerate("NED"))
    return off <= 10 ** -DP or "gust SD differs from banked sigma_ned by %.2e" % off


def work(job):
    tag, leg, src, gate = job
    try:
        st = leg_stats(read_samples(src))
        ok = check_gate(gate, st)
        return dict(corpus=tag, leg=leg, source=src, ok=ok is True, note="" if ok is True else ok, **st)
    except Exception as e:  # recorded, never silent; the run REFUSES on any not-ok row
        return {"corpus": tag, "leg": leg, "source": src, "ok": False, "note": "error: %s" % e}


def s3_index(tag, out):
    """(leg -> key) for one corpus from a read-only recursive listing, cached under <out>."""
    cache = os.path.join(out, "S3_INDEX_%s.csv" % tag)
    if os.path.exists(cache):
        return pd.read_csv(cache)
    ls = subprocess.run(["aws", "s3", "ls", "--recursive", BUCKET + S3PREFIX[tag]], capture_output=True, text=True, env=ENV)
    if ls.returncode != 0:
        raise SystemExit("REFUSE rc 5: s3 ls failed for %s: %s" % (tag, ls.stderr[-300:]))
    rows = [l.split(None, 3) for l in ls.stdout.splitlines() if l.endswith("/fdm_witness.jsonl")]
    df = pd.DataFrame(rows, columns=["date", "time", "bytes", "key"])
    df["leg"] = df.key.str.split("/").str[-2]
    df["bytes"] = df.bytes.astype(int)
    df[["leg", "key", "bytes"]].to_csv(cache, index=False)
    return df[["leg", "key", "bytes"]]


def jobs_for(tag, board, out):
    """One (tag, leg, source, gate_path) per board leg of the corpus; the gate is always the local sigma_gate.json."""
    root = R.resolve(tag)
    gates = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(os.path.join(root, "**", "sigma_gate.json"), recursive=True)}
    legs = board[board.corpus == tag].leg.tolist()
    missing = [l for l in legs if l not in gates]
    if missing:
        raise SystemExit("REFUSE rc 6: %s: %d legs have no local sigma_gate.json (e.g. %s)" % (tag, len(missing), missing[:2]))
    if tag in S3PREFIX:
        idx = s3_index(tag, out).set_index("leg").key
        return [(tag, l, BUCKET + idx[l], gates[l]) for l in legs if l in idx.index], [l for l in legs if l not in idx.index]
    loc = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(os.path.join(root, "**", "fdm_witness.jsonl"), recursive=True)}
    return [(tag, l, loc[l], gates[l]) for l in legs if l in loc], [l for l in legs if l not in loc]


def append_row(per, row):
    pd.DataFrame([row]).reindex(columns=PER_COLS).to_csv(per, mode="a", header=not os.path.exists(per), index=False)


def read_per(per):
    """Read by PER_COLS, never by the file's own header, so a file whose header came from a calm leg still loads."""
    return pd.read_csv(per, names=PER_COLS, header=None, skiprows=1)


def run_legs(tags, board, out, workers):
    per = os.path.join(out, "WIND_PER_LEG.csv")
    done = set(read_per(per).apply(lambda r: (r.corpus, r.leg), axis=1)) if os.path.exists(per) else set()
    for tag in tags:
        jobs, absent = jobs_for(tag, board, out)
        if absent:
            raise SystemExit("REFUSE rc 7: %s: %d legs have no witness object (e.g. %s)" % (tag, len(absent), absent[:2]))
        jobs = [j for j in jobs if (j[0], j[1]) not in done]
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
                append_row(per, row)
                if i % 100 == 0:
                    print("%s %d/%d" % (tag, i, len(jobs)), flush=True)
    return read_per(per)


def table(per):
    """Rows: S0; each stage at each sigma; pooled nine stages; strong wind. Columns in m/s with mph beside."""
    p = per.copy()
    p["stage_n"] = p.wind_label.map(lambda s: -1 if s == "S0" else int(s[1:]) // 2)
    rows = []
    def add(name, sigma, x):
        turb = x[x.n_samples > 0]
        v = {"steady_min": turb.steady_mag.min(), "steady_max": turb.steady_mag.max(), "gust_typical_p95": turb.gust_p95.mean(),
             "gust_largest": turb.gust_max.max(), "total_typical_p95": turb.total_p95.mean(), "total_extreme": turb.total_max.max()} if len(turb) else \
            {k: 0.0 for k in ("steady_min", "steady_max", "gust_typical_p95", "gust_largest", "total_typical_p95", "total_extreme")}
        r = {"wind stage": name, "turbulence sigma": sigma, "legs": len(x)}
        for k, val in v.items():
            r[k + "_ms"], r[k + "_mph"] = round(float(val), 2), round(float(val) * MPH, 1)
        rows.append(r)
    add("S0 calm control stage (turbulence off)", "0.00 (off)", p[p.wind_label == "S0"])
    for sigma in sorted(p[p.wind_label != "S0"].sigma.unique()):
        q = p[(p.sigma == sigma) & (p.wind_label != "S0")]
        for st in sorted(q.stage_n.unique()):
            add("%d m/s mean wind" % st, "%.2f" % sigma, q[q.stage_n == st])
        add("nine wind stages pooled, 0-8 m/s mean wind", "%.2f" % sigma, q)
        add("strong-wind (6-8 m/s) mean wind", "%.2f" % sigma, q[q.wind_label.isin(SW)])
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir"), ap.add_argument("--out"), ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only"), ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    board = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"), low_memory=False)
    out = a.out or os.path.join(a.run_dir, "comparative")
    os.makedirs(out, exist_ok=True)
    tags = a.only.split(",") if a.only else list(R.CORPORA)
    per = run_legs(tags, board, out, a.workers)
    bad = per[~per.ok.astype(bool)]
    if len(bad):
        print(bad[["corpus", "leg", "note"]].head(10).to_string())
        raise SystemExit("REFUSE rc 9: %d legs failed the gate reproduction or the read" % len(bad))
    per = per.merge(board[["corpus", "leg", "wind_label", "sigma"]], on=["corpus", "leg"])
    t = table(per)
    t.to_csv(os.path.join(out, "WIND_TABLE.csv"), index=False)
    open(os.path.join(out, "WIND_TABLE.md"), "w").write(_md(t))
    print("wind table: %d rows from %d legs (%d turbulent)" % (len(t), len(per), int((per.n_samples > 0).sum())))
    return 0


def _md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"


def selftest():
    import tempfile
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        a = rng.normal([6.0, 0.0, 0.0], [0.25, 0.25, 0.25], size=(2000, 3))
        w, g = os.path.join(td, "fdm_witness.jsonl"), os.path.join(td, "sigma_gate.json")
        open(w, "w").write("\n".join(json.dumps({"t": i, "wind_ned": list(map(float, v))}) for i, v in enumerate(a)) + "\n")
        st = leg_stats(read_samples(w))
        json.dump({"state": "MEASURED_PASS", "sigma_ned": [round(st["sd_N"], DP), round(st["sd_E"], DP), round(st["sd_D"], DP)]}, open(g, "w"))
        assert check_gate(g, st) is True
        json.dump({"state": "MEASURED_PASS", "sigma_ned": [0.3, 0.3, 0.3]}, open(g, "w"))
        assert check_gate(g, st) is not True
        assert abs(1.0 * MPH - 2.2369362920544) < 1e-9
        per = pd.DataFrame([dict(corpus="X", leg="S0_1_s", wind_label="S0", sigma=0.25, n_samples=0)] +
                           [dict(corpus="X", leg="W%d_1_s" % w_, wind_label="W%d" % w_, sigma=s, n_samples=10, steady_mag=w_ / 2.0,
                                 gust_p95=0.6, gust_max=1.2, total_p95=w_ / 2.0 + 0.5, total_max=w_ / 2.0 + 1.0)
                            for s in (0.25, 0.35) for w_ in range(0, 17, 2)])
        t = table(per)
        f = os.path.join(td, "per.csv")  # calm leg FIRST, then a wind leg, in both the fixed and the old ragged layout
        json.dump({"state": "CALM_CONFIRMED"}, open(g + "c", "w"))
        calm, windy = work(("X", "S0_1_s", os.devnull, g + "c")), dict(corpus="X", leg="W2_1_s", source=w, ok=True, note="", **st)
        append_row(f, calm), append_row(f, windy)
        open(f + "2", "w").write("corpus,leg,source,ok,note,n_samples\nX,S0_1_s,s,True,,0\nX,W2_1_s,s,True,,2000," + ",".join(["1.5"] * 8) + "\n")
        for q in (read_per(f), read_per(f + "2")):
            assert q.n_samples.tolist() == [0, 2000] and q.sd_D.notna().tolist() == [False, True] and list(q.columns) == PER_COLS
        assert len(t) == 1 + 2 * (9 + 2) and t.iloc[0].steady_max_ms == 0.0 and t[t["wind stage"].str.startswith("strong-wind")].steady_min_ms.tolist() == [6.0, 6.0]
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
