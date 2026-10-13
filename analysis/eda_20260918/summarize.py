#!/usr/bin/env python3
# =============================================================================
# . RESPONSIBILITY: turn CENSUS_legs.csv + CENSUS_config.json into SUMMARY.md: per-corpus
#   completeness, label agreement, the like-for-like pairing key set, channel census, residual-channel witness and the
#   run configuration. Every number is a pandas aggregate of the census; no outcome is computed.
# PUBLIC API: checks(df) -> (per_corpus DataFrame, findings list); main(argv=None) -> int. CLI: --out DIR
# SIDE-EFFECTS: writes <out>/SUMMARY.md and <out>/FINDINGS.json. Reads only.
# =============================================================================
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SIGMA = {"s025": 0.25, "s035": 0.35}
STAGE_WIND = {"S0": 0.0, "W0": 0.0, "W2": 1.0, "W4": 2.0, "W6": 3.0, "W8": 4.0, "W10": 5.0, "W12": 6.0, "W14": 7.0, "W16": 8.0}
CKPT = {"A2": "3d1cf6ca", "A2O": "3d1cf6ca", "S1": "34c7f116", "S1O": "34c7f116", "PIDO": "34c7f116", "PID": "a70c4a88"}


def md_table(t):
    """Markdown table without the optional tabulate dependency."""
    cols = list(t.columns)
    lines = ["| check | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        lines.append("| %s | " % idx + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def per_corpus(df):
    rows = []
    for c, g in df.groupby("corpus"):
        arm, sig = c.split("_")
        r = {"corpus": c, "legs": len(g), "row_errors": int(g.get("row_error", pd.Series(index=g.index, dtype=object)).notna().sum()),
             "cells": g.cell.nunique(), "seeds": g.name_seed.nunique(), "seed_min": int(g.name_seed.min()), "seed_max": int(g.name_seed.max()),
             "env_dirs=1": int((g.n_env_dirs == 1).sum()), "paired_env0": int(g.paired_env0_ok.sum())}
        for k in ["results", "leg_meta", "manifest", "metrics", "safety", "sigma_gate", "fdm", "episode_dir", "success_metrics",
                  "flight_state", "paired_channels", "epoch_stamp"]:
            r[k] = int((g["n_" + k] == 1).sum())
        r["lane_log"] = ",".join("%s:%d" % (k, v) for k, v in g.lane_log.value_counts().items())
        r["name=meta (seed,traj,stage)"] = int(((g.name_seed == g.meta_seed) & (g.name_traj == g.meta_traj) & (g.name_label == g.meta_stage)).sum())
        r["name=manifest seed"] = int((g.name_seed == g.man_seed).sum())
        r["ckpt expected"] = CKPT[arm]
        r["ckpt matches"] = int(g.meta_ckpt.astype(str).str.startswith(CKPT[arm]).sum())
        r["abort_pre_control"] = int(g.abort_pre_control.astype(str).eq("True").sum())
        # the label is written "0.250;0.250;0.250" (N;E;D) on windy legs and blank on S0: every axis must equal the corpus sigma
        axes = g.lbl_turbulence_sigma_ms.fillna("").astype(str).str.split(";")
        want = np.where(g.name_label == "S0", 0.0, SIGMA[sig])
        r["sigma label = corpus"] = int(sum(all(abs(float(x or 0.0) - w) < 1e-9 for x in ax) for ax, w in zip(axes, want)))
        r["wind label = stage"] = int(np.isclose(pd.to_numeric(g.lbl_wind_speed_ms, errors="coerce"), g.name_label.map(STAGE_WIND)).sum())
        r["profile = traj"] = int((g.lbl_leader_profile == g.name_traj).sum())
        r["cell label = stage"] = int((g.lbl_v3_cell_label == g.name_label).sum())
        r["gate passed"] = int(g.gate_passed.astype(str).eq("True").sum())
        r["gate states"] = ",".join("%s:%d" % (k, v) for k, v in g.gate_state.value_counts().items())
        r["gate malformed>0"] = int((pd.to_numeric(g.gate_malformed, errors="coerce") > 0).sum())
        r["fdm empty (S0)"] = int(((g.fdm_bytes == 0) & (g.name_label == "S0")).sum())
        r["fdm empty (windy)"] = int(((g.fdm_bytes == 0) & (g.name_label != "S0")).sum())
        r["fdm parse err"] = int(g.get("fdm_parse_error", pd.Series(index=g.index, dtype=object)).notna().sum())
        r["fs rows min/med/max"] = "%d/%d/%d" % (g.fs_rows.min(), g.fs_rows.median(), g.fs_rows.max())
        r["fs cols"] = ",".join(map(str, sorted(g.fs_cols.dropna().unique().astype(int))))
        r["fs colset"] = ",".join(sorted(g.fs_colset_sha.dropna().astype(str).unique()))
        r["fs inf cells"] = int(g.fs_inf_cells.sum())
        r["fs nan cells med"] = int(g.fs_nan_cells.median())
        r["fs dup ticks"] = int(g.fs_dup_ticks.sum())
        r["fs object cols"] = ",".join(sorted({x for x in g.fs_object_cols.dropna().astype(str) if x}))
        r["res_source"] = ",".join(sorted(g.res_source_set.dropna().unique()))
        r["res_applied all zero (0,1,2)"] = "%d,%d,%d" % tuple(int(g["zero_res_applied_%d" % i].astype(str).eq("True").sum()) for i in range(3))
        r["res_policy all zero (0,1,2)"] = "%d,%d,%d" % tuple(int(g["zero_res_policy_%d" % i].astype(str).eq("True").sum()) for i in range(3))
        r["res nan"] = int(sum(g["nan_res_applied_%d" % i].sum() + g["nan_res_policy_%d" % i].sum() for i in range(3)))
        r["ach types"] = ",".join(sorted(g.ach_types.dropna().unique()))
        r["instrument_gap"] = int(g.safety_instrument_gap.astype(str).eq("True").sum())
        r["metrics rows=1"] = int((g.metrics_rows == 1).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("corpus")


def pairing(df):
    """The like-for-like key: (stage, seed, trajectory). Every corpus must carry the same 600 keys."""
    keys = {c: set(zip(g.name_label, g.name_seed, g.name_traj)) for c, g in df.groupby("corpus")}
    ref = set.intersection(*keys.values())
    return {"common_keys": len(ref), "per_corpus": {c: {"keys": len(k), "missing_vs_common": len(ref - k), "extra_vs_common": len(k - ref)} for c, k in keys.items()},
            "seed_block": sorted({s for k in keys.values() for _, s, _ in k})}


def gust_identity(df):
    """Same key across arms at the same sigma: does the FDM witness record the same measured gust triple? Reported as a
    fact about the flights (known: warmup retries re-roll the stream), not as a verdict."""
    out = {}
    windy = df[df.name_label != "S0"].copy()
    windy["key"] = list(zip(windy.name_label, windy.name_seed, windy.name_traj))
    windy["trip"] = list(zip(windy.gate_sigma_n.round(6), windy.gate_sigma_e.round(6), windy.gate_sigma_d.round(6), windy.fdm_t_first.round(6)))
    for sig, g in windy.groupby(windy.corpus.str.split("_").str[1]):
        piv = g.pivot_table(index="key", columns="corpus", values="trip", aggfunc="first")
        n_ident = int((piv.nunique(axis=1) == 1).sum())
        out[sig] = {"keys": len(piv), "identical_across_all_arms": n_ident, "corpora": list(piv.columns)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = pd.read_csv(os.path.join(a.out, "CENSUS_legs.csv"), low_memory=False)
    cfg = json.load(open(os.path.join(a.out, "CENSUS_config.json")))
    pc = per_corpus(df)
    pr = pairing(df)
    gi = gust_identity(df)
    md = ["# — outcome-blind census of the benchmark corpora", "",
          "Source: CENSUS_legs.csv (%d rows, one per leg) and CENSUS_config.json, read from /home/ubuntu/flightdock_corpora/. No outcome is computed." % len(df), "",
          "## Per-corpus census (each cell is a count of legs, out of the `legs` column, unless labelled otherwise)", "", md_table(pc.T), "",
          "## Pairing key set: (stage, seed, trajectory)", "", "```", json.dumps(pr, indent=1), "```", "",
          "## Measured gust triple identical across arms at the same key (fact, not verdict)", "", "```", json.dumps(gi, indent=1), "```", "",
          "## Run configuration as witnessed", ""]
    for c in cfg:
        md += ["### %s" % c["corpus"], "", "- resolves to `%s`" % c["resolves_to"]]
        if "overlay_plus_lines" in c:
            md += ["- overlay dose %s" % c.get("overlay_dose")] + ["- `%s`" % l[:120] for l in c["overlay_plus_lines"]]
        else:
            md += ["- overlay witness: %s" % c.get("overlay_witness")]
        md += ["- venue checkpoint: `%s` sha `%s`" % (c.get("venue_ckpt"), c.get("venue_ckpt_sha256"))] if "venue_ckpt" in c else ["- venue witness: %s" % c.get("venue_witness")]
        md += [""]
    open(os.path.join(a.out, "SUMMARY.md"), "w").write("\n".join(md))
    json.dump({"pairing": pr, "gust_identity": gi}, open(os.path.join(a.out, "FINDINGS.json"), "w"), indent=1)
    print(pc.T.to_string())
    print(json.dumps(pr["per_corpus"]), json.dumps(gi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
