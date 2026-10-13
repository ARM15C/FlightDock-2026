#!/usr/bin/env python3
# =============================================================================
# . RESPONSIBILITY: one row per leg per corpus in /home/ubuntu/flightdock_corpora/, read
#   straight from the producer files with pandas/json. Records completeness, labels, pairing keys, channel census and
#   the configuration witnessed for the leg. It computes NO outcome: no docking rate, no dwell, no peak height, no
#   breakaway count. It reads no scoring or analysis tool.
# PUBLIC API: leg_row(corpus, legdir) -> dict; main(argv=None) -> int. CLI: --out DIR [--workers N] [--corpora ...]
# SIDE-EFFECTS: writes <out>/CENSUS_legs.csv and <out>/CENSUS_config.json. Reads only. Spawns worker processes.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

CORP = (_os.environ.get("FLIGHTDOCK_CORPORA", _REPO + "/corpora"))
ACH = ["actual_M2_achieved", "actual_M3a_coarse_achieved", "actual_M3a_fine_achieved", "actual_M3b_achieved"]
LABELS = ["leader_profile", "wind_speed_ms", "wind_from_deg", "turbulence_sigma_ms", "gps_cep_m", "v3_cell_label"]


def legdirs(corpus):
    """Leg directories: <cell>/<label>_<seed>_<traj>, cells nested (B25/W0_…) or hyphenated (B25-W0/W0_…)."""
    out = []
    for d in sorted(glob.glob(os.path.join(CORP, corpus, "B*", "*_*_*"))):
        b = os.path.basename(d)
        if os.path.isdir(d) and not b.startswith("episode_") and b.split("_")[1].isdigit():
            out.append(d)
    return out


def _one(path):
    h = glob.glob(path)
    return h[0] if len(h) == 1 else None


def first_last_line(path):
    with open(path, "rb") as f:
        first = f.readline()
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - 4096))
        last = f.read().rstrip(b"\n").split(b"\n")[-1]
    return first, last, n


def leg_row(corpus, d):
    cell, leg = os.path.basename(os.path.dirname(d)), os.path.basename(d)
    lab, seed_s, traj = leg.split("_", 2)
    r = {"corpus": corpus, "cell": cell, "leg": leg, "name_label": lab, "name_seed": int(seed_s), "name_traj": traj}
    envs = sorted(glob.glob(os.path.join(d, "env[0-9]*")))
    r["n_env_dirs"] = len(envs)
    r["paired_env0_ok"] = os.path.isdir(os.path.join(d, "paired", "env0"))
    for k, pat in [("results", "results_eid*.json"), ("leg_meta", "leg_meta.json"), ("manifest", "manifest_*.json"),
                   ("metrics", "metrics_*.csv"), ("safety", "safety_summary.json"), ("sigma_gate", "sigma_gate.json"),
                   ("fdm", "fdm_witness.jsonl"), ("reward_terms", "REWARD_TERMS_DECLARED.json"),
                   ("episode_dir", "episode_*"), ("success_metrics", "episode_*/success_metrics.json"),
                   ("flight_state", "env[0-9]*/flight_state_ep*.csv.gz"), ("paired_channels", "env[0-9]*/paired_channels_ep*.csv"),
                   ("epoch_stamp", "env[0-9]*/epoch_stamp_ep*.csv")]:
        r["n_" + k] = len(glob.glob(os.path.join(d, pat)))
    r["lane_log"] = ("leg" if os.path.exists(os.path.join(d, "lane.log")) else
                     "cell" if os.path.exists(os.path.join(os.path.dirname(d), "lane_%s_%s.log" % (seed_s, traj))) else "none")
    m = json.load(open(os.path.join(d, "leg_meta.json"))) if r["n_leg_meta"] else {}
    r.update({"meta_seed": m.get("seed"), "meta_traj": m.get("trajectory"), "meta_stage": m.get("stage_label"),
              "meta_ckpt": m.get("checkpoint_sha256"), "abort_pre_control": m.get("abort_pre_control")})
    mf = _one(os.path.join(d, "manifest_*.json"))
    if mf:
        mm = json.load(open(mf))
        r.update({"man_seed": mm.get("seed_value"), "man_stage": mm.get("stage"), "man_env_id": mm.get("env_id"),
                  "man_ckpt_path": mm.get("checkpoint_path"), "man_vecnorm_sha": mm.get("vecnormalize_sha256"),
                  "man_vecnorm_training_flag": mm.get("vecnormalize_training_flag"), "man_n_ep": mm.get("n_episodes_completed"),
                  "man_wave": (mm.get("venue_env") or {}).get("ARMB_WAVE")})
    rf = _one(os.path.join(d, "results_eid*.json"))
    if rf:
        rr = json.load(open(rf))
        eps = rr.get("episodes") or []
        r.update({"res_stage": rr.get("stage"), "res_n_episodes": rr.get("n_episodes"), "res_n_records": len(eps),
                  "res_ckpt_name": rr.get("checkpoint")})
        if eps:
            e = eps[0]
            r.update({"lbl_" + k: e.get(k) for k in LABELS})
            r["ach_types"] = ",".join(sorted({type(e.get(k)).__name__ + ":" + str(e.get(k)) for k in ACH}))
    sg = os.path.join(d, "sigma_gate.json")
    if os.path.exists(sg):
        g = json.load(open(sg))
        h = g.get("harvest") or {}
        r.update({"gate_passed": g.get("passed"), "gate_state": g.get("state"), "gate_n": g.get("n"),
                  "gate_cmd_sigma": (g.get("commanded_ned") or [None])[0],
                  "gate_sigma_n": (g.get("sigma_ned") or [None] * 3)[0], "gate_sigma_e": (g.get("sigma_ned") or [None] * 3)[1],
                  "gate_sigma_d": (g.get("sigma_ned") or [None] * 3)[2], "gate_kept": h.get("kept"), "gate_malformed": h.get("malformed"),
                  "gate_windless": h.get("windless")})
    fp = os.path.join(d, "fdm_witness.jsonl")
    if os.path.exists(fp):
        a, b, n = first_last_line(fp)
        r["fdm_bytes"] = n
        if n == 0:
            r["fdm_t_first"] = r["fdm_t_last"] = None   # the calm stage S0 writes an empty witness (harvest windless)
        else:
          try:
            ja, jb = json.loads(a), json.loads(b)
            r.update({"fdm_t_first": ja["fdm_timestamp"], "fdm_t_last": jb["fdm_timestamp"],
                      "fdm_wind_first_ms": float(np.linalg.norm(ja["wind_ned"][:2])), "fdm_wind_last_ms": float(np.linalg.norm(jb["wind_ned"][:2]))})
          except Exception as ex:  # a malformed first/last line is itself a finding
            r["fdm_parse_error"] = str(ex)[:80]
    sf = os.path.join(d, "safety_summary.json")
    if os.path.exists(sf):
        r["safety_instrument_gap"] = json.load(open(sf)).get("instrument_gap")
    fs = _one(os.path.join(d, "env[0-9]*", "flight_state_ep*.csv.gz"))
    if fs:
        df = pd.read_csv(fs, compression="gzip", low_memory=False)
        num = df.select_dtypes(include=[np.number])
        r.update({"fs_rows": len(df), "fs_cols": len(df.columns),
                  "fs_colset_sha": hashlib.sha256(",".join(df.columns).encode()).hexdigest()[:12],
                  "fs_sim_t_min": float(df.sim_time_s.min()), "fs_sim_t_max": float(df.sim_time_s.max()),
                  "fs_nan_cells": int(df.isna().sum().sum()), "fs_inf_cells": int(np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).sum()),
                  "fs_object_cols": ",".join(df.columns[df.dtypes == object]),
                  "fs_dup_ticks": int(df.tick_idx.duplicated().sum()) if "tick_idx" in df else None,
                  "fs_env_ids": ",".join(map(str, sorted(df.env_id.unique()))) if "env_id" in df else None})
        for c in ["res_policy_0", "res_policy_1", "res_policy_2", "res_applied_0", "res_applied_1", "res_applied_2"]:
            r["zero_" + c] = bool((df[c].astype(float).abs() < 1e-12).all()) if c in df else None
            r["nan_" + c] = int(df[c].isna().sum()) if c in df else None
        r["res_source_set"] = ",".join(map(str, sorted(df.res_source.astype(str).unique()))) if "res_source" in df else None
    mc = _one(os.path.join(d, "metrics_*.csv"))
    if mc:
        r["metrics_rows"] = len(pd.read_csv(mc))
    return r


def _work(args):
    try:
        return leg_row(*args)
    except Exception as ex:
        return {"corpus": args[0], "leg": os.path.basename(args[1]), "cell": os.path.basename(os.path.dirname(args[1])), "row_error": repr(ex)[:200]}


def config_witness(corpus):
    """Run-level configuration as witnessed: the '+' lines of the overlay diff (floor, offset, action scales) and the
    checkpoint the venue witness names. Absent files are reported as absent, never inferred."""
    root = os.path.join(CORP, corpus)
    out = {"corpus": corpus, "resolves_to": os.path.realpath(root)}
    ow = os.path.join(root, "OVERLAY_WITNESS.json")
    if os.path.exists(ow):
        d = json.load(open(ow))
        out["overlay_dose"] = d.get("dose")
        out["overlay_plus_lines"] = [l for l in d.get("diff", "").split("\n") if l.startswith("+") and not l.startswith("+++")]
    else:
        out["overlay_witness"] = "absent"
    vw = os.path.join(root, "VENUE_WITNESS.json")
    if os.path.exists(vw):
        d = json.load(open(vw))
        out.update({"venue_ckpt": d.get("ckpt"), "venue_ckpt_sha256": d.get("ckpt_sha256")})
    else:
        out["venue_witness"] = "absent"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--corpora", nargs="*")
    a = ap.parse_args(argv)
    corpora = a.corpora or sorted(c for c in os.listdir(CORP) if os.path.isdir(os.path.join(CORP, c)))
    os.makedirs(a.out, exist_ok=True)
    json.dump([config_witness(c) for c in corpora], open(os.path.join(a.out, "CENSUS_config.json"), "w"), indent=1)
    jobs = [(c, d) for c in corpora for d in legdirs(c)]
    print("corpora", corpora, "legs", len(jobs), flush=True)
    with Pool(a.workers) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(_work, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 300 == 0:
                print("done", i, "of", len(jobs), flush=True)
    df = pd.DataFrame(rows).sort_values(["corpus", "cell", "leg"])
    df.to_csv(os.path.join(a.out, "CENSUS_legs.csv"), index=False)
    print("wrote", len(df), "rows; row_errors", int(df.get("row_error", pd.Series(dtype=object)).notna().sum()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
