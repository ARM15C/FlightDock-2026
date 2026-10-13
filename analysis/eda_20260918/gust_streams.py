#!/usr/bin/env python3
# =============================================================================
# . RESPONSIBILITY: which gust random-draw stream each leg actually flew, and every key
#   (stage, flight seed, trajectory) where the arms at one sigma flew different streams, with the cause read from the
#   leg's own records. Reads ONLY: results_eid*.json (flown episode number) and each episode folder's
#   warmup_handoff_env*.jsonl (fields ep_idx, attempt, weather_seed, wind_seed, outcome, refused). No outcome field
#   of the flight is read. No scoring or analysis tool is imported.
# PUBLIC API: leg_streams(corpus, legdir) -> dict; classify(df) -> DataFrame; main(argv=None) -> int.
#   CLI: --out DIR [--workers N] [--corpora ...]
# SIDE-EFFECTS: writes <out>/GUST_STREAMS_legs.csv, GUST_STREAMS_keys.csv, GUST_STREAMS_SUMMARY.md. Reads only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd

CORP = (_os.environ.get("FLIGHTDOCK_CORPORA", _REPO + "/corpora"))
FIELDS = ("ep_idx", "attempt", "weather_seed", "wind_seed", "outcome", "refused")


def legdirs(corpus):
    out = []
    for d in sorted(glob.glob(os.path.join(CORP, corpus, "B*", "*_*_*"))):
        b = os.path.basename(d)
        if os.path.isdir(d) and not b.startswith("episode_") and b.split("_")[1].isdigit():
            out.append(d)
    return out


def handoff_lines(epdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(epdir, "warmup_handoff_env*.jsonl"))):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                j = json.loads(ln)
                rows.append({k: j.get(k) for k in FIELDS})
    return rows


def leg_streams(corpus, d):
    lab, seed, traj = os.path.basename(d).split("_", 2)
    r = {"corpus": corpus, "arm": corpus.split("_")[0], "sigma": corpus.split("_")[1], "key": "%s_%s_%s" % (lab, seed, traj),
         "stage": lab, "flight_seed": int(seed), "traj": traj}
    # results_eid numbers every leg's episode 1 (single-episode legs too), so it cannot name the flown folder. The
    # flown folder is the one whose finalized terminal sim time equals the flight-state record's last tick.
    folders = sorted(glob.glob(os.path.join(d, "episode_*")))
    r["n_episode_folders"] = len(folders)
    fs = glob.glob(os.path.join(d, "env[0-9]*", "flight_state_ep*.csv.gz"))
    t_end = float(pd.read_csv(fs[0], usecols=["sim_time_s"]).sim_time_s.max()) if len(fs) == 1 else None
    r["fs_sim_t_max"] = t_end
    term = {}
    for f in folders:
        p = os.path.join(f, "run_summary_finalized.json")
        term[f] = json.load(open(p)).get("terminal_sim_time_s") if os.path.exists(p) else None
    r["folder_terminal_t"] = ";".join("%s:%s" % (os.path.basename(f)[:11], term[f]) for f in folders)
    fl = [f for f in folders if term[f] is not None and t_end is not None and abs(term[f] - t_end) < 0.5]
    r["flown_folder_found"] = len(fl) == 1
    r["flown_ep"] = int(os.path.basename(fl[0]).split("_")[1]) if len(fl) == 1 else None
    lines = handoff_lines(fl[0]) if len(fl) == 1 else []
    r["n_attempts"] = len(lines)
    used = [x for x in lines if x["refused"] is False]
    r["n_accepted_attempts"] = len(used)
    u = used[-1] if used else (lines[-1] if lines else {})
    r.update({"flown_attempt": u.get("attempt"), "flown_weather_seed": u.get("weather_seed"), "flown_wind_seed": u.get("wind_seed"),
              "flown_attempt_outcome": u.get("outcome"),
              "attempt_seeds": ";".join("%s:%s" % (x["attempt"], x["weather_seed"]) for x in lines)})
    first = [x for f in folders for x in handoff_lines(f) if x["ep_idx"] == 0 and x["attempt"] == 1]
    r["ep0_attempt1_weather_seed"] = first[0]["weather_seed"] if first else None
    return r


def _work(a):
    try:
        return leg_streams(*a)
    except Exception as ex:
        return {"corpus": a[0], "key": os.path.basename(a[1]), "row_error": repr(ex)[:200]}


def classify(df):
    """Per (sigma, key): the stream most arms flew, and each arm that flew another one, with its cause."""
    rows = []
    for (sig, key), g in df.groupby(["sigma", "key"]):
        if g.stage.iloc[0] == "S0":
            continue
        seeds = g.flown_weather_seed.astype(str)
        maj = seeds.value_counts()
        if maj.empty:
            continue
        base = maj.index[0]
        base_att = g[seeds == base].flown_attempt.mode().iloc[0]
        for r in g[seeds != base].itertuples():
            if r.flown_ep not in (0, None):
                cause = "timebase retry (flew episode %s)" % r.flown_ep
            elif r.flown_attempt != base_att:
                cause = "warmup re-roll: flew attempt %s, majority flew attempt %s" % (r.flown_attempt, base_att)
            else:
                cause = "same episode and attempt, different stream"
            rows.append({"sigma": sig, "key": key, "n_arms": len(g), "n_distinct_streams": maj.size,
                         "majority_stream": base, "majority_n": int(maj.iloc[0]), "tie": bool(maj.size > 1 and maj.iloc[0] == maj.iloc[1]),
                         "odd_arm": r.arm, "odd_stream": r.flown_weather_seed, "cause": cause})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--corpora", nargs="*")
    a = ap.parse_args(argv)
    corpora = a.corpora or sorted(c for c in os.listdir(CORP) if os.path.isdir(os.path.join(CORP, c)))
    jobs = [(c, d) for c in corpora for d in legdirs(c)]
    print("corpora", corpora, "legs", len(jobs), flush=True)
    with Pool(a.workers) as p:
        df = pd.DataFrame(p.map(_work, jobs, chunksize=8)).sort_values(["sigma", "key", "arm"])
    df.to_csv(os.path.join(a.out, "GUST_STREAMS_legs.csv"), index=False)
    k = classify(df)
    k.to_csv(os.path.join(a.out, "GUST_STREAMS_keys.csv"), index=False)
    md = ["# Gust random-draw stream census (outcome-blind), ", "",
          "Legs read: %d; row errors: %d; corpora: %s" % (len(df), int(df.get("row_error", pd.Series(dtype=object)).notna().sum()), ", ".join(corpora)), ""]
    per = df.groupby("corpus").agg(legs=("key", "size"), flown_ep_not_0=("flown_ep", lambda s: int((s != 0).sum())),
                                   flown_folder_found=("flown_folder_found", "sum"), no_accepted_attempt=("n_accepted_attempts", lambda s: int((s == 0).sum())),
                                   reroll_legs=("n_attempts", lambda s: int((s > 1).sum())))
    md += ["## Per corpus", "", "```", per.to_string(), "```", ""]
    windy = df[df.stage != "S0"]
    for sig, g in windy.groupby("sigma"):
        nk = g.key.nunique()
        mism = k[k.sigma == sig].key.nunique() if len(k) else 0
        md += ["## sigma %s: %d windy keys, %d with every arm on one stream, %d with at least one arm on another" % (sig, nk, nk - mism, mism), ""]
        if len(k):
            md += ["```", k[k.sigma == sig].groupby("cause").size().to_string(), "```", "",
                   "Odd legs by arm:", "```", k[k.sigma == sig].groupby("odd_arm").size().to_string(), "```", ""]
    open(os.path.join(a.out, "GUST_STREAMS_SUMMARY.md"), "w").write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
