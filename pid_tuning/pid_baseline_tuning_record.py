#!/usr/bin/env python3
"""PID-baseline tuning record: every number from the two census files beside this script.

Early period (26 Nov 2025 to 23 Feb 2026): PID_TUNING_EARLY_CONTACT_CENSUS_20260930.csv, one row per flight log of the
numbered tuning tests. Late period (20 June to 19 July 2026): PID_TUNING_LATE_CONTACT_CENSUS_20260930.csv, one row per
finished flight. Writes PID_BASELINE_TUNING_RECORD.md. pandas throughout.
"""
import pathlib
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
EARLY = HERE / "PID_TUNING_EARLY_CONTACT_CENSUS_20260930.csv"
LATE = HERE / "PID_TUNING_LATE_CONTACT_CENSUS_20260930.csv"
OUT = HERE / "PID_BASELINE_TUNING_RECORD.md"
FINAL_FROM = "2026-07-03"

e = pd.read_csv(EARLY)
em = e.groupby("month").agg(tests=("tag", "nunique"), flights=("f", "size"), pre=("precontact_hold", "sum"),
                            creep=("creep", "sum"), contact=("contact", "sum")).reset_index()
tot = dict(tests=e.tag.nunique(), flights=len(e), pre=int(e.precontact_hold.sum()), creep=int(e.creep.sum()), contact=int(e.contact.sum()))
MON = {"2025-11": "Nov 2025", "2025-12": "Dec 2025", "2026-01": "Jan 2026", "2026-02": "Feb 2026"}

L = pd.read_csv(LATE); L["date"] = pd.to_datetime(L["date"])
L["traj"] = L.trajectory.str.capitalize()
L["wind"] = L.wind_mean_m_s.astype(int); L["turb"] = L.turbulence.map(lambda x: f"{x:.1f}")
ORDER = [("Straight", 0, "0.0"), ("Racetrack", 0, "0.0"), ("Straight", 1, "0.0"), ("Straight", 1, "0.1"), ("Racetrack", 1, "0.1"), ("Straight", 2, "0.1")]
g = L.groupby(["traj", "wind", "turb"]).agg(n=("run", "size"), c=("m3b_entry", "sum")).reset_index()
fin = L[L.date >= FINAL_FROM].groupby(["traj", "wind", "turb"]).agg(n=("run", "size"), c=("m3b_entry", "sum")).reset_index()
assert set(map(tuple, g[["traj", "wind", "turb"]].values)) == set(ORDER), "late census carries a condition the table does not list"

def frac(c, n): return f"{int(c)} of {int(n)} ({'100%' if c == n else f'{100*c/n:.1f}%'})"
late_rows = []
for t, w, s in ORDER:
    r = g[(g.traj == t) & (g.wind == w) & (g.turb == s)].iloc[0]
    f = fin[(fin.traj == t) & (fin.wind == w) & (fin.turb == s)]
    late_rows.append([t, str(w), s, frac(r.c, r.n), frac(f.iloc[0].c, f.iloc[0].n) if len(f) else "none flown"])
nL, cL = len(L), int(L.m3b_entry.sum()); F = L[L.date >= FINAL_FROM]; nF, cF = len(F), int(F.m3b_entry.sum())
late_rows.append(["All flights", "-", "-", frac(cL, nL), frac(cF, nF)])
d0, d1 = L.date.min().strftime("%-d %B"), L.date.max().strftime("%-d %B %Y")
early_rows = [[MON[r.month], str(r.tests), str(r.flights), str(r.pre), str(r.creep), str(r.contact)] for r in em.itertuples()]
early_rows.append(["Total", str(tot["tests"]), str(tot["flights"]), str(tot["pre"]), str(tot["creep"]), str(tot["contact"])])

def md(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])

text = f"""# PID-baseline Tuning Record

Early period: 26 Nov 2025 to 23 Feb 2026, zero wind and zero turbulence, {tot['tests']} numbered tuning tests ({tot['flights']} flights) in ArduPilot SITL.

{md(["Month", "Numbered Tests", "Flights", "Reached Hold Precontact", "Reached Docking Creep", "Reached Contact"], early_rows)}

Late period: {d0} to {d1}, mean wind 0 to 2 m/s, turbulence no higher than σ = 0.1, {nL} tuning flights in ArduPilot SITL.

{md(["Trajectory", "Mean wind speed (m/s)", "Turbulence Setting (σ)", "Reached Contact, all flights", "Reached Contact, final flights 3 to 19 Jul 2026"], late_rows)}

Notes

- Early period: a numbered test was often flown more than once; the three "Reached" columns count flights. "Reached Contact" = the state machine reached Hold Contact or an M3b entry exists in the flight's metrics record; "Reached Hold Precontact" and "Reached Docking Creep" = the state machine entered that state.
- Late period: "Reached Contact" = the M3b envelope was entered at least once (`m3b_entry`), no dwell required, overshoot allowed. Final flights = {FINAL_FROM} onward.
- Source files: `PID_TUNING_EARLY_CONTACT_CENSUS_20260930.csv` ({len(e)} rows, one per flight log) and `PID_TUNING_LATE_CONTACT_CENSUS_20260930.csv` ({nL} rows, one per flight). Both are beside this script; `pid_baseline_tuning_record.py` rebuilds this file from them.
"""
OUT.write_text(text)
print("wrote", OUT)
