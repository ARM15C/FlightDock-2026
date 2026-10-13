#!/usr/bin/env python3
# RESPONSIBILITY: fly ONE arbitrary candidate checkpoint over ALL 84 signed ARMBSIG grid-1 rows
#   (42 seed labels x sigma {0.20, 0.25}) at 24 lanes, so a candidate is scored on the SAME
#   instrument and SAME seed draw as ARM-B and judged by a4_2_compare.py. Neither existing driver
#   can: armbsig_driver.sh flies all 84 rows but HARD-WIRES its checkpoint (line 59) and passes
#   --force-zero-residual (which is what makes it ARM-B, not a policy); reramp_gate_wave.py takes
#   any --ckpt but flies only the 12 pinned rows. A NEW file, not a widening of either -- both stay
#   frozen as-flown. Row dealing is delegated to armbsig_manifest.assign (the ONE writer of slot
#   identity); the table is gated on its operator-countersigned sha; ZERO relay-seed spend, so the
#   ledger cursors must read +0 across the wave.
# PUBLIC API: table_rows(table_path) -> list[dict] | re-exported from
#             lanes(table_path, width) -> list[dict] | armbsig_exam_table.py --
#             lane_dir(out_dir, lane) -> str | the API is UNCHANGED
#             expected_by_sigma(table_path) -> dict | (+ this one, added 08-29)
#             sigma_env(env, sigma) -> dict
#             spawn_cmd(lane, ckpt, out_dir) -> list[str]
#             spawn_lane(lane, ckpt, out_dir, dry) -> (int, str)
#             score(out_dir, sigkey) -> list[dict]
#             tally(rows, expected) -> dict
#             cursors(ledger=LEDGER) -> dict
#             custody_measurable(ledger=LEDGER) -> bool
#             main(argv=None) -> int
# SIDE-EFFECTS: --dry-run renders lanes and spawns NOTHING. Otherwise spawns eval legs via
#   scripts/curriculum_eval_spawner.py, writing under --out-dir only, and runs the per-leg
#   REALIZED-SIGMA gate in-loop via wave_sigma_hook (RB-42) -- which writes sigma_gate.json and
#   fdm_witness.jsonl into that lane's dir and REFUSES the leg on non-zero rc, leaving it in the
#   denominator. Reads ledger cursors before
#   and after and REFUSES on movement. The ledger path is an ARGUMENT (--ledger) defaulting
#   byte-for-byte to LEDGER, because a box may hold only a COPY ledger as its custody of
#   record; it is a way to NAME the ledger, never a way to skip custody. Draws no seed,
#   writes no kills nothing.
# SIZE: soft 150 / hard 190 LoC; function cap 40. The selftest lives in the sibling
#   armbsig_exam_wave_selftest.py purely to respect these caps; NO cap was raised.
# COVERAGE: armbsig_exam_wave_selftest.py, EXECUTED, with every guard driven to its refusal.
#   2026-08-29: the table gate, row count, lane dealing and lane-dir naming moved to the sibling
#   armbsig_exam_table.py -- a SPLIT, not a raise, taken because this file stood at 187 code
#   against its hard 190 and the next change had nowhere to go. Behaviour is unchanged and the
#   84-row default deals byte-identical lanes (equivalence smoke in armbsig_exam_table_selftest).
import argparse, glob, importlib.util, json, os, subprocess, sys

FD = "/home/ubuntu/FlightDock"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(FD, "flightdock_native_venv/bin/python")
SPAWNER = os.path.join(FD, "scripts/curriculum_eval_spawner.py")
SPAWN_GUARD = os.path.join(FD, "analysis/armbtop_20260817T1240Z/armbtop_spawn_guard.py")
SIGMA_HOOK = os.path.join(HERE, "wave_sigma_hook.py")   # RB-42: the per-leg realized-sigma gate
CRIT, ATTRITION_BAR, WIDTH = "v2_real_events", 0.145, 24
# env_contract A7 admits exactly these triples; anything else must ride a DECLARED ack.
A7 = ("0.10,0.10,0.10", "0.20,0.20,0.20", "0.30,0.30,0.30")
ACK = ("ARMBSIG candidate-selection exam (A4.2): sigma %s is OFF the A7 taxonomy by construction. "
       "It is the B25 half of the signed ARM-B bank and a candidate must fly the SAME cell to be "
       "its comparator. Declared, not widened, never rounded.")
BANNED_FLAG = "--force-zero-residual"    # the flag that makes a run ARM-B instead of a policy
LEDGER = os.path.join(FD, "analysis/c006_campaign_20260728T110000Z/logs_w1418/"
                          "gate_relay_ledger.jsonl")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# The countersigned table -- its sha gate, its row count, its lane dealing and its lane-dir
# naming -- lives in the sibling armbsig_exam_table.py so this file stays inside its DECLARED
# caps. Re-exported BY NAME so the public API, every caller and the selftest are unchanged, and
# so exactly ONE Refusal class exists (loaded once here, not per call).
_T = _load(os.path.join(HERE, "armbsig_exam_table.py"), "armbsig_exam_table")
Refusal, TABLE, N_ROWS = _T.Refusal, _T.TABLE, _T.N_ROWS
table_rows, lanes, lane_dir = _T.table_rows, _T.lanes, _T.lane_dir
expected_by_sigma = _T.expected_by_sigma


def sigma_env(env, sigma):
    """Two decimals always ('0.2' is REFUSED by the contract). ACK set IFF the triple is off the
    A7 taxonomy: 0.20 is ON it and carries none; 0.25 is off it and must."""
    s = "%.2f" % float(sigma)
    triple = "%s,%s,%s" % (s, s, s)
    env["FD_V3_SIGMA_OVERRIDE_MS"] = triple
    env.pop("FD_V3_SIGMA_OVERRIDE_MS_ACK", None)
    if triple not in A7:
        env["FD_V3_SIGMA_OVERRIDE_MS_ACK"] = ACK % s
    return env


def spawn_cmd(lane, ckpt, out_dir):
    """NOTE what is ABSENT: --force-zero-residual would fly ARM-B and call it a candidate."""
    return [PY, SPAWNER, "--checkpoint", ckpt, "--stage-label", lane["label"],
            "--out-dir", os.path.dirname(lane_dir(out_dir, lane)),
            "--seeds", str(lane["seed"]), "--env-id", str(lane["env_id"]),
            "--traj-map", "%s:%s" % (lane["seed"], lane["traj"])]


def spawn_lane(lane, ckpt, out_dir, dry):
    d = lane_dir(out_dir, lane)
    os.makedirs(d, exist_ok=True)
    if glob.glob(os.path.join(d, "results_eid*.json")):
        return 0, "SKIP (already flown)"
    cmd = spawn_cmd(lane, ckpt, out_dir)
    if BANNED_FLAG in cmd:
        raise Refusal("%s reached the spawn argv -- that flies the BASELINE, not a candidate"
                      % BANNED_FLAG)
    if dry:
        return 0, "DRY " + " ".join(cmd)
    env = sigma_env(dict(os.environ), lane["sigma_ms"])
    log = os.path.join(d, "lane.log")
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=FD, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return rc, "spawner rc=%d" % rc
    g = subprocess.run([PY, SPAWN_GUARD, "--lane-log", log], capture_output=True, text=True)
    if g.returncode != 0:                 # SPAWNGUARD-782: rc 0 is not proof the leg flew
        return 14, "spawn guard REFUSED: %s" % g.stdout.strip()[:160]
    # RB-42: gate the REALIZED turbulence, in-loop -- the FDM stream is truncated at the next
    # episode, so this is the only moment this lane's sigma is measurable (RB-41 amendment 1).
    src = _load(SIGMA_HOOK, "wave_sigma_hook").gate(d, lane["env_id"], lane["sigma_ms"], log)
    if src != 0:
        return 20 + src, ("realized-sigma gate rc=%d (2=UNSAMPLED/INCONCLUSIVE, 3=OUT-OF-BAND); "
                          "leg stays in the denominator" % src)
    return 0, "flown"


def score(out_dir, sigkey):
    """The instrument of record -- the same one that produced the bank being compared against."""
    sys.path.insert(0, os.path.join(FD, "scripts"))
    import curriculum_gate_runner as cgr
    adj, _oc = cgr._frozen()
    root, rows = os.path.join(out_dir, sigkey), []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "results_eid*.json")):
            r = cgr.leg_verdict(d, adj, CRIT)
            r["leg"] = name
            rows.append(r)
    return rows


def tally(rows, expected):
    gradable = [r for r in rows if r.get("gradable")]
    viol = [r["leg"] for r in gradable
            if any(str(x).startswith("truth_ceiling_bust") for x in (r.get("fail_reasons") or []))]
    attr = 1.0 - (len(gradable) / expected) if expected else 1.0
    return {"n_expected": expected, "n_gradable": len(gradable),
            "k": sum(1 for r in gradable if r.get("passed")),
            "zone_violations": len(viol), "violation_legs": sorted(viol),
            "attrition": round(attr, 4), "attrition_over_bar": attr > ATTRITION_BAR}


def cursors(ledger=LEDGER):
    """Relay ledger cursors. Every row is an already-signed label, so a wave must move NOTHING
    (a relay-driven equivalent would exhaust the block, as K-mid did at 11:14Z)."""
    n6 = _load(os.path.join(FD, "analysis/n6auth_20260822T2130Z/n6_auth_launch.py"), "n6")
    n6.LEDGER = ledger
    return n6.cursors()          # {} when the ledger is ABSENT -- see custody_measurable()


def custody_measurable(ledger=LEDGER):
    """{} == {} would 'pass' on a box with no ledger at all: missing evidence is INCONCLUSIVE."""
    return os.path.isfile(ledger)


def sitl_count():
    n = 0
    for e in glob.glob("/proc/[0-9]*/exe"):
        try:
            if "arduplane" in os.path.realpath(e):
                n += 1
        except OSError:
            pass
    return n


def _preflight(a):
    """Every refusal that must happen BEFORE a lane is spawned."""
    if not (a.ckpt and a.out_dir):
        raise Refusal("--ckpt and --out-dir are required")
    if not a.dry_run and a.slot is None and sitl_count() >= a.width:   # a slot child shares the box
        raise Refusal("%d SITL already alive -- this box is not free" % sitl_count())
    lns = lanes(a.table, a.width)
    dirs = {lane_dir(a.out_dir, l) for l in lns}
    if len(dirs) != len(lns):                    # the 84->42 collapse, refused structurally
        raise Refusal("lane dirs collapsed: %d distinct for %d rows" % (len(dirs), len(lns)))
    measured = custody_measurable(a.ledger)
    if not a.dry_run and not measured:
        raise Refusal("relay ledger absent (%s): seed custody cannot be MEASURED, so a live "
                      "wave is refused" % a.ledger)
    return lns, dirs, measured


def _fan_out(a, argv):
    """E-PAR: all --width slot children at once via the ONE shared fan-out; the caller then
    runs its own sequential pass, in which every flown row SKIPs and the pool is tallied."""
    base = [x for x in (sys.argv[1:] if argv is None else argv) if x != "--parallel"]
    rcs = _load(os.path.join(HERE, "wave_fanout.py"), "wave_fanout").fanout(
        os.path.abspath(__file__), base, a.width, a.out_dir, cwd=FD)
    if any(rcs.values()):
        raise Refusal("slot children returned non-zero: %s" % {k: v for k, v in rcs.items() if v})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fly one checkpoint over every signed ARMBSIG row")
    ap.add_argument("--ckpt"); ap.add_argument("--out-dir")
    ap.add_argument("--table", default=TABLE); ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--ledger", default=LEDGER,
                    help="custody-of-record ledger for THIS box (default: the canonical "
                         "campaign ledger). Never a way to skip custody: a live wave is "
                         "still REFUSED when the named ledger is absent.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--slot", type=int, help="fly ONLY the rows dealt to env_id N (a fan-out child)")
    ap.add_argument("--parallel", action="store_true",
                    help="E-PAR: fan out --width slot children via wave_fanout, then POOL")
    a = ap.parse_args(argv)
    if a.selftest:
        return _load(os.path.join(HERE, "armbsig_exam_wave_selftest.py"), "aews").selftest()
    lns, dirs, measured = _preflight(a)
    before = cursors(a.ledger)
    if a.parallel:
        _fan_out(a, argv)
    if a.slot is not None:
        lns = [l for l in lns if l["env_id"] == a.slot]
    results = [(l, ) + spawn_lane(l, a.ckpt, a.out_dir, a.dry_run) for l in lns]
    after = cursors(a.ledger)
    if measured and before != after:             # ZERO spend, witnessed not asserted
        raise Refusal("ledger cursors MOVED %s -> %s; this wave must spend nothing"
                      % (before, after))
    bad = [(l["label"], l["seed"], note) for l, rc, note in results if rc != 0]
    out = {"dry_run": a.dry_run, "rows": len(lns), "lane_dirs": len(dirs), "ledger": a.ledger,
           "cursors_before": before, "cursors_after": after, "custody_measured": measured,
           "cursor_delta_zero": (before == after) if measured else "NOT MEASURED (no ledger here)",
           "failed": bad}
    if not a.dry_run and a.slot is None:                 # only the POOLED pass tallies and publishes
        for sk, n in sorted(expected_by_sigma(a.table).items()):   # counted, never halved
            out[sk] = tally(score(a.out_dir, sk), n)
        os.makedirs(a.out_dir, exist_ok=True)
        json.dump(out, open(os.path.join(a.out_dir, "EXAM_WAVE.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
