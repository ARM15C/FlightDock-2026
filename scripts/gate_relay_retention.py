#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: RC3 RETENTION-SCREEN EXECUTOR (T-74 slice 2). At a sealed
#   gate whose panel is non-empty, flies the retention panel on the SAME
#   frozen checkpoint: per stratum a screen of N=6 (3/3 trajectory balance,
#   seeds from the frozen `screening` block); triggered strata (k<=4/6,
#   DIAGNOSTIC) escalate to a confirmation of N=12 disjoint seeds (frozen
#   `confirmation` block) which REPLACES the screen; a CONFIRMED strike
#   (k<=9/12) is recorded in the append-only strike ledger and the caller
#   HOLDS the gate verdict + pages the operator (MANUAL-demotion election,
#   2026-07-28 — clearing attempts and demotion run only under ).
#   Adjudication is curriculum_gate_runner --mode retention, consumed AS-IS.
#   NEVER clears a strike, never publishes anything itself.
# PUBLIC API: stratum_labels, run_panel. No CLI (called by gate_relay_core).
# SIDE-EFFECTS: spawns eval legs via curriculum_eval_spawner.py under
#   <legs_dir>/retention/; appends to the seed ledger + strike ledger; writes
#   retention spec/adjudication JSONs. Nothing else.
# CAPS: file soft 150 / hard 200 LoC; function cap 60.
# COVERAGE: tests/unit/test_gate_relay_retention.py (label map; balanced
#   traj-map; screen->confirm escalation wiring on a fixture adjudication;
#   strike-ledger append; no-clear invariant).
# =============================================================================
import json
import os
import subprocess
import sys

FD = "/home/ubuntu/FlightDock"
SPAWNER = os.path.join(FD, "scripts", "curriculum_eval_spawner.py")
RUNNER = os.path.join(FD, "scripts", "curriculum_gate_runner.py")
# S-2 (2026-08-06): this WAS a hand-maintained mirror of config.STAGE_LABELS_V3,
# re-typed as a literal. That is the exact defect that let the docs carry three
# contradictory ladders — and a mislabelled retention leg does not raise, it
# produces a PLAUSIBLE WRONG LABEL. The map is now IMPORTED, so a renumber can
# never desynchronise it. Kept as a module-level name for call-site stability.
sys.path.insert(0, FD)
from src.drl.config import STAGE_LABELS_V3 as STAGE_LABELS  # noqa: E402
SCREEN_N, CONFIRM_N = 6, 12


def stratum_labels(panel):
    """Panel stage indices -> stage labels (mirrors config.STAGE_LABELS_V3)."""
    return [STAGE_LABELS[s] for s in panel]


def _balanced_tmap(seeds):
    return {int(s): ("straight" if k % 2 == 0 else "racetrack")
            for k, s in enumerate(seeds)}


def _spawn_stratum(checkpoint, label, seeds, out_dir, lanes, tmap=None):
    """One stratum's legs, lane-parallel, SAME frozen checkpoint. True iff
    every spawner process exits 0. `tmap` overrides the index-parity default
    for R-3 replacement legs, which must mirror what was quarantined."""
    model = os.path.join(checkpoint, "model.zip")
    vec = os.path.join(checkpoint, "vecnormalize.pkl")
    tmap = _balanced_tmap(seeds) if tmap is None else tmap
    os.makedirs(out_dir, exist_ok=True)
    procs = []
    lanes = max(1, min(lanes, len(seeds)))
    chunks = [seeds[i::lanes] for i in range(lanes)]
    # env-ids start at 1 — env0 is the training lane (see lane_split note in
    # gate_relay_core.py; same collision class, independently minted here).
    pairs = list(enumerate((c for c in chunks if c), 1))
    floor = int(os.environ.get("RELAY_MIN_ENV_ID", "1"))
    if any(e < floor for e, _ in pairs):  # fail closed, no spawn, no verdict
        print(f"RETENTION HALT: lane env-id below floor {floor}", file=sys.stderr)
        return False
    # FIXAWIRE-719: this is a SECOND relay path that spawns eval legs and had
    # no contract assertion at all. gate_relay_core asserts for its own path;
    # a retention panel reaching here by any other route was unchecked. Fail
    # closed before the first lane spawns, never after.
    sys.path.insert(0, FD)
    from src.drl import env_contract
    env_contract.assert_contract(env_contract.EVAL)
    for env_id, chunk in pairs:
        cmd = [sys.executable, SPAWNER, "--checkpoint", model,
               "--stage-label", label, "--out-dir", out_dir,
               "--seeds", ",".join(str(s) for s in chunk),
               "--env-id", str(env_id),
               "--traj-map", ",".join(f"{s}:{tmap[s]}" for s in chunk)]
        if os.path.isfile(vec):
            cmd += ["--vecnormalize", vec]
        procs.append(subprocess.Popen(cmd))
    # T-155: wait ALL. `all(p.wait()==0 for ...)` short-circuited, leaving
    # siblings unwaited and still landing legs in a terminal-declared tree.
    rcs = [p.wait() for p in procs]
    for rc in (r for r in rcs if r != 0):
        print(f"RETENTION lane rc={rc} ({label})", file=sys.stderr)
    # T-322/A-1: 20 = spawner PARTIAL-LEGS sentinel (evidence written; the
    # screen replaces those legs). Any other nonzero rc still fails the
    # stratum spawn, fail-closed — see gate_relay_core._spawn_lanes.
    return all(rc in (0, 20) for rc in rcs)


def _adjudicate(legs_root, strata, phase, prior_strikes, out_path):
    """runner --mode retention, consumed AS-IS. Returns dict or None."""
    spec = {"strata": strata, "phase": phase, "prior_strikes": prior_strikes}
    spec_path = out_path.replace(".json", "_spec.json")
    with open(spec_path, "w") as fh:
        json.dump(spec, fh)
    rc = subprocess.run([sys.executable, RUNNER, "--mode", "retention",
                        "--legs-dir", legs_root, "--spec", spec_path,
                        "--out", out_path]).returncode
    try:
        return json.load(open(out_path)) if rc == 0 else None
    except Exception:
        return None


def _strike_key(rec):
    return (rec.get("gate"), rec.get("stratum"), rec.get("checkpoint"))


def _prior_strikes(strike_ledger):
    """Strikes minus matching operator CLEAR records (R-3).

    A CLEAR line ({"record": "CLEAR", gate, stratum, checkpoint}) cancels ONE
    matching strike. Before R-3 this counted every non-blank line, so a clear
    annotation would have been miscounted as a SECOND strike and armed
    strikes_demote=2 (T-127). Fail-closed: an unparseable line COUNTS as a
    strike — corruption must never reduce the count.
    """
    if not os.path.isfile(strike_ledger):
        return 0
    strikes, bad = [], 0
    for ln in open(strike_ledger):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            bad += 1
            continue
        if rec.get("record") == "CLEAR":
            if _strike_key(rec) in strikes:
                strikes.remove(_strike_key(rec))
        else:
            strikes.append(_strike_key(rec))
    return len(strikes) + bad


def _ensure_clean(stratum_dir, lab, want_n, req, lanes, next_seeds, block,
                  bridge_fn):
    """R-3: classify the stratum and replace FAULT/UNSAMPLED legs until
    want_n clean, trajectory balance preserved. (ok, why). Replacement seeds
    are drawn per round with an explicit `replace:` purpose, so the ledger
    shows exactly what contamination cost. Escape hatch (rollback):
    FD_V3_FAULTSCREEN_DISABLE=1 skips screening entirely, read at use."""
    if os.environ.get("FD_V3_FAULTSCREEN_DISABLE", "0").strip() == "1":
        return True, "faultscreen_disabled_by_flag"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gate_relay_faultscreen as fs
    classify = fs.load_classifier()

    def spawn_replacements(need):
        n = sum(need.values())
        seeds = next_seeds(block, n, f"replace:{req['label']}:{lab}")
        if not _spawn_stratum(req["checkpoint"], lab, seeds, stratum_dir,
                              lanes, tmap=fs.explicit_tmap(seeds, need)):
            return False
        bridge_fn(stratum_dir)
        return True

    half = want_n // 2
    ok, why, _ = fs.ensure_clean_panel(
        stratum_dir, {"straight": half, "racetrack": half},
        classify, spawn_replacements)
    return ok, (why if ok else f"faultscreen:{lab}:{why}")


def run_panel(req, panel, legs_dir, lanes, next_seeds, strike_ledger,
              bridge_fn):
    """Execute the full RC3 panel for one gate. Returns a dict:
    {clean: bool, halt_reason: str|None, screen: ..., confirm: ...}.
    clean=True => caller may publish the gate verdict.
    clean=False => caller HOLDS (no verdict) and pages the operator."""
    labels = stratum_labels(panel)
    root = os.path.join(legs_dir, "retention")
    for lab in labels:
        seeds = next_seeds("screening", SCREEN_N, f"screen:{req['label']}:{lab}")
        if not _spawn_stratum(req["checkpoint"], lab, seeds,
                              os.path.join(root, lab), lanes):
            return {"clean": False, "halt_reason": f"screen_spawn_fail:{lab}"}
        bridge_fn(os.path.join(root, lab))
        ok, why = _ensure_clean(os.path.join(root, lab), lab, SCREEN_N, req,
                                lanes, next_seeds, "screening", bridge_fn)
        if not ok:
            return {"clean": False, "halt_reason": why}
    screen = _adjudicate(root, labels, "screen", _prior_strikes(strike_ledger),
                         os.path.join(root, "screen_adjudication.json"))
    if screen is None or screen.get("invalid_protocol"):
        return {"clean": False, "halt_reason": "screen_adjudication_invalid",
                "screen": screen}
    triggered = screen.get("triggered", [])
    if not triggered:
        return {"clean": True, "halt_reason": None, "screen": screen}
    conf_root = os.path.join(legs_dir, "retention_confirm")
    for lab in triggered:
        seeds = next_seeds("confirmation", CONFIRM_N,
                           f"confirm:{req['label']}:{lab}")
        if not _spawn_stratum(req["checkpoint"], lab, seeds,
                              os.path.join(conf_root, lab), lanes):
            return {"clean": False, "halt_reason": f"confirm_spawn_fail:{lab}",
                    "screen": screen}
        bridge_fn(os.path.join(conf_root, lab))
        ok, why = _ensure_clean(os.path.join(conf_root, lab), lab, CONFIRM_N,
                                req, lanes, next_seeds, "confirmation",
                                bridge_fn)
        if not ok:
            return {"clean": False, "halt_reason": why, "screen": screen}
    confirm = _adjudicate(conf_root, triggered, "confirm",
                          _prior_strikes(strike_ledger),
                          os.path.join(conf_root, "confirm_adjudication.json"))
    if confirm is None or confirm.get("invalid_protocol"):
        return {"clean": False, "halt_reason": "confirm_adjudication_invalid",
                "screen": screen, "confirm": confirm}
    strikes = confirm.get("confirmed_strikes", [])
    for lab in strikes:
        with open(strike_ledger, "a") as fh:
            fh.write(json.dumps({"gate": req["label"], "stratum": lab,
                                 "checkpoint": req.get("checkpoint")}) + "\n")
    if strikes:
        # Option-A: a confirmed strike BLOCKS promotion. Under the elected
        # MANUAL regime everything after this point is.
        return {"clean": False,
                "halt_reason": f"confirmed_strike:{','.join(strikes)}",
                "screen": screen, "confirm": confirm}
    return {"clean": True, "halt_reason": None,
            "screen": screen, "confirm": confirm}
