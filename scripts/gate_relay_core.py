#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Curriculum-v3 GATE-RELAY CORE (T-74). Processes ONE sealed
#   gate request end-to-end: read logs/gate_requests/<label>.json (written by
#   CurriculumCallback._v3_gate_logic), draw fresh frozen seeds (SEED_BLOCKS
#   cursor, append-only ledger), fan sealed eval legs across lanes via
#   curriculum_eval_spawner.py (explicit --traj-map so lane splits preserve
#   the GLOBAL k%2 trajectory balance), adjudicate via curriculum_gate_runner
#   --mode gate, and atomically publish logs/gate_verdicts/<label>.json (the
#   ONLY input that can promote training). Retention panel per RC3: strata
#   {S0, W0, prev, one rotating older} + trained anchors at anchor gates —
#   composition rule VERIFIED to reproduce the generator's 91 stratum-screens
#   (unit-pinned; 85 pre-A5-diagonal, 67 pre-T-191-append); screens run on the
#   SAME frozen checkpoint; a triggered
#   screen escalates to confirmation; a CONFIRMED strike blocks the verdict
#   and pages the operator (manual-demotion election 2026-07-28).
#   NEVER synthesizes a verdict: any spawn/adjudication failure leaves NO
#   verdict file (training pauses safely) and exits nonzero.
#   R-1 (2026-07-30): legs land in a PER-ATTEMPT dir and EVERY terminal
#   outcome — success and every failure rc — is marked, so a failed gate
#   pages a human instead of re-dispatching into its own polluted leg tree.
#   Bookkeeping lives in gate_relay_state.py; see that file for the incident.
# PUBLIC API: load_request, next_seeds (re-export from gate_relay_seeds,
#   T-157), global_traj_map, lane_split, panel_for_stage, panel_total_check,
#   write_verdict_atomic, process_gate, main. CLI: --request PATH --legs-root
#   DIR [--lanes N] [--dry-run] [--seed-blocks PATH] [--ledger PATH] [--verdicts-dir DIR].
# SIDE-EFFECTS: appends to the relay ledger (append-only JSONL); creates ONE
#   per-attempt leg dir + spawns eval subprocesses (via the spawner); writes
#   ONE verdict file atomically (tmp+mv); writes <request>.processed on
#   success (legacy) and <request>.terminal on EVERY terminal outcome.
#   Dry runs write NEITHER marker — a rehearsal must not seal a real gate.
# CAPS: file soft 330 / hard 360 LoC (code lines, excl. comments/blanks);
#   function cap 65. HARD RAISED 340 -> 360 + fn 60 -> 65 by E-5/T-275
#   (08-05), DECLARED: generator-derived PANEL_TOTAL (the guard keeps its
#   independent comparison) + panel_budget_s + the PANEL_ETA intake stamp
#   (+2 in process_gate) took the file 322 -> 349.
#   Prior raise: HARD 320 -> 340 by E-4 (08-05), DECLARED: the
#   grade_and_publish extraction (ONE named bridge->grade->publish flow for
#   relay AND drivers — the T-273 hand-mirror class) plus its driver-mode
#   lifecycle guards took the file 301 -> 322; behavior on the relay path is
#   byte-identical (pinned by test_grade_and_publish_is_the_only_publish_path).
#   Prior raise: HARD 300 -> 320 by T-262, DECLARED not squeezed:
#   the B2 sentinel on the fail-closed path (+8 code) took the file 293 -> 301,
#   a NEW hard breach caught by tools/check_loc_caps.py. The alternative was
#   trimming the operator-facing message to land on exactly 300 — a zero-
#   headroom trap that buys nothing. Same convention T-237 used on the test
#   module. R-1 took this 239 -> 259: over SOFT, under HARD. The
#   bookkeeping itself lives in gate_relay_state.py precisely so this file
#   absorbed only call sites.
#   DEBT INCURRED, LABELLED (R-1): process_gate() was ALREADY over the
#   function cap before this change (67 code lines vs cap 60) and R-1 takes
#   it to 78. Splitting it is STRUCTURAL and would be a second class in a
#   BEHAVIORAL dispatch, so it is registered as follow-up, not done here.
#   Not silently exceeded — measured, stated, and tracked. Now enforced
#   mechanically: `tools/check_loc_caps.py` reads these very numbers and
#   fails on any NEW breach, with this one recorded in the accepted baseline
#   at analysis/loccaps_20260730T232000Z/ACCEPTED_LOC_BREACHES.json.
# COVERAGE: tests/unit/test_gate_relay_core.py (malformed request; seed
#   cursor/ledger; traj-map balance under lane split; atomic verdict + label
#   echo; duplicate/processed guard; panel rule == 91 enforced in main(), both
#   directions incl. the B2 HOLD sentinel on the fail-closed path) and
#   tests/unit/test_gate_relay_state.py (attempt isolation; terminal markers).
# =============================================================================
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_relay_state as state       # noqa: E402 (sibling script module)

FD = "/home/ubuntu/FlightDock"
SPAWNER = os.path.join(FD, "scripts", "curriculum_eval_spawner.py")
RUNNER = os.path.join(FD, "scripts", "curriculum_gate_runner.py")
SEED_BLOCKS_DEFAULT = os.path.join(
    FD, "analysis/curriculumv3_20260717T025122Z/SEED_BLOCKS.json")
GATE_N = {"mid12": 12, "anchor30": 30}
# S-2 (2026-08-06): these WERE hand-copied literals — (0, 9, 23) and 25 — and
# they silently went stale the moment a rung was inserted. They are now IMPORTED
# from the ladder authority, the same pattern this file already uses for
# PANEL_TOTAL ("no hand-copied integer to re-arm the T-262 drift"). A pure
# duplicate with no independent-transcription rationale earns no second home.
sys.path.insert(0, FD)
from src.drl.config import (                     # noqa: E402
    V3_ANCHOR_STAGES as ANCHOR_STAGES,
    STAGE_LABELS_V3 as _STAGE_LABELS_V3,
)
N_STAGES = len(_STAGE_LABELS_V3)     # S0, W0..W12, D1a, D1, W13..W16, D2,
                                     # W17..W20, C1 (S-2 diagonal + D1a insert)
# E-5 (T-275): PANEL_TOTAL is IMPORTED from the generator authority
# (tools/curriculum_v3_totals.py::totals) — no hand-copied integer to re-arm
# the T-262 drift. The rc=5 guard stays DISCRIMINATING: it compares core's own
# panel_for_stage-derived sum (panel_total_check) against this independent
# generator value — deriving both sides from one function would build a guard
# that cannot fire.
def _generator_panel_total():
    import importlib.util
    p = os.path.join(FD, "tools", "curriculum_v3_totals.py")
    spec = importlib.util.spec_from_file_location("curriculum_v3_totals", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.totals()["eval_episode_ledger"]
               ["retention_stratum_screen_count"])


PANEL_TOTAL = _generator_panel_total()
TRAJS = ("straight", "racetrack")
# E-5 (T-275): panel wall-time budgets = measured norm x 1.5 (mid12 ~3.5 h
# median request->verdict; anchor30 = W8's 6 h). Past budget the relay pages
# PANEL_OVERDUE — a stalled panel becomes visible in minutes, not 5.5 h
# (the W12 rc=5 stall). Advisory only: nothing is auto-aborted.
#
# S-1 (2026-08-06): this table is MIRRORED in src/drl/callbacks.py as
# PANEL_WALL_BUDGET_S, which DERIVES the learner's hard gate-pause timeout from
# it (max + GATE_PAUSE_MARGIN_S). The two are pinned to EXACT equality by
# tests/unit/test_gate_pause_outlives_panel_budget.py. Changing a value here
# without changing it there fails the suite by design — before D1 they diverged
# silently and an 18,505 s panel outlived an 18,000 s pause, halting the window.
# If you edit this table, run that test.
PANEL_BUDGET_S = {"mid12": int(3.5 * 3600 * 1.5),
                  "anchor30": int(6.0 * 3600 * 1.5)}


def panel_budget_s(req):
    """Budget seconds for a request (0 = unknown/no budget). Test seam:
    FD_RELAY_BUDGET_OVERRIDE_S forces a value so the watchdog can be
    exercised in BOTH directions without a multi-hour wait."""
    ov = os.environ.get("FD_RELAY_BUDGET_OVERRIDE_S", "").strip()
    if ov:
        return int(ov)
    return PANEL_BUDGET_S.get((req or {}).get("gate_type"), 0)


def load_request(path):
    """Request JSON -> (dict|None, problems). Malformed => (None, [...])."""
    problems = []
    try:
        with open(path) as fh:
            req = json.load(fh)
    except Exception as e:
        return None, [f"unreadable: {type(e).__name__}"]
    for key in ("stage", "label", "gate_type", "checkpoint"):
        if req.get(key) in (None, ""):
            problems.append(f"missing:{key}")
    if not problems and req["gate_type"] not in GATE_N:
        problems.append(f"unknown_gate_type:{req['gate_type']}")
    if not problems and not os.path.isdir(str(req["checkpoint"])):
        problems.append("checkpoint_dir_absent")
    return (None, problems) if problems else (req, [])


# T-157: seed custody moved VERBATIM to
# gate_relay_seeds.py — core.py was at its 300-line hard cap and the T-156
# single-writer lock needs room there. Re-export keeps every caller
# (process_gate, ws1r1_confirm.sh, tests) working unchanged.
from gate_relay_seeds import next_seeds, pinned_seeds  # noqa: E402,F401


def global_traj_map(seeds):
    """GLOBAL k%2 assignment, frozen before any lane split — build_legs()
    restarts k%2 per spawner process, so lane subsets MUST carry an explicit
    --traj-map or balance silently breaks (e.g. 12 lanes x 1 seed = 12 straight)."""
    return {int(s): TRAJS[k % 2] for k, s in enumerate(seeds)}


def lane_split(seeds, lanes):
    """Round-robin seeds across lanes -> [(env_id, [seeds...]), ...].

    env-ids start at 1: env0 is the TRAINING lane, and every eval leg's SITL
    launcher pkills its own port offset — an env0 eval lane kills a live
    training run (C006 S0 gate, 2026-07-28). Eval lanes must start ABOVE the
    training n_envs (campaign pins n_envs=1, so 1..N is safe)."""
    lanes = max(1, min(int(lanes), len(seeds)))
    out = [[] for _ in range(lanes)]
    for k, s in enumerate(seeds):
        out[k % lanes].append(s)
    return [(i, chunk) for i, chunk in enumerate(out, 1) if chunk]


def panel_for_stage(stage, rotation_cursor=0):
    """RC3 retention panel at the gate for `stage`: {S0, W0, prev, ONE
    rotating older} restricted to mastered (< stage), deduped, + trained
    anchors at anchor gates. Reproduces the generator's 91 stratum-screens
    across s=0..24 (panel_total_check; A5 diagonal)."""
    mastered = set(range(stage))
    panel = [s for s in (0, 1, stage - 1) if s in mastered]
    rot_pool = sorted(mastered - set(panel))
    if rot_pool:
        panel.append(rot_pool[rotation_cursor % len(rot_pool)])
    if stage in ANCHOR_STAGES:
        panel += [a for a in ANCHOR_STAGES if a in mastered and a not in panel]
    return sorted(set(panel))


def panel_total_check():
    """Sum of panel sizes over all 25 gates must equal the generator-emitted
    91 stratum-screens (curriculum_v3_totals.py; 85 pre-A5-diagonal)."""
    return sum(len(panel_for_stage(s)) for s in range(N_STAGES))


def write_verdict_atomic(verdicts_dir, label, payload):
    """tmp+mv publish; refuses a payload whose label does not echo the
    request (wrong-label verdict is a named failure mode)."""
    if payload.get("label") != label:
        raise ValueError(f"label mismatch: payload={payload.get('label')!r} "
                         f"request={label!r}")
    if "passed" not in payload:
        raise ValueError("verdict payload missing decisive key 'passed'")
    os.makedirs(verdicts_dir, exist_ok=True)
    tmp = os.path.join(verdicts_dir, f".{label}.json.tmp")
    final = os.path.join(verdicts_dir, f"{label}.json")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, final)
    return final


def bridge_paired_layout(legs_dir):
    """The FROZEN dwell instrument (t0_adjudicate -> dwell_intersection ->
    facetskip load) reads <leg>/paired/env0/paired_channels_ep0001.csv; the
    spawner routes the CSV to <leg>/env<K>/ (lane K). Bridge = the Part-A
    shadow-scoring precedent: symlink <leg>/paired/env0 -> the leg's real
    env dir. Idempotent; legs without a paired CSV are left alone (the
    liveness guard fails them, correctly). The link is RELATIVE (T-317)."""
    # WHY RELATIVE. The link used to be absolute, which killed it the moment
    # the leg moved -- four MEASURED incidents from that one fact: quarantine
    # os.rename (2026-07-31), the W11 panel move (T-233), the 24xl->dev t240
    # mirror (25 dangling legs), and the dev-side copy of the 210-leg ARM-B
    # corpus (210 with a CSV, zero links). Two were patched downstream by
    # re-pointing AFTER the move; this fixes the cause. `../env<K>` resolves
    # identically in place and survives every move.
    bridged = 0
    for leg in sorted(os.listdir(legs_dir)):
        leg_dir = os.path.join(legs_dir, leg)
        if not os.path.isdir(leg_dir):
            continue
        link = os.path.join(leg_dir, "paired", "env0")
        if os.path.exists(link):
            continue
        # A DANGLING link is not ABSENT: os.path.exists() is False for both,
        # so this used to fall through to os.symlink() and raise
        # FileExistsError -- i.e. crash on exactly the trees needing repair.
        if os.path.lexists(link):
            os.remove(link)
        src = None
        for d in sorted(os.listdir(leg_dir)):
            if d.startswith("env") and os.path.isfile(
                    os.path.join(leg_dir, d, "paired_channels_ep0001.csv")):
                src = os.path.join(leg_dir, d)
                break
        if src is None:
            continue
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.relpath(src, os.path.dirname(link)), link)
        bridged += 1
    return bridged


def _spawn_lanes(req, seeds, tmap, lanes, legs_dir, dry_run):
    """Fan spawner processes across lanes; wait for ALL; nonzero rc from any
    lane => False (caller aborts WITHOUT writing a verdict)."""
    vecnorm = os.path.join(req["checkpoint"], "vecnormalize.pkl")
    model = os.path.join(req["checkpoint"], "model.zip")
    procs = []
    pairs = lane_split(seeds, lanes)
    floor = int(os.environ.get("RELAY_MIN_ENV_ID", "1"))
    if any(e < floor for e, _ in pairs):  # env0 = training lane; fail closed
        print(f"RELAY HALT: lane env-id below floor {floor}", file=sys.stderr)
        return False
    for env_id, chunk in pairs:
        # traj-map filtered PER LANE: the spawner hard-errors on map seeds
        # absent from its --seeds (fail-closed by design; rehearsal-caught).
        tmap_arg = ",".join(f"{s}:{tmap[s]}" for s in chunk)
        cmd = [sys.executable, SPAWNER, "--checkpoint", model,
               "--stage-label", req["label"], "--out-dir", legs_dir,
               "--seeds", ",".join(str(s) for s in chunk),
               "--env-id", str(env_id), "--traj-map", tmap_arg]
        if os.path.isfile(vecnorm):
            cmd += ["--vecnormalize", vecnorm]
        if dry_run:
            cmd.append("--dry-run")
        procs.append(subprocess.Popen(cmd))
    # T-155: wait ALL. The docstring above always said "wait for ALL", but
    # `all(p.wait()==0 for ...)` short-circuits — a nonzero early lane left
    # later children unwaited and still landing legs in a tree about to be
    # declared terminal. Collect every rc first, then decide.
    rcs = [p.wait() for p in procs]
    # T-322/A-1: rc=20 is the spawner's PARTIAL-LEGS sentinel — some legs
    # failed but every leg's evidence exists, so the faultscreen can classify
    # and REPLACE them. Treating it as lane failure turned one dead leg into
    # "RELAY ABORT — NO verdict written" and a stalled training pause. Any
    # OTHER nonzero rc (crash 1, usage 2, signal <0) still aborts, fail-closed.
    return all(rc in (0, 20) for rc in rcs)


def _gate_ensure_clean(req, legs_dir, lanes, ledger_path, seed_blocks_path):
    """R-3: T-93-classify the gate legs; quarantine FAULT/UNSAMPLED and
    replace from the frozen full_gate block until N clean, GLOBAL trajectory
    balance preserved. (ok, why). WS1-r0's gate carried 6/12 fault legs
    (T-126) — at that incidence an unscreened gate adjudicates an instrument,
    not a policy. Escape (rollback): FD_V3_FAULTSCREEN_DISABLE=1, read at use."""
    if os.environ.get("FD_V3_FAULTSCREEN_DISABLE", "0").strip() == "1":
        return True, "faultscreen_disabled_by_flag"
    import gate_relay_faultscreen as fs
    classify = fs.load_classifier()
    n = GATE_N[req["gate_type"]]

    def spawn_replacements(need):
        k = sum(need.values())
        seeds = next_seeds(ledger_path, "full_gate", k,
                           f"replace:gate:{req['label']}", seed_blocks_path)
        if not _spawn_lanes(req, seeds, fs.explicit_tmap(seeds, need),
                            lanes, legs_dir, False):
            return False
        bridge_paired_layout(legs_dir)
        return True

    ok, why, _ = fs.ensure_clean_panel(
        legs_dir, {"straight": n // 2, "racetrack": n // 2},
        classify, spawn_replacements)
    return ok, why


def _retention_hold(request_path, legs_dir, label, attempt, reason, detail,
                    verdicts_dir=None):
    """The ONE place a retention non-clean outcome becomes terminal: halt file
    + operator page + terminal marker. A HOLD is a page, never a retry.

    T-237/B2: also drops a HOLD sentinel in the learner's log_dir so the pause
    can tell DEAD from LATE (verdicts_dir optional => older callers unchanged).
    """
    with open(os.path.join(legs_dir, f"RETENTION_HALT_{label}.json"), "w") as f:
        json.dump({"label": label, "halt_reason": detail}, f)
    if verdicts_dir:
        state.hold_sentinel(verdicts_dir, label, 6, reason,
                            attempt=attempt, halt_reason=detail,
                            legs_dir=legs_dir)
    print(f"RELAY HOLD {label}: {reason} ({detail}) — NO verdict, operator "
          f"page", file=sys.stderr)
    return state.terminal_state(request_path, 6, reason, label=label,
                                attempt=attempt, halt_reason=detail)


def process_gate(request_path, legs_root, lanes, dry_run,
                 seed_blocks_path, ledger_path, verdicts_dir):
    """ONE request end-to-end. Returns 0 on published verdict (or clean
    dry-run), nonzero otherwise — with NO verdict file on any failure."""
    req, problems = load_request(request_path)
    if problems:
        print(f"RELAY SKIP {request_path}: {problems}", file=sys.stderr)
        if dry_run:                      # a rehearsal must not seal the gate
            return 2
        return state.terminal_state(request_path, 2, "malformed_request",
                                    problems=problems)
    label, n = req["label"], GATE_N[req["gate_type"]]
    # R-1: legs land in a PER-ATTEMPT dir. A shared `gate_{label}` with
    # exist_ok=True let a retry pool its legs with the previous attempt's, so
    # the retention screen counted every attempt at once, failed its N check,
    # and re-dispatched forever (36 legs in one gate_W2/, 84 seeds burned).
    legs_dir, attempt = state.attempt_dir(os.path.join(legs_root,
                                                       f"gate_{label}"))
    budget = panel_budget_s(req)        # E-5: expected-completion stamp
    with open(os.path.join(legs_dir, "PANEL_ETA.json"), "w") as fh:
        json.dump({"label": label, "attempt": attempt,
                   "spawned_epoch": int(time.time()), "budget_s": budget,
                   "deadline_epoch": int(time.time()) + budget,
                   "note": "advisory watchdog budget (T-275); nothing "
                           "auto-aborts — the relay pages PANEL_OVERDUE"},
                  fh, indent=1)
    # T-363: a request carrying "seeds" RE-FLIES that exact panel in that
    # exact order (pairing = position, via global_traj_map); cursor unmoved.
    # Absent => the normal fresh draw, byte-identical. Faultscreen
    # REPLACEMENTS still draw fresh (line ~312) — disclosed, not silent.
    seeds = (pinned_seeds(ledger_path, req["seeds"], n, f"refly:{label}",
                          seed_blocks_path) if req.get("seeds") else
             next_seeds(ledger_path, "full_gate", n, f"gate:{label}",
                        seed_blocks_path))
    tmap = global_traj_map(seeds)
    if not _spawn_lanes(req, seeds, tmap, lanes, legs_dir, dry_run):
        print(f"RELAY ABORT {label}: lane failure — NO verdict written",
              file=sys.stderr)
        if dry_run:                      # a rehearsal must not seal the gate
            return 3
        return state.terminal_state(request_path, 3, "lane_failure",
                                    label=label, attempt=attempt)
    if dry_run:
        print(json.dumps({"dry_run": True, "label": label, "n": n,
                          "seeds": seeds, "panel": panel_for_stage(req["stage"])}))
        return 0
    panel = panel_for_stage(req["stage"])
    if panel:
        # RC3 panel on the SAME frozen checkpoint (executor = slice 2).
        # Any non-clean outcome => NO verdict (training pauses) + HALT file
        # for the operator page (MANUAL-demotion election 2026-07-28).
        import gate_relay_retention as ret
        try:
            res = ret.run_panel(
                req, panel, legs_dir, lanes,
                lambda blk, n, why: next_seeds(ledger_path, blk, n, why,
                                               seed_blocks_path),
                os.path.join(os.path.dirname(ledger_path),
                             "strike_ledger.jsonl"),
                bridge_paired_layout)
        except (KeyboardInterrupt, SystemExit):
            raise            # operator kill / seed exhaustion: never a HOLD
        except Exception as exc:
            # T-154: an escape here skipped terminal_state => no .terminal
            # marker => the poller re-dispatched forever, burning fresh seeds
            # every pass (the pre-R-1 loop, through a vector R-1 left open).
            return _retention_hold(request_path, legs_dir, label, attempt,
                                   f"retention_exception:{type(exc).__name__}",
                                   str(exc), verdicts_dir)
        with open(os.path.join(legs_dir, "retention_result.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        if not res["clean"]:
            return _retention_hold(request_path, legs_dir, label, attempt,
                                   "retention_hold", res["halt_reason"],
                                   verdicts_dir)
    return grade_and_publish(req, request_path, label, attempt, seeds,
                             legs_dir, lanes, ledger_path, seed_blocks_path,
                             verdicts_dir)


def grade_and_publish(req, request_path, label, attempt, seeds, legs_dir,
                      lanes, ledger_path, seed_blocks_path, verdicts_dir,
                      decorate=None, request_side_effects=True):
    """E-4: the ONE bridge -> ensure-clean -> spec -> grade -> publish flow.
    The T-273 false FAIL happened because a completion driver hand-mirrored
    this sequence and missed the bridge step — so the sequence now has ONE
    name, called by the relay (process_gate) and by any completion/repair
    driver alike. `decorate(verdict)` lets a driver stamp provenance (e.g.
    amends/repair) before the atomic publish; `request_side_effects=False`
    suppresses the .processed marker + terminal_state for T-233-style
    amendment drivers that must not touch the original request lifecycle."""
    bridge_paired_layout(legs_dir)      # frozen-instrument layout (precedent)
    ok, why = _gate_ensure_clean(req, legs_dir, lanes, ledger_path,
                                 seed_blocks_path)
    if not ok:
        print(f"RELAY HOLD {label}: gate faultscreen {why} — NO verdict "
              f"written, operator page", file=sys.stderr)
        if not request_side_effects:
            return 7
        state.hold_sentinel(verdicts_dir, label, 7, f"gate_faultscreen:{why}",
                            attempt=attempt, legs_dir=legs_dir)
        return state.terminal_state(request_path, 7, f"gate_faultscreen:{why}",
                                    label=label, attempt=attempt)
    spec = os.path.join(legs_dir, "gate_spec.json")
    # T-362: safety_criteria passes request -> spec verbatim when present;
    # absent stays absent (runner's v2 default), so pre-A-9 requests are
    # byte-identical. Without this the runner NEVER sees an A-9 criterion.
    body = {"gate_type": req["gate_type"]}
    if "safety_criteria" in req:
        body["safety_criteria"] = req["safety_criteria"]
    with open(spec, "w") as fh:
        json.dump(body, fh)
    out = os.path.join(legs_dir, "gate_adjudication.json")
    rc = subprocess.run([sys.executable, RUNNER, "--mode", "gate",
                         "--legs-dir", legs_dir, "--spec", spec,
                         "--out", out]).returncode
    try:
        verdict = json.load(open(out))
    except Exception:
        verdict = None
    if rc != 0 or verdict is None:
        print(f"RELAY ABORT {label}: runner rc={rc} — NO verdict written",
              file=sys.stderr)
        if not request_side_effects:
            return 4
        return state.terminal_state(request_path, 4, "runner_failed",
                                    label=label, attempt=attempt, runner_rc=rc)
    verdict["label"] = label
    verdict["relay"] = {"request": request_path, "seeds": seeds,
                        "legs_dir": legs_dir, "attempt": attempt}
    if decorate is not None:
        decorate(verdict)
    write_verdict_atomic(verdicts_dir, label, verdict)
    if request_side_effects:
        with open(request_path + ".processed", "w") as fh:  # legacy marker
            fh.write("published\n")
        state.terminal_state(request_path, 0, "published",
                             label=label, attempt=attempt)
    print(f"RELAY PUBLISHED {label}: passed={verdict['passed']} "
          f"k={verdict.get('k')}/n={verdict.get('n')}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="v3 gate-relay core (T-74)")
    ap.add_argument("--request", required=True)
    ap.add_argument("--legs-root", required=True)
    ap.add_argument("--lanes", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-blocks", default=SEED_BLOCKS_DEFAULT)
    ap.add_argument("--ledger", default=os.path.join(FD, "logs/gate_relay_ledger.jsonl"))
    ap.add_argument("--verdicts-dir", default=os.path.join(FD, "logs/gate_verdicts"))
    a = ap.parse_args(argv)
    # F-1: this process's environment IS the eval lanes' environment — legs
    # are spawned with no env= and inherit it wholesale. Assert the contract
    # here, before any lane spawns, so a relay started from a bare shell
    # cannot fly gate legs on a different env than training. role=eval
    # requires FD_V3_REHEARSAL_DISABLE=1 (sealed gates fly the current cell).
    # FIXAWIRE-719: the FD_CURRICULUM_V3 guard that used to wrap this call is
    # REMOVED. It made the contract fail-OPEN in exactly the case it exists to
    # catch — a relay started from a bare shell has V3 unset, so it skipped the
    # assertion and flew gate legs on an unverified environment.
    sys.path.insert(0, FD)
    from src.drl import env_contract
    env_contract.assert_contract(env_contract.EVAL)
    if os.path.exists(a.request + ".processed"):
        print(f"RELAY SKIP {a.request}: already processed", file=sys.stderr)
        return 0
    # R-1: ANY terminal state blocks re-dispatch, not just success. A HOLD or
    # an abort stays put until a human removes the marker; retrying one on a
    # timer burned 12 fresh seeds per pass and could never clear itself.
    if state.is_terminal(a.request):
        rec = state.read_terminal(a.request) or {}
        print(f"RELAY SKIP {a.request}: terminal rc={rec.get('rc', '?')} "
              f"({rec.get('reason', 'unreadable marker')}) — remove "
              f"{a.request}{state.TERMINAL_SUFFIX} to re-dispatch",
              file=sys.stderr)
        return 0
    if panel_total_check() != PANEL_TOTAL:
        # T-262/B2: rc=5 ALONE IS INVISIBLE TO THE LEARNER. Its pause predicate
        # is "no verdict published yet", so a fail-closed relay looked identical
        # to a merely slow panel: W12 burned the whole 300-min window and then
        # failed open (measured 2026-08-05 — the same shape B2 was built to stop,
        # reached by a vector B2 did not screen). Publish the DEAD signal that
        # _v3_read_hold already knows how to read, and mark the request terminal
        # so it stays a HOLD until a human clears it.
        label = os.path.basename(a.request)[:-5]
        why = "panel_total_check=%d expected=%d" % (panel_total_check(),
                                                    PANEL_TOTAL)
        state.hold_sentinel(a.verdicts_dir, label, 5, "panel_rule_drift",
                            detail=why)
        state.terminal_state(a.request, 5, "panel_rule_drift", label=label)
        print("RELAY HALT: retention panel rule no longer reproduces the "
              "generator's %d stratum-screens (%s) — refusing to run"
              % (PANEL_TOTAL, why), file=sys.stderr)
        return 5
    return process_gate(a.request, a.legs_root, a.lanes, a.dry_run,
                        a.seed_blocks, a.ledger, a.verdicts_dir)


if __name__ == "__main__":
    sys.exit(main())
