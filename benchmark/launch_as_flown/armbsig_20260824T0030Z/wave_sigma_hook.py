#!/usr/bin/env python3
# RESPONSIBILITY: run the per-leg REALIZED-SIGMA gate for one exam lane, exactly as ARM-B's own
#   driver ran it (armbsig_driver.sh:387-398), so a candidate corpus carries the same
#   `sigma_gate.json` + `fdm_witness.jsonl` its comparator carries. RB-41: armbsig_exam_wave.py
#   never invoked the gate, so every candidate cell read "sigma UNSAMPLED" and the exam measured
#   none of the turbulence it flew. RB-41 AMENDMENT 1: logs/fdm_env<N>.jsonl is opened mode "w"
#   per FDM launch (tools/json_fdm/plane6dof_fdm.py:284), i.e. TRUNCATED at the next episode --
#   so the gate MUST run inside the per-lane loop, before that lane's slot flies its next row.
#   Retro-fitting it after a wave is impossible, not merely awkward. A sibling module rather than
#   inline code purely to respect armbsig_exam_wave.py's DECLARED caps; NO cap was raised.
# PUBLIC API: band(sigma) -> float the driver's band rule, evaluated at this sigma
#             gate(leg_dir, env_id, sigma, log_path) -> int the gate's rc (0 PASS, 2
#                 UNSAMPLED/INCONCLUSIVE, 3 OUT-OF-BAND/ASSERT_FAIL), read from .returncode and
#                 NEVER through a pipe (feedback_exit_code_never_through_a_pipe)
# SIDE-EFFECTS: spawns leg_sigma_gate.py as a subprocess (which writes <leg_dir>/sigma_gate.json
#   and <leg_dir>/fdm_witness.jsonl and reads logs/fdm_env<N>.jsonl), appending its stdout+stderr
#   to the lane log. Draws no seed, touches no ledger, kills nothing, writes nothing else.
# SIZE: soft 45 / hard 70 LoC; function cap 25.
# COVERAGE: wave_sigma_hook_selftest.py, EXECUTED, with the band pinned against the driver's own
#   arithmetic and a non-zero rc driven through to a refusal.
import os, subprocess

FD = "/home/ubuntu/FlightDock"
PY = os.path.join(FD, "flightdock_native_venv/bin/python")
GATE = os.path.join(FD, "analysis/armb_refly_20260807T1830Z/leg_sigma_gate.py")
#: armbsig_driver.sh:387 VERBATIM -- sd(s) = 0.04015*(s/0.30), band = 2.3877*sd(s), 4dp via awk
#: "%.4f". sigma 0.20 -> 0.0639, 0.25 -> 0.0799: STRICTLY TIGHTER than A-10's signed 0.10 at
#: every sigma flown here, so it can only refuse MORE legs, never manufacture a pass.
SD_UNIT, K = 0.04015, 2.3877


def band(sigma):
    return round(K * SD_UNIT * (float(sigma) / 0.30), 4)


def gate(leg_dir, env_id, sigma, log_path):
    """Gate ONE lane. rc is returned, never swallowed: the caller refuses the leg on non-zero and
    the leg STAYS IN THE DENOMINATOR (armbsig_driver.sh's own rule -- a refused leg is attrition,
    never a deletion). No --paired-csv: ARM-B's driver passed none, and a candidate gated by a
    different instrument is not comparable to it."""
    cmd = [PY, GATE, "--leg", leg_dir, "--env-id", str(env_id),
           "--commanded", str(sigma), "--band", str(band(sigma))]
    with open(log_path, "a") as fh:
        fh.write("\n# --- realized-sigma gate (RB-42) ---\n# " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, cwd=FD, stdout=fh,
                              stderr=subprocess.STDOUT).returncode
