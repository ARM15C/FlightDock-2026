#!/usr/bin/env python3
# RESPONSIBILITY: the per-leg REALIZED-SIGMA gate for one BENCH600 lane -- TWIN of
#   analysis/armbsig_20260824T0030Z/wave_sigma_hook.py with ONE declared delta, the S0 branch:
#     THE S0 BRANCH: on a row commanded
#     sigma 0.00 the parent's band collapses to band(0.00) = 0.0000 -- a zero-width comparison that
#     passes only on a bit-exact zero and is a DIFFERENT INSTRUMENT from the one that judged the
#     ARM-B ladder's own S0 rung (CALM_CONFIRMED via --paired-csv, armblad025_driver.sh:131-141).
#     A candidate paired to that rung must be gated by the SAME tier, so here sigma 0.00 forwards
#     the leg's OWN paired-channels CSV (<leg>/env<N>/paired_channels_ep0001.csv -- the overlay
#     lane layout, measured on eval3 wave_A2PVOFF025) and then PROVES the calm tier engaged with
#     the flown armblad025_calm_witness (rc 5 = never ran = rung UNGATED => the leg is REFUSED).
#   Every sigma > 0 takes the parent's path byte-for-byte. The parent is NOT edited.
# PUBLIC API: band(sigma) -> float; gate(leg_dir, env_id, sigma, log_path) -> int
#   rc 0 PASS (MEASURED_PASS or CALM_CONFIRMED). 2 UNSAMPLED/INCONCLUSIVE. 3 OUT-OF-BAND /
#   delivered wind on a calm cell. 16 S0 paired CSV unreadable (calm tier cannot engage).
#   17 S0 leg on which the calm tier NEVER RAN (silent tier-1 fallback). A refused leg stays in
#   the denominator (attrition), never deleted.
# SIDE-EFFECTS: spawns leg_sigma_gate.py (writes <leg>/sigma_gate.json + fdm_witness.jsonl) and,
#   on S0 only, armblad025_calm_witness.py (read-only); appends both to the lane log. Draws no
#   seed, touches no ledger, kills nothing, imports neither src/ nor torch.
# SIZE: soft 45 / hard 70 code-lines; function cap 25.
# COVERAGE: bench600_selfcheck.py -- S0 with a readable calm CSV engages the calm tier (witness
#   rc 0 read back), S0 with the CSV absent refuses 16, S0 whose gate never entered the calm
#   tier refuses 17, sigma 0.25 takes the parent path with the parent's band (pinned 0.0799).
import os
import subprocess

FD = "/home/ubuntu/FlightDock"
PY = os.path.join(FD, "flightdock_native_venv/bin/python")
GATE = os.path.join(FD, "analysis/armb_refly_20260807T1830Z/leg_sigma_gate.py")
CALM_WITNESS = os.path.join(FD, "analysis/armblad025_20260830T1600Z/armblad025_calm_witness.py")
SD_UNIT, K = 0.04015, 2.3877          # armbsig_driver.sh:387 VERBATIM, as in the parent
RC_CSV_ABSENT, RC_CALM_NEVER_RAN = 16, 17


def band(sigma):
    return round(K * SD_UNIT * (float(sigma) / 0.30), 4)


def paired_csv(leg_dir, env_id):
    """The leg's OWN paired-channels file under the overlay lane layout."""
    return os.path.join(leg_dir, "env%s" % env_id, "paired_channels_ep0001.csv")


def gate(leg_dir, env_id, sigma, log_path):
    calm = float(sigma) == 0.0
    cmd = [PY, GATE, "--leg", leg_dir, "--env-id", str(env_id),
           "--commanded", str(sigma), "--band", str(band(sigma))]
    if calm:
        csv = paired_csv(leg_dir, env_id)
        if not os.access(csv, os.R_OK):
            with open(log_path, "a") as fh:
                fh.write("\n# S0 REFUSED rc %d: paired CSV unreadable at %s -- the calm tier "
                         "cannot engage and the rung would be UNGATED\n" % (RC_CSV_ABSENT, csv))
            return RC_CSV_ABSENT
        cmd += ["--paired-csv", csv]
    with open(log_path, "a") as fh:
        fh.write("\n# --- realized-sigma gate (RB-42%s) ---\n# %s\n"
                 % (" + S6.1 calm tier" if calm else "", " ".join(cmd)))
        fh.flush()
        rc = subprocess.run(cmd, cwd=FD, stdout=fh, stderr=subprocess.STDOUT).returncode
        if calm:
            wrc = subprocess.run([PY, CALM_WITNESS, "--leg", leg_dir], cwd=FD, stdout=fh,
                                 stderr=subprocess.STDOUT).returncode
            if wrc == 5:
                fh.write("# S0 REFUSED rc %d: calm tier NEVER RAN (silent tier-1 fallback)\n"
                         % RC_CALM_NEVER_RAN)
                return RC_CALM_NEVER_RAN
    return rc
