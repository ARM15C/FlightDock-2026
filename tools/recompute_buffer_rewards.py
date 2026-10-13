# =============================================================================
# RESPONSIBILITY: offline replay-buffer reward recompute -- the CLI and the
#   I/O. Loads a banked SB3 ReplayBuffer, adds ONE term's additive delta to
#   its stored rewards, writes a NEW pickle plus a JSON report, and proves the
#   banked input was not touched. WHICH term, WHICH axes and WHAT must hold
#   before any reward moves all live in tools/reprice_terms.py -- split out by
#   R1ARM-723 when the "precontact" term pushed this file through its HARD
#   cap. Applying two terms = two chained runs, each with its own report; the
#   reports' input/output shas are the chain.
# PUBLIC API: compute_delta(buffer, term) -> np.ndarray; main(argv) -> int.
# SIDE-EFFECTS: reads the input pickle; writes ONE new buffer pickle (never
#   the input -- identical in/out paths and existing outputs are refused;
#   input bytes are hash-asserted unchanged after the write) and ONE JSON
#   report.
# CAPS: file soft 100 / hard 130 code-LoC; functions <= 60.
#   Was soft 150 / hard 200 as a single file; the split reset it to what this
#   half actually is. The HARD cap was never raised to fit code.
# COVERAGE: tests/unit/test_recompute_buffer_rewards.py (synthetic round-trip,
#   contamination refusal, input-immutability, overwrite refusal).
# =============================================================================
"""Fail-closed provenance: refuses buffers whose observations exceed the
OBS_CLIP rail (|obs| > 1.1 + 1e-6) -- VecNormalize-normalized transitions
(the inert DAW/bridge seeding paths) would decode to garbage under obs*50."""
import argparse
import hashlib
import json
import sys

import numpy as np

from src.drl import config
from tools.reprice_terms import TERMS, assert_decode_is_sane, axes

_RAIL = 1.1 + 1e-6


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size(buf):
    return buf.buffer_size if buf.full else buf.pos


def compute_delta(buf, term="ceiling"):
    """Per-transition shaping delta from stored endpoint observations.
    Raises ValueError on normalized-obs contamination (fail-closed).
    `term` selects the axis binding from reprice_terms.TERMS; the
    original ceiling-only behavior."""
    n = _size(buf)
    obs, nxt = buf.observations[:n], buf.next_observations[:n]
    mx = float(max(np.abs(obs).max(), np.abs(nxt).max()))
    if mx > _RAIL:
        raise ValueError(
            "provenance: |obs| max %.4f exceeds OBS_CLIP rail %.4f — buffer "
            "contains VecNormalize-normalized transitions; refusing" % (mx, _RAIL))
    ax = axes(term)
    fn = np.vectorize(TERMS[term]["fn"], otypes=[np.float64])
    if TERMS[term].get("point"):
        # POINT: every axis at s'. Validated against an identity this delta
        # never computes, BEFORE any reward is touched -- raises on mismatch,
        # so a direct caller is as fail-closed as main().
        assert_decode_is_sane(nxt, ax)
        return fn(*[nxt[..., i] * nm for i, nm in ax])
    idx, norm = ax[0]                       # PAIR: one axis, (s, s')
    return fn(obs[..., idx] * norm, nxt[..., idx] * norm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer", required=True, help="banked replay_buffer .pkl")
    ap.add_argument("--out", required=True, help="NEW output .pkl (must not exist)")
    ap.add_argument("--report", default=None, help="JSON report path "
                    "(default: <out>.report.json)")
    ap.add_argument("--term", default="ceiling", choices=sorted(TERMS),
                    help="which shaping term to apply (default: ceiling)")
    a = ap.parse_args(argv)
    import os
    if os.path.abspath(a.out) == os.path.abspath(a.buffer):
        print("REFUSED: --out equals --buffer (banked input is never overwritten)")
        return 2
    if os.path.exists(a.out):
        print("REFUSED: --out already exists: %s" % a.out)
        return 2
    from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

    in_sha = _sha256(a.buffer)
    buf = load_from_pkl(a.buffer)
    n = _size(buf)
    delta = compute_delta(buf, a.term)
    r_before = buf.rewards[:n].copy()
    buf.rewards[:n] = r_before + delta.astype(buf.rewards.dtype)
    save_to_pkl(a.out, buf)
    if _sha256(a.buffer) != in_sha:
        print("FATAL: input file bytes changed — investigate immediately")
        return 3
    ad, rb, ra = np.abs(delta), np.abs(r_before), np.abs(buf.rewards[:n])
    _ax = axes(a.term)
    report = {
        "input": a.buffer, "input_sha256": in_sha,
        "output": a.out, "output_sha256": _sha256(a.out),
        "transitions": int(n), "full": bool(buf.full),
        "term": a.term, "mode": "point" if TERMS[a.term].get("point") else "pair",
        "axes": [{"feature": f, "index": i, "norm": nm}
                 for (f, _), (i, nm) in zip(TERMS[a.term]["axes"], _ax)],
        # LEGACY singular keys, emitted verbatim for single-axis terms. Banked
        # campaign code gates on them (w1419_gb_verify.py:73 refuses a report
        # whose obs_feature is not the axis it staged), so dropping them would
        # break a GB gate for a cosmetic rename. Absent for multi-axis terms,
        # where no single feature is THE axis and a guessed one would lie.
        **({"obs_feature": TERMS[a.term]["axes"][0][0],
            "obs_index": _ax[0][0], "obs_norm": _ax[0][1]}
           if len(_ax) == 1 else {}),
        "decode_audit": (assert_decode_is_sane(buf.next_observations[:n], _ax)
                         if TERMS[a.term].get("point") else {}),
        "constants": TERMS[a.term]["constants"](),
        "delta": {"nonzero_frac": float((delta != 0).mean()),
                  "abs_p50": float(np.percentile(ad, 50)),
                  "abs_p95": float(np.percentile(ad, 95)),
                  "abs_p99": float(np.percentile(ad, 99)),
                  "abs_max": float(ad.max()), "mean": float(delta.mean())},
        "abs_reward_p50_before_after": [float(np.percentile(rb, 50)),
                                        float(np.percentile(ra, 50))],
        "abs_reward_p95_before_after": [float(np.percentile(rb, 95)),
                                        float(np.percentile(ra, 95))],
    }
    rp = a.report or (a.out + ".report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["delta"], indent=1))
    print("OK: %d transitions -> %s (report %s)" % (n, a.out, rp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
