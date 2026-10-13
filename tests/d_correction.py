#!/usr/bin/env python3
"""
d_correction.py — architecture-D terminal UWB correction (production runtime).
================================================================================
RESPONSIBILITY
    Memoryless per-tick algebraic correction of the GPS-relative boom-tip
    vector (behind, cross, below) using the 4 boom-frame UWB ranges.
    Small-delta Gauss-Newton least squares (1-2 iterations) with a Tikhonov
    prior on the GPS estimate — per-axis observability weighting falls out of
    the prior (directions the range Jacobian does not constrain stay at GPS).
    Range-consistency reject gate drops outlier ranges before the solve.
    Productionization of the step-3 validated prototype
    (analysis/step3_dalign_20260611T170357Z/s3b_replay/d_correction_proto.py);
    math identical. Consumed by baseline_formation_controller Hunk-D(ii)
    under FD_D_CORRECTION=1 + FD_GEOM_GPS_DIRECT=1 (step-4, R7.1 waiver).
FORBIDDEN BY CONSTRUCTION
    No Kalman. No carried state (module holds no mutable solution state).
    No covariance/P-matrix propagation. No truth injection (correct() sees
    only gps_rel + measured ranges + sigmas).
PUBLIC API
    correct(gps_rel, ranges, sigmas, rng=42, trust_m=0.0, trust_mode="reject")
        -> (corrected_rel ndarray(3), diagnostics dict)
        rng accepted for API parity with the validated prototype; correct()
        is DETERMINISTIC and generates no noise (rng unused).
        trust_m (SIGCLEAN Hunk-D1, 2026-06-12): per-axis trust region on the
        solved delta. 0.0 (default) = disabled = byte-identical math. When
        > 0 and max|delta| exceeds it, trust_mode decides (stateless):
        'reject' (default) returns the GPS triple unchanged (trust_exceeded=
        True, delta zeroed, delta_pre_trust preserved) — byte-identical to
        the pre-DFIX path. 'clamp' (DFIX Hunk-D2, 2026-06-13, d_census
        ed2c4606) scales the delta onto the trust boundary, direction
        preserved (trust_clamped=True) — removes the measured 4-5 Hz
        apply/reject consumed-signal discontinuity (0.43-0.51 m steps).
        Caller (controller Hunk-C1/C3) reads FD_D_TRUST_M / FD_D_TRUST_MODE
        env; this module never reads env.
    synthetic_selftest(rng=42) -> dict (known truth + sigma0.1 noise; asserts
        recovery + determinism; all noise via the rng argument, never the
        global numpy random state)
    make_rng(seed_or_gen) -> numpy.random.Generator
SIDE EFFECTS
    Pure computation; no I/O of any kind; no env reads.
CAPS file soft 200 / hard 300 lines; functions <= 40 LoC.
COVERAGE tests/unit/test_d_correction.py + py_compile.
ANCHORS must equal uwb_measurement_emitter's boom-frame anchor set:
    {0:(0,0,0)=boom tip, 1:(0,0.5,0), 2:(0,-0.5,0), 3:(0,0,0.3)} (behind, cross, below).
"""
import numpy as np

ANCHORS = np.array([
    [0.0, 0.0, 0.0],    # 0: boom tip
    [0.0, 0.5, 0.0],    # 1
    [0.0, -0.5, 0.0],   # 2
    [0.0, 0.0, 0.3],    # 3
])
CONSISTENCY_GATE_M = 2.0
GN_ITERS = 2


def make_rng(seed_or_gen=42):
    """numpy Generator from an int seed or pass-through Generator."""
    if isinstance(seed_or_gen, np.random.Generator):
        return seed_or_gen
    return np.random.default_rng(seed_or_gen)


def correct(gps_rel, ranges, sigmas, rng=42, trust_m=0.0, trust_mode="reject"):
    """One memoryless correction step.

    Args:
        gps_rel: (behind, cross, below) GPS-relative boom-tip vector [m].
        ranges: dict {anchor_id: measured_range_m} (subset of 0..3 allowed).
        sigmas: dict with 'uwb' (range sigma, e.g. 0.1) and per-axis GPS prior
                 sigmas 'gps_behind', 'gps_cross', 'gps_below' [m].
        rng: accepted for prototype API parity; UNUSED (deterministic).
        trust_m: per-axis trust region on |delta|; 0.0 = disabled (identical
                 math). Exceeding it triggers trust_mode (stateless).
        trust_mode: 'reject' (default, byte-identical: zero the delta) or
                 'clamp' (DFIX Hunk-D2: scale delta onto the trust boundary,
                 direction preserved — removes the apply/reject discontinuity).
    Returns:
        (corrected_rel ndarray(3), diagnostics dict)
    """
    p = np.asarray(gps_rel, dtype=float)
    ids = [i for i in sorted(ranges) if np.isfinite(ranges[i])]
    a = ANCHORS[ids]
    r = np.array([ranges[i] for i in ids], dtype=float)
    pred = np.linalg.norm(p - a, axis=1)
    keep = np.abs(r - pred) <= CONSISTENCY_GATE_M
    n_rej = int((~keep).sum())
    a, r = a[keep], r[keep]
    prior_sig = np.array([sigmas["gps_behind"], sigmas["gps_cross"], sigmas["gps_below"]], float)
    lam = np.diag(1.0 / np.maximum(prior_sig, 1e-6) ** 2)
    w = 1.0 / max(sigmas.get("uwb", 0.1), 1e-6) ** 2
    delta = np.zeros(3)
    resid_pre = resid_post = float("nan")
    if len(r) > 0:
        for _ in range(GN_ITERS):
            q = p + delta
            d = np.linalg.norm(q - a, axis=1)
            d = np.maximum(d, 1e-6)
            jac = (q - a) / d[:, None]
            resid = r - d
            lhs = w * jac.T @ jac + lam
            rhs = w * jac.T @ resid - lam @ delta
            delta = delta + np.linalg.solve(lhs, rhs)
        d0 = np.linalg.norm(p - a, axis=1)
        resid_pre = float(np.sqrt(np.mean((r - d0) ** 2)))
        dpost = np.linalg.norm(p + delta - a, axis=1)
        resid_post = float(np.sqrt(np.mean((r - dpost) ** 2)))
    diag = {"n_ranges_used": int(len(r)), "n_rejected": n_rej,
            "delta": delta.copy(), "resid_rms_pre": resid_pre,
            "resid_rms_post": resid_post, "trust_exceeded": False,
            "trust_clamped": False}
    if trust_m > 0.0 and float(np.max(np.abs(delta))) > trust_m:
        diag["delta_pre_trust"] = delta.copy()
        if trust_mode == "clamp":
            # DFIX Hunk-D2 (2026-06-13, d_census ed2c4606): scale the solution
            # onto the trust boundary instead of zeroing it — direction
            # preserved, consumed signal CONTINUOUS at the boundary (the
            # binary apply/reject switching at 4-5 Hz was measured injecting
            # 0.43-0.51 m consumed steps = the S1 driver). Stateless.
            delta = delta * (trust_m / float(np.max(np.abs(delta))))
            diag["trust_clamped"] = True
            diag["delta"] = delta.copy()
            return p + delta, diag
        # SIGCLEAN Hunk-D1: solution outside the GPS-prior trust region —
        # reject wholesale (default mode; byte-identical decision path).
        diag["trust_exceeded"] = True
        diag["delta"] = np.zeros(3)
        return p.copy(), diag
    return p + delta, diag


def _selftest_case(truth, gps_sig, rng, n=400, uwb_sig=0.1):
    """Monte-Carlo one geometry point; median |err| per axis + |radial err|,
    pre/post. Radial = error projected on the truth direction (the combination
    range-only anchors actually observe at standoff distances)."""
    truth = np.asarray(truth, float)
    u = truth / np.linalg.norm(truth)
    sig = {"uwb": uwb_sig, "gps_behind": gps_sig[0],
           "gps_cross": gps_sig[1], "gps_below": gps_sig[2]}
    pre, post, pre_r, post_r = [], [], [], []
    for _ in range(n):
        gps = truth + rng.normal(0, gps_sig)
        tr = np.linalg.norm(truth - ANCHORS, axis=1)
        ranges = {i: tr[i] + rng.normal(0, uwb_sig) for i in range(4)}
        corr, _ = correct(gps, ranges, sig)
        pre.append(np.abs(gps - truth))
        post.append(np.abs(corr - truth))
        pre_r.append(abs(float((gps - truth) @ u)))
        post_r.append(abs(float((corr - truth) @ u)))
    return (np.median(pre, axis=0), np.median(post, axis=0),
            float(np.median(pre_r)), float(np.median(post_r)))


def synthetic_selftest(rng=42):
    """Known truth + sigma0.1 ranges; asserts recovery + determinism.

    Physics note: at the M2 standoff (~10 m range) the radial direction is
    ~92% below-axis, so range-only anchors observe the RADIAL combination —
    behind-axis observability strengthens as range closes toward M3a/M3b.
    Asserted claims: regime-1 (close range) behind accuracy; regime-2
    (standoff) radial improvement WITHOUT behind degradation."""
    g = make_rng(rng)
    pre1, post1, _, _ = _selftest_case([0.5, 0.2, 0.4], [0.5, 0.5, 0.75], g)
    pre2, post2, prer2, postr2 = _selftest_case([4.0, 0.3, 9.7], [0.5, 0.5, 0.75], g)
    assert post1[0] <= 0.15, f"close-range behind err {post1[0]:.3f} > 0.15"
    assert np.all(post1 <= pre1 + 1e-9), "regime-1 correction must not degrade"
    assert postr2 < prer2, "regime-2 RADIAL err must improve over GPS"
    assert post2[0] <= pre2[0] * 1.15, "regime-2 behind err materially degraded"
    a1 = _selftest_case([4.0, 0.3, 9.7], [0.5, 0.5, 0.75], make_rng(123), n=50)
    a2 = _selftest_case([4.0, 0.3, 9.7], [0.5, 0.5, 0.75], make_rng(123), n=50)
    assert all(np.allclose(x, y) for x, y in zip(a1, a2)), "determinism violated"
    return {"regime1_pre": pre1.tolist(), "regime1_post": post1.tolist(),
            "regime2_pre": pre2.tolist(), "regime2_post": post2.tolist(),
            "regime2_radial_pre_post": [prer2, postr2]}


if __name__ == "__main__":
    print("[d_correction] selftest:", synthetic_selftest())
