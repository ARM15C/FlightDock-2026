#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T-344 statistics layer for tools/pvt_paired_census.py. Split
#   out of that file when it crossed its declared soft cap -- the cap was NOT
#   raised (feedback_never_commit_a_hard_breach / caps-estimates-run-low).
#   Holds the CLEAN-vs-FAULT contrast, Holm-Bonferroni multiplicity, the
#   Hodges-Lehmann shift estimator, and the PRE-DECLARED test families. The
#   PRIMARY test carries a REGISTERED PREDICTION OF NO DIFFERENCE (see
#   analysis/pvtprov_20260808T225910Z/FINDINGS.md §0.4): T-93 FAULT is an
#   along-track TRUTH-channel defect, so it does not enter the below axis, and
#   the banked T-194 control measured below unchanged. A significant result
#   therefore REFUTES the banked account rather than confirming a hunch.
# PUBLIC API: holm(pvals), hl_shift(a, b), class_contrast(rows, metric, view,
#   alpha, m, field), contrasts(panel, alpha), selfcheck_stats(),
#   SECONDARY_FAMILY, BEHIND_CE.
# SIDE-EFFECTS: none. Pure functions over in-memory rows; no file, env,
#   subprocess, or network access of any kind.
# CAPS: file soft 160 / hard 220 code-LoC; function LoC cap 45.
# COVERAGE: selfcheck_stats() -- S4 (class test significant AND permuted-label
#   null) and S5 (Holm must be able to FLIP a raw-significant verdict); both
#   arms required. Invoked by the census tool's fail-closed --selfcheck.
# =============================================================================
import numpy as np
from scipy.stats import mannwhitneyu

BEHIND_CE = "ctrl_minus_env_behind_CONFOUNDED_descriptive"
# Family fixed in FINDINGS.md §0.5 BEFORE the instrument ran. m = 7.
SECONDARY_FAMILY = (("env_below", "abs_p95"), ("ctrl_cross", "abs_p95"),
                    ("env_cross", "abs_p95"),
                    ("ctrl_minus_env_below", "abs_p95"),
                    (BEHIND_CE, "abs_p95"),
                    ("ctrl_minus_env_cross", "abs_p95"),
                    ("ctrl_below", "signed_median"))


def holm(pvals):
    """Holm-Bonferroni. Returns adjusted p in the INPUT order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def hl_shift(a, b):
    """Hodges-Lehmann shift (median pairwise difference) + a distribution-free
    CI from the pairwise-difference order statistics (normal approximation)."""
    d = np.sort((np.asarray(a)[:, None] - np.asarray(b)[None, :]).ravel())
    n1, n2, tot = len(a), len(b), len(d)
    se = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    k = int(round(n1 * n2 / 2.0 - 1.96 * se))
    lo = d[max(0, k)] if tot else float("nan")
    hi = d[min(tot - 1, tot - 1 - max(0, k))] if tot else float("nan")
    return (round(float(np.median(d)), 4), round(float(lo), 4),
            round(float(hi), 4))


def class_contrast(rows, metric, view, alpha=0.05, m=1, field="abs_p95"):
    """CLEAN vs FAULT on one per-leg metric. m = DECLARED family size; the
    Holm-adjusted verdict is what is reported. Fewer than 3 legs a side is
    UNSAMPLED -- never a p-value on a degenerate group."""
    grp = {}
    for r in rows:
        blk = r.get("views", {}).get(view, {}).get(metric)
        lab = r.get("class_recomputed")
        if blk and lab in ("CLEAN", "FAULT"):
            grp.setdefault(lab, []).append(blk[field])
    if len(grp.get("CLEAN", [])) < 3 or len(grp.get("FAULT", [])) < 3:
        return {"metric": metric, "field": field, "verdict": "UNSAMPLED",
                "n": {k: len(v) for k, v in grp.items()}}
    u, p = mannwhitneyu(grp["CLEAN"], grp["FAULT"], alternative="two-sided")
    shift, lo, hi = hl_shift(grp["CLEAN"], grp["FAULT"])
    n1, n2 = len(grp["CLEAN"]), len(grp["FAULT"])
    return {"metric": metric, "field": field, "view": view,
            "n_clean": n1, "n_fault": n2,
            "median_clean": round(float(np.median(grp["CLEAN"])), 4),
            "median_fault": round(float(np.median(grp["FAULT"])), 4),
            "U": float(u), "p_raw": round(float(p), 6), "family_m": m,
            "hodges_lehmann": shift, "hl_ci95": [lo, hi],
            "rank_biserial": round(1.0 - 2.0 * u / (n1 * n2), 4)}


def contrasts(panel, alpha=0.05):
    """PRIMARY (m=1, registered prediction NO DIFFERENCE) + the pre-declared
    SECONDARY family under Holm. A raw p<alpha with Holm p>alpha reports
    NOT SIGNIFICANT -- the correction is decisive, not decorative."""
    rows = panel["rows"]
    pri = class_contrast(rows, "ctrl_below", "filtered", alpha, m=1)
    pri["registered_prediction"] = "NO_DIFFERENCE"
    pri["p_holm"] = pri.get("p_raw")
    pri["verdict"] = ("SIGNIFICANT_refutes_T194_account"
                      if pri.get("p_raw", 1.0) < alpha
                      else "NULL_confirms_T194_retires_my_claim")
    sec = [class_contrast(rows, k, "filtered", alpha, m=len(SECONDARY_FAMILY),
                          field=f) for k, f in SECONDARY_FAMILY]
    ok = [s for s in sec if "p_raw" in s]
    for s, adj in zip(ok, holm([s["p_raw"] for s in ok])):
        s["p_holm"] = round(float(adj), 6)
        s["verdict"] = "SIGNIFICANT" if adj < alpha else "NOT_SIGNIFICANT"
        s["flipped_by_holm"] = bool(s["p_raw"] < alpha <= adj)
    return {"primary": pri, "secondary_family": sec,
            "family_m": len(SECONDARY_FAMILY), "alpha": alpha}


def selfcheck_stats():
    """S4 class test both ways; S5 Holm actually flips a verdict."""
    def mk(vals):
        return [{"class_recomputed": c,
                 "views": {"filtered": {"ctrl_below": {"abs_p95": v,
                                                       "n": 999}}}}
                for c, v in vals]
    sep = mk([("CLEAN", 0.1 + i * 0.01) for i in range(8)]
             + [("FAULT", 0.9 + i * 0.01) for i in range(8)])
    r = class_contrast(sep, "ctrl_below", "filtered", m=1)
    assert r["p_raw"] < 0.05, "S4: separated classes must be significant"
    same = mk([("CLEAN" if i % 2 else "FAULT", 0.5 + (i % 3) * 0.01)
               for i in range(16)])
    r2 = class_contrast(same, "ctrl_below", "filtered", m=1)
    assert r2["p_raw"] > 0.05, "S4 disagree arm: permuted labels significant"
    adj = holm([0.04, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    assert 0.04 < 0.05 <= adj[0], \
        "S5: Holm must be able to FLIP a raw-significant result"
    return "S4/S5 class-test + multiplicity PASS (both arms)"
