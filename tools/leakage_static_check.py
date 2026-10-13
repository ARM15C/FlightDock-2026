#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T0-SL zero-runtime tranche (goals-v4 G3/G4) — STATIC
#   truth-reachability check over the declared training-path surface. Two
#   passes: (1) TOKEN SCAN: truth-field tokens (declared naming convention)
#   must not appear in training-path files (actor/critic/reward/replay/
#   observation/policy surface); (2) IMPORT REACHABILITY: no training-path
#   module may (transitively) import a declared truth-source module. Both
#   passes apply the G3 SYNTHESIS WHITELIST, which is now a CLOSED, digest-
#   pinned, DENY-BY-DEFAULT artifact (docs/truth_synthesis_whitelist.json) —
#   the single source of truth for permitted truth->sensor paths (goals-v4 G3
#   edit #7). Deny-by-default = the checker FAILS CLOSED (rc=2) if the artifact
#   is missing, unreadable, digest-mismatched, not flagged deny_by_default, or
#   whitelists a module that is not itself a declared truth module (phantom).
#   Expanding what is permitted therefore requires an auditable artifact edit +
#   a digest recompute (both visible in git) — it cannot be done silently in
#   code. Result claims are EVIDENCE under declared conventions, not formal
#   proof (Python dynamic access; the runtime probe closes the gap — PROBE_SPEC,
#   Dispatch P).
# PUBLIC API: token_scan(files, tokens, whitelist_sites) -> [finding];
#   import_edges(py_files) -> {module: set(imports)};
#   reachable_truth(edges, roots, truth_modules, whitelist_modules) -> [path];
#   canonical_digest(doc) -> hexdigest; load_whitelist(repo_root) ->
#   (modules, token_sites, meta) | raises WhitelistError;
#   run_declared_config(repo_root) -> (findings, paths, report_lines);
#   main(argv=None) -> int (0 clean / 1 violations / 2 config error).
# SIDE-EFFECTS: reads source files + the whitelist JSON; prints report to
#   stdout. NO writes, no imports of scanned modules (ast only), no network, no
#   runtime effect — nothing in the runtime tree imports this tool.
# CAPS: file soft 210 / hard 300 lines; functions <= 50 lines.
# COVERAGE: tests/unit/test_leakage_static_check.py — two-direction fixtures
#   (planted violation CAUGHT; whitelisted synthesis PASSES) + deny-by-default
#   digest/fail-closed fixtures + py_compile.
# =============================================================================
import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys

# --- DECLARED CONVENTIONS (FlightDock; goals-v4 G3/G4) ----------------------
# Truth-field naming convention: raw_* paired-truth channels + SIM_STATE truth
# taps. Tokens are matched as whole words. This is the DETECTION surface (what
# counts as truth) — distinct from the permit-list, which lives in the closed
# deny-by-default JSON artifact loaded at run time.
TRUTH_TOKENS = [
    r"raw_behind_m", r"raw_cross_m", r"raw_below_m", r"raw_range_m",
    r"raw_m3b", r"raw_geom", r"sim_state_truth", r"truth_behind",
    r"truth_below", r"truth_cross", r"leader_truth",
]
# Training-path surface (actor/critic/reward/replay/observation/policy). SB3
# internals are out-of-tree; this is the in-tree surface those consume.
TRAINING_PATH_FILES = [
    "src/drl/reward_mrrl.py",
    "src/drl/residual_action.py",
    "src/drl/observation_builder.py",
    "src/drl/train.py",
    "src/drl/curriculum.py",
    "src/drl/config.py",
]
# Declared truth-source modules (module name -> why it is truth-bearing).
TRUTH_MODULES = {
    "uwb_measurement_emitter": "truth-sourced UWB synthesis (whitelisted)",
    "plane6dof_fdm": "FDM truth state",
}
# Closed deny-by-default whitelist artifact (single source of truth).
WHITELIST_JSON = os.path.join("docs", "truth_synthesis_whitelist.json")


class WhitelistError(Exception):
    """Raised when the closed whitelist is absent/tampered/invalid => rc=2."""


def canonical_digest(doc):
    """sha256 over the artifact with _meta.digest removed (sort_keys, compact).
    Shared verbatim by the artifact generator and the fail-closed load path."""
    d = copy.deepcopy(doc)
    (d.get("_meta") or {}).pop("digest", None)
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_whitelist(repo_root):
    """Load + VERIFY the closed deny-by-default whitelist. Fail closed: any
    missing/tampered/invalid condition raises WhitelistError (=> rc=2), never a
    silent permissive fallback."""
    path = os.path.join(repo_root, WHITELIST_JSON)
    if not os.path.exists(path):
        raise WhitelistError(f"whitelist artifact MISSING ({WHITELIST_JSON}) — "
                             "deny-by-default refuses to certify")
    try:
        doc = json.loads(open(path, encoding="utf-8").read())
    except (json.JSONDecodeError, OSError) as e:
        raise WhitelistError(f"whitelist unreadable: {e}")
    if doc.get("deny_by_default") is not True:
        raise WhitelistError("whitelist missing 'deny_by_default: true'")
    stored = (doc.get("_meta") or {}).get("digest")
    calc = canonical_digest(doc)
    if stored != calc:
        raise WhitelistError(f"whitelist DIGEST MISMATCH (laundering guard): "
                             f"stored={stored} calc={calc}")
    modules, token_sites = set(), set()
    for e in doc.get("entries", []):
        kind = e.get("kind")
        if kind == "module":
            mod = e.get("source_module")
            if mod not in TRUTH_MODULES:
                raise WhitelistError(f"whitelisted module '{mod}' is not a "
                                     "declared TRUTH_MODULE (phantom entry)")
            modules.add(mod)
        elif kind == "token_site":
            token_sites.add((e.get("file"), e.get("token")))
        else:
            raise WhitelistError(f"unknown whitelist entry kind: {kind!r}")
    return modules, token_sites, doc.get("_meta") or {}


def token_scan(files, tokens, whitelist_sites):
    """Whole-word truth-token hits in training-path files (comments included:
    a commented truth access is still a smell worth surfacing)."""
    pats = [(t, re.compile(r"\b" + t + r"\b")) for t in tokens]
    findings = []
    for path in files:
        if not os.path.exists(path):
            findings.append((path, "<missing>", 0,
                             "declared training-path file NOT FOUND"))
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f, 1):
                for tok, pat in pats:
                    if pat.search(line) and (path, tok) not in whitelist_sites:
                        findings.append((path, tok, ln, line.strip()[:100]))
    return findings


def import_edges(py_files):
    """{module_basename: set(imported module basenames)} via ast (no exec)."""
    edges = {}
    for path in py_files:
        mod = os.path.splitext(os.path.basename(path))[0]
        edges.setdefault(mod, set())
        try:
            tree = ast.parse(open(path, encoding="utf-8",
                                  errors="replace").read())
        except SyntaxError:
            edges[mod].add("<SYNTAX-ERROR>")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    edges[mod].add(a.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom) and node.module:
                edges[mod].add(node.module.split(".")[-1])
    return edges


def reachable_truth(edges, roots, truth_modules, whitelist_modules):
    """Import paths root->...->truth_module, whitelist-filtered."""
    hits = []
    for root in roots:
        stack, seen = [(root, [root])], set()
        while stack:
            mod, path = stack.pop()
            if mod in seen:
                continue
            seen.add(mod)
            for nxt in sorted(edges.get(mod, ())):
                if nxt in truth_modules and nxt not in whitelist_modules:
                    hits.append(path + [nxt])
                elif nxt in edges:
                    stack.append((nxt, path + [nxt]))
    return hits


def run_declared_config(repo_root):
    wl_modules, wl_sites, meta = load_whitelist(repo_root)  # raises => rc=2
    files = [os.path.join(repo_root, f) for f in TRAINING_PATH_FILES]
    findings = token_scan(files, TRUTH_TOKENS, wl_sites)
    src_dirs = [os.path.join(repo_root, "src")]
    py_files = [os.path.join(dp, fn) for d in src_dirs
                for dp, _, fns in os.walk(d) for fn in fns
                if fn.endswith(".py")]
    edges = import_edges(py_files)
    roots = [os.path.splitext(os.path.basename(f))[0]
             for f in TRAINING_PATH_FILES]
    paths = reachable_truth(edges, roots, set(TRUTH_MODULES), wl_modules)
    lines = ["T0-SL STATIC CHECK (declared conventions; evidence, not proof)",
             f"whitelist: {WHITELIST_JSON} deny_by_default digest="
             f"{(meta.get('digest') or '')[:12]}.. modules={sorted(wl_modules)}",
             f"training-path files: {len(files)} | truth tokens: "
             f"{len(TRUTH_TOKENS)} | token sites: {len(wl_sites)}",
             f"token findings: {len(findings)} | import-reach findings: "
             f"{len(paths)}"]
    for p, tok, ln, txt in findings:
        lines.append(f"  TOKEN {p}:{ln} [{tok}] {txt}")
    for path in paths:
        lines.append("  IMPORT " + " -> ".join(path))
    lines.append("VERDICT: " + ("VIOLATIONS FOUND" if findings or paths else
                                "no truth-reachability found under declared"
                                " conventions + closed whitelist"))
    return findings, paths, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="T0-SL static leakage check")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    try:
        findings, paths, lines = run_declared_config(args.repo_root)
    except WhitelistError as e:
        print("T0-SL CONFIG ERROR (deny-by-default, fail-closed): " + str(e))
        return 2
    print("\n".join(lines))
    return 1 if (findings or paths) else 0


if __name__ == "__main__":
    sys.exit(main())
