#!/usr/bin/env python3
# RESPONSIBILITY: the countersigned exam TABLE and nothing else -- gate it on the operator's
#   sha, return its rows, deal those rows into lanes through armbsig_manifest.assign (the ONE
#   writer of slot identity), and name each lane's SIGMA-SCOPED directory. Split out of
#   armbsig_exam_wave.py 2026-08-29 with NO behaviour change, because that file stood at 187
#   code-lines against a DECLARED hard 190 and caps are split, never raised. This file is the
#   only place that decides how many rows a given countersigned table is allowed to have.
# PUBLIC API: Refusal
#             table_rows(table_path) -> list[dict]
#             lanes(table_path, width) -> list[dict]
#             lane_dir(out_dir, lane) -> str
#             expected_by_sigma(table_path) -> {sigma_key: n_rows}
#   T5 (2026-08-30): a countersigned table carrying any sigma outside armbsig_grids.ALLOWED_SIGMAS is
#   REFUSED here -- the signature admits the table, the venue rule still refuses it.
# SIDE-EFFECTS: reads the table and the prereg carrying its countersignature. Writes nothing,
#   spawns nothing, draws no seed, touches no ledger, kills nothing.
# SIZE: soft 60 / hard 90 code-lines; function cap 40.
# COVERAGE: armbsig_exam_table_selftest.py (EXECUTED, every guard driven to its refusal) and the
#   A11 caps limb of armbsig_selfcheck.py. armbsig_exam_wave_selftest.py continues to exercise
#   these three functions through the wave module's re-exports, unchanged.
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "SEED_CELL_TABLE_ARMBSIG.json")
N_ROWS = 84                      # the fall-back for a signed table of an unlisted shape
# A countersignature ADMITS a table; every consumer still pins its SHAPE. The 1W grid widened
# B25-W0 from 6 legs to 12 on the operator's 2026-08-29 20:15Z countersignature, so its table
# carries 90 rows -- and this launcher refused the first candidate 1W wave at spawn ("table has
# 90 rows, expected 84", C34) because 84 was a module constant. The expected count is therefore
# keyed on THE VERY SHA the countersignature gate returns, never on the file's own claim about
# itself: an UNSIGNED table of any size cannot reach this map, and a signed table of an unlisted
# shape falls back to 84 and is refused. Fail-closed in both directions.
ROWS_BY_SHA = {
    "968fc0727886c63da572ef91a8abb6096cfa09a2df3089b0f892a0acb31fc96a": 84,  # grid 1, the default
    "ebb6531275bded36d3f33d21d07a76d215eb15a85a59b7361843a0d7ff050531": 84,  # grid 3, sigma 0.35
    "bd61150e36f12dd41a110d07dc19f593887bc1c1913879e3509ccdf5fe277c93": 90,  # grid 1W, B25-W0 x12
    "b8ed7b5fa13a643825f1d5346fef6069b70813f2ab2f12b24e9f023cf5f4031b": 90,  # grid B25R, sigma 0.25 only (T5)
    "c3426387e5a60834baaf18f9ddd74049878e553284f3d12ab1b826c3b10ab658": 90,  # grid 6 = G6, sigma 0.25 only, 90 new labels
    "ce9ac35737ba02f2474aafdfb335d83023d0b4c45e5ee99da744753ca7212a2c": 90,  # grid 7 = G7, sigma 0.25 only, 90 new labels (H2 option c)
}


class Refusal(Exception):
    """Fail-closed. Every guard here exists because its absence was measured once."""


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _grids():
    return _load(os.path.join(HERE, "armbsig_grids.py"), "armbsig_grids")


def table_rows(table_path=TABLE):
    """Gate the table on the operator's countersigned sha, then return its rows. The expected
    row count is keyed on THAT sha -- the signature is what says how big this table may be."""
    man = _load(os.path.join(HERE, "armbsig_manifest.py"), "armbsig_manifest")
    sha = man.assert_countersigned(table_path)    # refuses an unsigned / drifted table
    rows = json.load(open(table_path))["legs"]
    why = _grids().sigma_refusal(rows, "countersigned table")     # T5: a purged sigma never flies
    if why:
        raise Refusal(why)
    want = ROWS_BY_SHA.get(sha, N_ROWS)
    if len(rows) != want:
        raise Refusal("table has %d rows, expected %d for countersigned sha %s"
                      % (len(rows), want, sha[:12]))
    return rows


def lanes(table_path=TABLE, width=24):
    """Deal via armbsig_manifest.assign -- the ONE writer of slot identity (depths 3-4)."""
    man = _load(os.path.join(HERE, "armbsig_manifest.py"), "armbsig_manifest")
    return man.assign(table_rows(table_path), width)


def lane_dir(out_dir, lane):
    """SIGMA-SCOPED: the same 30 seeds fly at BOTH sigmas, so without the sigma key the 84 rows
    collapse onto 42 names and half the grid silently skips with every gate green."""
    return os.path.join(out_dir, str(lane["cell"]).split("-")[0],
                        "%s_%s_%s" % (lane["label"], lane["seed"], lane["traj"]))


def expected_by_sigma(table_path=TABLE):
    """How many rows each SIGMA half of this table expects. Replaces N_ROWS // 2, which is a
    half only while the halves are equal: 1W's are 42 and 48, and grid 3 has one half of 84."""
    out = {}
    for r in table_rows(table_path):
        k = str(r["cell"]).split("-")[0]
        out[k] = out.get(k, 0) + 1
    return out
