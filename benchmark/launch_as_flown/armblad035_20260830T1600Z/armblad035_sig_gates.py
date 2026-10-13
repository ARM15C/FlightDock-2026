"""armblad035_sig_gates -- the operator's four ARMBLAD-035 signatures, and their consumers.
RESPONSIBILITY
  One owner for every signature this campaign reads. COPY, not import, of
  baseline025_sig_gates.py: the parent's regexes and paths are module-level constants bound to
  BASELINE-025, so they cannot be composed for a different campaign without parameterising a
  file that has already flown. L2 -- the two must diverge without either editing the other.
  EVERY SIGNATURE HAS A CONSUMER, AND EVERY CONSUMER IS NAMED:
  * A ARMBLAD035-<yyyymmdd> the prereg is signed at all -> assert_signed_prereg
  * B ARMBLAD035-SEEDS- the table's FILE-BYTES sha -> assert_countersigned
  * C ARMBLAD035-SIGMABAND- the realized-sigma band rule -> assert_sigma_band_ruled
  * D ARMBLAD035-S0- OPTIONAL; the still-air rung -> s0_rung_signed
  A signature no gate consumes cannot refuse anything (T-322 / ARMBTOP-GATE-674).

  *** C IS NEW AND HAS NO PARENT. *** The band rule is `2.3877 * 0.04015 * (sigma/0.30)`, which
  is 0.0799 at the parent's 0.25 and 0.1118 at this campaign's 0.35. At 0.35 that is LOOSER
  than A-10's signed 0.10, and two source comments (armbsig_driver.sh:382,
  wave_sigma_hook.py:27) assert the opposite -- true for grid 1, never updated when grid 2 flew
  above 0.30. The SCALED 0.1118 for symmetry with the parent, which applies
  A-10's RULE rather than A-10's NUMBER. This gate makes that ruling a launch precondition
  instead of a comment, and refuses a document that does not carry it.

  WIDTH: the parent read it from its own S18 clause. ARMBLAD-035 carries no width block, so
  width is NOT a signature here -- it is the wave launcher's argument, defaulting to 24, and
  the manifest records what was used. Declared so the missing block is a decision, not a gap.

  ALL FOUR PATTERNS ARE COLUMN-0 ANCHORED (re.M + '^'). The prereg carries INDENTED templates
  of the same text by design; a pattern without '^' matches a template and passes on an
  UNSIGNED document -- which is the failure this anchoring exists to prevent.
PUBLIC API
  PREREG. TABLE. MANIFEST. die(rc, msg). file_sha(path)
  . assert_signed_prereg(doc) -> str # prereg sha; refuses rc 1 unsigned
  . assert_countersigned(table_path, doc) -> str # the countersigned sha; refuses on drift
  . assert_sigma_band_ruled(doc) -> dict # {"mode": "SCALED"|"CLAMPED", "band": float}
  . s0_rung_signed(doc) -> bool # never refuses; presence only
SIDE-EFFECTS
  Reads the prereg and the table READ-ONLY. Writes nothing, spawns nothing, kills nothing.
CAPS: file soft 85 / hard 105 code-lines; function cap 25. Chosen at file creation from the
  measured size; no existing cap moved. DISCLOSED: the parent defines die/file_sha/_doc
  TWICE -- a leftover from its own cap-split, harmless there because the second definition
  simply wins. The duplicate is NOT propagated here: carrying dead code forward because the
  source had it is how a lineage accumulates debt. The parent is not edited (L2).
COVERAGE: armblad035_sig_limbs.py -- every gate driven BOTH directions.
"""
import hashlib
import os
import re
import sys

FD = "/home/ubuntu/FlightDock"
HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "ARMBLAD035_PREREG.md")
TABLE = os.path.join(HERE, "SEED_CELL_TABLE_ARMBLAD035.json")
MANIFEST = os.path.join(HERE, "SLOT_MANIFEST_ARMBLAD035.json")


def die(rc, msg):
    print("ARMBLAD035 SIG GATE REFUSED (%d): %s" % (rc, msg), file=sys.stderr)
    sys.exit(rc)


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _doc(doc):
    if not os.path.isfile(doc):
        die(2, "prereg absent at %s" % doc)
    return open(doc).read()


SIG_PREREG_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD035-\d{8}\b", re.M)
SIG_SEEDS_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD035-SEEDS-[^\n]*?([0-9a-f]{64})", re.M)
SIG_BAND_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-[^\n]*?\b(SCALED|CLAMPED)\b",
                         re.M)
SIG_S0_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD035-S0-", re.M)
#: A-10's rule, not A-10's number. 0.30 is where sd was MEASURED (R3: 210 legs, 630 axis
#: draws, sd 0.04015), so it is the reference the ratio is taken against -- never the
#: campaign's own sigma. band(0.30) = 0.0959 -> the signed 0.10.
K_ANY_AXIS, SD_UNIT, SD_REF_SIGMA = 2.3877, 0.04015, 0.30
CLAMP = 0.10


def scaled_band(sigma):
    return round(K_ANY_AXIS * SD_UNIT * (float(sigma) / SD_REF_SIGMA), 4)


def assert_signed_prereg(doc=PREREG):
    """S15. Without this the campaign has no authority at all."""
    if not SIG_PREREG_RE.search(_doc(doc)):
        die(1, "prereg NOT signed: no column-0 'OPERATOR-SIGNATURE: ARMBLAD035-<yyyymmdd>' "
               "line in %s. The indented template does not count." % os.path.basename(doc))
    return file_sha(doc)


def assert_sigma_band_ruled(doc=PREREG, sigma=0.35):
    """Block C. NEW in ARMBLAD-035 and it has no parent gate. There is no defensible default:
    SCALED applies A-10's RULE (0.1118 at sigma 0.35, matching the parent's +/-32% relative);
    CLAMPED applies A-10's NUMBER (0.10). Refuses a document that rules neither."""
    m = SIG_BAND_RE.search(_doc(doc))
    if not m:
        die(1, "no column-0 'OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-' clause naming SCALED "
               "or CLAMPED in %s. At sigma 0.35 the scaled band (0.1118) is LOOSER than A-10's "
               "signed 0.10 and the source comments claim the opposite, so this campaign may "
               "not pick one silently." % os.path.basename(doc))
    mode = m.group(1)
    return {"mode": mode, "band": scaled_band(sigma) if mode == "SCALED" else CLAMP,
            "doc_sha256": file_sha(doc)}


def assert_countersigned(table_path=TABLE, doc=PREREG):
    """S16. The countersignature binds a SHA, never a filename: a table regenerated or swapped
    after signature no longer matches and everything downstream refuses."""
    m = SIG_SEEDS_RE.search(_doc(doc))
    if not m:
        die(1, "seed table sha NOT countersigned: no column-0 ARMBLAD035-SEEDS- clause "
               "carrying a 64-hex sha in %s (prereg block B)" % os.path.basename(doc))
    want, got = m.group(1), file_sha(table_path)
    if want != got:
        die(1, "seed table sha drift: countersigned=%s on-disk=%s" % (want, got))
    return got


def s0_rung_signed(doc=PREREG):
    """S17, OPTIONAL. Reported, never required. Unsigned => the readout renders false/missed
    abort rates NOT MEASURED and the grid flies unchanged (prereg S11)."""
    return bool(SIG_S0_RE.search(_doc(doc)))


