#!/usr/bin/env python3
"""armblad025_sig_gates -- the operator's four ARMBLAD-025 signatures, and their consumers.

RESPONSIBILITY
  One owner for every signature this campaign reads. Split out of armblad025_manifest.py
  2026-08-26 at 144 code-lines against a declared hard 120 -- caps are SPLIT, never raised.

  EVERY SIGNATURE HAS A CONSUMER, AND EVERY CONSUMER IS NAMED:
  * S15 ARMBLAD025-<yyyymmdd> the prereg is signed at all -> assert_signed_prereg
  * S16 ARMBLAD025-SEEDS- the table's FILE-BYTES sha -> assert_countersigned
  * S18 ARMBLAD025-WIDTH-<N> the fan-out width -> signed_width
  * S17 ARMBLAD025-ABORT-WARRANT- OPTIONAL; presence REPORTED -> abort_warrant_signed
  A signature no gate consumes cannot refuse anything (T-322 / ARMBTOP-GATE-674). S17 is read
  here even though it gates no leg, so the manifest RECORDS whether this campaign may compute
  false/missed abort rates and the readout cannot invent that authority for itself.

  ALL FOUR PATTERNS ARE COLUMN-0 ANCHORED (re.M + '^'). The prereg carries INDENTED templates
  of the same text by design; a pattern without '^' matches a template and passes on an
  UNSIGNED document -- which is the failure this anchoring exists to prevent.

PUBLIC API
  PREREG. TABLE. MANIFEST. die(rc, msg). file_sha(path)
  . assert_signed_prereg(doc) -> str # prereg sha; refuses rc 1 unsigned
  . signed_width(doc) -> dict # {"width": int, "doc_sha256": str}
  . assert_countersigned(table_path, doc) -> str # the countersigned sha; refuses on drift
  . abort_warrant_signed(doc) -> bool # never refuses; presence only

SIDE-EFFECTS
  Reads the prereg and the table READ-ONLY. Writes nothing, spawns nothing, kills nothing.

CAPS: file soft 80 / hard 100 code-lines; function cap 25. DISCLOSED: the cap is CHOSEN here
  at file creation from the measured 87 code-lines of the split -- a cap being set, not an
  established cap being raised. No existing cap moved.
COVERAGE: armblad025_manifest_limbs.py -- every gate driven BOTH directions.
"""
import hashlib
import os
import re
import sys

FD = "/home/ubuntu/FlightDock"
HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "ARMBLAD025_PREREG.md")
TABLE = os.path.join(HERE, "SEED_CELL_TABLE_ARMBLAD025.json")
MANIFEST = os.path.join(HERE, "SLOT_MANIFEST_ARMBLAD025.json")


def die(rc, msg):
    print("ARMBLAD025 SIG GATE REFUSED (%d): %s" % (rc, msg), file=sys.stderr)
    sys.exit(rc)


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _doc(doc):
    if not os.path.isfile(doc):
        die(2, "prereg absent at %s" % doc)
    return open(doc).read()


SIG_PREREG_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD025-\d{8}\b", re.M)
SIG_SEEDS_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD025-SEEDS-[^\n]*?([0-9a-f]{64})", re.M)
SIG_WIDTH_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD025-WIDTH-<?(\d+)>?", re.M)
SIG_ABORT_RE = re.compile(r"^OPERATOR-SIGNATURE: ARMBLAD025-ABORT-WARRANT-", re.M)


def die(rc, msg):
    print("ARMBLAD025 MANIFEST REFUSED (%d): %s" % (rc, msg), file=sys.stderr)
    sys.exit(rc)


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _doc(doc):
    if not os.path.isfile(doc):
        die(2, "prereg absent at %s" % doc)
    return open(doc).read()


def assert_signed_prereg(doc=PREREG):
    """S15. Without this the campaign has no authority at all."""
    if not SIG_PREREG_RE.search(_doc(doc)):
        die(1, "prereg NOT signed: no column-0 'OPERATOR-SIGNATURE: ARMBLAD025-<yyyymmdd>' "
               "line in %s. The indented template does not count." % os.path.basename(doc))
    return file_sha(doc)


def signed_width(doc=PREREG):
    """S18. The width is an operator act, not mine to choose."""
    m = SIG_WIDTH_RE.search(_doc(doc))
    if not m:
        die(1, "no column-0 'OPERATOR-SIGNATURE: ARMBLAD025-WIDTH-<N>' clause in %s"
            % os.path.basename(doc))
    return {"width": int(m.group(1)), "doc_sha256": file_sha(doc)}


def assert_countersigned(table_path=TABLE, doc=PREREG):
    """S16. The countersignature binds a SHA, never a filename: a table regenerated or swapped
    after signature no longer matches and everything downstream refuses."""
    m = SIG_SEEDS_RE.search(_doc(doc))
    if not m:
        die(1, "seed table sha NOT countersigned: no column-0 ARMBLAD025-SEEDS- clause "
               "carrying a 64-hex sha in %s (prereg S16)" % os.path.basename(doc))
    want, got = m.group(1), file_sha(table_path)
    if want != got:
        die(1, "seed table sha drift: countersigned=%s on-disk=%s" % (want, got))
    return got


def abort_warrant_signed(doc=PREREG):
    """S17, OPTIONAL. Reported, never required. Unsigned => the readout renders false/missed
    abort rates NOT MEASURED and the grid flies unchanged (prereg S11)."""
    return bool(SIG_ABORT_RE.search(_doc(doc)))


