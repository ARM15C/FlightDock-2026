"""FlightDock DRL Pipeline — Residual MRRL with TD3 (primary), SAC (fallback), or PPO (defense).

Path setup: baseline modules (geometry, formation_control_laws, etc.) live
in the tests/ directory. Adding tests/ to sys.path here lets all DRL
modules import them transparently.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
