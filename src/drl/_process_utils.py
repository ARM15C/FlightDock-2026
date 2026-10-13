"""Shared process-management utilities for the DRL pipeline.

Provides the SIGKILL-on-parent-death preexec function used by both
``env_formation_sitl.py`` and ``sitl_pool.py`` when launching SITL
and leader-broadcaster subprocesses.
"""

from __future__ import annotations

import ctypes
import os
import signal

# Cache libc handle at module scope — NOT inside preexec_fn.
# ctypes.CDLL(None) uses dlopen(NULL), returning all loaded symbols.
# Works on glibc, musl, macOS, and BSD.
_LIBC = ctypes.CDLL(None, use_errno=True)
_PR_SET_PDEATHSIG = 1


def pdeathsig_preexec():
    """preexec_fn: set auto-SIGKILL on parent death.

    Runs in the child after fork(), before exec(). Only async-signal-safe
    operations are used. os.setsid() is handled by start_new_session=True.
    """
    result = _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if result != 0:
        err = ctypes.get_errno()
        msg = f"[Warning] prctl(PR_SET_PDEATHSIG) failed: errno={err}\n"
        os.write(2, msg.encode())
    # Race check: if parent already died between fork() and prctl(),
    # child is reparented to init (ppid=1). Self-terminate immediately.
    if os.getppid() == 1:
        os.write(2, b"[Warning] Parent already dead at fork -- self-terminating\n")
        os._exit(1)
