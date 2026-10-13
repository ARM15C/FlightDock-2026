"""Per-step transition-stream dump for training-path determinism A/B runs.

=============================================================================
RESPONSIBILITY: dispatch #305 determinism instrument. When registered (train.py
  gates registration on the FD_STEP_STREAM_DUMP env var, default-OFF), writes
  one JSONL row per collected env step — normalized obs, raw (pre-VecNormalize)
  obs, policy action, buffer action, reward, done, scalar info fields — plus a
  meta row at start and a final replay-buffer pickle at training end. Two runs
  with identical config/seed can then be compared field-by-field to decide
  training-path (SIGSTOP-gated env) replay determinism.
PUBLIC API: StepStreamDumpCallback(out_dir) — an SB3 BaseCallback.
SIDE-EFFECTS: writes <out_dir>/step_stream.jsonl and
  <out_dir>/replay_buffer_final.pkl ONLY. Reads callback locals; never mutates
  model/env state; no network/SITL interaction. OFF path (env var unset):
  train.py never imports this module — byte-identical training behavior.
CAPS: file soft 110 / hard 140 LoC; per-function <= 40 LoC.
COVERAGE: py_compile + tests/unit/test_step_stream_dump.py (row count == steps,
  meta first, floats round-trip exactly); a run with 0 rows is visible as a
  missing/empty stream file, never a silent pass.
=============================================================================
"""
import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_SCALARS = (int, float, bool, str, np.integer, np.floating, np.bool_)


def _tolist(x):
    """Exact float64 widening -> JSON round-trips bit-exactly via repr."""
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


class StepStreamDumpCallback(BaseCallback):
    def __init__(self, out_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._out_dir = out_dir
        self._path = os.path.join(out_dir, "step_stream.jsonl")
        self._fh = None
        self._rows = 0

    def _orig_obs(self):
        """Unnormalized last obs from the VecNormalize layer, if present."""
        get = getattr(self.training_env, "get_original_obs", None)
        try:
            return _tolist(get()) if get is not None else None
        except Exception:
            return None

    def _on_training_start(self) -> None:
        os.makedirs(self._out_dir, exist_ok=True)
        self._fh = open(self._path, "w", buffering=1)
        meta = {
            "row": "meta",
            "n_envs": int(getattr(self.training_env, "num_envs", 1)),
            "first_obs": _tolist(getattr(self.model, "_last_obs", None)),
            "first_obs_raw": self._orig_obs(),
        }
        self._fh.write(json.dumps(meta) + "\n")

    def _on_step(self) -> bool:
        loc = self.locals
        row = {"t": int(self.num_timesteps)}
        for src, dst in (("new_obs", "obs"), ("actions", "action"),
                         ("buffer_actions", "buffer_action"),
                         ("rewards", "reward")):
            v = loc.get(src)
            if v is not None:
                row[dst] = _tolist(v)
        dones = loc.get("dones")
        if dones is not None:
            row["done"] = [bool(b) for b in np.asarray(dones).reshape(-1)]
        row["obs_raw"] = self._orig_obs()
        infos = loc.get("infos")
        if infos:
            row["info"] = {k: (v.item() if isinstance(v, np.generic) else v)
                           for k, v in infos[0].items()
                           if isinstance(v, _SCALARS)}
        self._fh.write(json.dumps(row) + "\n")
        self._rows += 1
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps({"row": "end", "steps": self._rows}) + "\n")
            self._fh.close()
            self._fh = None
        save = getattr(self.model, "save_replay_buffer", None)
        if save is not None:
            try:
                save(os.path.join(self._out_dir, "replay_buffer_final"))
            except Exception as exc:  # never fail the run on a dump error
                print(f"[StepStreamDump] replay-buffer save failed: {exc}",
                      flush=True)
