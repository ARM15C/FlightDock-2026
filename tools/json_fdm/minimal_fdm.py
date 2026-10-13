#!/usr/bin/env python3
"""minimal_fdm — minimal external JSON-SITL FDM scaffold for JSON-lockstep Phase-1.

RESPONSIBILITY
    Speak ArduPilot's JSON SITL physics-backend protocol over UDP so a Phase-1
    lockstep HANDSHAKE can be validated WITHOUT any aircraft dynamics. The scaffold
      - listens on UDP :9002 for ArduPilot servo packets,
      - validates the packet magic (18458 = 16-channel, 29569 = 32-channel),
      - parses {magic, frame_rate, frame_count, pwm[N]},
      - maintains a DETERMINISTIC FDM timestamp advanced by dt = 1/frame_rate per
        accepted frame (NEVER wall-clock; lockstep requires monotone physics time),
      - replies with the mandatory SIM_JSON state fields, with no_time_sync ABSENT
        so lockstep stays ON.
    This is protocol/clock scaffolding ONLY. It emits a STATIC level placeholder
    state (position = initial NED; zero velocity / gyro / accel_body / attitude)
    matching the SIM_JSON readme example. It makes NO aircraft-fidelity claim,
    performs NO flight and NO docking, and has NO leader/controller/truth coupling.
    Field names verified against ardupilot_formation/libraries/SITL/SIM_JSON.{h,cpp}
    (HEAD 4687de5b, JSON Phase-1).

PUBLIC API
    parse_servo_packet(data: bytes) -> dict | None
        Validate magic + decode; return {magic, n, frame_rate, frame_count, pwm}
        or None if malformed / unknown magic / truncated.
    next_timestamp(prev_s: float, frame_rate: int) -> float
        Pure deterministic clock step: prev_s + 1/frame_rate (rate>0) else prev_s.
    build_fdm_response(timestamp_s, ned=(0,0,0)) -> bytes
        Newline-terminated JSON line carrying the mandatory SIM_JSON fields.
    MinimalFDM(bind_host, bind_port, initial_ned, max_frames, log_jsonl_path,
               recv_timeout_s)
        .bind() -> int # create+bind socket, return bound port (lets a caller
                         #   learn an ephemeral port before run() receives)
        .run() -> dict # blocking UDP loop; returns a summary dict on exit
    main(argv=None) -> int # CLI entry point

SIDE EFFECTS
    Binds ONE UDP socket on bind_host:bind_port (default 0.0.0.0:9002) and sends
    JSON replies to the servo-packet sender address (recvfrom). Optionally appends
    one JSON line per accepted frame to --log-jsonl. Prints a startup + summary
    line to stdout. SENDS to no other host. NO SITL spawn. NO source write. NO
    imports of training / reward / observation / replay / env / fuser / controller
    / truth. Importing this module and constructing MinimalFDM bind nothing.

CAPS
    file soft 200 / hard 300 LoC; function LoC cap 60; localhost/LAN UDP only.

COVERAGE
    tests/unit/test_json_fdm_packet.py (T1..T10).
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

# --- Protocol constants (ardupilot_formation/libraries/SITL/SIM_JSON.h:43-55) ---
MAGIC_16: int = 18458            # servo_packet_16: magic, frame_rate, frame_count, pwm[16]
MAGIC_32: int = 29569            # servo_packet_32: ...pwm[32]
HEADER_FMT: str = "<HHI"         # uint16 magic, uint16 frame_rate, uint32 frame_count
HEADER_LEN: int = struct.calcsize(HEADER_FMT)        # 8 bytes
PWM_N: Dict[int, int] = {MAGIC_16: 16, MAGIC_32: 32}
PACKET_LEN: Dict[int, int] = {m: HEADER_LEN + n * 2 for m, n in PWM_N.items()}  # 40 / 72
DEFAULT_PORT: int = 9002


def parse_servo_packet(data: bytes) -> Optional[Dict]:
    """Validate magic + decode an ArduPilot JSON servo packet. None if malformed."""
    if not data or len(data) < HEADER_LEN:
        return None
    magic = struct.unpack_from("<H", data, 0)[0]
    n = PWM_N.get(magic)
    if n is None or len(data) < PACKET_LEN[magic]:
        return None  # unknown/bad magic or truncated -> reject
    _, frame_rate, frame_count = struct.unpack_from(HEADER_FMT, data, 0)
    pwm = list(struct.unpack_from("<%dH" % n, data, HEADER_LEN))
    return {"magic": magic, "n": n, "frame_rate": frame_rate,
            "frame_count": frame_count, "pwm": pwm}


def next_timestamp(prev_s: float, frame_rate: int) -> float:
    """Deterministic lockstep clock: advance by dt=1/frame_rate. No wall-clock."""
    if frame_rate and frame_rate > 0:
        return prev_s + 1.0 / float(frame_rate)
    return prev_s


def build_fdm_response(timestamp_s: float,
                       ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bytes:
    """Mandatory SIM_JSON state, newline-terminated. no_time_sync ABSENT (lockstep ON).

    STATIC placeholder (no dynamics): zero gyro/accel_body/attitude/velocity, fixed
    position = initial NED. Key order + accel_body=[0,0,0] mirror the SIM_JSON readme
    example (examples/JSON/readme.md:59); accel_body here is a non-physical placeholder,
    NOT a specific-force/gravity model (Phase-1 is protocol/clock only).
    """
    state = {
        "timestamp": float(timestamp_s),                 # absolute physics seconds (f64)
        "imu": {"gyro": [0.0, 0.0, 0.0],                 # body rates rad/s
                "accel_body": [0.0, 0.0, 0.0]},          # body accel m/s^2 (placeholder)
        "position": [float(ned[0]), float(ned[1]), float(ned[2])],  # NED m
        "attitude": [0.0, 0.0, 0.0],                     # roll,pitch,yaw rad (level)
        "velocity": [0.0, 0.0, 0.0],                     # NED m/s
    }
    return (json.dumps(state, separators=(",", ":")) + "\n").encode("ascii")


class MinimalFDM:
    """UDP JSON-SITL FDM loop: deterministic clock + static placeholder state."""

    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = DEFAULT_PORT,
                 initial_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 max_frames: Optional[int] = None,
                 log_jsonl_path: Optional[str] = None,
                 recv_timeout_s: Optional[float] = None) -> None:
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.initial_ned = tuple(float(x) for x in initial_ned)
        self.max_frames = max_frames
        self.log_jsonl_path = log_jsonl_path
        self.recv_timeout_s = recv_timeout_s
        self.fdm_timestamp_s = 0.0
        self.accepted = 0
        self.rejected = 0
        self.last_frame_count: Optional[int] = None
        self.sock: Optional[socket.socket] = None  # bound only in bind()/run() (T10)

    def bind(self) -> int:
        """Create + bind the UDP socket; return the bound port (supports port 0)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.bind_host, self.bind_port))
        if self.recv_timeout_s:
            s.settimeout(self.recv_timeout_s)
        self.sock = s
        self.bind_port = s.getsockname()[1]
        return self.bind_port

    def _log(self, fh, pkt: Dict, dt: float) -> None:
        if fh is None:
            return
        pwm = pkt["pwm"]
        rec = {"frame_count": pkt["frame_count"], "frame_rate": pkt["frame_rate"],
               "dt": dt, "fdm_timestamp": self.fdm_timestamp_s, "pwm_n": pkt["n"],
               "pwm_min": min(pwm), "pwm_max": max(pwm),
               "pwm_mean": sum(pwm) / len(pwm) if pwm else 0,
               "fields_complete": True,
               "wall_rx_time": time.time()}  # DIAGNOSTIC ONLY (never feeds the clock)
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()

    def run(self) -> Dict:
        """Blocking receive/reply loop. Binds if not already bound. Returns summary."""
        fh = open(self.log_jsonl_path, "w") if self.log_jsonl_path else None
        if self.sock is None:
            self.bind()
        print("minimal_fdm: listening udp://%s:%d max_frames=%s" %
              (self.bind_host, self.bind_port, self.max_frames), flush=True)
        try:
            while self.max_frames is None or self.accepted < self.max_frames:
                try:
                    data, addr = self.sock.recvfrom(2048)
                except socket.timeout:
                    break
                pkt = parse_servo_packet(data)
                if pkt is None:
                    self.rejected += 1
                    continue
                dt = (1.0 / pkt["frame_rate"]) if pkt["frame_rate"] else 0.0
                self.fdm_timestamp_s = next_timestamp(self.fdm_timestamp_s, pkt["frame_rate"])
                self.sock.sendto(build_fdm_response(self.fdm_timestamp_s, self.initial_ned), addr)
                self.accepted += 1
                self.last_frame_count = pkt["frame_count"]
                self._log(fh, pkt, dt)
        finally:
            if self.sock is not None:
                self.sock.close()
            if fh is not None:
                fh.close()
        summary = {"accepted": self.accepted, "rejected": self.rejected,
                   "fdm_timestamp_s": self.fdm_timestamp_s,
                   "last_frame_count": self.last_frame_count}
        print("minimal_fdm: done %s" % json.dumps(summary, separators=(",", ":")), flush=True)
        return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Minimal ArduPilot JSON-SITL FDM scaffold (Phase-1).")
    ap.add_argument("--bind-host", default="0.0.0.0")
    ap.add_argument("--bind-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--initial-ned-n", type=float, default=0.0)
    ap.add_argument("--initial-ned-e", type=float, default=0.0)
    ap.add_argument("--initial-ned-d", type=float, default=0.0)
    ap.add_argument("--recv-timeout-s", type=float, default=None,
                    help="optional socket recv timeout; loop exits on timeout (bounded smoke)")
    args = ap.parse_args(argv)
    MinimalFDM(bind_host=args.bind_host, bind_port=args.bind_port,
               initial_ned=(args.initial_ned_n, args.initial_ned_e, args.initial_ned_d),
               max_frames=args.max_frames, log_jsonl_path=args.log_jsonl,
               recv_timeout_s=args.recv_timeout_s).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
