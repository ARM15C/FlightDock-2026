#!/usr/bin/env python3
"""Training completion notification -- called by run_drl_training.sh."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--metrics-dir", type=str, default=None,
                        help="Directory containing metrics.csv (--log-dir from train.py)")
    args = parser.parse_args()

    status = "SUCCESS" if args.exit_code == 0 else f"FAILED (exit {args.exit_code})"

    # Write status JSON
    status_file = Path(args.log_file).parent / f"status_{args.timestamp}.json"
    status_data = {
        "timestamp": args.timestamp,
        "exit_code": args.exit_code,
        "status": status,
        "log_file": args.log_file,
        "completed_at": datetime.now().isoformat(),
    }

    # Try to extract last metrics line
    metrics_csv = None
    if args.metrics_dir:
        metrics_csv = os.path.join(args.metrics_dir, "metrics.csv")
    if metrics_csv is None or not os.path.exists(metrics_csv):
        # Fallback: search common locations
        for candidate in [
            os.path.join(str(Path(args.log_file).parent.parent), "metrics.csv"),
            os.path.join(str(Path(args.log_file).parent), "metrics.csv"),
        ]:
            if os.path.exists(candidate):
                metrics_csv = candidate
                break
    if metrics_csv and os.path.exists(metrics_csv):
        with open(metrics_csv) as f:
            lines = f.readlines()
            if len(lines) > 1:
                status_data["last_metric_line"] = lines[-1].strip()
                status_data["total_episodes"] = len(lines) - 1

    status_file.write_text(json.dumps(status_data, indent=2))
    print(f"[notify] Status: {status}")
    print(f"[notify] Written to {status_file}")

    # Desktop notification (WSL/Linux)
    title = f"FlightDock Training {status}"
    body = f"Exit code: {args.exit_code}\nLog: {args.log_file}"
    try:
        subprocess.run(
            ["notify-send", title, body],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Email notification (optional)
    email = os.environ.get("FLIGHTDOCK_NOTIFY_EMAIL")
    if email:
        try:
            import smtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg["Subject"] = title
            msg["To"] = email
            msg["From"] = f"flightdock@{os.uname().nodename}"
            msg.set_content(
                f"{body}\n\nStatus JSON:\n{json.dumps(status_data, indent=2)}"
            )
            smtp_host = os.environ.get("FLIGHTDOCK_SMTP_HOST", "localhost")
            with smtplib.SMTP(smtp_host) as s:
                s.send_message(msg)
        except Exception as e:
            print(f"[notify] Email failed: {e}")


if __name__ == "__main__":
    main()
