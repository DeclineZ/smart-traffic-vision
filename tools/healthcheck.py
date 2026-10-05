"""
Exit status for service supervisors (systemd ExecStartPost/timers, Docker
HEALTHCHECK, Windows scheduled task) from the runner's --health-file.

  0  healthy: file updated within --max-age and every camera usable
  1  degraded: running, but a camera is unusable or controller output is blocked
  2  unhealthy: file missing/stale (process hung or dead) or no usable camera

    .venv\\Scripts\\python.exe -m tools.healthcheck --file /run/traffic-vision/health.json --max-age 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def check(path: str, max_age: float, now: float | None = None, stale_only: bool = False) -> tuple[int, str]:
    now = now if now is not None else time.time()
    try:
        age = now - os.path.getmtime(path)
        with open(path, encoding="utf-8") as f:
            h = json.load(f)
    except (OSError, ValueError) as e:
        return 2, f"health file unreadable: {e}"
    if age > max_age:
        return 2, f"health file is {age:.0f}s old (process hung or stopped)"
    if stale_only:
        return 0, f"process alive (health updated {age:.0f}s ago)"
    valid, total = h.get("validCameras", 0), h.get("totalCameras", 0)
    bad = [f"{c.get('name')}={c.get('status')}" for c in h.get("cameras", []) if c.get("status") not in ("ok", "degraded")]
    if total and valid == 0:
        return 2, "no usable camera: " + ", ".join(bad)
    controller = h.get("controllerDelivery")
    if controller and controller.get("state") != "publishing":
        return 1, f"controller output {controller.get('state')}: " + ", ".join(controller.get("blockers", []))
    if bad:
        return 1, "degraded: " + ", ".join(bad)
    return 0, f"ok: {valid}/{total} cameras, sequence {h.get('sequence')}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--max-age", type=float, default=10.0)
    ap.add_argument("--stale-only", action="store_true",
                    help="only report a missing/stale file (process liveness), ignore camera state")
    args = ap.parse_args(argv)
    code, msg = check(args.file, args.max_age, stale_only=args.stale_only)
    print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main())
