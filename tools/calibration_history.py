"""
List and restore calibration revisions saved by the segmentor.

    .venv\\Scripts\\python.exe -m tools.calibration_history list config/config_north.json
    .venv\\Scripts\\python.exe -m tools.calibration_history restore config/config_north.json 20261004T120000123456Z

A restore validates the archived revision (lanes and gates) first, archives the
current file as a revision of its own, then replaces the config atomically.
Restart the runner afterwards; it does not reload calibration while running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segmentor import calibration_history_dir, validate_gates  # noqa: E402
from trt_pipeline.lane_validation import validate_config_file  # noqa: E402


def list_revisions(config_path: str) -> list[dict]:
    hist = calibration_history_dir(config_path)
    out = []
    if not os.path.isdir(hist):
        return out
    current = None
    try:
        with open(config_path, encoding="utf-8") as f:
            current = json.load(f).get("calibration", {}).get("revision")
    except (OSError, ValueError):
        pass
    for name in sorted(os.listdir(hist)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(hist, name)
        try:
            with open(path, encoding="utf-8") as f:
                calib = json.load(f).get("calibration", {})
        except (OSError, ValueError):
            calib = {}
        rev = name[:-5]
        out.append({"revision": rev, "saved_by": calib.get("saved_by"), "saved_at": calib.get("saved_at"),
                    "current": rev == current, "path": path})
    return out


def restore(config_path: str, revision: str) -> int:
    src = os.path.join(calibration_history_dir(config_path), f"{revision}.json")
    if not os.path.exists(src):
        print(f"[ERROR] revision {revision} not found in {calibration_history_dir(config_path)}", file=sys.stderr)
        return 1
    report = validate_config_file(src, context=f"revision {revision}")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    gate_errors = validate_gates(data.get("gates", []))
    if report.errors or gate_errors:
        for e in report.errors:
            print(f"[ERROR] {e.lane_id or 'config'}: {e.reason}", file=sys.stderr)
        for e in gate_errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        print("[ERROR] archived revision is invalid; not restored", file=sys.stderr)
        return 1

    hist = calibration_history_dir(config_path)
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cur_rev = (json.load(f).get("calibration") or {}).get("revision") or "unversioned"
        keep = os.path.join(hist, f"{cur_rev}.json")
        if not os.path.exists(keep):
            shutil.copy2(config_path, keep)

    d = os.path.dirname(os.path.abspath(config_path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".restore-", suffix=".json")
    os.close(fd)
    shutil.copy2(src, tmp)
    os.replace(tmp, config_path)
    print(f"[OK] restored {config_path} to revision {revision}. Restart the runner to apply it.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("config")
    rs = sub.add_parser("restore")
    rs.add_argument("config")
    rs.add_argument("revision")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        revs = list_revisions(args.config)
        if not revs:
            print("no archived revisions")
        for r in revs:
            print(f"{'*' if r['current'] else ' '} {r['revision']}  {r['saved_at'] or '-'}  {r['saved_by'] or '-'}")
        return 0
    return restore(args.config, args.revision)


if __name__ == "__main__":
    sys.exit(main())
