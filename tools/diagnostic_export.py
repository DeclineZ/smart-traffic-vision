"""
Bundle what is needed to review an incident into one zip:
  * recorded payloads in the time window (from --record-payloads JSONL, incl. rotated files)
  * the calibration files in use (with revision) and their reference images
  * model / engine hashes and engine build metadata
  * the latest health file and the tail of the runner log
  * installed package versions of the running environment

    .venv\\Scripts\\python.exe -m tools.diagnostic_export --payloads logs/payloads.jsonl \\
        --from 2026-10-12T07:30:00+07:00 --to 2026-10-12T07:45:00+07:00 --log logs/vision.log \\
        --health /run/traffic-vision/health.json --out incident-0730.zip

The bundle contains no camera images except calibration reference frames and
no stream credentials (sources are recorded redacted in payload meta/logs).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_report() -> str:
    """Installed packages (from metadata; works without pip) plus torch/TensorRT/GPU versions."""
    import importlib.metadata as md
    import platform

    pkgs = sorted({f"{d.metadata.get('Name')}=={d.version}" for d in md.distributions() if d.metadata.get("Name")},
                  key=str.lower)
    lines = [f"# {platform.platform()} Python {platform.python_version()}"]
    try:
        import torch
        lines.append(f"# torch {torch.__version__} cuda={torch.version.cuda} "
                     f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    except Exception:
        pass
    return "\n".join(lines + pkgs) + "\n"


def payloads_in_window(path: str, t_from: float, t_to: float):
    files = sorted(glob.glob(path + ".*"), reverse=True) + [path]
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                p = rec.get("payload", rec)
                t = ts(p.get("observedAt") or p.get("timestamp"))
                if t_from <= t <= t_to:
                    yield line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payloads", required=True)
    ap.add_argument("--from", dest="t_from", required=True)
    ap.add_argument("--to", dest="t_to", required=True)
    ap.add_argument("--configs", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "config", "config_*.json"))))
    ap.add_argument("--models", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "models", "yolo26s_thai_traffic.*"))))
    ap.add_argument("--log", default=None)
    ap.add_argument("--health", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t_from, t_to = ts(args.t_from), ts(args.t_to)
    manifest = {"window": [args.t_from, args.t_to], "createdAt": datetime.now().astimezone().isoformat(),
                "configs": {}, "models": {}}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        lines = list(payloads_in_window(args.payloads, t_from, t_to))
        z.writestr("payloads.jsonl", "".join(lines))
        manifest["payloadCount"] = len(lines)

        for cfg in args.configs:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
            z.write(cfg, f"config/{os.path.basename(cfg)}")
            calib = data.get("calibration", {})
            manifest["configs"][os.path.basename(cfg)] = {"revision": calib.get("revision"), "sha256": sha256(cfg)}
            ref = calib.get("reference_image")
            ref_path = os.path.join(os.path.dirname(cfg), ref) if ref else None
            if ref_path and os.path.exists(ref_path):
                z.write(ref_path, f"config/{ref}")

        for m in args.models:
            if m.endswith(".json"):
                z.write(m, f"models/{os.path.basename(m)}")
            elif os.path.exists(m):
                manifest["models"][os.path.basename(m)] = {"sha256": sha256(m), "bytes": os.path.getsize(m)}

        if args.health and os.path.exists(args.health):
            z.write(args.health, "health.json")
        if args.log and os.path.exists(args.log):
            with open(args.log, "rb") as f:
                f.seek(max(0, os.path.getsize(args.log) - 2_000_000))
                z.writestr("runner.log.tail", f.read())

        z.writestr("environment.txt", environment_report())
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    print(f"wrote {args.out}: {manifest['payloadCount']} payloads, {len(manifest['configs'])} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
