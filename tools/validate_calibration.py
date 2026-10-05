"""
Offline Calibration Validator CLI.
Validates lane polygons for geometric validity, coordinate correctness,
and reports intra-camera lane overlaps across intersection camera configurations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.lane_validation import validate_config_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline validation tool for traffic lane calibration geometry."
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="One or more JSON configuration file paths to validate. Defaults to the five bundled camera configs.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if args.configs is not None and len(args.configs) > 0:
        config_paths = [os.path.abspath(p) for p in args.configs]
    else:
        # Default bundled configs resolved relative to repo root
        default_names = [
            "config_north.json",
            "config_south.json",
            "config_east.json",
            "config_west.json",
            "config_northeast.json",
        ]
        config_paths = [str(repo_root / "config" / name) for name in default_names]

    total_errors = 0
    total_warnings = 0
    files_with_errors = 0
    all_gate_ids: set = set()

    print("=" * 65)
    print("Lane Calibration Geometry Validation")
    print("=" * 65)

    for cfg_path in config_paths:
        try:
            display_name = os.path.relpath(cfg_path, repo_root)
        except ValueError:
            display_name = cfg_path

        print(f"\nConfiguration: {display_name}")

        report = validate_config_file(cfg_path)

        if report.warnings:
            for warn in report.warnings:
                print(f"  [WARNING] {warn.message}")
                total_warnings += 1

        gate_errors = []
        if report.raw_config is not None:
            from tools.segmentor import validate_gates
            gate_errors = validate_gates(report.raw_config.get("gates", []))
            if not gate_errors:
                gate_ids = [g.get("gate_id") for g in report.raw_config.get("gates", [])]
                dupes = sorted({g for g in gate_ids if g in all_gate_ids})
                gate_errors = [f"gate_id '{g}' is also used by another camera" for g in dupes]
                all_gate_ids.update(gate_ids)

        if report.errors or gate_errors:
            files_with_errors += 1
            for err in report.errors:
                target = f"Lane '{err.lane_id}'" if err.lane_id else "Config"
                print(f"  [ERROR] {target}: {err.reason}")
                total_errors += 1
            for msg in gate_errors:
                print(f"  [ERROR] Gate: {msg}")
                total_errors += 1
        elif not report.warnings:
            print("  [OK] All configured lanes are valid. No overlaps detected.")

    print("\n" + "=" * 65)
    print(f"Summary: {len(config_paths)} file(s) checked | {total_errors} error(s) | {total_warnings} warning(s)")

    if files_with_errors > 0:
        print("Status: FAILED - Invalid lane geometry detected.")
        return 1

    print("Status: PASSED - All lane configurations are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
