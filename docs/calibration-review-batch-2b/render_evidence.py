"""Render before/after evidence for the Batch 2B E1/NE1 closing-vertex repairs.

Run from the repository root:
    .venv\\Scripts\\python.exe docs\\calibration-review-batch-2b\\render_evidence.py

The ORIGINAL polygons are embedded below so the evidence can be regenerated
after the configuration files have been repaired.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

ORIGINAL = {
    "E1": [[2, 909], [483, 712], [813, 498], [892, 411], [862, 352], [789, 314],
           [828, 302], [926, 339], [1041, 392], [1089, 434], [1074, 588],
           [987, 752], [854, 878], [627, 1029], [4, 910], [3, 908]],
    "NE1": [[652, 267], [1818, 994], [1332, 1034], [640, 273], [654, 267]],
}
REMOVED_INDEX = {"E1": 15, "NE1": 4}

CASES = [
    # lane id, config, video, frame seconds, close-up centre, close-up half size
    ("E1", "config/config_east.json", "videos/cam03_east.avi", 60.0, (4, 909), 14),
    ("NE1", "config/config_northeast.json", "videos/cam45_northeast.avi", 60.0, (650, 269), 14),
]

COL_ORIG = (0, 0, 255)
COL_NEW = (0, 220, 0)
COL_OTHER = (255, 200, 0)
COL_REMOVED = (255, 0, 255)


def read_frame(video, seconds):
    cap = cv2.VideoCapture(str(REPO / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read frame from {video}")
    return frame


def draw_poly(img, pts, color, labels=True, scale=1.0, offset=(0, 0), thickness=2):
    arr = np.array([[(x - offset[0]) * scale, (y - offset[1]) * scale] for x, y in pts], np.int32)
    cv2.polylines(img, [arr], True, color, thickness, cv2.LINE_AA)
    if labels:
        for i, (x, y) in enumerate(arr):
            cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 if scale == 1 else 0.7, color, 2, cv2.LINE_AA)


def caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    report = {}
    for lane_id, cfg_path, video, sec, centre, half in CASES:
        cfg = json.loads((REPO / cfg_path).read_text(encoding="utf-8"))
        lanes = cfg["lane_metrics"]["lanes"]
        original = ORIGINAL[lane_id]
        repaired = lanes[lane_id]["polygon"]
        others = {k: v["polygon"] for k, v in lanes.items() if k != lane_id}
        frame = read_frame(video, sec)

        for tag, pts, col in (("before", original, COL_ORIG), ("after", repaired, COL_NEW)):
            img = frame.copy()
            for oid, opts in others.items():
                draw_poly(img, opts, COL_OTHER, labels=False)
                cx, cy = np.mean(opts, axis=0).astype(int)
                cv2.putText(img, oid, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_OTHER, 2)
            draw_poly(img, pts, col)
            caption(img, f"{lane_id} {tag} | {video} @ {sec:.0f}s | valid={Polygon(pts).is_valid}")
            cv2.imwrite(str(OUT / f"{lane_id}_{tag}_full.png"), img)

        # Close-up around the closing vertices, upscaled for legibility.
        scale = 24
        x0, y0 = centre[0] - half, centre[1] - half
        crop = frame[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half]
        pad = np.zeros((2 * half, 2 * half, 3), np.uint8)
        pad[max(-y0, 0):max(-y0, 0) + crop.shape[0], max(-x0, 0):max(-x0, 0) + crop.shape[1]] = crop
        tiles = []
        for tag, pts, col in (("before", original, COL_ORIG), ("after", repaired, COL_NEW)):
            tile = cv2.resize(pad, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            draw_poly(tile, pts, col, scale=scale, offset=(x0, y0), thickness=3)
            if tag == "before":
                rx, ry = original[REMOVED_INDEX[lane_id]]
                cv2.circle(tile, (int((rx - x0) * scale), int((ry - y0) * scale)), 14, COL_REMOVED, 3)
            caption(tile, f"{lane_id} {tag} close-up ({2*half}x{2*half}px at x{scale})")
            tiles.append(tile)
        cv2.imwrite(str(OUT / f"{lane_id}_closeup_before_after.png"), np.hstack(tiles))

        report[lane_id] = {
            "config": cfg_path,
            "video": video,
            "frame_seconds": sec,
            "removed_vertex": {"index": REMOVED_INDEX[lane_id], "point": original[REMOVED_INDEX[lane_id]]},
            "before": {"vertices": len(original), "area": Polygon(original).area,
                       "validity": explain_validity(Polygon(original))},
            "after": {"vertices": len(repaired), "area": Polygon(repaired).area,
                      "validity": explain_validity(Polygon(repaired))},
            "overlaps_after": {oid: round(Polygon(repaired).intersection(Polygon(o)).area, 2)
                               for oid, o in others.items()
                               if Polygon(repaired).intersection(Polygon(o)).area > 0},
        }
    (OUT / "repair_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
