# Batch 2B: minimal E1/NE1 closing-vertex repairs

Date: 4 October 2026. Status: **awaiting visual review**. Geometry validity alone does not establish correct physical lane coverage.

## Changes

Only two JSON fields changed: `config/config_east.json` → `lanes.E1.polygon` and `config/config_northeast.json` → `lanes.NE1.polygon`. In each case one trailing vertex was removed. All other vertices, polygons, gates and settings are byte-identical; the remaining vertices keep their original order.

| Lane | Removed vertex | Distance to vertex 0 | Vertices | Shoelace area (px²) | Validity | Overlaps after |
|---|---|---|---|---|---|---|
| E1 | index 15 `[3, 908]` | 1.4 px from `[2, 909]` | 16 → 15 | 301,681.0 → 301,682.5 (+1.5) | self-intersection at (3.245, 908.490) → valid | none (E1 is the only east lane) |
| NE1 | index 4 `[654, 267]` | 2.0 px from `[652, 267]` | 5 → 4 | 206,617.0 → 206,623.0 (+6.0) | self-intersection at (652.815, 267.508) → valid | NE2: 33.80 px² (new warning) |

The bad vertex in both lanes was a duplicate click near the start point when the polygon was closed. Its closing edge crossed the first edge, making a "bow-tie" a few pixels wide. Only the stray vertex was removed. No automatic repair (`buffer(0)`, `make_valid`) was used.

Rejected alternatives:

- E1, remove index 14 `[4,910]`: still self-intersecting.
- E1, remove 14 and 15: also valid, but removes area 191.5 px² more than needed.
- NE1, remove index 3 `[640,273]`: still self-intersecting.
- NE1, remove 3 and 4: also valid, but changes area by 6,636 px² and moves the left boundary.

The NE1/NE2 overlap (33.8 px², at the far apex near the vanishing point) was hidden while NE1 was invalid; it is not caused by the repair. With invalid NE1, the `buffer(0)` interpretation gave 33.3 px². Under Batch 1 first-lane-wins arbitration, vehicles in that sliver are assigned to NE1.

## Validation

- `python -m tools.validate_calibration`: 0 errors, 6 warnings (the 5 previously known overlaps plus NE1/NE2 33.8 px²).
- Unit suite: 70 tests passed.
- A JSON comparison against `HEAD` confirmed that everything except the two polygon fields is identical.

## Evidence

Regenerate with `.venv\Scripts\python.exe docs\calibration-review-batch-2b\render_evidence.py` (original polygons are embedded in the script). Frames: 60 s into `videos/cam03_east.avi` and `videos/cam45_northeast.avi` (1920×1080).

- `E1_before_full.png`, `E1_after_full.png`, `NE1_before_full.png`, `NE1_after_full.png`: full-frame overlays with vertex labels; other lanes in light blue.
- `E1_closeup_before_after.png`, `NE1_closeup_before_after.png`: 28×28 px region at ×24. The removed vertex is circled in magenta.
- `repair_summary.json`: machine-readable before/after figures.

## Items for the reviewer (not changed in this batch)

1. **E1 coverage.** The polygon runs from the left kerb to the yellow centre line. It touches the bodies of the vehicles parked along the left kerb near vertices 2–3. Check whether parked vehicles' box centres fall inside E1 over a longer clip; if they do, they will be reported as queued.
2. **NE1 coverage.** The left boundary follows the drainage grating and kerb rather than a lane marking. Confirm that NE1 is a usable traffic lane and not shoulder or parking.
3. **Far apex.** All NE lanes converge near (650–760, 255–275), where vehicles are a few pixels tall. Consider truncating lanes at a validated detection range rather than at the vanishing point. Counts beyond that range are lower bounds at best.
4. **North vs northeast ownership** is still undefined (review report, "Cross-camera coverage").
