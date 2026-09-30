"""
Shared lane geometry validation module.
Enforces geometric validity, coordinate correctness, and overlap detection
across offline validation CLI, production startup preflight, and calibration saving.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.validation import explain_validity


@dataclass
class LaneValidationError:
    context: str
    lane_id: str
    reason: str


@dataclass
class LaneValidationWarning:
    context: str
    lane_a: str
    lane_b: str
    overlap_area: float
    message: str


@dataclass
class ValidationReport:
    errors: List[LaneValidationError] = field(default_factory=list)
    warnings: List[LaneValidationWarning] = field(default_factory=list)
    valid_polygons: Dict[str, Polygon] = field(default_factory=dict)
    raw_config: Optional[Dict[str, Any]] = None

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_single_polygon(
    poly_candidate: Any,
    lane_id: str = "unknown",
    context: str = "",
) -> Tuple[Optional[Polygon], List[str]]:
    """
    Validates a single candidate lane polygon.

    Checks:
      1. Coordinate structure: iterable of (x, y) coordinate pairs in a list or tuple.
      2. Numeric and finite: x, y must be int or float (not bool), and finite (no NaN, inf).
      3. Vertices: at least three distinct vertices (excluding an optional closing vertex).
      4. Positive area: area must be strictly positive (> 0.0).
      5. Geometry validity & simplicity: valid and simple according to Shapely.

    Does NOT silently repair polygons (no buffer(0), make_valid, vertex sorting, or convex hulls).
    """
    if poly_candidate is None:
        return None, ["Polygon coordinates cannot be None"]

    if not isinstance(poly_candidate, (list, tuple)):
        return None, [
            f"Polygon coordinates must be a list/tuple of coordinate pairs, got {type(poly_candidate).__name__}"
        ]

    if len(poly_candidate) == 0:
        return None, ["Polygon coordinates list is empty"]

    coords = poly_candidate

    pts: List[Tuple[float, float]] = []
    for i, pt in enumerate(coords):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None, [f"Vertex {i} must be a pair of (x, y) coordinates, got {pt!r}"]

        x, y = pt[0], pt[1]
        # Reject booleans (bool is a subclass of int in Python)
        if isinstance(x, bool) or isinstance(y, bool):
            return None, [f"Vertex {i} coordinates must be numeric, got boolean"]

        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return None, [
                f"Vertex {i} coordinates must be numeric, got ({type(x).__name__}, {type(y).__name__})"
            ]

        if not (math.isfinite(x) and math.isfinite(y)):
            return None, [f"Vertex {i} coordinates must be finite (got x={x}, y={y})"]

        pts.append((float(x), float(y)))

    # Check distinct vertices (excluding optional closing point)
    if len(pts) > 1 and pts[0] == pts[-1]:
        open_pts = pts[:-1]
    else:
        open_pts = pts

    distinct_vertices = set(open_pts)
    if len(distinct_vertices) < 3:
        return None, [
            f"Polygon must have at least 3 distinct vertices, found {len(distinct_vertices)}"
        ]

    try:
        poly = Polygon(pts)
    except Exception as e:
        return None, [f"Failed to construct Shapely Polygon: {e}"]

    errors = []
    if poly.area <= 0.0:
        errors.append(f"Polygon area must be positive, got {poly.area}")

    if not poly.is_valid:
        explanation = explain_validity(poly)
        errors.append(f"Invalid polygon geometry: {explanation}")
    elif not poly.is_simple:
        errors.append("Polygon is not simple (contains self-intersections or self-tangencies)")

    if errors:
        return None, errors

    return poly, []


def validate_camera_lanes(
    lanes: Any,
    context: str = "",
) -> ValidationReport:
    """
    Validates all lane polygons within one camera configuration.
    Enforces that 'lanes' is an object and each lane entry is an object containing 'polygon'.
    Detects intra-camera positive-area overlaps.
    Does NOT compare polygons across different cameras.
    An empty lanes object is allowed (e.g. gate-only cameras).
    """
    if not isinstance(lanes, dict):
        return ValidationReport(
            errors=[
                LaneValidationError(
                    context=context,
                    lane_id="",
                    reason=f"'lanes' collection must be an object, got {type(lanes).__name__}",
                )
            ]
        )

    report = ValidationReport()

    if len(lanes) == 0:
        return report

    for lane_id, l_info in lanes.items():
        if not isinstance(l_info, dict):
            report.errors.append(
                LaneValidationError(
                    context=context,
                    lane_id=str(lane_id),
                    reason=f"Lane entry must be an object, got {type(l_info).__name__}",
                )
            )
            continue

        if "polygon" not in l_info:
            report.errors.append(
                LaneValidationError(
                    context=context,
                    lane_id=str(lane_id),
                    reason="Missing 'polygon' key in lane entry",
                )
            )
            continue

        poly_candidate = l_info["polygon"]
        poly, errs = validate_single_polygon(poly_candidate, lane_id=str(lane_id), context=context)
        if errs:
            for err in errs:
                report.errors.append(
                    LaneValidationError(
                        context=context,
                        lane_id=str(lane_id),
                        reason=err,
                    )
                )
        else:
            assert poly is not None
            report.valid_polygons[str(lane_id)] = poly

    # Check pairwise overlaps only among individually valid polygons within this single camera
    lane_ids = list(report.valid_polygons.keys())
    for i in range(len(lane_ids)):
        for j in range(i + 1, len(lane_ids)):
            id_a = lane_ids[i]
            id_b = lane_ids[j]
            poly_a = report.valid_polygons[id_a]
            poly_b = report.valid_polygons[id_b]

            try:
                if poly_a.intersects(poly_b):
                    inter = poly_a.intersection(poly_b)
                    if inter.area > 0.0:
                        area_val = float(inter.area)
                        if area_val >= 0.01:
                            overlap_display = f"{round(area_val, 2)}"
                        else:
                            overlap_display = f"{area_val:.6g}"
                        report.warnings.append(
                            LaneValidationWarning(
                                context=context,
                                lane_a=id_a,
                                lane_b=id_b,
                                overlap_area=area_val,
                                message=(
                                    f"Lanes '{id_a}' and '{id_b}' overlap by {overlap_display} sq px "
                                    "(requires calibration review)"
                                ),
                            )
                        )
            except Exception as e:
                report.warnings.append(
                    LaneValidationWarning(
                        context=context,
                        lane_a=id_a,
                        lane_b=id_b,
                        overlap_area=0.0,
                        message=(
                            f"Overlap calculation could not complete between lanes '{id_a}' and '{id_b}': {e}"
                        ),
                    )
                )

    return report


def validate_config_file(config_path: str, context: Optional[str] = None) -> ValidationReport:
    """
    Loads and validates lane geometry from a JSON configuration file.
    Validates container types: JSON root must be an object, lane_metrics (if present)
    must be an object, and lanes (if present) must be an object.
    """
    if context is None:
        context = os.path.basename(config_path)

    if not os.path.exists(config_path):
        return ValidationReport(
            errors=[LaneValidationError(context=context, lane_id="", reason=f"File not found: {config_path}")]
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ValidationReport(
            errors=[LaneValidationError(context=context, lane_id="", reason=f"Invalid JSON: {e}")]
        )

    if not isinstance(data, dict):
        return ValidationReport(
            errors=[
                LaneValidationError(
                    context=context,
                    lane_id="",
                    reason=f"JSON root must be an object, got {type(data).__name__}",
                )
            ]
        )

    if "lane_metrics" in data:
        lm = data["lane_metrics"]
        if not isinstance(lm, dict):
            return ValidationReport(
                errors=[
                    LaneValidationError(
                        context=context,
                        lane_id="",
                        reason=f"'lane_metrics' must be an object, got {type(lm).__name__}",
                    )
                ]
            )
        if "lanes" in lm:
            lanes = lm["lanes"]
            if not isinstance(lanes, dict):
                return ValidationReport(
                    errors=[
                        LaneValidationError(
                            context=context,
                            lane_id="",
                            reason=f"'lanes' collection must be an object, got {type(lanes).__name__}",
                        )
                    ]
                )
        else:
            lanes = {}
    else:
        lanes = {}

    report = validate_camera_lanes(lanes, context=context)
    report.raw_config = data
    return report
