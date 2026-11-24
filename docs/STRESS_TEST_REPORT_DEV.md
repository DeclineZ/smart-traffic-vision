# Stress Test & Audit Report (`dev` @ `bd1741c`)

- **Branch:** `stress-test` (branched from `origin/dev` commit `bd1741cfd4faed26f165ebe793c9fb683995af37`)
- **Date:** 2026-11-24
- **Scope:** Offline unit, fuzz, multi-camera config, and logic-probe stress testing against `dev` after PR #7 (`corridor counting, outgoing road`).
  - *Note:* Stress tests ran with mocked detections (`FakeModel`) and simulated packets on macOS (`darwin`), not live RTSP streams or full YOLO/TensorRT inference against `.avi` files.
  - *No application code was modified on this branch.*

---

## 1. Test Execution Summary

| Check | Command | Result | Notes |
|---|---|---|---|
| Calibration geometry check | `.venv/bin/python tools/validate_calibration.py` | **PASSED** | 6 configs checked (`north`, `south`, `east`, `west`, `northeast`, `north_upstream`); 0 errors, 5 overlap warnings (`south`, `west`, `northeast`) |
| Existing `dev` unit suite | `.venv/bin/python -m unittest discover -s tests -v` | **232 / 232 PASSED** (`8.499s`) | Baseline suite passes |
| New stress & bug-probe suite | `.venv/bin/python -m unittest tests.test_stress_dev -v` | **13 ran: 4 passed, 9 failed** (`1.887s`) | Fuzz/soak & 6-cam startup passed; 9 bug probes failed as expected, reproducing defects in `dev` |

### Passing Stress & Platform Checks (`tests/test_stress_dev.py`)
- `test_fuzz_corridor_estimator_10k_steps`: 10,000 randomized steps of `CorridorQueueEstimator` (random vehicles, `dt` spikes, `dt=0`, borrowing, spillback toggles, empty resets) — **passed** (invariants held).
- `test_fuzz_byetrack_and_motion_5k_frames`: 5,000 frames of `ByteTrack` + `MotionStateClassifier` with rapid track churn — **passed** (bounded memory, valid states).
- `test_six_camera_real_configs_startup_and_step`: Loaded all 6 real configs (`north`, `south`, `east`, `west`, `northeast`, `north_upstream`) with `MainControllerAdapter` (`config/controller_main.json`) and stepped 20 frames — **passed**.
- `test_09_stream_cameras_import_on_non_windows`: Confirmed `stream_cameras.py` raises `ModuleNotFoundError: No module named 'msvcrt'` on non-Windows hosts — **passed** (platform defect confirmed).

---

## 2. Confirmed Bugs & Loopholes Found on `dev`

### High Severity

#### 1. Thai-model `truck` (`class_id = 3`) misclassified as `motorcycle` at gates
- **Location:** `trt_pipeline/gates.py` (`GateFlowManager.update_tracks`, L403–L408) -> `trt_pipeline/corridor.py` (`is_motorcycle`, L42–L47)
- **Cause:** `update_tracks` forwards raw integer `cls_id` to `corridor_estimator.record_gate_crossing`. `is_motorcycle(3)` returns `True` (assuming COCO class `3 = motorcycle`), whereas the custom Thai traffic model maps `0: car, 1: motorcycle, 2: bus, 3: truck, 4: three_wheeler`.
- **Impact:** Every truck crossing an ingress/stopline gate skips Virtual Cart token enqueueing and is accounted under `moto_pools` instead of `car_pools` / lane flow.
- **Reproduced by:** `test_01_thai_model_truck_class_id_3_misclassified_as_motorcycle_at_gate` (`AssertionError: 0 != 1`).

#### 2. Stale / offline Cam 2 keeps feeding frozen metrics & causes false spillback
- **Location:** `run_multi_camera.py` (`BatchedCameraPipeline.step`, L842–L885)
- **Cause:** The corridor sync loop only filters on `not c.health_invalid and c.frame_invalid_reason is None and not c.maintenance_active`. When an upstream camera goes silent (`status == "stale"` or `"offline"`), those flags stay `False`, so `c.metrics` keeps reporting the last frame's vehicles indefinitely.
- **Impact:** Stale Cam 2 data keeps inflating `vis_cam2` and holding `stopped_cam2 = True`, causing false Spillback Override (`15 + vis_cam2`).
- **Reproduced by:** `test_03_stale_cam2_without_health_invalid_keeps_feeding_metrics` (`AssertionError: 1 != 0` while `status == "stale"`).

#### 3. Health-invalid or maintenance camera leaves stuck `vis_cam2` / `stopped_cam2` / `spillback`
- **Location:** `trt_pipeline/corridor.py` (`CorridorQueueEstimator.update_visual_occupancy`, L556–L579 & L522–L555)
- **Cause:** When `northeast` is `health_invalid` or in maintenance, `run_multi_camera.py` omits `NE1` from `lane_vis`. In `update_visual_occupancy`, `elif target_lane not in self.reverse_mappings:` prevents `vis_cam2` / `stopped_cam2` from resetting because `N1` is in `reverse_mappings`. Similarly, if Cam 1 is missing (`c1 is None`), `vis_cam1` and `stopped_cam1` are not cleared.
- **Impact:** Fake spillback (`spillback = True`) and frozen cart state persist after camera failure.
- **Reproduced by:** `test_02_cam2_health_invalid_leaves_stale_counts_and_spillback_in_corridor` (`AssertionError: True is not false`).

#### 4. Schema 2.0 `count` invariant broken & motorcycles dropped when dual-cam/cart activates
- **Location:** `run_multi_camera.py` (`BatchedCameraPipeline.build_payload`, L1023–L1030)
- **Cause:** When `has_dual_cam_or_cart` becomes `True`, `lane["count"]` is overwritten with `demand_count` (which excludes motorcycles), while `queuedCount`, `movingCount`, `unknownStateCount`, and `classes` still come from `cam.metrics.snapshot()` (which include motorcycles).
- **Impact:** Violates `docs/CONTROLLER_CONTRACT.md` invariant (`count == queuedCount + movingCount + unknownStateCount`). In a lane with 3 motorcycles + 1 car (`count = 4`), 1 upstream car appearing in Cam 2 causes `lane["count"]` to drop from `4` to `2`.
- **Reproduced by:** `test_04_payload_count_invariant_and_motorcycle_drop` (`AssertionError: 2 != 4`).

#### 5. Cam 1 tracker ID switches silently drain Virtual Cart tokens
- **Location:** `trt_pipeline/corridor.py` (`CorridorQueueEstimator.update_visual_occupancy`, L531–L539)
- **Cause:** `seen_cam1_tracks` triggers `handover_token()` whenever any unseen `track_id` appears in Cam 1 `car_tracks`, with no ID-switch protection.
- **Impact:** A tracker ID switch inside Cam 1 (e.g., temporary occlusion while `cars_total` stays constant) consumes a blind-spot token or borrows/steals one from an adjacent frozen lane (`N2`).
- **Reproduced by:** `test_08_cam1_id_switch_drains_virtual_cart` (`AssertionError: 0 != 1`).

---

### Medium Severity

#### 6. Token `ttl_remaining_sec` double-subtracts elapsed time
- **Location:** `trt_pipeline/corridor.py` (`_tick_carts` L370–L371, `get_telemetry` L843)
- **Cause:** `_tick_carts` decrements `tk.ttl -= dt` and increments `tk.age += dt`, then `get_telemetry` computes `max(0.0, t.ttl - t.age)` instead of `t.ttl`.
- **Impact:** After `4.5s` of an `8.5s` TTL, `ttl_remaining_sec` reports `0.0` instead of `4.0`.
- **Reproduced by:** `test_05_token_ttl_remaining_sec_double_subtraction` (`AssertionError: 0.0 != 4.0`).

#### 7. `GateFlowManager.get_mqtt_telemetry` divides `flow_rate` by `1.0s` instead of `window_s`
- **Location:** `trt_pipeline/gates.py` (`get_mqtt_telemetry`, L528 & L543)
- **Cause:** Calls `self.corridor_estimator.get_telemetry()` without `interval_sec=window_s`.
- **Impact:** `by_direction[*]["flow_rate"]` and `by_lane[*]["flow_rate"]` are scaled by `1.0s` instead of the actual publication interval (e.g., `5` outflows over `5.0s` reported `5.0` instead of `~1.0`).
- **Reproduced by:** `test_06_mqtt_telemetry_flow_rate_uses_wrong_interval` (`AssertionError: 5.0 not less than 1.2`).

#### 8. `handover_token` Priority 1 matches `age >= 0.0` before Priority 2 (`age >= 0.7s`)
- **Location:** `trt_pipeline/corridor.py` (`handover_token`, L306–L321)
- **Cause:** `for min_age in ((1.0, 0.0) if require_min_age else (0.0,)):` falls back to `min_age = 0.0` inside Priority 1 before Priority 2 (`age >= 0.7s`) is checked.
- **Impact:** A newly created `0.1s` exact-class token hijacks handover ahead of an older `5.0s` token in FIFO.
- **Reproduced by:** `test_07_handover_priority_1_vs_2_age_rule` (`AssertionError: 2 != 1`).

---

### Low / Portability Severity

#### 9. `segments_intersect` misses crossings directly through gate endpoints
- **Location:** `trt_pipeline/gates.py` (`ccw` / `segments_intersect`, L33–L45 vs `get_segment_intersection`, L48–L66)
- **Cause:** Strict `>` in `ccw` returns `False` for collinear/endpoint intersections where `get_segment_intersection` returns a valid point.
- **Reproduced by:** `test_10_virtual_gate_endpoint_crossing_inconsistency` (`AssertionError: False is not true`).

#### 10. `stream_cameras.py` crashes on non-Windows platforms
- **Location:** `stream_cameras.py` (L22)
- **Cause:** Unconditional `import msvcrt`.
- **Reproduced by:** `test_09_stream_cameras_import_on_non_windows` (`ModuleNotFoundError: No module named 'msvcrt'` on macOS).

---

## 3. How to Re-run
```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest tests.test_stress_dev -v
```
