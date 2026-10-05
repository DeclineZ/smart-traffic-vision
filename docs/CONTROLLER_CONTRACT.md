# Vision → controller contract (schema 2.0)

This describes the canonical rich measurement payload used by shadow output
and local records. `--output-mode controller` instead sends the existing main
format through the [main compatibility adapter](CONTROLLER_MAIN_COMPATIBILITY.md).
No controller code or migration is required for that path. Vision tests a real
rich payload in `tests/fixtures/vision_payload_v2.json`. The acceptance requirements
below apply to an optional future rich-schema consumer.

## Topics

| Topic | Publisher | QoS / retain | Content |
|---|---|---|---|
| `traffic/counts/shadow` | vision runner by default, once per `--pub-interval` (default 1 s) | 1, not retained | measurement message below; shadow subscribers only |
| `traffic/health/shadow/<sourceId>` | vision runner by default (each publish), broker (last will) | 1, **retained** | shadow health summary; `{"status":"offline"}` when the runner dies |
| `traffic/counts` | vision with `--output-mode controller` | 0, not retained | complete fresh occupancy in main's existing format; see the main adapter document |
| `traffic/health/<sourceId>` | vision with `--output-mode controller`, broker last will | live health 0, last will 1; retained | diagnostic health; main does not subscribe |
| `traffic/decision/v1` | backend | 1, not retained | signal decisions (unchanged) |

## Measurement message

```jsonc
{
  "schemaVersion": "2.0",
  "intersectionId": "INT-001",
  "cameraId": "VISION-INT-001",        // = sourceId; kept for v1 readers
  "sourceId": "VISION-INT-001",        // vision identity; legacy wire cameraId is CAM-01
  "sessionId": "uuid",                 // new on every runner start
  "sequence": 42,                      // +1 per message within a session
  "timestamp": "…Z",                   // observation time (= observedAt); freshness is judged on this
  "observedAt": "…Z",                  // oldest last-observation time among VALID cameras
  "publishedAt": "…Z",
  "meta": { "model": {"path": "…", "sha256": "…"}, "mode": "live|replay", "anchor": {...}, "queueUnits": {...}, ... },
  "cameras": [
    { "cameraId": "INT-001-CAM-N", "name": "north", "status": "ok", "reason": null,
      "observedAt": "…Z", "ageMs": 84, "calibrationRevision": "…", "reconnects": 0, "framesDropped": 0, ... }
  ],
  "lanes": [
    { "laneId": "N1", "direction": "N", "role": "queue", "cameraId": "INT-001-CAM-N",
      "valid": true, "invalidReason": null, "observedAt": "…Z",
      "count": 3, "queuedCount": 1, "movingCount": 2, "unknownStateCount": 0,
      "vehicles": { "queued": {"cars": 1, "motorbike": 0}, "moving": {...}, "unknown": {...} },
      "classes": { "car": 3, "motorcycle": 0, "bus": 0, "truck": 0, "three_wheeler": 0 } },
    { "laneId": "S1", "direction": "S", "cameraId": "INT-001-CAM-S",
      "valid": false, "invalidReason": "stale",
      "count": null, "queuedCount": null, "movingCount": null, "unknownStateCount": null, "vehicles": null, "classes": null }
  ],
  "traffic_flow": {
    "interval": {
      "windowStart": "…Z", "windowEnd": "…Z", "windowSec": 1.02,
      "gates": [ { "gateId": "GATE_N_STOPLINE", "type": "stopline", "direction": "N", "cameraId": "INT-001-CAM-N",
                   "count": 1, "observedSec": 1.0, "coverage": 0.98, "valid": true } ],
      "by_direction": {
        "N": { "arrivals":   { "count": 0, "observedSec": 1.0, "ratePerSec": 0.0, "valid": true },
               "departures": { "count": 1, "observedSec": 1.0, "ratePerSec": 0.98, "valid": true } },
        "S": { "arrivals": null, "departures": {...} }
      }
    },
    "by_direction": { "N": { "inflow": 0, "cleared": 1, "live_discharge_rate": 0.98 } }
  }
}
```

## Definitions

- **Occupancy (`count`)** is every tracked vehicle whose road-contact point (bottom-centre of its box, configurable per camera as `lane_metrics.anchor`) lies in the lane polygon at the observation time. It is a current snapshot, not an interval union. Each vehicle counts in at most one lane: the first lane in configuration order whose polygon covers the point, boundary included.
- **`queuedCount` / `movingCount` / `unknownStateCount`** split occupancy by motion state. The invariant is `count = queuedCount + movingCount + unknownStateCount`, and `count = sum(classes)`.
  - A vehicle is **queued** after its speed stays below `enter_speed` for `enter_duration_s`. It returns to **moving** only after exceeding `exit_speed` for `exit_duration_s` (hysteresis).
  - Speed is measured over `window_s` on the camera's observation clock. Its units are box-heights/s by default (perspective-normalised), or m/s when the camera config has a road-plane `homography`. The units are in `meta.queueUnits`.
  - **Unknown** means the track has less than `min_history_s` of history, or is slow but not yet slow for long enough.
  - Defaults are `enter_speed 0.15`, `exit_speed 0.35`, `enter_duration_s 1.5`, `exit_duration_s 0.5`, `window_s 1.0` and `min_history_s 0.5`. They are configurable under `lane_metrics.queue`. **They are engineering defaults, not values validated on labelled clips.**
- **Lane role.**
  - `queue` (default): the lane is part of the approach's stop-line queue.
  - `upstream`: the lane is further back on the approach. It is reported for arrivals and spill-back context. The receiver must exclude it from stop-line queue totals and coverage to avoid double counting.

  INT-001's northeast lanes are `upstream` because that camera looks at the north approach from behind the north camera. Confirm this ownership on site.
- **Validity.** A lane is valid only when its camera's status is `ok` or `degraded` (sharpness warning). Otherwise every count field is `null` and `invalidReason` says why: `starting`, `tracker_warmup`, `stale`, `offline`, `reconnecting`, `resolution_mismatch`, `no_signal`, `frozen_feed`, `camera_shifted`, `low_detail` or the operator's maintenance reason (status `maintenance`, set with `--maintenance-file`).
  - **A valid zero is a measured empty lane. An invalid lane is unknown and must never be read as zero.**
  - A camera becomes `stale` when its last processed observation is older than `--max-observation-age` (default 1.5 s). Frames older than `--max-frame-age` (default 1 s) are dropped before inference.
  - Local frame/observation age uses monotonic time; wall-clock corrections do not extend freshness. Payload times remain UTC, so the receiver must reject future timestamps and monitor host clock synchronization.
  - A low-detail, low-contrast view is unusable (`low_detail`), with 2 s of clear frames required for recovery. Mild low detail remains a usable warning. Per-camera `camera_health` settings configure thresholds such as `obstruction_std` and `recovery_duration_s`; validate them on site.
  - RTSP observation wall time is local decode time. Backlog inside a camera/NVR is not measurable from this timestamp; verify end-to-end age on site.
  - Maintenance pauses analytics for the affected camera. Leaving maintenance requires fresh observations and tracker confirmation. An invalid maintenance file disables measurements with `maintenance_config_error` until corrected or removed.
- **Gates** count a track once per gate, when its contact point crosses the line in the gate's forward direction by at least 2 px.
  - Gate continuity is bounded: a track missing for up to 1 s can still be counted, but longer gaps are not bridged.
  - Counts are **per publish interval**; `observedSec` is how much of that interval the gate's camera was actually processed.
  - A gate whose coverage is below 0.8, whose camera is unusable, or whose interval contains a temporal reset is `valid: false` with `count: null`.
  - Blind gaps earn no coverage; coverage is clipped to the current publication interval. Successful delivery starts a new interval, including its validity flags.
  - A direction with no gate of a type has `arrivals`/`departures` = `null`, meaning **not instrumented**, not zero.
  - After a failed publish, the interval keeps accumulating, so no crossings are lost.
  - The legacy `traffic_flow.by_direction` block only lists instrumented directions, and it now holds interval counts, not cumulative ones.
- **Discontinuities.** A reconnect or file loop starts a new camera epoch, which resets that camera's tracker, motion, class votes and gate history. A runner restart produces a new `sessionId`.

## Consumer acceptance requirements (controller owner)

1. **Reject the message** in any of these cases:
   - the intersection is unknown;
   - `sourceId` is not registered to that intersection;
   - a lane is not in the intersection's `topology.lanes`, or appears twice;
   - a count is non-integer or negative, or the invariant is broken;
   - an invalid lane carries numbers;
   - the timestamp is more than `MAX_FUTURE_SKEW_MS` in the future;
   - `(sessionId, sequence)` is a duplicate or out of order.

   Any legacy mode must be explicit. Log and count rejection reasons. Once a source changes session, reject messages from retired sessions too.
2. **Ignore `role: "upstream"` lanes** for queue, occupancy and coverage. **Use the right field.** Queue strategies use `queuedCount`: per-direction totals, and per-lane maxima for max-pressure. The "empty approach → skip phase" logic uses occupancy (`count`), because during green the traffic is moving.
3. **Unknown is not zero.** Invalid lanes are left out of every total. Require the complete expected lane set from the intersection registry; an omitted lane is missing coverage too. Validate freshness per lane. If any approach served by a phase is not fully valid, use the established fixed-time fallback with a reason.
4. **Freshness** is `now − timestamp` (the observation time) against `FRESHNESS_MAX_AGE_MS` (2000 ms). The age budget is observation age at publish (≈0.1–0.3 s) + publish interval (1 s) + delivery, which leaves about 0.5 s of margin. Both hosts must run NTP.
5. **Measured discharge** requires a valid gate and a queue at the interval's start. The entire observation interval must lie within the served green; checking the phase only when the message arrives is insufficient. Keep timing use disabled until this is verified.
6. **Every automatic green**, including early empty-approach skips, respects the phase's configured `min_green_sec` / `max_green_sec`.
7. **Persistence.** `traffic_readings.measurements` keeps per-lane q/m/u, validity, camera status and interval flow for every reading. `lane_counts` / `direction_totals` hold occupancy of valid lanes only.

## Versioning

- A breaking change increments the major version. Consumers accept `2.x` and reject other majors.
- Additive fields are minor-version changes and must not change the meaning of existing fields.
