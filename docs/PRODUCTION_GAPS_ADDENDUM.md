# Production gaps: addendum to the readiness review

Reviewed 4 October 2026, after Batches 1, 2A and 2B. The audit below is historical.
Only vision remediation is authorized. Shared broker, controller, database and
dashboard changes remain their owners' responsibility; candidate uncommitted
changes do not close those release gaps. Current status and acceptance work are
in [VISION_ONLY_RELEASE.md](VISION_ONLY_RELEASE.md) and the opening section of
[PRODUCTION_READINESS_REVIEW.md](PRODUCTION_READINESS_REVIEW.md). Nothing was run
against live RTSP cameras, a shared broker or a signal cabinet.

## A. Safety boundary with the physical signal (highest priority)

1. **Neither repository contains the device that drives the lamps.** The backend publishes phase decisions to MQTT `traffic/decision/v1` (`smart-traffic-sys/backend/src/services/decision.service.js:358`). The receiver is told to "serve this much more green, then run yellow and all-red" (`decision.service.js:683-696`). Nothing in either repository defines:
   - what that receiver is;
   - what it does when decisions stop arriving;
   - whether a hardware conflict monitor (MMU) independently prevents conflicting greens;
   - whether actual lamp state is reported back.

   The dashboard shows the backend's *intended* phase, not confirmed lamp state. Before field use, document and test the following:
   - The local controller enforces min green, intergreen and conflicts by itself.
   - The local controller reverts to its own timing plan or flash on loss of communication.
   - The backend only *requests* timings.
2. **There is no pedestrian phase handling.** Searching the controller for pedestrian/crosswalk logic found nothing. Minimum green must cover the pedestrian crossing time wherever crosswalks exist. A skip-phase or short adaptive green can otherwise cut a crossing short.
3. **Emergency-vehicle preemption and railway or other interlocks are not represented.** Confirm whether any site needs them.

## B. Anyone on the network can drive the signals

The audit flagged MQTT TLS/ACL from the vision side. The larger exposure is on the controller side:

- The broker listens on `0.0.0.0:1883` with `allow_anonymous true`, and compose publishes the port (`smart-traffic-sys/docker-compose.yml:35`). Any host that can reach it can:
  - publish fake counts to `traffic/counts`, which directly changes green times;
  - publish straight to `traffic/decision/v1`, which is the signal command topic.
- Postgres is published on `5432` with `postgres/postgres`.
- `JWT_SECRET=dev-only-not-for-production` has no startup guard that refuses a default secret.
- The receiver accepts any `intersectionId` string and creates a memory entry for it (`trafficMemory.model.js:5`).
- MediaMTX RTSP/WebRTC (`8554`, `8889`) has no authentication.

Separate topics per role, add per-client credentials and ACLs (vision may only write counts, only the backend may write decisions), use TLS, keep database and broker off public interfaces, and fail startup on default secrets.

## C. Time and clock correctness

- Freshness compares the **vision host's** payload timestamp with the **backend host's** `Date.now()` (`trafficMemory.model.js:7`, `decision.service.js:129`). No NTP requirement is documented.
- A vision clock running >2 s behind puts the controller in permanent fallback.
- A clock running ahead makes old data look fresh. A future-dated message gets a negative age and passes the freshness check until real time catches up.
- Reject timestamps beyond a skew tolerance, record receive time, and require/monitor NTP on both hosts.
- Vision gate rates use `time.time()` (`run_multi_camera.py:571`, `gates.py:225`). An NTP step distorts discharge rates. Use a monotonic clock for durations and wall time only for labelling.

## D. Configuration and deployment specifics

- **No production camera configuration exists.** All five configs point at local `.avi` files. RTSP sources can only be passed with `--videos`. Credentials would then appear in the process list and in `StreamBufferWorker` start-up logs (`stream.py:84` logs the full source). Add a source registry with secret references and URL redaction.
- **No GPU check.** `run_multi_camera.py:690-698` silently falls back to CPU (or MPS) when CUDA is unavailable and still publishes. In production that should be a startup failure or a published degraded state.
- **Environment silently overrides CLI.** `MQTT_URL`/`MQTT_TOPIC` take precedence over an explicit `--mqtt-broker`/`--mqtt-topic` (`publisher.py:56-62`), the reverse of the usual convention. Log the effective values.
- **Model weights are not version-controlled.** `models/*.pt` is not tracked by git or any artifact store. Distribute weights through a registry and verify the SHA-256 at startup.
- **Logging is stdout only.** Vision has no file logging or rotation. After the first broker outage, `MQTTPublisher._warned_no_conn` is never reset, so later outages produce no publish-side warning.
- **One exception stops all output.** Any uncaught exception in the batch loop (CUDA OOM, driver reset, malformed frame) ends the process for all cameras. Pair the planned supervisor with per-iteration error containment and a crash counter.

## E. Controller-side testing and operations

- **The backend has no automated tests.** There is no `test` script in `backend/package.json`, and it makes the signal decisions. The contract tests planned for the controller field fix should start a backend test suite: aggregators, freshness/fallback, phase timing, manual override transitions.
- **No alerting path.** `system_events` records `data_stale` and similar events, but nothing notifies an operator. Define who is told, how, and within what time, when an approach falls back or a camera goes dark.
- **Restart mid-phase is undefined.** Backend phase timing is held in in-memory timers. Define and test what the street sees when the backend restarts during green, yellow or all-red. This depends on A.1.

## F. Field, privacy and environment

- **Privacy (PDPA).** Frames show readable licence plates and pedestrians. Define recording retention, who may view streams and recordings, and access logging. The repository already contains `tools/face_detector`; decide whether blurring is required for stored or exported footage.
- **No northeast night footage.** Night clips exist for east, north, south and west but not northeast (`videos/`). That camera supplies the only east stopline gate. Include it in the withheld night evaluation.
- **Roadside hardware.** Cabinet thermal limits, UPS ride-through, and camera lens maintenance (dust, rain, insects, IR at night) should be part of the site acceptance plan. They affect detection more than model tuning does.

## G. Calibration coverage observations from Batch 2B

See [calibration-review-batch-2b/README.md](calibration-review-batch-2b/README.md):

- E1 may include kerb-parked vehicles.
- NE1's left edge follows the drain grating rather than a lane line.
- Northeast lanes extend to the vanishing point, where vehicles are a few pixels tall. Truncate lanes at a validated detection range.
