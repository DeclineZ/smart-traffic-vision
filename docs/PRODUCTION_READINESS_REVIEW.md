# Vision production readiness review

Reviewed 29 September 2026. Scope: `smart-traffic-vision`, with the backend/controller and frontend in `smart-traffic-sys` for integration context. `smart-traffic-last-repo` was excluded. Application code, model weights and calibration files were not changed. This review adds this report and offline verification evidence.

**Verdict: suitable for continued testing and a monitored shadow deployment; not ready to supply unattended adaptive signal decisions.** The highest-impact change is to make every lane reading a fresh, well-defined observation with explicit validity. Current messages can represent departed vehicles, duplicate vehicles, or old frames while still looking like current queue measurements.

The architecture has useful foundations: bounded application frame queues, one model shared across cameras, separate trackers, temporal class voting, directed gates, asynchronous MQTT, calibration tooling, and controller fallback to fixed-cycle operation when data is stale. Keep those foundations. Fix the measurement contract and failure handling before optimizing with TensorRT.

## Vision-only release status (4 October 2026)

**Vision now supports the controller's unchanged committed main through an
adapter.** Only vision changes are authorized. Controller mode projects complete,
fresh stop-line occupancy into main's existing `CAM-01` message format and
withholds the entire count snapshot on a required-camera failure. Upstream
regions are excluded. No controller code changes or new migration are required.
CLI defaults remain shadow; the service example explicitly selects controller
mode. [Compatibility, startup and pilot limits](CONTROLLER_MAIN_COMPATIBILITY.md).

Additional review reproduced and fixed vision reconnect false zeros, bright
obstruction handling, dead ingestion threads after backend exceptions, stale
motion evidence, wall-clock freshness errors, maintenance transitions, gate
validity/coverage errors and commissioning-tool gaps. **189 vision tests pass**
including the main-adapter checks (171 before that adapter). The
real PyTorch model passed a new two-minute, five-recording smoke test with two
withheld-frame outages and recovery, zero errors and 113 payloads
([report](soak/vision-only-2min-pytorch.json)). No MQTT messages were sent.
A final one-minute run after the freshness correction also passed both outages
and recovery ([report](soak/vision-only-final-pytorch.json)). The stricter
TensorRT acceptance tool passed a small real-frame sample for batches 1–5
([report](trt-parity/vision-only-fp32-smoke.json)).

The optional controller candidate still has early skips below minimum green, incomplete
expected-lane coverage, retired-session acceptance, discharge intervals spanning
red time and health changes that do not reach open dashboards. These are open
controller-owner tasks. The earlier cross-repository status below records
candidate work, not delivered controller features or production acceptance.
The unchanged main also lacks minimum-green enforcement on allocation and early
skip. The adapter does not certify physical signal safety. Its pinned-main tests
exercise real intake, aggregation, database-write preparation, all four algorithms
and scheduled stale fallback without broker/database/hardware I/O. The new
[real-model smoke](soak/controller-main-1min-pytorch.json) checks required-camera
suppression and upstream-camera independence; all software checks are recorded in
[the full suite output](soak/controller-main-tests.txt).

## Earlier remediation status (4 October 2026; superseded where noted above)

The earlier remediation included candidate changes in both repositories. The
original audit text further down is unchanged. **Suitable for monitored shadow
evaluation; not ready to drive signals unattended.** Receiver fixes, database,
dashboard and shared-broker changes are outside the vision release scope and
cannot be assumed available. Remaining work includes:

- field accuracy on withheld, labelled clips;
- on-site calibration and coverage review;
- a defined, tested safety boundary with the physical signal controller;
- tests on the deployment hardware, network and broker.

### Verification performed

| Check | Result |
|---|---|
| Vision unit/runner tests (`python -m unittest discover -s tests`) | Earlier remediation: **139 passed** (36 at the start of the audit). Additional regression coverage is recorded in the vision-only handoff. |
| Controller tests (`npm test` in `smart-traffic-sys/backend`, new) | **27 candidate tests passed**; additional safety/coverage defects were reproduced during review. These tests and changes are not an authorized vision deliverable. |
| Mutation checks: each original defect re-injected | All caught: assign-then-filter tracking, frame-based speed, invalid lanes published as numbers, gate interval reset on failed publish, unbounded gap bridging, queue from `count` instead of `queuedCount`, missing min-green clamp |
| Real model, all five recordings, CUDA, PyTorch | Pipeline runs end to end; count invariant held on every lane of every payload; a real payload is the shared contract fixture |
| TensorRT export (FP32, dynamic batch 1–8, RTX 5060 Laptop, TRT 11.2) | Built. Parity on 30 frames per camera at batch sizes 1–5: ≥ 99.89 % boxes matched, mean IoU 0.9996, class agreement 100 %, lane-occupancy MAE ≤ 0.006 (never off by more than 1). Latency p50 / p99 for 5 cameras: 22.2 / 25.9 ms vs 46.8 / 66.8 ms PyTorch. [Report](trt-parity/fp32_rtx5060_report.json) |
| Soak, TensorRT engine, real recordings, two injected camera outages | First run (15 min, [report](soak/soak-15min-trt-fp32.json)): 0 errors, bounded state, but healthy cameras were wrongly flagged `camera_shifted` (see the next row). After the fix, second run (12 min, [report](soak/soak-12min-trt-fp32.json)) **passed**: 0 errors, 8,782 inference batches, step p50/p99 55/90 ms, RSS flat (2,168 → 2,063 MB), ≤ 48 motion tracks per camera. During each outage only that camera's lanes were unknown, never zero, and every other camera stayed valid. This is not a substitute for the 24–72 h field soak. |
| Camera-shift detector on recordings | The first soak found that whole-image phase correlation flagged fixed cameras as shifted (hundreds of px, caused by traffic). Replaced with ORB + RANSAC and 3 consecutive confirmations. Over 60 checks spanning 40 min per camera, max measured shift is 4.2 px; a synthetic 47 px shift measures 47–48 px on every camera. Night frames against a day reference are inconclusive (never flagged). |
| Calibration validator | 0 errors, 6 overlap warnings (5 known + NE1/NE2 33.8 px² exposed by the Batch 2B repair) |
| Frontend | Changed files parse (esbuild). **Not rendered**: the dashboard's dependencies are only installed inside its Docker volume and Docker was not running. |
| Database / broker / backend process | **Not run** (Docker unavailable). Migration 012 and the persistence changes are untested against PostgreSQL. |

### Release blockers

| # | Finding | Status | What changed |
|---|---|---|---|
| 1 | One failed camera stops every camera | **Fixed in code** | Each camera is processed independently and reconnects on its own. Live readers open with timeouts and reconnect with backoff; `start()` never blocks. Startup is inside the cleanup boundary. A silent camera's lanes are published `valid:false` with `null` counts. Per-camera status goes in every payload, and a retained health topic carries a last will. Verified with fakes and the soak outages; **not yet with real RTSP cameras/NVRs**. |
| 2 | Freshness describes publish time | **Vision corrected; source latency unverified** | Frames carry local decode time; the runner takes the newest frame and drops locally stale frames. Per-lane observation time and validity are explicit. Black, frozen, severe low-detail and shifted views (ORB/RANSAC) are detected. Camera/NVR backlog and health thresholds need field verification. |
| 3 | Lane interval union | **Fixed** (Batch 1) | Lanes now also carry queued/moving/unknown state per vehicle and per-class counts. |
| 4 | Controller uses `count` for queues | **Producer ready; receiver open** | Vision provides `queuedCount` and honest unknowns. Queue aggregation, expected-lane completeness, freshness, skip safety, persistence and discharge learning require controller-owner remediation and acceptance. Contract in [CONTROLLER_CONTRACT.md](CONTROLLER_CONTRACT.md). |
| 5 | Invalid / overlapping calibration | **Partly fixed** | E1/NE1 repaired (Batch 2B, awaiting your visual review). Lanes use the road-contact point, consistent with gates; boundary points count. Resolution, revision, operator and reference image are recorded, with history and rollback. **Open:** 6 overlap warnings and the E1/NE1 coverage questions need on-site review; reference frames come from the recordings and must be retaken live. |
| 6 | `MULTI-CAM` not registered | **Controller-owner task** | Vision emits `VISION-INT-001` and `INT-001-CAM-*`. Candidate migration 012 is unapproved, outside vision and not verified against the database. Register and test these identities before live delivery. |

### Accuracy and algorithms

| Finding | Status |
|---|---|
| Field accuracy unproven | **Open.** [`tools/eval_counts.py`](../tools/eval_counts.py) computes per-lane occupancy/queue MAE, bias, p95, false zeros, empty-lane false positives, unknown share and gate precision/recall from recorded payloads plus a labelled clip. No withheld labelled clips exist yet; northeast has no night footage. |
| Queue classification FPS/perspective dependent, memory growth | **Fixed in code.** Time-based speed on each camera's observation clock, normalised by box height (or metres with a road-plane `homography`), with enter/exit hysteresis and an explicit `unknown` state. Held boxes on skipped frames no longer touch motion or occupancy. Histories are pruned by time (bounded in tests and soak). **Thresholds are defaults, not validated on labelled clips.** |
| Tracking | **Fixed in code.** Threshold-aware assignment (reference ByteTrack `cost_limit` objective). Note: under that objective the audit's 2×2 example correctly yields one match; the real defect is the assign-then-filter case `[[0.88,0.81,0.67],[0.96,0.93,0.75]]`, which returned zero matches and is now a regression test. The detector runs at conf 0.10 so the low-confidence stage receives input. Class/score come from the matched detection; SORT uses one-to-one IoU matching. **Thresholds still need tuning on labelled clips.** |
| Cross-camera coverage (north vs northeast) | **Default chosen, needs site confirmation.** Lanes have a `role`; NE1–NE3 are `upstream`, excluded from queue totals and coverage, and shown with ↑ on the dashboard. |
| Gate outputs | **Fixed in code.** Interval counts with window, observed seconds and coverage validity; `null` = not instrumented; explicit `target_dir`; unique IDs enforced; bounded 1 s continuity; 2 px minimum motion; time-bounded de-duplication; per-camera reset on discontinuity; session-cumulative counts kept for the HUD only. **Open:** only north has an ingress gate; no per-lane gates or turning movements. |

### Runtime, delivery and deployment

| Finding | Status |
|---|---|
| Unbounded `track_histories` | Fixed (see queue classification). |
| 2 s publish = 2 s freshness | Fixed: publish every 1 s; freshness judged on observation time; budget documented in the contract. |
| Receiver validation, duplicates, ordering | Controller-owner task. Candidate validation still lacks complete expected-lane coverage and rejection of retired sessions. |
| TLS / auth / last will | Vision supports TLS/auth and isolated retained shadow health/last will. Broker ACLs/listeners, backend credential guards and dev compose bindings are outside vision; candidate shared-service changes are not an approved release dependency. |
| Publish failure ignored | Fixed: delivery counters and acknowledgement age go into the health message; gate intervals continue across failed publishes; nothing stale is queued. |
| Ignored config, silent fallbacks | Fixed: effective settings are logged and sent in payload meta; ignored JSON keys are logged; strict camera/config/source counts; missing model is fatal; `--model-sha256`; no silent CPU fallback; CLI beats environment for MQTT settings; `--check-config`; `--sources-file` keeps credentials off the command line; URLs are redacted in logs. |
| Replay vs live | Fixed: replay uses media time from `--replay-fps` (warns when metadata reports > 60 FPS), and a file loop starts a new epoch that resets temporal state. |
| Service, restart, lock, soak | Added: `deploy/` (systemd unit with restart and hardening, liveness timer, env/sources templates, commissioning checklist), `--log-file` and `--record-payloads` (rotating), `--health-file` with `tools/healthcheck.py`, error containment that exits for supervisor restart after 50 consecutive errors, `requirements-lock.txt`, `tools/soak.py`. **Open:** the 24–72 h soak on field hardware with real cameras, broker restarts and clock jumps. Note that `torch` in the verified venv has a broken install record (no metadata). |
| TensorRT | Export CLI fixed; engine metadata records source hash and builder versions; FP32 engine verified (see above). **FP16 with TensorRT 11 needs `nvidia-modelopt`** (not installed). The engine must be rebuilt and parity-checked on the deployment GPU. |

### Operator tools (เจ้าหน้าที่)

| Tool | Status |
|---|---|
| Camera/measurement health panel (Thai) | Candidate dashboard work is outside vision, not rendered and not released. Live health changes need a broadcast fix and acceptance testing by its owner. Vision supplies health metadata and a local overlay. |
| Explainable overlay | Added to the runner display: counted vehicles show lane and Q/M/U state with a contact-point dot; uncounted tracks are grey; tiles show camera status and age; missing cameras get a placeholder tile. |
| Calibration validation / versioning / rollback | Added (see blocker 5). |
| Incident review | Partly done: rotating payload record, `tools/diagnostic_export.py` (payloads in a window, calibration, hashes, health, log tail, environment) and `tools/eval_counts.py`. **Open:** a replay UI with overlays and a manual correction workflow. |
| Maintenance mode | Added: `--maintenance-file` (`{"camera": "reason"}`, re-read when changed) publishes the camera as `maintenance` with unknown lanes, so the controller falls back. |
| Alerting | Candidate backend webhook is outside vision and not a delivered vision feature. |

### Controller safety additions

- Candidate allocation clamping fixes the max-pressure 3 s result against a 5 s minimum, but **automatic early skips still bypass minimum green**. Controller-owner acceptance remains open.
- A non-numeric `FRESHNESS_MAX_AGE_MS` is treated as stale instead of always fresh.
- `stopDecisionLoop()` runs on shutdown.

### Still open — needs people, hardware or decisions

1. **Signal safety boundary** (addendum A): the device that drives the lamps is outside both repos. Before any field control, document and test that the local controller enforces min green, intergreen and conflicts, reverts to its own plan or flash on lost communication, and reports actual lamp state.
2. **Pedestrian phases and preemption**: not modelled.
3. **Labelled withheld evaluation**: continuous clips (day/night/rain/queues) labelled per lane and per gate event, then run `tools/eval_counts.py`; agree acceptance thresholds with the controller team.
4. **On-site calibration**: overlaps, E1/NE1 coverage, lane range, NE upstream ownership, live reference frames, and an optional road-plane homography.
5. **Field hardware**: rebuild and parity-check the TensorRT engine (install `nvidia-modelopt` for FP16), 24–72 h soak with real cameras, broker and backend, NTP on all hosts, a real database insert, rendering the dashboard.
6. **PDPA / privacy, roadside hardware** (addendum F).

---

## What was verified

- All **36 existing unit tests passed** using the repository's Python environment.
- **31 Python source files passed syntax parsing.** This does not prove every optional/legacy module imports correctly.
- The exact `models/yolo26s_thai_traffic.pt` loaded on CUDA, exposed the expected five classes, and processed one batch containing a frame from each of the five configured videos. All returned boxes were finite. This was a smoke test, not a performance or accuracy benchmark; startup emitted an NMS time-limit warning, which warrants a warmed-up sustained test.
- All five configured videos opened at 1920 × 1080. Their reported file frame rate is **150 FPS**.
- Offline probes reproduced category double counting, obsolete interval counts, FPS-dependent queue classification, centroid/contact-point disagreement, FIFO frame retrieval, the all-camera barrier, a tracker assignment failure, a missed gate crossing after an observation gap, invalid/overlapping polygons, and the broken export CLI.
- A direct Node check fed `count=10, queuedCount=0, movingCount=10` into the controller aggregators. Both the direction total and busiest-lane count became **10**.
- Model hashes were compared with existing evaluation reports. Those evaluations were inspected, not rerun.

Evidence: [offline probe results](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/production-audit-2026-09-29/evidence.json), [reproducible offline probes](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/production-audit-2026-09-29/verify_audit.py).

No live RTSP cameras, broker, database or physical signal controller were exercised. No messages were published. Network recovery, actual installed database records, continuous throughput and field accuracy remain unverified.

## Release blockers

### 1. One failed camera stops every camera's output

**P1, confirmed control flow and offline reproduction.** The runner removes one frame from every available worker, then discards the entire batch when even one camera has no frame. This happens before inference, publishing and display updates. A disconnected camera therefore stops healthy approaches too; multiple slow cameras also add sequential waits. There is no independent health publication.

The production `StreamBufferWorker` retries `read()` on the same capture after failure. It does not reopen the connection, use bounded open/read timeouts, or apply reconnect backoff. Opening a source is synchronous, and startup occurs outside the runner's cleanup `try/finally`. A startup failure can leave earlier workers running until process exit.

**Change:** process available cameras with explicit camera-to-result mapping and a bounded batch deadline. Independently publish camera health. Reconnect each camera with backoff and reset its temporal state after discontinuities. Put startup under cleanup protection. Treat an unavailable approach as unknown and let the controller use an agreed fallback, rather than substituting zero vehicles.

Evidence: [batch barrier](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:381), [capture startup and failure handling](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/stream.py:57), [startup cleanup boundary](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:366).

### 2. Freshness describes publication time, not the observed traffic

**P1, confirmed design gap.** The capture worker produces a local timestamp but the runner discards it. The payload stamps the current time when publishing. Delayed decoded frames can consequently pass the controller's freshness check. `get_frame()` returns the oldest queued item, despite its latest-frame description. A two-frame Python queue does not bound camera, network and decoder buffering. The capture loop also deliberately paces reads, which can allow upstream backlog when a live source is faster than the configured ingestion rate.

**Change:** continuously drain live capture, consume the newest suitable frame, and carry source presentation/capture time where available plus local receive time. Enforce an age limit before inference and publication. Track per-camera latency, dropped frames, reconnects and frame continuity. Detect frozen, black or obstructed views and camera movement; publish degraded/invalid status. Repeated imagery needs a careful check so a genuinely stationary scene is not automatically called frozen.

Use timeout properties at capture open with a supported backend; OpenCV documents its FFmpeg/GStreamer open/read timeout settings as **open-only**. The older `video_stream.py` sets these after opening, and its reconnect code is not the path used by `main.py run`. [OpenCV video I/O documentation](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html).

Evidence: [discarded timestamps](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:392), [publisher timestamp](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/payload.py:199), [paced capture and FIFO retrieval](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/stream.py:93).

### 3. Lane messages mix interval observations with current occupancy

**P1, reproduced.** `LaneMetricsManager` accumulates track IDs for the entire publication interval. A vehicle that leaves the lane remains until reset. If it changes lane, it can remain in both lanes. If a track fragments, the same physical vehicle can acquire multiple counted IDs. If its class changes from car to motorcycle, registration removes the previous movement state only within the new category, so one ID is counted twice. The probe produced `count=2` for one ID.

This is neither a current queue snapshot nor a clean arrival count. It is an interval union of IDs with their last registered states, presented as a snapshot.

**Change:** maintain one current record per camera/track, with exactly one lane, category and movement state. Rebuild occupancy at the observation time or use an explicitly defined short smoothing interval and disappearance grace period. Keep gate crossings as separate interval flow measurements. Prune/reset stale tracks, including when an entire frame is empty.

Evidence: [registration and snapshot](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/payload.py:88), [empty-frame early return](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:305), [publication reset](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:552).

### 4. The controller uses the wrong field for queue decisions

**P1, reproduced across the boundary.** Vision sends total `count`, `queuedCount`, `movingCount`, and class breakdowns. The controller's queued aggregator reads **`lane.count`**, and so does its busiest-lane aggregator. Queue-based, simple-cycle and max-pressure decisions therefore use counts that include moving traffic. The primary seeded strategy is `MAXPRESSURE_SWITCHING_LOSS`, which uses the busiest lane per direction; other strategies sum directions.

The README says the controller uses `traffic_flow.by_direction.live_discharge_rate` and upstream `inflow`. The reviewed controller code does not consume either field. Its discharge assumptions still come from configured phase values. Flow and queue/class detail are also omitted from persisted `traffic_readings`, so history cannot fully reconstruct these measurements.

**Change:** agree on occupancy, queue and flow semantics, then update both producer and consumer together. If a strategy deliberately wants all approaching vehicles, name that input explicitly. Do not change `count` to mean queued vehicles silently. Add a producer-to-controller contract test and persist the measurement fields needed to explain decisions.

Evidence: [direction aggregation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/utils/reliableQueuedAggregator.js:29), [busiest-lane aggregation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/utils/reliableQueuedAggregator.js:146), [static flow calculation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/brain/maxPressureSwitchingLoss.js:10), [database projection](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/repositories/trafficReading.repo.js:50).

### 5. Current calibration contains invalid and overlapping lanes

**P1, verified against the checked-in JSON.** `E1` and `NE1` are self-intersecting polygons. Several lanes overlap: the largest detected overlap is about 2,441 square pixels between `S2` and `S3`. The lane evaluator registers a track in every containing polygon. No single-lane arbitration prevents duplicates. Polygon boundaries are excluded by `contains_xy`, creating another ambiguity for lane-divider traffic.

Lane membership and queue motion use bounding-box **centres**, while gate crossings use **bottom centres**. The README's road-contact-point claim does not match lane assignment. A synthetic vehicle with its road contact inside a lane but box centre outside was missed entirely.

**Change:** repair geometry with a camera preview; reject invalid polygons, duplicate IDs and unintended overlaps on save and startup. Use a consistent, validated road-contact convention and stable lane assignment with boundary handling. Record calibration resolution, camera identity and revision; reject or deliberately transform mismatched stream resolutions. Add camera-shift detection.

Evidence: [east geometry](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/config/config_east.json:63), [northeast geometry](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/config/config_northeast.json:63), [lane assignment](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:308), [calibration save](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/tools/segmentor.py:142).

### 6. The default publisher identity does not match the seeded database

**P1 for a deployment using the supplied migrations; live database not inspected.** Vision hardcodes `cameraId="MULTI-CAM"`. The controller's seed creates `CAM-01` and the CCTV camera IDs, but no `MULTI-CAM`. `traffic_readings.camera_id` has a foreign key to `cameras`. Thus fresh seeded installations reject and drop vision history rows even while in-memory decisions and dashboard updates can appear to work.

**Change:** define a registered vision-source identity per intersection and preserve contributing camera IDs on lanes. If keeping an aggregate publisher ID, register it explicitly and validate its intersection ownership. Verify an actual insert as part of deployment acceptance.

Evidence: [hardcoded publisher ID](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:272), [camera seeds](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/migrations/007_seed_intersection_001.sql:28), [foreign key](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/migrations/003_traffic_readings.sql:17).

## Accuracy and algorithm findings

### The model runs, but its field counting accuracy is not established

The installed model SHA-256 is `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`, matching the baseline reports below. Its classes are car, motorcycle, bus, truck and three_wheeler.

| Existing evaluation for this exact checkpoint | Frames | mAP50 | Precision | Recall |
|---|---:|---:|---:|---:|
| Primary validation | 130 | 94.40% | 81.33% | 90.20% |
| Verified local CCTV diagnostic | 42 | 46.44% | 79.58% | 41.82% |

These are **detection metrics, not car-count accuracy**. The diagnostic evaluation used confidence 0.25 at image size 640, whereas the runner defaults to 0.20. Small-object recall in that diagnostic is only **34.92%**, northeast recall **35.84%**, and class-agnostic recall **45.57%**. Many misses are therefore not merely confusion between vehicle types. The different scenes, object sizes and labels need investigation; neither table row proves deployment performance.

The primary manifest contains 107 frames labelled `old_train_split` and 23 `old_val_split`. That is a historical-exposure concern for this baseline checkpoint, even if the set is held out from newer candidate training. The 42-frame diagnostic explicitly disclaims independent-test status: seven frames are near historical training frames and 35 have unproven checkpoint exposure. Do not present either result as an independent production accuracy guarantee.

Evidence: [primary model evaluation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/runs/eval/remapping_expansion_experiment/baseline_primary_val/eval_results.json), [CCTV diagnostic](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/runs/eval/remapping_expansion_experiment/baseline_diagnostic/eval_results.json), [primary sample lineage](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/experiment_b9_dataset_a/primary_val_manifest.json), [diagnostic lineage notice](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/eval_snapshot_v1/README.md).

**Recommended experiment:** create a genuinely withheld set of continuous clips from new dates/time blocks, with all five cameras, daytime/nighttime, rain/glare, heavy queues, motorcycles filtering, buses occluding cars, and empty roads. Label current per-lane occupancy, queued vehicles and gate events, not only boxes. Compare 640 versus higher resolution or carefully deduplicated ROI crops; measure the improvement in distant-vehicle recall against actual camera-to-message latency. Prioritize distant dense queues before adding more rare class taxonomy. Do not promote a candidate solely for slightly higher mAP.

### Queue classification changes with FPS and perspective

**P1 for reliable queued counts.** Speed is pixels per processed batch, using the first and last of up to 15 positions. It ignores elapsed seconds, road perspective and repeated held boxes on skipped-inference frames. At the same 30 pixels/second, the probe returned moving at 10 FPS and queued at 30 FPS. Vehicles far from the camera can look slow in pixels even when moving normally. New tracks initially count as moving; there is no stopped-duration requirement or hysteresis for creeping queues.

Use elapsed timestamps and road-plane calibration where possible, or validated perspective-aware thresholds. Require a sustained low-speed interval and separate thresholds for entering/leaving the queue state. Test stopped motorcycles, bumper-to-bumper creeping, camera vibration and detection jitter. Distinguish queued vehicles from parked vehicles and unrelated lanes. [Queue classifier](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:279).

### Tracking has avoidable identity and class errors

**P2, with one reproduced assignment defect.** The custom ByteTrack implementation performs unconstrained assignment and then discards matches over the threshold. A cost matrix `[[0.1, 0.6], [0.6, 0.8]]` with threshold 0.7 returned one match although two valid cross-matches exist. This can fragment tracks in dense scenes. Use threshold-aware assignment with unmatched options and validate on occlusions.

The detector's default confidence 0.20 discards the 0.10–0.20 detections before ByteTrack can use its configured low-confidence stage. New tracks still require 0.40, so a far vehicle consistently below 0.40 never initializes. Tune these thresholds together on labelled clips.

After ByteTrack has associated detections, the runner reassigns classes using nearest detection centre without a one-to-one or distance constraint. This can borrow a neighbouring vehicle's class. Preserve the tracker's matched class/confidence for voting instead. The voting filter is useful, but cannot repair identity mistakes. Skipped frames hold the last boxes; the advertised continuous Kalman propagation is not actually executed there.

Evidence: [assignment](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/algorithm/byetrack.py:404), [tracker defaults](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:174), [class reassociation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:436), [held boxes](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:497).

### Cross-camera coverage is not yet a defined measurement model

**P1 validation requirement.** North and northeast both publish lane measurements under direction `N`. They have separate trackers and no world-coordinate fusion or ownership map. If their views overlap, the same vehicles can be counted twice; if northeast is upstream, it should not automatically be treated as an additional stopline queue. The physical overlap must be checked on-site. Direction-summing strategies add both; the busiest-lane strategy treats them as separate candidates.

Define which camera owns each physical lane segment. Use northeast as upstream arrivals unless a validated partition/fusion scheme makes it part of queue occupancy. You do not need cross-camera vehicle re-identification merely to count a queue if you can partition coverage clearly. [Northeast lane mapping](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/config/config_northeast.json:63).

### Gate outputs need observation windows, coverage and phase context

**P2 now; P1 before using them to set green duration.** `inflow` and `cleared` are cumulative since startup; discharge is clearances divided by time since interval reset, with no green-state input. Zero discharge during red is not a measured saturation capacity. Two-second windows are noisy, especially with motorcycles and mixed vehicle sizes. Rates currently aggregate an entire approach; a controller dividing the busiest lane's queue by an approach-wide rate would mix units.

The manager drops a missing track's previous point whenever other tracks remain; an occluded track reappearing across a gate can therefore miss its crossing. The offline probe reproduced this. Conversely, bridging long gaps without a limit would introduce false crossings, so use a bounded continuity rule. Add crossing hysteresis/minimum motion and test jitter near lines, turnbacks, line endpoints and ID switches.

Only north currently has an ingress gate. Other directions' emitted zeros therefore also mean **not instrumented**, not proven no arrivals. East's stopline gate is on the northeast camera, so a default four-camera run has no east discharge measurement. Gate direction fallback silently becomes `N` for unrecognized IDs, and duplicate gate IDs overwrite the aggregate dictionary while remaining in per-camera lists. Require explicit movement/direction and globally unique IDs.

Send interval start/end, observation duration, interval event counts, coverage/validity, and a restart/session ID. Keep cumulative counts optional and reset-aware. Estimate saturation discharge over adequate observed green periods with a queue present, sample support and bounded smoothing. Do not infer turning movements from current approach-level gates alone.

Evidence: [gap cleanup](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/gates.py:211), [rate and cumulative metrics](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/gates.py:218), [gate registration](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/gates.py:160).

## Runtime, delivery and deployment gaps

| Priority | Finding | Recommended change |
|---|---|---|
| P1 | `track_histories` never removes completed IDs, so long-running traffic grows memory indefinitely. | Bound state by last observation time and tracker lifecycle; test a sustained stream of new IDs. [Motion history](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:281). |
| P1 | The default 2-second publish interval matches the controller's default 2-second freshness limit, leaving no worst-case delivery/scheduling margin. | Agree on an age budget using capture age plus measured latency/jitter; shorten publication interval or change freshness policy deliberately. [Freshness check](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/services/decision.service.js:120). |
| P1 | The receiver checks only that top-level fields exist. It accepts inconsistent counts, unknown lanes, duplicates, out-of-order readings and problematic timestamps into memory. QoS 1 delivery can repeat messages. | Versioned schema, finite nonnegative integers, field-sum checks, camera/lane registry, session+sequence deduplication, per-source ordering, clock-skew checks and validity-aware fallback. [Validation](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/services/incomingTraffic.service.js:11), [memory insertion](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/backend/src/models/trafficMemory.model.js:5). |
| P1 before field networking | Vision has username/password support but no TLS setup or last-will health; the supplied broker allows anonymous access. | Broker authentication and topic ACLs, validated TLS, restricted network access, retained health/LWT and redacted RTSP URLs in logs. Setting an `mqtts://` string alone does not configure TLS here. [Publisher](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/trt_pipeline/publisher.py:53), [development broker](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/mosquitto/mosquitto.conf). |
| P2 | Publish failures return false, but the runner ignores the result and resets metrics. There is no acknowledgement-age metric or explicit backlog limit/expiry. | Track sent/acknowledged/dropped counters and last successful delivery. Keep occupancy delivery latest-only with expiry; preserve flow events/counters separately if historical loss matters. Do not replay old occupancy as current. [Publish/reset](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:552). |
| P2 | Config fields for tracker parameters, model, MQTT and enable flags are largely ignored by the main runner, which uses CLI/hardcoded values. Missing custom videos/configs silently fall back to presets; `--videos` does not determine camera count. Missing main weights silently select YOLOv8s. | Validate one effective configuration at startup, exact source/config cardinality and known camera names. Fail clearly on missing production weights. Show effective model, confidence, tracker, source and calibration revision to operators. [Argument resolution](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:599). |
| P2 | File playback runs at target processing FPS, not source timestamps; EOF loops without resetting tracks or counts. | Explicit replay mode using media timestamps, reset/epoch on seek/loop, and separate test topic/intersection. With reported 150-FPS files and a 25-FPS runner, playback is six times slower than metadata time if that metadata is correct. Verify the actual recording clock before using wall-clock flow rates from replays. [File ingest/pacing](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/run_multi_camera.py:383). |
| P2 | No production service definition, restart/watchdog policy, dependency lock or automated release pipeline was found in vision. Dependencies are broad lower bounds; the local successful environment uses a development PyTorch build. | Package a reproducible supported environment, pin tested versions, startup preflight, automatic restart, health/readiness reporting, log rotation and rollback. Test on the actual field GPU and camera codecs. |

Paho provides TLS, last-will and queue-limit APIs; these should be configured intentionally. QoS acknowledgement is not proof that the controller validated, stored or used a reading. [Paho client documentation](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).

### TensorRT

The current export command fails immediately: `export_trt.main()` uses `parser` without constructing it. The offline probe confirmed `NameError: name 'parser' is not defined`. Fix [the CLI initialization](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/export_trt.py:89) before export testing.

The main runner already loads `.engine` through Ultralytics. Keep that path and verify FP16 export on the deployment GPU, supported batch profiles, camera outages producing smaller batches, image sizes, model class metadata and numerical/count parity. The legacy `TRTModel` path is separate and should not be assumed interchangeable. Ultralytics documents TensorRT export options for precision and dynamic batching. [TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt).

TensorRT can improve inference throughput. It cannot correct stale frames, lane geometry, queue semantics, counting duplicates or missing small vehicles. Compare warmed-up end-to-end p50/p95/p99 latency and per-lane accuracy before and after export, including preprocessing, decoding, tracking and publishing.

## Proposed controller information contract

This is a proposal requiring coordinated producer/consumer changes. Preserve compatibility through an explicit schema version.

| Information | Meaning and controller use |
|---|---|
| `schemaVersion`, intersection/source ID, `sessionId`, sequence | Validate routing, reject duplicates, distinguish restart from out-of-order delivery. |
| Per-camera observed/received/processed/published times, age and status | Distinguish fresh measurements from delayed, missing or degraded cameras. Do not rely on a single aggregate publication timestamp. |
| Lane ID, physical coverage/movement, contributing camera, calibration revision | Establish which road segment was measured and prevent cross-camera duplication. |
| `count`, `queuedCount`, `movingCount`, optionally `unknownStateCount` | Current occupancy and movement state with documented definitions. A valid empty lane is zero; an unavailable observation is unknown/invalid. Define the count-sum invariant when an unknown state is included. |
| Per-class current counts | Preserve car, motorcycle, bus, truck and three_wheeler where reliable. If using passenger-car equivalents, calibrate weights and keep both raw vehicles and weighted units; bus/truck/motorcycle saturation effects differ. |
| Arrival and stopline crossing counts, window start/end and observed seconds | Compute flow in explicit vehicles/second; communicate missing gate coverage separately from zero events. |
| Per-lane validity and reason | Allow fallback when coverage is obstructed, calibration changed or observations are stale. Detection confidence alone is not count confidence. |
| Model ID/hash, configuration revision | Explain differences after deployment and allow rollback comparisons. |

Add queue-tail position/length and `queueBeyondVisibleArea` after calibration, because a queue extending beyond the field of view makes the visible count a lower bound. Downstream occupancy/spillback can be useful for avoiding a blocked exit, but needs additional calibrated coverage. Turning movement demand and upstream arrival ETA require lane connectivity/travel-time evidence; current gate totals do not provide them.

## Tools for เจ้าหน้าที่

1. **Camera and measurement health panel.** Show normal, delayed, disconnected, obstructed and calibration-changed states in Thai, the age of the last usable reading, affected approaches, and whether the controller has entered fallback. The current vehicle-count view keeps the last received values without an age-based stale treatment, so old numbers can appear live. [Current count state](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/frontend/src/hooks/useCamera.js), [count view](E:/Work/Projects/AdaptiveTrafficControl/smart-traffic-sys/frontend/src/components/VehicleCount.jsx).
2. **Explainable count overlay.** Show exactly which vehicles contribute to each lane count, queued/moving/unknown state, lane boundaries, and gate direction. Include a last-update time. Operators should be able to distinguish an incorrect polygon from a detector miss.
3. **Calibration validation and rollback.** Extend the existing segmentor with invalid/overlap checks, explicit approach/movement selection, resolution checks, sample crossing playback, versioned save, operator identity and rollback. The current `.bak` backup is useful but is overwritten on successive saves; saves are not atomically validated or applied to the running process.
4. **Incident review and count correction workflow.** Select a time window, replay overlays and the exact payload, add manual ground-truth counts, and export a short diagnostic package with model/config IDs. Feed these corrections into the labelled regression set. Preserve explicit distinction between a review correction and live control input.
5. **Safe camera maintenance mode.** Mark a camera/approach unavailable with a reason while cleaning or recalibrating it. Surface the resulting controller fallback and recovery criteria. Reuse the controller's existing manual override and audit trail rather than creating an unlogged signal-control shortcut in vision.

## Suggested release order and acceptance tests

**First:** repair geometry; define current lane occupancy and queues; prevent category/lane duplicates; fix the controller field choice and registered source ID. Add contract tests that include empty, moving-only, queued-only, missing-camera and duplicate-message cases.

**Second:** implement independent camera recovery, observation timestamps/expiry, per-lane validity, bounded track state and authenticated delivery. Run a 24–72-hour soak with camera disconnect/reconnect, frozen video, slow source, broker restart, application restart, clock jumps, packet loss, resolution changes and GPU overload. Verify that healthy cameras continue, invalid input never becomes a valid zero, fallback engages as agreed, memory stays bounded, and recovery does not generate spurious gate events. These durations are proposed engineering acceptance targets, not a certification standard.

**Third:** validate end-to-end counting on withheld continuous footage. Report per-lane occupancy/queue MAE, signed bias and p95 absolute error; empty-lane false positives/false zeros; gate-event precision/recall; time-to-update; and tracking ID switches. Break down by camera, distance, day/night, weather, class and congestion. Avoid percentage error alone on near-empty lanes. Compare the decisions driven by vision with those driven by human-labelled counts, and agree acceptable error/delay thresholds with the controller team.

**Then:** export TensorRT, repeat numerical/count parity checks, and measure sustained throughput with all intended cameras on the field hardware. Start with shadow operation, recording what vision would have caused while the existing controller policy remains in charge. Promote only after the failure tests and agreed accuracy criteria pass.

The supplementary style scanner checked 11 pipeline files and reported six mechanical findings, score 10, with no high-severity findings. That score is not a readiness measure: some flags concern third-party API names or optional telemetry handling. The reproduced measurement and integration defects above determine the release verdict.
