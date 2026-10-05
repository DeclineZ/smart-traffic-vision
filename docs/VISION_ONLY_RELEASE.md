# Vision-only release for existing controller main

Updated 4 October 2026. Only `smart-traffic-vision` is authorized for changes.
Existing uncommitted changes in `smart-traffic-sys` have been preserved and are
not an approved dependency or a delivered controller fix. Nothing has been
committed, pushed or deployed.

## Can vision ship independently?

**Vision now supports the unchanged controller main through a producer-side
adapter.** `--output-mode controller` emits main's existing message format using
its already registered counting-only source `CAM-01`. No controller migration
or candidate working-tree changes are required. See
[main compatibility and startup](CONTROLLER_MAIN_COMPATIBILITY.md).

All nine calibrated stop-line regions retain their occupancy counts; northeast
upstream regions are excluded. Any missing, invalid or stale required reading
withholds the whole live count snapshot. Main's existing freshness check then
selects fixed-cycle fallback at the next scheduled decision. Health and rich
schema 2.0 records continue locally. An upstream-only outage does not block the
stop-line measurements. Unknown counts are never replaced with zero.

CLI and soak output default to isolated shadow topics. The deployment example
explicitly selects controller mode and live sources. Separate MQTT client IDs
keep shadow and live clients distinct. During shadow evaluation keep the existing
live producer; at cutover stop main's mock generator and other count producers for
this intersection. The existing dashboard holds old numbers during an outage,
so operators must also monitor vision's health/delivery state.

This establishes software compatibility, not acceptance of unattended signal
operation. No live MQTT messages, database writes or physical signal commands
were tested.

## Vision corrections in this review

- Reconnects, long observation gaps and image failures clear previous temporal
  state and observation timestamps. The first fresh frame is inferred even with
  frame skipping; tracker confirmation is reported as unknown during warmup.
- Local frame/observation age uses a monotonic clock so wall-clock corrections
  cannot keep a silent camera's readings fresh. UTC observation times remain in
  the payload for receiver-side freshness and require synchronized hosts.
- Low-detail, low-contrast bright obstructions are unusable. Recovery needs 2 s
  of clear observations; mild sharpness loss remains a warning. Thresholds are
  configurable per camera. Inconclusive shift checks break a confirmation run;
  an already confirmed shift stays invalid until the view is restored.
- Capture open/read/metadata exceptions release the capture and retry with
  backoff. Read failures no longer kill the camera thread. Shutdown avoids
  releasing a native capture concurrently with its reader. Replay failures also
  report offline and clean up the capture.
- Motion evidence restarts after an unobserved gap. Gate counts become null when
  camera health/maintenance/warmup or interval coverage makes them unknown; blind
  gaps earn no coverage and publication boundaries cannot reuse prior coverage.
- Maintenance transitions reset temporal state and require fresh observations on
  exit. Malformed, unreadable or misspelled-camera maintenance files disable
  measurements until corrected instead of silently leaving cameras active.
- The soak accepts field sources, bounds latency-history memory, observes
  payloads even when broker delivery fails, checks every affected lane throughout
  outages and requires recovery. The evaluator includes empty-gate false
  crossings, per-lane observation times and unobserved samples/intervals.
- TensorRT acceptance now checks extra engine detections and class agreement as
  well as matched boxes and lane occupancy.
- Linux install instructions give the service user access to configuration and
  the broker CA. Service start-limit settings are in the correct unit section;
  shutdown allows bounded waits for all five camera readers.
  Health-file write failures now propagate to the logged processing-error path
  after temporary-file cleanup, rather than being silently ignored.

## Integration and signal-owner checks

The adapter removes the payload, unknown-count, upstream and source-registration
dependencies on candidate controller changes. Its source profile checks complete
coverage of the calibrated stop-line region set and a conservative age budget.
Operational acceptance still needs one producer per intersection, the existing
`CAM-01` database row, synchronized clocks, measured delivery latency, real
database persistence and an operator-visible outage check.

Main's baseline minimum-green behavior remains a signal-safety issue: allocation
can be 3 s for a phase configured with a 5 s minimum, and early empty-phase skip
can also interrupt below that minimum. Suppression reaches fixed fallback at the
next decision rather than immediately. The physical receiver's intergreen,
conflict protection, lost-command behavior and pedestrian timings are outside
vision's scope. Confirm those boundaries before a pilot that changes lights.
Changing truthful vehicle counts cannot repair these controller behaviors.

The earlier candidate v2 controller also had coverage, session-retirement,
discharge-interval and health-display defects. They remain review findings for
that optional candidate; it is not used by this vision release.

## Evidence and remaining field checks

The offline suite passes **189 tests**, including main-adapter checks, recorded in
[the test output](soak/controller-main-tests.txt) (171 tests before this adapter).
Syntax checks passed for the modified modules. The new regression tests first reproduced the
false-zero, obstruction, dead-reader, motion-gap and gate-validity defects before
the production fixes. Tests use production runner/analytics paths with offline
sources, a fake model and publisher.

[Two-minute real-model smoke report](soak/vision-only-2min-pytorch.json): all five
recordings, `yolo26s_thai_traffic.pt`, CUDA, 113 payloads, 1,661 inference batches,
zero errors. East and northeast each had a 10 s withheld-frame outage: every
affected lane was unknown after the freshness grace period, other cameras stayed
usable and both recovered. No MQTT messages were sent. RSS was stable after
startup (about 1.9 GB). This is a short smoke test, not a long-soak certificate or
proof of network reconnection.

After the monotonic-freshness correction, a
[final one-minute real-model smoke](soak/vision-only-final-pytorch.json) also
passed: 54 payloads, 682 inference batches, zero errors, two 8 s outages with
all affected lanes unknown after the grace period and both cameras recovered.
No other camera became unusable; no MQTT messages were sent.

[TensorRT acceptance smoke](trt-parity/vision-only-fp32-smoke.json): 3 sampled
frames per camera, batch sizes 1–5, all acceptance checks passed. Box matching
and class agreement were 100%, mean IoU about 0.9996, and lane occupancy matched
exactly. This small sample checks the stricter tool and is not an accuracy or
performance certificate. Model weights, calibration coordinates and the engine
were not changed by this follow-up. A code scan found no unresolved bug-class
findings in the eight reviewed implementation modules; two existing descriptive
metric names account for a cosmetic score of 2.

Still required: labelled withheld day/night/rain/queue clips and agreed error
limits; visual/on-site E1/NE1 coverage and six lane overlaps; northeast upstream
ownership; live reference images; true camera/NVR latency; physical RTSP/broker
disconnect tests; 24–72 h on the deployment hardware; TensorRT rebuilt and checked
on that GPU. Live frame timestamps are local decode times and cannot expose
backlog before the frame reaches this process.

The [main-adapter real-model smoke](soak/controller-main-1min-pytorch.json) uses
all five recordings with the actual model and an in-memory publisher. It checks
that an east outage suppresses the whole controller output, a northeast outage
does not block stop-line output, every affected rich measurement becomes unknown
and both cameras recover. A real nine-region wire message is preserved in
`tests/fixtures/vision_payload_main.json` and accepted by the unmodified main
intake/aggregation/decision harness. The harness also verifies database-write
parameters and the scheduled stale-data fallback, with all external I/O stubbed.
The one-minute run produced 55 rich snapshots, 47 accepted wire snapshots and
8 suppressed attempts, with 694 inference batches and zero errors. Both outage
projection checks passed and no other camera became unusable. This is a smoke
test, not a field soak or live network test.
