# Vision output for the existing controller main

Target: `smart-traffic-sys/main`, commit
`7a2a1f4683abcc4ca6ec95e8918230b3cbd7a8fe`. Recheck the pinned fixtures when
main's receiver, aggregation or decision behavior changes. No controller code
or new migration is required for this adapter.

## Start vision against main

Use live camera sources and the existing controller's broker:

```bash
python main.py run --cameras all --sources-file /etc/traffic-vision/sources.json \
  --output-mode controller --controller-config config/controller_main.json \
  --mqtt-broker mqtt://controller-host:1883 \
  --health-file /run/traffic-vision/health.json \
  --record-payloads /var/log/traffic-vision/payloads.jsonl --display
```

Adjust paths for Windows. The default model remains `yolo26s_thai_traffic.pt`.
TLS and credentials remain supported; use the listener actually configured by
the broker owner. This command does not provision the broker or start anything
in the controller repo. CLI output still defaults to shadow when the explicit
controller option is absent. Recorded footage is rejected in controller mode;
`--allow-replay-controller` is solely for isolated offline tests.

Before replacing the producer, the operator must stop the mock generator's
`INT-001` messages and any other producer for this intersection. Main's Compose
configuration starts a mock generator by default. Two producers would mix
simulated and measured traffic and prevent outage freshness from expiring.
Keep the current producer during shadow evaluation; shadow output does not
update main's dashboard.

## What main receives

The adapter emits the existing format on `traffic/counts`:

```json
{
  "intersectionId": "INT-001",
  "cameraId": "CAM-01",
  "timestamp": "2026-10-04T12:00:00.000Z",
  "meta": {"frameId": 42, "countKind": "occupancy"},
  "lanes": [{"laneId": "N1", "direction": "N", "count": 3}]
}
```

The example shows one lane for readability; each actual message contains **all
nine configured stop-line regions**: N1–N3, E1, S1–S3 and W1–W2. Northeast's NE1–NE3
are upstream and are excluded. No region is fabricated or merged. Main's seeded
topology lists eight lanes, but its actual receiver, JSON persistence and
aggregation read the lanes supplied in the message rather than that list.
Offline tests exercise this with twelve lanes too, including a third lane that
must remain the busiest lane. Confirm on site that the calibrated regions cover
all relevant traffic; a profile cannot detect a physical lane outside the view.

`CAM-01` is the uppercase counting-only source already seeded for `INT-001` by
migration 007. It is separate from the lowercase CCTV camera `cam-01`. No new
source registration is required for this seeded intersection. Verify that the
deployed database has that existing row; a different intersection requires its
own profile and registered counting source.

**`count` retains its original meaning: current occupancy**, including moving
vehicles. Main uses occupancy for its traffic decisions. Vision does not replace
it with stopped-only `queuedCount` or interval crossings. Rich queue, class,
gate and health measurements remain in the canonical schema 2.0 payload,
local records and shadow output; main does not consume them. Physical camera
IDs and vision session/sequence are included in wire metadata for tracing,
although main does not validate those fields.

## Failure and freshness behavior

`config/controller_main.json` pins the expected lane IDs/directions. Startup
rejects a missing, additional or differently directed stop-line region. Runtime
requires every expected reading to be valid with a nonnegative integer count
and a timezone-qualified observation time no more than 750 ms old. An upstream
outage does not block healthy stop-line counts. A real, observed empty region
can have count zero.

If any required reading is unavailable, stale, warming up, obstructed, shifted
or in maintenance, **the whole live count snapshot is withheld**. Main never
receives a partial snapshot or an unknown count substituted with zero. Health
and local recording continue and state the blockers. Recovery resumes complete
snapshots. Wire timestamps use the oldest required lane observation, not send
time or an upstream observation.

The profile assumes main's configured freshness limit is 2,000 ms and publication
interval is 1 s. Startup requires source age plus publication interval to leave
room within that limit for delivery. Verify the deployed freshness setting,
clock synchronization and actual network delays. Counts use QoS 0, are not
retained and are not queued while disconnected; this avoids QoS 1 replay of old
snapshots after reconnect. `controllerDelivery.accepted` means the publisher
accepted the message locally, not confirmed database or broker receipt.

With one producer and no retained/replayed counts, main sees stale data after
its freshness limit expires. It chooses fixed-cycle fallback **at its next
scheduled decision**; this does not immediately interrupt an active green.
Main's dashboard also holds the last numbers during silence without dimming
them. Operators must watch vision's preview, health file and delivery state;
the existing dashboard alone cannot identify this outage.

## Evidence and pilot limits

`tests/test_controller_main.py` tests the actual vision publication path,
suppression, recovery, maintenance, replay guard and profile mismatch. The Node
harness runs unmodified committed main intake, memory, aggregator, database-write
preparation, registry, all four algorithms and scheduled stale fallback. Source
snapshots are pinned by SHA-256 and compared with the actual controller commit
when that checkout is present. Only database, broker, clock and timer I/O are
replaced; no controller code is patched. Node is needed for these compatibility
tests, not for running vision.

This establishes software-contract compatibility. The isolated transport tool
checks actual RTSP and MQTT recovery against loopback services; it cannot verify
field camera/NVR latency. A real database insert, dashboard rendering and field
hardware run remain separate acceptance checks. Validate the selected model's
tail latency and observation freshness on dedicated field hardware using the
[production soak and accuracy checks](HARDWARE_BENCHMARK_GUIDE.md).

The baseline controller can allocate 3 s against a phase minimum of 5 s and
its automatic empty-phase skip can also end green before that minimum. Vision
cannot repair signal timing by changing truthful counts. Confirm timing,
intergreen/conflict protection, lost-command behavior and pedestrian safety
with the signal owners before a pilot that changes physical lights. A monitored
counting/shadow pilot can evaluate vision first. Labelled withheld footage,
on-site E1/NE1 calibration and overlap review, upstream ownership, NVR latency
and a 24–72 h field soak remain necessary.

Pinned source hashes normalize CRLF to LF; actual source changes still fail
verification. Model warmup precedes camera/MQTT startup; the watchdog allows
60 seconds for startup only. Canonical and wire observation freshness limits
remain unchanged. Proposed receiver improvements are recorded in the
[system-owner change plan](SYSTEM_OWNER_CHANGE_PLAN.md).
