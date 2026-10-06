# Proposed system changes if system-owner permission becomes available

Prepared 5 October 2026 against `smart-traffic-sys/main`, revision
`7a2a1f4683abcc4ca6ec95e8918230b3cbd7a8fe`.

This is a proposed work plan for the system owners. It records what we would
change, why it matters and how we would check it. It does not authorize or
implement changes in `smart-traffic-sys`.

Vision already supports unchanged main through its legacy adapter. These
proposals would strengthen the receiving system and let it use more of vision's
measurements. They are not prerequisites for the existing software contract.
Physical signal operation has additional safety requirements, and richer queue
measurements need accuracy acceptance before influencing timing.

Vision setup and remaining commissioning checks are in the [README](../README.md)
and [validation guide](HARDWARE_BENCHMARK_GUIDE.md).
The existing wire format is described in
[CONTROLLER_MAIN_COMPATIBILITY.md](CONTROLLER_MAIN_COMPATIBILITY.md); the optional
rich format is in [CONTROLLER_CONTRACT.md](CONTROLLER_CONTRACT.md).

## Order of work

| Batch | Proposed change | Priority |
|---|---|---|
| 1 | Enforce phase timing limits and protect automatic skips | Required before accepting physical signal control |
| 2 | Validate legacy messages, coverage, ordering and producer identity | Protect decisions without changing vision's current wire format |
| 3 | Show freshness, camera health and fallback clearly to operators | Make a monitored pilot observable |
| 4 | Separate simulation from deployment and secure broker access | Prevent mixed or unauthorized inputs |
| 5 | Add an opt-in rich measurement consumer | Enable queue-aware decisions and explicit unknown lanes |
| 6 | Persist rich measurements and decision provenance | Make decisions traceable and performance measurable |
| 7 | Exercise the deployed backend, database and dashboard together | Establish integration acceptance |
| 8 | Evaluate measured discharge, arrivals and spill-back | Later experiment after measurement and safety acceptance |

Each batch should be a small, separately reviewable change with its own checks.
The owners could accept batches 1–4 while retaining the legacy contract. Batches
5–6 are a coordinated extension, and batch 8 should remain disabled by default.

## 1. Enforce phase timing and protect early skips

**Current behavior.** The max-pressure algorithm clamps green time between a
reaction-time setting and a global maximum. With a 3-second reaction time it can
allocate 3 seconds even when the registry specifies a 5-second phase minimum.
The automatic empty-approach skip invokes the next decision immediately, without
waiting for that minimum or explicitly scheduling the outgoing clearance in
that skip path. Manual transitions already have separate behavior and must keep
their ownership protections.

**Proposed change.** Validate phase configuration and apply the selected phase's
minimum and maximum at the shared automatic decision boundary, so every
algorithm and fixed-time fallback receives the same protection. Reject invalid
phase IDs, nonfinite durations and inconsistent timing configuration. A clamp
alone is insufficient: automatic skips must use elapsed green time, wait for the
minimum, and pass through the approved yellow/all-red transition. Preserve
manual-mode ownership and its audit trail; verify every transition route.

**Acceptance checks.** Fake-clock tests cover low/high algorithm outputs,
fallback, rapid zero readings, skip requests before/at/after minimum green,
manual mode and multiple intersections. A skip cannot publish a conflicting next
phase before the required transition finishes. Hardware simulation then verifies
command loss, duplicate commands, restart and acknowledgment behavior with the
signal owner. Software timers do not establish the physical controller's safety.

## 2. Harden the existing legacy receiver

**Current behavior.** Intake checks that a few top-level fields exist and that
`lanes` is an array. It does not enforce complete coverage, lane directions,
integer counts, observation-time bounds or ordering before updating memory and
the dashboard. Several paths use `count || 0`, so unsuitable inputs can become
apparent empty lanes. Memory takes the last arrival as the latest reading.

**Proposed change.** Validate before changing control state, persisting traffic
or broadcasting counts. Require an authorized source for the intersection,
known unique lane IDs with correct directions, nonnegative integer counts, a
valid UTC observation time and a configured age/future-skew limit. Reject an
incomplete legacy snapshot because that format cannot represent unknown lanes.
Check the complete expected stop-line set from an owner-approved coverage
profile; do not infer completeness from whichever lanes arrived.

INT-001's seeded topology lists eight lanes while vision measures nine stop-line
regions. Main currently accepts all nine. Before enforcing topology validation,
the owners must reconcile the physical coverage and registry: update the
approved configuration if all nine regions are correct. Do not discard a real
region or invent replacement counts to satisfy the old seed.

Authorize one active counting producer per intersection/coverage set. Use
vision's session and sequence metadata when present to reject duplicates and
out-of-order messages. Retire previous sessions when an authorized restart is
accepted, and define bounded retention/restart handling so delayed messages from
retired sessions cannot reactivate them. Invalid messages must not advance
ordering state or retire a healthy session. A configured legacy source without
that metadata needs an explicit timestamp-based compatibility policy; timestamps
alone cannot prove exactly-once delivery.

**Acceptance checks.** Exercise malformed arrays, missing/extra/duplicate lanes,
direction changes, null/negative/fractional/string counts, future/stale times,
duplicate sequences, reordered arrivals, session restart and retired-session
replay. Rejected readings must not update memory, refresh freshness, trigger an
empty-phase skip or appear as accepted traffic in the dashboard/database. Test
the agreed nine-region profile and preserve the busiest-lane aggregation.

## 3. Put freshness and camera health on the operator dashboard

**Current behavior.** The camera stream updates on incoming count messages. The
frontend retains its last numbers when messages stop; an open dashboard can
therefore display old counts while its connection still looks healthy. Main
does not subscribe to vision's diagnostic health topic.

**Proposed change.** Consume health separately from counts and expire both by
their own timestamps. Health must never extend a traffic observation's validity.
Publish freshness/fallback updates to already-open dashboard streams even when
no count message arrives. Add a browser-side age timer as well, so a stalled
backend stream cannot leave numbers looking current.

For เจ้าหน้าที่, show physical camera identity, usable/stale/offline/maintenance
state, last observation age, the reason counts are withheld, and whether the
controller is adaptive or in fixed-time fallback. Use Thai labels, visible text
alongside color, and a distinct unknown display rather than `0`. Clearly label
occupancy versus queued vehicles. Keep upstream readings separate from stop-line
totals. If remote maintenance controls are added, require operator authorization,
an audit record and fresh observations before returning to service.

**Acceptance checks.** Keep the dashboard open, stop a required camera, stop only
the upstream camera, kill vision, interrupt MQTT and interrupt the browser's
stream. State must change without a reload, last-known numbers must be marked
stale, and recovery must use new observations. Switching intersections must not
mix their counts or health. Backend/stream connectivity and camera health must
remain distinguishable.

## 4. Separate simulation and deployment; restrict broker access

**Current behavior.** Compose starts a mock count producer by default. The
bundled MQTT listener allows anonymous access on all interfaces. Vision's TLS
support cannot secure a broker whose deployment does not configure it.

**Proposed change.** Make the mock generator an explicit development profile or
separate simulation deployment, and prevent mock INT-001 traffic from entering
the live count stream. Add a production configuration using owner-approved
network boundaries, broker authentication and topic access rules, plus TLS where
required by that deployment. Separate permission to publish counts from
permission to publish signal commands. Fail production startup on known
development secrets or missing required configuration; keep local development
usable through explicit settings. Expose broker and producer status to operators.

**Acceptance checks.** Production startup does not start the mock producer.
Unauthorized clients cannot publish counts or commands; the vision account can
publish only its allowed topics. Verify TLS/certificate failures where enabled,
broker reconnect and old/retained message handling. A rejected or second producer
cannot keep data falsely fresh after the approved producer stops.

## 5. Add an opt-in consumer for vision's rich contract

**Current behavior.** Main's aggregation uses occupancy `count`, including in
functions named for queued traffic. It does not use `queuedCount`, lane validity,
camera health or interval gates. Vision therefore sends complete legacy
occupancy and withholds the whole snapshot when one required reading is invalid.

**Proposed change.** First consume schema 2.0 on an isolated shadow path, validate
it and compare decisions without sending physical commands. Keep legacy support
explicit and preserve the meaning of `count`; changing it silently to queued
vehicles would break existing consumers. Validate schema version, source/session,
per-lane observation times, count invariants and registered coverage.

Queue-based algorithms would use validated `queuedCount`; an empty-approach skip
would use occupancy, since vehicles moving through green are still present.
`unknownStateCount > 0` must not establish that a lane has no demand. Choose and
test an owner-approved conservative policy for uncertain motion state before
queue values affect timing. Exclude upstream regions from stop-line demand and
coverage. Aggregate snapshots with the declared time-window policy rather than
adding repeated occupancy readings as if they were arrivals.

Unknown/missing/stale required lanes must invalidate adaptive demand. Initially,
use whole-intersection fixed-time fallback for incomplete required coverage.
Retain valid readings for diagnostics, but do not let them imply that missing
approaches are empty. Any later partial adaptive policy needs its own acceptance.
Report fallback reasons immediately; transition the active signal according to
the approved timing policy rather than abruptly cutting green on a health event.

**Acceptance checks.** A failed east camera leaves east unknown, never zero;
healthy cameras remain visible, while adaptive timing uses the approved fallback.
A northeast-only upstream outage does not invalidate complete stop-line demand.
Test omitted lanes, unknown motion state, per-lane stale observations inside an
otherwise fresh message, window transitions, and tracker warmup after recovery.
Legacy and rich modes must produce deliberately labelled, comparable results.

## 6. Persist measurements and the evidence behind each decision

**Current behavior.** Traffic persistence stores lane occupancy and direction
totals. It deliberately leaves the raw payload column unwritten, so queue-state
splits, camera validity, session/sequence and interval flow are not preserved in
those reading rows. The existing bounded asynchronous write queue is useful and
should remain: a database outage must not block the decision loop.

**Proposed change.** Add a forward migration for versioned measurement data and
provenance, with an agreed retention/storage policy. Keep legacy occupancy
columns compatible; preserve unknowns explicitly in rich data rather than
coercing them to zero. Record observation/receipt times, source/session/sequence,
model and calibration revision, lane validity, queue-state splits and gate
intervals. Link decisions to their accepted observations, aggregation policy and
fallback reason. Display write failures/dropped records separately from current
camera health and current control status.

Retain `CAM-01` for the existing legacy counting source. If registering rich
source and physical-camera IDs, use intersection-specific identities and verify
ownership/foreign keys. Do not reuse CAM-02…CAM-05 from other intersections or
apply the old experimental migration without reconciliation. A new migration
must upgrade deployed databases; editing an already-applied migration is not
an upgrade strategy.

**Acceptance checks.** Apply the migration to a disposable database with current
main migrations and representative existing rows. Insert/retrieve both formats;
verify null validity and provenance survive. Test database outage, recovery,
write-queue overflow, unknown references and restart without blocking decisions.
Check storage volume and retention with the owners before keeping full payloads.

## 7. Establish real system integration tests and commissioning evidence

**Current evidence.** Vision's pinned-main harness exercises real controller
modules with external I/O replaced. Local RTSP/MQTT recovery is tested, but the
deployed backend process, PostgreSQL writes and rendered dashboard have not been
verified by this review. Passing the harness is not that integration evidence.

**Proposed change.** Add controller-owned regression tests and CI, using the
vision wire/rich fixtures as versioned contract inputs. Run an isolated backend,
temporary broker and disposable PostgreSQL through intake, aggregation,
decision, persistence and dashboard streaming. Add browser checks for stale and
recovered data. Use a signal simulator until the hardware owner approves field
testing. Record rejected messages, observation ages, fallback transitions,
write-queue losses and the distinction between planned and acknowledged phases.

**Acceptance checks.** Run fresh traffic, malformed traffic, incomplete coverage,
camera silence, broker loss, delayed/reordered traffic, database loss and backend
restart. Assert both the decision and the operator-visible explanation. Then
complete a 24–72 hour shadow soak on the target hardware with physical disconnect
tests, synchronized clocks and measured network/NVR delays. Field safety tests
must cover the approved physical controller, pedestrian phases and recovery
policy; their timings cannot be chosen from vision counts alone.

## 8. Evaluate flow-based improvements after the earlier batches

Once labelled count accuracy and gate coverage are accepted, use arrivals,
departures and upstream occupancy to study discharge rates and spill-back.
Record estimates first and compare them with observed traffic before enabling
their use in timing. Do not interpret an uninstrumented gate as zero flow.

A learned discharge sample needs valid gate coverage, demand available at the
interval's start, and an interval entirely inside the relevant green. A message
arriving during green may describe crossings from red, a previous phase or an
outage. Exclude intervals spanning those boundaries unless measurements support
an accurate split. Instrumentation and lane/turn mapping must be agreed on site.

**Acceptance checks.** Compare estimates against labelled crossings and actual
phase history, including no gate, no queue, short greens, red/green boundaries,
camera gaps and recovery. Keep experimental learning behind an explicit disabled
flag until it improves agreed metrics without violating the timing protections.

## Likely system files involved

Paths below are relative to `smart-traffic-sys`. These are change locations, not
files modified by this document.

| Area | Main locations |
|---|---|
| Timing and fallback | `backend/src/services/decision.service.js`, `backend/src/services/skipPhase.service.js`, `backend/src/brain/*`, `backend/src/config/intersectionRegistry.js` |
| Intake and aggregation | `backend/src/services/incomingTraffic.service.js`, `backend/src/models/trafficMemory.model.js`, `backend/src/utils/reliableQueuedAggregator.js` |
| Health subscriptions | `backend/src/server.js`, `backend/src/config/mqtt.config.js`, camera/dashboard services and controllers |
| Operator display | `frontend/src/hooks/useCamera.js`, `frontend/src/hooks/useDashboard.js`, `frontend/src/pages/DashboardPage.jsx`, count/status components |
| Persistence | A new `backend/migrations/*.sql`, `backend/src/repositories/trafficReading.repo.js`, decision repositories; retain `writeQueue.js` behavior |
| Deployment | `docker-compose.yml`, `mosquitto/mosquitto.conf`, environment examples and deployment documentation |
| Verification | Controller-owned tests, backend test command/CI, shared fixtures and disposable integration services |

## Smallest request to take to the system owners

Start with the timing guard/automatic-skip fix, strict legacy intake and complete
coverage validation, visible stale-data state, and production/mock separation.
These address current risks while keeping vision's deployed message format.
Request rich-schema queue control and persistence as a separate, opt-in change.

Even if all these software changes were accepted, field accuracy, live
calibration, dedicated hardware throughput and physical signal/pedestrian
acceptance would still be required. Editing the system repository alone would
not complete production readiness.
