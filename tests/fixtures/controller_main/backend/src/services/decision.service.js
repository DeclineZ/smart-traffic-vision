const { randomUUID } = require('node:crypto');
const { getSnapshots } = require('../models/trafficMemory.model');
const { getMqttClient } = require('../config/mqtt.config');
const {
    getIntersection,
    getIntersections,
    getDefaultIntersectionId,
    toControllerConfig,
} = require('../config/intersectionRegistry');
const { getController } = require('../brain');
const {
    getReliableQueuedAggregation,
    getReliableLaneMaxByDirection,
} = require('../utils/reliableQueuedAggregator');
const { insertDecision } = require('../repositories/decision.repo');
const { closeOpenPhase, openPhase } = require('../repositories/phaseExecution.repo');
const { logEvent, logEventThrottled, resetThrottle } = require('../repositories/systemEvent.repo');
const manualRepo = require('../repositories/manualOverride.repo');
const { waitForIdle } = require('../repositories/writeQueue');

const DECISION_TOPIC = process.env.DECISION_TOPIC;

/**
 * Everything the decision loop remembers about one intersection.
 *
 * This used to be a set of module-level variables, which is exactly as many
 * intersections as the process could control: one. Nothing here is shared
 * between intersections — including `lastServedMs`, which the Max-Pressure
 * controller otherwise keeps in a module-level object, so one intersection's
 * red-wait timers would drive another's Max-Red override.
 *
 * @type {Map<string, object>}
 */
const runtimes = new Map();
const locks = new Map();

function runExclusive(key, operation) {
    const current = (locks.get(key) || Promise.resolve()).catch(() => {}).then(operation);
    const release = () => {
        if (locks.get(key) === current) locks.delete(key);
    };
    locks.set(key, current);
    current.then(release, release);
    return current;
}

function createRuntime(intersection) {
    return {
        config: intersection,
        controller: getController(intersection.controllerName),
        controllerConfig: toControllerConfig(intersection),
        timer: null,
        lastPhase: null,
        currentGreenTimeSec: null,
        lastDecisionPayload: null,
        phaseStartedAtMs: null,
        // Why the phase that is running now will end, and what caused the decision
        // that replaces it. triggerSkip() sets these; the next publishDecision()
        // records them and resets them. Both are constrained by CHECKs in
        // 004_decisions.sql — including decisions_manual_needs_actor, which is why
        // the operator's id has to travel with a manual skip rather than being NULL.
        pendingEndReason: 'natural',
        pendingSource: 'auto',
        pendingActorUserId: null,
        lastServedMs: {},
        // Manual mode is persistent state for this intersection. It is kept in
        // memory for fast status/SSE updates and mirrored by override_sessions
        // in PostgreSQL so it can be restored after a backend restart.
        controlMode: 'auto',
        manualSession: null,
        manualTransition: null,
        transitionTimer: null,
        manualError: null,
    };
}

function getRuntime(intersectionId) {
    const existing = runtimes.get(intersectionId);
    if (existing) return existing;

    const intersection = getIntersection(intersectionId);
    if (!intersection) return null;

    const runtime = createRuntime(intersection);
    runtimes.set(intersectionId, runtime);
    return runtime;
}

function phaseTiming(runtime, phaseId) {
    return runtime.config.phases.find((p) => p.phase === phaseId) ?? null;
}

/** Timing of the phase now running, or the intersection's first phase before one is. */
function currentTiming(runtime) {
    return phaseTiming(runtime, runtime.lastPhase) ?? runtime.config.phases[0];
}

function clearAutoTimer(runtime) {
    if (runtime.timer) clearTimeout(runtime.timer);
    runtime.timer = null;
}

function clearTransitionTimer(runtime) {
    if (runtime.transitionTimer) clearTimeout(runtime.transitionTimer);
    runtime.transitionTimer = null;
}

function autoSignalPhase(runtime) {
    if (!runtime.phaseStartedAtMs || !Number.isFinite(Number(runtime.currentGreenTimeSec))) {
        return 'green';
    }

    const elapsedSec = (Date.now() - runtime.phaseStartedAtMs) / 1000;
    const timing = currentTiming(runtime);
    if (elapsedSec >= Number(runtime.currentGreenTimeSec) + timing.yellowSec) return 'allred';
    if (elapsedSec >= Number(runtime.currentGreenTimeSec)) return 'yellow';
    return 'green';
}

function aggregateWindow(snapshots, windowSec, freshnessMaxAgeMs = Number(process.env.FRESHNESS_MAX_AGE_MS)) {
    const now = Date.now();

    // No snapshots at all
    if (!snapshots || snapshots.length === 0) return null;

    // Check freshness using the latest snapshot we have (most robust)
    const latestSnap = snapshots[snapshots.length - 1];
    const ageMs = now - Number(latestSnap.tsMs || 0);

    if (!Number.isFinite(ageMs) || ageMs > freshnessMaxAgeMs) {
        // Data is stale (older than 2s default) => treat as "no data"
        console.log(
            `[DECISION] stale data: latest age=${ageMs}ms > ${freshnessMaxAgeMs}ms`,
        );
        // Throttled: this is checked on every incoming payload, so an hour of
        // missing data would otherwise be an hour of identical rows.
        logEventThrottled('data_stale', 'warning', latestSnap.payload?.intersectionId, {
            ageMs,
            thresholdMs: freshnessMaxAgeMs,
        });
        return null;
    }

    // Data is flowing again — let the next outage record its own onset.
    resetThrottle('data_stale', latestSnap.payload?.intersectionId);

    // Normal aggregation window
    const start = now - windowSec * 1000;
    const window = snapshots.filter((s) => s.tsMs >= start);
    const reliable = getReliableQueuedAggregation(window, {
        outlierToleranceRatio: 0.7, // 0.7 => keep values within ±70% of median
        minSamples: 3, // if less then 3 samples, then use the latest snapshot
    });
    console.log(`[DEBUG] reliable aggregation: ${JSON.stringify(reliable)}`);

    if (!reliable) return null;

    const laneMax = getReliableLaneMaxByDirection(window, {
        outlierToleranceRatio: 0.7,
        minSamples: 3,
    });

    return {
        latest: window[window.length - 1].payload, // for debug the latest snapshot only
        laneTotals: reliable.laneTotals,
        laneMaxByDirection: laneMax ? laneMax.maxByDir : { N: 0, S: 0, E: 0, W: 0 },
        laneMedians: laneMax ? laneMax.laneMedians : {},
        sampleCount: reliable.sampleCount,
        reliability: reliable.reliability,
    };
}

function getStatus(intersectionId = getDefaultIntersectionId()) {
    const runtime = getRuntime(intersectionId);
    if (!runtime) return null;

    const phaseIds = runtime.config.phases.map((p) => p.phase);
    let nextPhase = null;
    const strategy = runtime.lastDecisionPayload?.decision?.strategy;
    if (runtime.lastPhase && runtime.controlMode === 'auto' && strategy !== 'MAXPRESSURE_SWITCHING_LOSS') {
        nextPhase = phaseIds[(phaseIds.indexOf(runtime.lastPhase) + 1) % phaseIds.length];
    }

    const timing = currentTiming(runtime);
    const transition = runtime.manualTransition
        ? {
            phase: runtime.manualTransition.stage,
            targetPhase: runtime.manualTransition.targetPhase,
            reason: runtime.manualTransition.reason,
            startedAt: new Date(runtime.manualTransition.startedAtMs).toISOString(),
            endsAt: new Date(runtime.manualTransition.endsAtMs).toISOString(),
        }
        : null;
    const signalPhase = transition?.phase
        ?? (runtime.controlMode === 'manual' ? 'green' : autoSignalPhase(runtime));
    const manualSession = runtime.manualSession
        ? {
            id: runtime.manualSession.id,
            ownerUsername: runtime.manualSession.ownerUsername,
            startedAt: runtime.manualSession.startedAt,
        }
        : null;

    return {
        intersectionId: runtime.config.id,
        intersectionName: runtime.config.name,
        currentPhase: runtime.lastPhase,
        nextPhase,
        greenTimeSec: runtime.currentGreenTimeSec,
        // The dashboard needs these to draw the countdown. Sending them keeps the
        // backend the only place they are defined — hardcoding them in the
        // frontend meant a change here silently desynced the operator's clock.
        // They come from intersection_phases rather than a global env var, because
        // five intersections cannot share one yellow time (ADR-004).
        yellowTimeSec: timing.yellowSec,
        allRedTimeSec: timing.allRedSec,
        phaseStartedAt: runtime.phaseStartedAtMs
            ? new Date(runtime.phaseStartedAtMs).toISOString()
            : null,
        lastDecision: runtime.lastDecisionPayload ? runtime.lastDecisionPayload.decision : null,
        controlMode: transition ? 'transition' : runtime.controlMode,
        manualHold: runtime.controlMode === 'manual',
        manualSession,
        signalPhase,
        transition,
        manualError: runtime.manualError,
    };
}

/**
 * Every intersection's status in one object, keyed by id.
 *
 * The dashboard needs all of them at once even though it shows one: the tab bar
 * carries a live phase indicator for each. Five statuses are a few hundred
 * bytes, so one shared stream costs less than five subscriptions that each have
 * to be torn down and rebuilt on every tab switch.
 */
function getAllStatuses() {
    const statuses = {};
    for (const intersection of getIntersections()) {
        statuses[intersection.id] = getStatus(intersection.id);
    }
    return statuses;
}

const sseClients = new Set();

function statusSnapshot() {
    return { intersections: getAllStatuses() };
}

function broadcastSse() {
    const data = JSON.stringify(statusSnapshot());
    for (const client of sseClients) {
        client.write(`data: ${data}\n\n`);
    }
}

function registerSseClient(res) {
    sseClients.add(res);
    res.on('close', () => sseClients.delete(res));
}

/**
 * @param {object}  options
 * @param {string}  [options.intersectionId] defaults to the first intersection
 * @param {'auto'|'manual'} [options.mode]
 * @param {number|null} [options.actorUserId] required for a manual skip: a
 *        decision with source='skip_manual' and no actor is rejected by the
 *        decisions_manual_needs_actor constraint.
 */
function triggerSkip({ intersectionId = getDefaultIntersectionId(), mode = 'auto', actorUserId = null } = {}) {
    const runtime = getRuntime(intersectionId);
    if (!runtime) return null;

    // Fully manual mode intentionally has no automatic skip-phase path. The
    // operator changes phase through the manual command endpoint, and the
    // current phase remains held until then.
    if (runtime.controlMode !== 'auto' || runtime.manualTransition) {
        logEvent('phase_skip_rejected', 'info', runtime.config.id, {
            mode,
            phase: runtime.lastPhase,
            actorUserId,
            reason: 'manual_override_active',
        });
        return getStatus(runtime.config.id);
    }

    // Recorded against the phase being cut short; pendingSource describes the
    // decision that replaces it.
    //
    // A manual skip does not take effect immediately — it shortens the green and
    // lets the timer run out — so the automatic skip check keeps running in the
    // gap and used to overwrite this slot before the decision was written. The
    // operator's action was then filed as `skip_auto` with no actor, which is
    // precisely what decisions_manual_needs_actor exists to prevent. Once a
    // person has asked for the phase to end, that is why it ended.
    const manualPending = runtime.pendingSource === 'skip_manual';
    if (mode === 'manual' || !manualPending) {
        runtime.pendingEndReason = mode === 'manual' ? 'skipped_manual' : 'skipped_auto';
        runtime.pendingSource = mode === 'manual' ? 'skip_manual' : 'skip_auto';
        runtime.pendingActorUserId = mode === 'manual' ? actorUserId : null;
    }

    logEvent('phase_skipped', 'info', runtime.config.id, {
        mode,
        phase: runtime.lastPhase,
        actorUserId,
        // true when this automatic skip deferred to an operator's pending one.
        deferredToManual: mode !== 'manual' && manualPending,
    });

    if (mode === 'manual') {
        if (!runtime.phaseStartedAtMs || !runtime.currentGreenTimeSec) {
            console.log(`[DECISION] ${runtime.config.id} manual skip — no active phase, running next decision immediately`);
            if (runtime.timer) clearTimeout(runtime.timer);
            runDecisionForIntersection(runtime.config.id).catch((err) => {
                console.error(`[DECISION] ${runtime.config.id} manual skip error: ${err.message}`);
            });
            return getStatus(runtime.config.id);
        }

        const timing = currentTiming(runtime);
        const elapsedSec = (Date.now() - runtime.phaseStartedAtMs) / 1000;
        const remainingGreenSec = runtime.currentGreenTimeSec - elapsedSec;
        const minGreenSec = timing.minGreenSec;

        if (remainingGreenSec > minGreenSec) {
            console.log(`[DECISION] ${runtime.config.id} manual skip — truncating green from ${remainingGreenSec.toFixed(1)}s to ${minGreenSec}s`);
            runtime.currentGreenTimeSec = Math.round(elapsedSec + minGreenSec);

            const newDelayMs = (minGreenSec + timing.yellowSec + timing.allRedSec) * 1000;

            scheduleNextDecision(runtime, newDelayMs);
            broadcastSse();
        } else {
            console.log(`[DECISION] ${runtime.config.id} manual skip — remaining green (${remainingGreenSec.toFixed(1)}s) <= ${minGreenSec}s, finishing naturally`);
            // The request was declined, so the phase really does end naturally
            // and the next decision is an ordinary automatic one.
            runtime.pendingEndReason = 'natural';
            runtime.pendingSource = 'auto';
            runtime.pendingActorUserId = null;
        }

        return getStatus(runtime.config.id);
    }

    console.log(`[DECISION] ${runtime.config.id} skip triggered — running next decision immediately`);
    if (runtime.timer) clearTimeout(runtime.timer);
    runDecisionForIntersection(runtime.config.id).catch((err) => {
        console.error(`[DECISION] ${runtime.config.id} skip error: ${err.message}`);
    });
    return getStatus(runtime.config.id);
}

function publishDecision(runtime, payload, log = {}) {
    const client = getMqttClient();
    client.publish(DECISION_TOPIC, JSON.stringify(payload), {
        qos: 1,
        retain: false,
    });

    const startedAt = new Date();
    const { intersectionId, decision } = payload;
    const fromPhase = decision.meta?.from ?? runtime.lastDecisionPayload?.decision?.phase ?? null;

    runtime.currentGreenTimeSec = decision.greenTimeSec;
    runtime.lastDecisionPayload = payload;
    runtime.phaseStartedAtMs = startedAt.getTime();
    runtime.controlMode = 'auto';
    runtime.manualError = null;
    broadcastSse();

    // Timing of the phase being started, not a global constant: the five
    // intersections do not share a yellow or all-red time (ADR-004).
    const timing = phaseTiming(runtime, decision.phase) ?? runtime.config.phases[0];
    const totalMs =
        (runtime.currentGreenTimeSec + timing.yellowSec + timing.allRedSec) * 1000;

    // Order matters and the write queue preserves it: the decision has to exist
    // before phase_executions can reference it, and the previous phase has to be
    // closed before a new one opens or the one-open-per-intersection index
    // rejects the insert.
    insertDecision({
        decisionId: payload.decisionId,
        decidedAt: startedAt,
        intersectionId,
        strategy: decision.strategy,
        phase: decision.phase,
        fromPhase,
        greenTimeSec: decision.greenTimeSec,
        // A fallback overrides the pending source: fallback_reason is only
        // allowed alongside source='fallback' (decisions_fallback_reason_needs_source).
        source: log.source || runtime.pendingSource,
        fallbackReason: log.fallbackReason,
        actorUserId: runtime.pendingActorUserId,
        agg: log.agg,
        meta: decision.meta,
    });

    // The moment one phase ends is the moment the next begins, so both sides use
    // the same timestamp — otherwise the gap shows up as drift.
    closeOpenPhase(intersectionId, runtime.pendingEndReason, startedAt);
    runtime.pendingEndReason = 'natural';
    runtime.pendingSource = 'auto';
    runtime.pendingActorUserId = null;

    openPhase({
        intersectionId,
        decisionId: payload.decisionId,
        phase: decision.phase,
        fromPhase,
        startedAt,
        plannedGreenSec: runtime.currentGreenTimeSec,
        plannedTotalSec: totalMs / 1000,
    });

    console.log(
        `[NEXT] ${intersectionId} decision in ${totalMs}ms (green=${runtime.currentGreenTimeSec}s)`,
    );

    scheduleNextDecision(runtime, totalMs);
}

function scheduleNextDecision(runtime, delayMs) {
    if (runtime.timer) {
        clearTimeout(runtime.timer);
    }

    if (runtime.controlMode !== 'auto' || runtime.manualTransition) return;

    runtime.timer = setTimeout(() => {
        runtime.timer = null;
        runDecisionForIntersection(runtime.config.id).catch((err) => {
            console.error(`[DECISION] ${runtime.config.id} loop error: ${err.message}`);
        });
    }, delayMs);
}

function runDecisionForIntersection(intersectionId, aggregationWindowSec = null) {
    return runExclusive(intersectionId, () => runDecisionForIntersectionUnsafe(intersectionId, aggregationWindowSec));
}

function runDecisionForIntersectionUnsafe(intersectionId, aggregationWindowSec = null) {
    const runtime = getRuntime(intersectionId);
    if (!runtime) {
        console.error(`[DECISION] unknown intersection ${intersectionId} — no decision made`);
        return;
    }

    if (runtime.controlMode !== 'auto' || runtime.manualTransition) return;

    if (aggregationWindowSec === null) {
        aggregationWindowSec = runtime.config.aggWindowSec;
    }

    const snapshots = getSnapshots(intersectionId);
    const agg = aggregateWindow(snapshots, aggregationWindowSec, runtime.config.freshnessMaxAgeMs);

    // Shared by both branches so the controller sees this intersection's own
    // red-wait timers rather than a module-level object shared by all five.
    const state = { lastPhase: runtime.lastPhase, lastServedMs: runtime.lastServedMs };

    // If agg is null => fallback (or you can use FIXED_CYCLE as fallback controller)
    if (!agg) {
        const fixed = getController('FIXED_CYCLE');
        const decision = fixed.decide({
            intersectionId,
            agg: null,
            intersectionConfig: runtime.controllerConfig,
            state,
            nowMs: Date.now(),
        });

        const decisionPayload = {
            version: '1.0',
            decisionId: randomUUID(),
            timestamp: new Date().toISOString(),
            intersectionId,
            decision,
        };

        logEventThrottled('fallback_triggered', 'warning', intersectionId, {
            reason: 'stale_data',
            phase: decision.phase,
        });

        runtime.lastPhase = decision.phase;
        require('./skipPhase.service').setCurrentPhase(intersectionId, decision.phase);
        publishDecision(runtime, decisionPayload, { source: 'fallback', fallbackReason: 'stale_data' });
        return;
    }
    // Adaptive / selected controller
    const decision = runtime.controller.decide({
        intersectionId,
        agg,
        intersectionConfig: runtime.controllerConfig,
        state,
        nowMs: Date.now(),
    });

    const decisionPayload = {
        version: '1.0',
        decisionId: randomUUID(),
        timestamp: new Date().toISOString(),
        intersectionId,
        decision,
    };

    runtime.lastPhase = decision.phase;
    require('./skipPhase.service').setCurrentPhase(intersectionId, decision.phase);
    publishDecision(runtime, decisionPayload, { agg });
}

// Each intersection's first decision is offset from the last, so five loops
// with the same phase lengths do not settle into deciding, publishing and
// writing on the same tick for ever.
const STAGGER_MS = Number(process.env.DECISION_STAGGER_MS || 200);

function startDecisionLoop() {
    const intersections = getIntersections();
    if (intersections.length === 0) {
        console.error('[DECISION] registry is empty — no intersection to control');
        return;
    }

    const initialDelayMs = Number(process.env.INITIAL_DECISION_DELAY_MS);

    console.log(
        `Decision loop starting | ${intersections.length} intersection(s) | topic=${DECISION_TOPIC} | initial delay=${initialDelayMs}ms | stagger=${STAGGER_MS}ms`,
    );

    intersections.forEach((intersection, index) => {
        const runtime = getRuntime(intersection.id);
        const delayMs = initialDelayMs + index * STAGGER_MS;
        console.log(
            `[DECISION] ${runtime.config.id} | controller=${runtime.config.controllerName} | window=${runtime.config.aggWindowSec}s | phases=${runtime.config.phases.length} | first decision in ${delayMs}ms`,
        );
        if (runtime.controlMode === 'auto') scheduleNextDecision(runtime, delayMs);
    });
}

function operatorError(code, message, details = {}) {
    const err = new Error(message);
    err.code = code;
    Object.assign(err, details);
    return err;
}

async function waitForAuditWrites(message = 'automatic audit writes are still queued') {
    if (!(await waitForIdle(10))) throw operatorError('DB_QUEUE_BUSY', message);
}

function publishManualDecision(payload) {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => resolve(false), 2000);
        getMqttClient().publish(DECISION_TOPIC, JSON.stringify(payload), {
            qos: 1,
            retain: false,
        }, (err) => {
            clearTimeout(timeout);
            if (err) reject(err);
            else resolve(true);
        });
    });
}

async function publishAndRememberManual(runtime, payload) {
    runtime.lastPhase = payload.decision.phase;
    runtime.currentGreenTimeSec = null;
    runtime.lastDecisionPayload = payload;
    runtime.phaseStartedAtMs = new Date(payload.timestamp).getTime();
    runtime.manualError = null;
    require('./skipPhase.service').setCurrentPhase(runtime.config.id, payload.decision.phase);
    broadcastSse();

    try {
        const published = await publishManualDecision(payload);
        if (!published) {
            runtime.manualError = 'บันทึกคำสั่งแล้ว แต่ยังส่งคำสั่งไป MQTT ไม่สำเร็จ';
            logEvent('manual_mqtt_publish_timeout', 'error', runtime.config.id, {
                decisionId: payload.decisionId,
            });
            broadcastSse();
        }
    } catch (err) {
        runtime.manualError = `บันทึกคำสั่งแล้ว แต่ส่ง MQTT ไม่สำเร็จ: ${err.message}`;
        logEvent('manual_mqtt_publish_failed', 'error', runtime.config.id, {
            decisionId: payload.decisionId,
            error: err.message,
        });
        broadcastSse();
    }
}

async function applyManualPhase(runtime, { session, userId, phase, fromPhase, recordCommand = false, meta = null }) {
    const decisionId = randomUUID();
    const timestamp = new Date();
    try {
        await waitForAuditWrites('automatic audit writes are still queued; manual command was not applied');
        await manualRepo.applyManualPhase({
            decisionId,
            decisionAt: timestamp,
            intersectionId: runtime.config.id,
            sessionId: session.id,
            userId,
            phase,
            fromPhase,
            recordCommand,
            meta: meta || { from: fromPhase || null },
        });
    } catch (err) {
        if (recordCommand) {
            await manualRepo.rejectCommand({
                sessionId: session.id,
                intersectionId: runtime.config.id,
                userId,
                requestedPhase: phase,
                reason: `apply_failed:${err.code || 'database_error'}`,
            }).catch(() => {});
        }
        throw err;
    }

    const payload = {
        version: '1.1',
        decisionId,
        timestamp: timestamp.toISOString(),
        intersectionId: runtime.config.id,
        decision: {
            strategy: 'MANUAL_OVERRIDE',
            source: 'manual',
            phase,
            greenTimeSec: null,
            manualHold: true,
            meta: { from: fromPhase || null },
        },
    };
    await publishAndRememberManual(runtime, payload);
}

async function activateManualSession(runtime, { session, userId, action }) {
    const intersectionId = runtime.config.id;
    const phase = runtime.lastPhase;
    runtime.manualSession = session;
    runtime.controlMode = 'manual';
    runtime.manualError = null;

    try {
        await applyManualPhase(runtime, {
            session,
            userId,
            phase,
            fromPhase: phase,
            recordCommand: true,
            meta: { from: phase, action },
        });
        logEvent(action === 'start' ? 'manual_session_started' : 'manual_session_takeover', 'info', intersectionId, {
            sessionId: session.id,
            actorUserId: userId,
            phase,
            ...(action === 'takeover' ? { reason: 'takeover' } : {}),
        });
        return getStatus(intersectionId);
    } catch (err) {
        await manualRepo.closeSessionBySystem({ sessionId: session.id, reason: `${action}_failed` }).catch(() => {});
        runtime.manualSession = null;
        runtime.controlMode = 'auto';
        scheduleNextDecision(runtime, 1000);
        throw err;
    }
}

function scheduleTransitionStep(runtime, expectedStage, delayMs) {
    runtime.transitionTimer = setTimeout(() => {
        runtime.transitionTimer = null;
        advanceManualTransition(runtime.config.id, expectedStage).catch((err) => {
            console.error(`[MANUAL] ${runtime.config.id} transition error: ${err.message}`);
        });
    }, Math.max(0, delayMs));
}

/**
 * Tells the signal to end the phase it is holding and run its clearance.
 *
 * A manual hold is published with greenTimeSec null, which the receiver reads
 * as "keep this phase until told otherwise" — so it has nothing to start a
 * yellow from. Without this message the backend would sit through yellow and
 * all-red locally, show them on the dashboard, and then publish the next phase,
 * leaving the street to go green to green with no clearance at all.
 *
 * greenTimeSec 0 is deliberately the same mechanism automatic decisions already
 * use: serve this much more green, then run yellow and all-red. Zero more
 * seconds is exactly a clearance, so nothing new has to be understood — but the
 * receiver must read 0 as a number and not as a missing value.
 */
function publishClearance(runtime, { yellowSec, allRedSec, actorUserId, reason }) {
    const payload = {
        version: '1.1',
        decisionId: randomUUID(),
        timestamp: new Date().toISOString(),
        intersectionId: runtime.config.id,
        decision: {
            strategy: 'MANUAL_OVERRIDE',
            source: 'manual',
            phase: runtime.lastPhase,
            greenTimeSec: 0,
            manualHold: false,
            meta: { from: runtime.lastPhase, clearance: true, reason, yellowSec, allRedSec },
        },
    };

    try {
        getMqttClient().publish(DECISION_TOPIC, JSON.stringify(payload), { qos: 1, retain: false });
    } catch (err) {
        runtime.manualError = `ส่งคำสั่งไฟเหลืองไป MQTT ไม่สำเร็จ: ${err.message}`;
        logEvent('manual_clearance_publish_failed', 'error', runtime.config.id, {
            phase: runtime.lastPhase,
            error: err.message,
        });
        return;
    }

    // Recorded as an event rather than a decision: it starts no phase execution
    // and serves no green, so it is not an intent the decisions table describes.
    // It is still what went out on the wire, so it has to be traceable.
    logEvent('manual_clearance_published', 'info', runtime.config.id, {
        decisionId: payload.decisionId,
        phase: runtime.lastPhase,
        yellowSec,
        allRedSec,
        actorUserId,
        reason,
    });
}

function beginTransition(runtime, { targetPhase = null, actorUserId = null, reason }) {
    clearAutoTimer(runtime);
    clearTransitionTimer(runtime);

    const timing = currentTiming(runtime) || { yellowSec: 0, allRedSec: 0 };
    const yellowSec = Number(timing.yellowSec || 0);
    const allRedSec = Number(timing.allRedSec || 0);
    const now = Date.now();
    runtime.manualTransition = {
        stage: 'yellow',
        targetPhase,
        actorUserId,
        reason,
        fromPhase: runtime.lastPhase,
        startedAtMs: now,
        endsAtMs: now + (yellowSec + allRedSec) * 1000,
        yellowSec,
        allRedSec,
    };
    runtime.controlMode = 'transition';
    // Sent before the local timers start, so the signal runs its clearance over
    // the same seconds the dashboard is showing rather than after them.
    publishClearance(runtime, { yellowSec, allRedSec, actorUserId, reason });
    broadcastSse();
    scheduleTransitionStep(runtime, 'yellow', yellowSec * 1000);
}

async function advanceManualTransition(intersectionId, expectedStage) {
    return runExclusive(intersectionId, async () => {
        const runtime = getRuntime(intersectionId);
        const transition = runtime?.manualTransition;
        if (!runtime || !transition || transition.stage !== expectedStage) return;

        if (expectedStage === 'yellow') {
            transition.stage = 'allred';
            broadcastSse();
            scheduleTransitionStep(
                runtime,
                'allred',
                Math.max(0, transition.endsAtMs - Date.now()),
            );
            return;
        }

        if (transition.reason === 'return_to_auto') {
            runtime.manualTransition = null;
            runtime.controlMode = 'auto';
            runtime.manualSession = null;
            runtime.manualError = null;
            runtime.pendingEndReason = 'override';
            runtime.pendingSource = 'auto';
            runtime.pendingActorUserId = null;
            broadcastSse();
            // Already inside this intersection's mutex; avoid acquiring it again.
            runDecisionForIntersectionUnsafe(intersectionId);
            return;
        }

        const session = runtime.manualSession;
        const targetPhase = transition.targetPhase;
        const fromPhase = transition.fromPhase;
        try {
            await applyManualPhase(runtime, {
                session,
                userId: transition.actorUserId,
                phase: targetPhase,
                fromPhase,
                recordCommand: true,
                meta: { from: fromPhase || null },
            });
            runtime.manualTransition = null;
            runtime.controlMode = 'manual';
            broadcastSse();
        } catch (err) {
            runtime.manualTransition = null;
            runtime.controlMode = 'manual';
            runtime.manualError = `เปลี่ยน phase ไม่สำเร็จ: ${err.message}`;
            logEvent('manual_command_apply_failed', 'error', intersectionId, {
                phase: targetPhase,
                error: err.message,
            });
            broadcastSse();
        }
    });
}

async function openManualOverride({ intersectionId = getDefaultIntersectionId(), userId, action }) {
    return runExclusive(intersectionId, async () => {
        const runtime = getRuntime(intersectionId);
        if (!runtime) throw operatorError('UNKNOWN_INTERSECTION', 'unknown intersection');
        if (runtime.manualTransition) {
            throw operatorError('TRANSITION_IN_PROGRESS', 'intersection is in transition', { status: getStatus(intersectionId) });
        }

        const existing = await manualRepo.getOpenSession(intersectionId);
        if (existing?.startedBy === userId && runtime.controlMode === 'manual') {
            runtime.manualSession = existing;
            return getStatus(intersectionId);
        }
        if (existing && action === 'start') {
            throw operatorError('OVERRIDE_SESSION_EXISTS', 'another operator owns the manual session', {
                status: getStatus(intersectionId),
            });
        }
        if (!existing && action === 'takeover') {
            throw operatorError('NO_OVERRIDE_SESSION', 'there is no open manual session to take over');
        }
        if (!runtime.lastPhase) {
            throw operatorError('ACTIVE_PHASE_REQUIRED', 'cannot enter manual mode without an active phase');
        }
        // Entering manual publishes "hold this phase", which would cut short a
        // clearance the signal is already running and send it back to green
        // without the all-red finishing. Waiting the couple of seconds for the
        // automatic cycle to reach green is the difference between a hold and a
        // conflicting movement. Checked here rather than in the browser: the
        // dashboard would be guessing from its own clock, and only the backend
        // knows when the phase actually started.
        if (autoSignalPhase(runtime) !== 'green') {
            throw operatorError(
                'CLEARANCE_IN_PROGRESS',
                'แยกนี้กำลังอยู่ในช่วงไฟเหลือง/all-red — รอไฟเขียวรอบถัดไปแล้วกดใหม่',
                { status: getStatus(intersectionId) },
            );
        }

        clearAutoTimer(runtime);
        await waitForAuditWrites();

        const session = action === 'start'
            ? await manualRepo.createSession({ intersectionId, userId })
            : await manualRepo.takeoverSession({ intersectionId, userId });
        return activateManualSession(runtime, { session, userId, action });
    });
}

async function commandManualOverride({ intersectionId = getDefaultIntersectionId(), userId, phase }) {
    return runExclusive(intersectionId, async () => {
        const runtime = getRuntime(intersectionId);
        if (!runtime) throw operatorError('UNKNOWN_INTERSECTION', 'unknown intersection');
        const requestedPhase = typeof phase === 'string' ? phase.trim() : '';
        const session = await manualRepo.getOpenSession(intersectionId);
        const reject = async (reason, code, message) => {
            await manualRepo.rejectCommand({
                sessionId: session?.id,
                intersectionId,
                userId,
                requestedPhase,
                reason,
            }).catch(() => {});
            throw operatorError(code, message, { status: getStatus(intersectionId) });
        };

        if (!session) {
            return reject('no_manual_session', 'NO_OVERRIDE_SESSION', 'manual session is not open');
        }
        if (session.startedBy !== userId) {
            return reject('not_session_owner', 'NOT_SESSION_OWNER',
                'only the current session owner can issue phase commands');
        }
        if (runtime.manualTransition) {
            return reject('transition_in_progress', 'TRANSITION_IN_PROGRESS',
                'another manual transition is still running');
        }
        if (runtime.controlMode !== 'manual' || runtime.manualSession?.id !== session.id) {
            return reject('manual_session_not_active', 'MANUAL_SESSION_NOT_ACTIVE',
                'manual session is not active in this backend process');
        }

        if (!phaseTiming(runtime, requestedPhase)) {
            return reject('unknown_phase', 'UNKNOWN_PHASE',
                `phase "${requestedPhase}" is not configured for this intersection`);
        }
        if (!runtime.lastPhase) {
            return reject('no_active_phase', 'ACTIVE_PHASE_REQUIRED', 'there is no active phase to transition from');
        }

        if (requestedPhase === runtime.lastPhase) {
            await applyManualPhase(runtime, {
                session,
                userId,
                phase: requestedPhase,
                fromPhase: runtime.lastPhase,
                recordCommand: true,
                meta: { from: runtime.lastPhase, action: 'reassert' },
            });
            return getStatus(intersectionId);
        }

        beginTransition(runtime, {
            targetPhase: requestedPhase,
            actorUserId: userId,
            reason: 'phase_change',
        });
        return getStatus(intersectionId);
    });
}

async function stopManualOverride({ intersectionId = getDefaultIntersectionId(), userId }) {
    return runExclusive(intersectionId, async () => {
        const runtime = getRuntime(intersectionId);
        if (!runtime) throw operatorError('UNKNOWN_INTERSECTION', 'unknown intersection');
        if (runtime.manualTransition) {
            throw operatorError('TRANSITION_IN_PROGRESS', 'cannot return to auto during a transition', {
                status: getStatus(intersectionId),
            });
        }

        const session = await manualRepo.getOpenSession(intersectionId);
        if (!session) {
            if (runtime.controlMode === 'manual') {
                runtime.controlMode = 'auto';
                runtime.manualSession = null;
            }
            return getStatus(intersectionId);
        }
        if (session.startedBy !== userId) {
            throw operatorError('NOT_SESSION_OWNER', 'only the current session owner can return to auto', {
                status: getStatus(intersectionId),
            });
        }

        await waitForAuditWrites();
        await manualRepo.closeSession({ sessionId: session.id, userId, reason: 'operator_exit' });
        runtime.manualSession = null;

        if (!runtime.lastPhase) {
            runtime.controlMode = 'auto';
            runDecisionForIntersectionUnsafe(intersectionId);
            return getStatus(intersectionId);
        }

        beginTransition(runtime, {
            targetPhase: null,
            actorUserId: userId,
            reason: 'return_to_auto',
        });
        return getStatus(intersectionId);
    });
}

/** Rehydrate open manual sessions before the automatic loop schedules timers. */
async function restoreManualSessions() {
    let sessions;
    try {
        sessions = await manualRepo.getOpenSessions();
    } catch (err) {
        console.error(`[MANUAL] could not load open sessions: ${err.message}`);
        return;
    }

    for (const session of sessions) {
        await runExclusive(session.intersectionId, async () => {
            const runtime = getRuntime(session.intersectionId);
            if (!runtime) {
                await manualRepo.closeSessionBySystem({
                    sessionId: session.id,
                    reason: 'recovery_unknown_intersection',
                }).catch(() => {});
                return;
            }

            const latest = await manualRepo.getLatestManualState(session.intersectionId);
            if (!latest || !phaseTiming(runtime, latest.phase)) {
                await manualRepo.closeSessionBySystem({
                    sessionId: session.id,
                    reason: 'recovery_no_phase',
                }).catch(() => {});
                logEvent('manual_session_recovery_failed', 'error', session.intersectionId, {
                    sessionId: session.id,
                    reason: 'phase_not_restorable',
                    phase: latest?.phase || null,
                });
                return;
            }

            clearAutoTimer(runtime);
            runtime.manualSession = session;
            runtime.controlMode = 'manual';
            runtime.lastPhase = latest.phase;
            runtime.currentGreenTimeSec = null;
            runtime.manualError = null;
            try {
                await applyManualPhase(runtime, {
                    session,
                    userId: session.startedBy,
                    phase: latest.phase,
                    fromPhase: latest.phase,
                    meta: { from: latest.phase, action: 'restart_restore' },
                });
                logEvent('manual_session_restored', 'info', session.intersectionId, {
                    sessionId: session.id,
                    phase: latest.phase,
                });
            } catch (err) {
                await manualRepo.closeSessionBySystem({
                    sessionId: session.id,
                    reason: 'recovery_apply_failed',
                }).catch(() => {});
                runtime.manualSession = null;
                runtime.controlMode = 'auto';
                runtime.lastPhase = null;
                runtime.currentGreenTimeSec = null;
                logEvent('manual_session_recovery_failed', 'error', session.intersectionId, {
                    sessionId: session.id,
                    reason: err.message,
                });
            }
        }).catch((err) => {
            console.error(`[MANUAL] restore ${session.intersectionId} failed: ${err.message}`);
        });
    }
}

module.exports = {
    startDecisionLoop,
    aggregateWindow,
    triggerSkip,
    getStatus,
    getAllStatuses,
    statusSnapshot,
    registerSseClient,
    openManualOverride,
    commandManualOverride,
    stopManualOverride,
    restoreManualSessions,
    isManualOverride: (intersectionId) => {
        const runtime = getRuntime(intersectionId);
        return Boolean(runtime && (runtime.controlMode === 'manual' || runtime.manualTransition));
    },
};
