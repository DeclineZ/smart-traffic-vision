// Ends a green phase early when the direction being served has been empty for
// several consecutive readings.
//
// The zero counter is per intersection. As one module-level counter it meant a
// quiet approach at one intersection could skip a phase at another — the kind
// of fault that looks like a controller bug and is not.

const { getSnapshots } = require('../models/trafficMemory.model');
const { getDirFromPhase } = require('../brain/simpleCycleBased.controller');
const { getIntersection } = require('../config/intersectionRegistry');

const THRESHOLD = Number(process.env.SKIP_PHASE_ZERO_THRESHOLD || 2);

/** @type {Map<string, {currentPhase: string|null, consecutiveZeroCount: number}>} */
const trackers = new Map();

function getTracker(intersectionId) {
    let tracker = trackers.get(intersectionId);
    if (!tracker) {
        tracker = { currentPhase: null, consecutiveZeroCount: 0 };
        trackers.set(intersectionId, tracker);
    }
    return tracker;
}

function setCurrentPhase(intersectionId, phase) {
    const tracker = getTracker(intersectionId);
    tracker.currentPhase = phase;
    tracker.consecutiveZeroCount = 0;
    console.log(`[SKIP_PHASE] ${intersectionId} tracking phase=${phase}, counter reset`);
}

function onIncomingPayload(intersectionId) {
    const tracker = trackers.get(intersectionId);
    if (!tracker?.currentPhase) return;

    // Manual mode owns the phase completely. Adaptive decisions and the
    // automatic zero-queue skip must stay silent until the operator returns to
    // auto; otherwise an incoming MQTT reading could undo a manual hold.
    const decisionService = require('./decision.service');
    if (decisionService.isManualOverride(intersectionId)) return;

    const dir = getDirFromPhase(tracker.currentPhase);
    const config = getIntersection(intersectionId);
    if (!config) return;

    // Lazy require to avoid circular dependency
    const snapshots = getSnapshots(intersectionId);
    const agg = decisionService.aggregateWindow(snapshots, config.aggWindowSec, config.freshnessMaxAgeMs);

    if (!agg) return;

    const count = agg.laneTotals[dir] || 0;

    if (count === 0) {
        tracker.consecutiveZeroCount++;
        console.log(
            `[SKIP_PHASE] ${intersectionId} dir=${dir} count=0 (${tracker.consecutiveZeroCount}/${THRESHOLD})`,
        );
    } else {
        if (tracker.consecutiveZeroCount > 0) {
            console.log(`[SKIP_PHASE] ${intersectionId} dir=${dir} count=${count}, counter reset`);
        }
        tracker.consecutiveZeroCount = 0;
    }

    if (tracker.consecutiveZeroCount >= THRESHOLD) {
        console.log(
            `[SKIP_PHASE] ${intersectionId} direction ${dir} had 0 cars for ${THRESHOLD} consecutive checks, skipping ${tracker.currentPhase}`,
        );
        tracker.consecutiveZeroCount = 0;

        try {
            decisionService.triggerSkip({ intersectionId });
        } catch (err) {
            console.error(`[SKIP_PHASE] ${intersectionId} automatic skip failed: ${err.message}`);
        }
    }
}

module.exports = { setCurrentPhase, onIncomingPayload };
