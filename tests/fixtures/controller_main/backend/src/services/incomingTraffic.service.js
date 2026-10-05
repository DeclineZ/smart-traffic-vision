const { logTraffic } = require('../utils/logger');
const { addSnapshot, getLatestSnapshot } = require('../models/trafficMemory.model');
const { insertReading } = require('../repositories/trafficReading.repo');

// One set of listeners per intersection. A single shared set sent every
// intersection's lane counts to every open dashboard, so the counts on screen
// would be whichever intersection reported last.
/** @type {Map<string, Set<import('http').ServerResponse>>} */
const cameraClients = new Map();

function basicValidate(payload) {
    if (!payload) throw new Error('Missing payload');
    if (!payload.intersectionId) throw new Error('Missing intersectionId');
    if (!payload.cameraId) throw new Error('Missing cameraId');
    if (!payload.timestamp) throw new Error('Missing timestamp');
    if (!payload.lanes || !Array.isArray(payload.lanes))
        throw new Error('Missing lanes[]');
    return true;
}

function buildCameraData(payload) {
    const directionTotals = {};
    for (const lane of payload.lanes) {
        const dir = lane.direction;
        directionTotals[dir] = (directionTotals[dir] || 0) + Number(lane.count || 0);
    }
    return {
        intersectionId: payload.intersectionId,
        cameraId: payload.cameraId,
        timestamp: payload.timestamp,
        lanes: payload.lanes,
        directionTotals,
    };
}

function broadcastCamera(payload) {
    const clients = cameraClients.get(payload.intersectionId);
    if (!clients || clients.size === 0) return;

    const data = JSON.stringify(buildCameraData(payload));
    for (const client of clients) {
        client.write(`data: ${data}\n\n`);
    }
}

function registerCameraClient(intersectionId, res) {
    let clients = cameraClients.get(intersectionId);
    if (!clients) {
        clients = new Set();
        cameraClients.set(intersectionId, clients);
    }
    clients.add(res);

    // Switching tabs closes one stream and opens another, so an empty set is
    // the common case rather than the rare one — drop it instead of keeping an
    // entry per intersection the operator has ever looked at.
    res.on('close', () => {
        clients.delete(res);
        if (clients.size === 0) cameraClients.delete(intersectionId);
    });
}

function getLatestCameraData(intersectionId) {
    const snap = getLatestSnapshot(intersectionId);
    if (!snap) return null;
    return buildCameraData(snap.payload);
}

async function handleIncomingCounts(payload) {
    basicValidate(payload);
    addSnapshot(payload);
    broadcastCamera(payload);
    logTraffic('CAR_COUNTS_RECEIVED', payload);

    // Reuses the direction totals buildCameraData() already computed for the
    // SSE stream rather than summing the lanes a second time. Queued, so a
    // database outage cannot slow down or break MQTT ingest.
    insertReading(payload, buildCameraData(payload).directionTotals);

    const total = payload.lanes.reduce(
        (sum, lane) => sum + Number(lane.count || 0),
        0,
    );

    return {
        intersectionId: payload.intersectionId,
        cameraId: payload.cameraId,
        totalCars: total,
        laneCount: payload.lanes.length,
        timestamp: payload.timestamp,
    };
}

module.exports = { handleIncomingCounts, registerCameraClient, getLatestCameraData };
