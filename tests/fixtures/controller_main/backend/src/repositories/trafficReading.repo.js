// The highest-volume table: one row per camera payload, ~216k rows/day per
// intersection. See SCHEMA.md for why it is partitioned by month.

const { enqueue } = require('./writeQueue');
const { logEventThrottled } = require('./systemEvent.repo');

// total_n/e/s/w are generated columns — listing them here would be an error.
// raw is deliberately left NULL; see the note in 003_traffic_readings.sql.
const INSERT_SQL = `
    INSERT INTO traffic_readings
        (recorded_at, intersection_id, camera_id, frame_id, lane_counts, direction_totals)
    VALUES ($1, $2, $3, $4, $5, $6)
`;

const MAX_CLOCK_SKEW_MS = 24 * 60 * 60 * 1000;

/**
 * A camera clock that is days off would drop rows into
 * traffic_readings_default, and a month with rows sitting in the default
 * partition can no longer have its own partition attached. Server time is the
 * safer wrong answer: ingested_at already records it anyway.
 */
function resolveRecordedAt(timestamp, intersectionId, cameraId) {
    const parsed = Date.parse(timestamp);
    const now = Date.now();

    if (!Number.isFinite(parsed) || Math.abs(now - parsed) > MAX_CLOCK_SKEW_MS) {
        logEventThrottled('clock_skew', 'warning', intersectionId, {
            cameraId,
            timestamp,
            usedServerTime: true,
        }, 5 * 60_000);
        return new Date(now);
    }

    return new Date(parsed);
}

function toLaneCounts(lanes) {
    const counts = {};
    for (const lane of lanes) {
        if (!lane?.laneId) continue;
        counts[lane.laneId] = Number(lane.count || 0);
    }
    return counts;
}

/**
 * @param {object} payload         the raw MQTT message
 * @param {object} directionTotals from buildCameraData(), already computed
 */
function insertReading(payload, directionTotals) {
    const recordedAt = resolveRecordedAt(
        payload.timestamp,
        payload.intersectionId,
        payload.cameraId,
    );

    enqueue({
        sql: INSERT_SQL,
        params: [
            recordedAt,
            payload.intersectionId,
            payload.cameraId,
            payload.meta?.frameId || null,
            JSON.stringify(toLaneCounts(payload.lanes)),
            JSON.stringify(directionTotals),
        ],
        kind: 'traffic_reading',
        key: payload.cameraId,
        intersectionId: payload.intersectionId,
    });
}

module.exports = { insertReading };
