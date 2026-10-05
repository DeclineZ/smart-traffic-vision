function median(values) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Removes points that are too far from the median (robust against one bad frame).
function removeOutliersByMedian(values, toleranceRatio = 0.7) {
    // Bucket = [8, 9, 8, 0, 8]
    // median = 8
    // toleranceRatio = 0.7 => keep values within ±70% of median
    // 0 is eliminated as outlier because abs(0-8)/8 = 1 > 0.7
    if (values.length <= 2) return values;
    const med = median(values);

    // If median is 0, "ratio vs med" breaks; keep values and let median handle it.
    if (med === 0) return values;

    return values.filter((v) => Math.abs(v - med) / med <= toleranceRatio);
}

/**
 * Build per-direction totals per snapshot:
 * total(dir) = sum over lanes with that direction of lane.count
 */
function totalsByDirectionFromPayload(payload) {
    const totals = { N: 0, S: 0, E: 0, W: 0 };
    const lanes = payload?.lanes;
    if (!Array.isArray(lanes)) return totals;

    for (const lane of lanes) {
        const dir = lane?.direction;
        if (!Object.prototype.hasOwnProperty.call(totals, dir)) continue;

        totals[dir] += Number(lane?.count || 0);
    }

    return totals;
}

/**
 * Get a "reliable" queued state within the aggregation window.
 * - Uses median filtering per direction across frames
 * - Optional outlier removal before median
 *
 * @param {Array<{tsMs:number, payload:any}>} windowSnapshots
 * @param {object} [opts]
 * @param {number} [opts.outlierToleranceRatio=0.7]  // 0.7 => keep values within ±70% of median
 * @param {number} [opts.minSamples=3]               // need at least N snapshots to use median robustly
 * @returns {{
 *   latestTsMs: number,
 *   laneTotals: {N:number,S:number,E:number,W:number},
 *   perDirSamples: {N:number,S:number,E:number,W:number},
 *   sampleCount: number,
 *   reliability: string
 * } | null}
 */
function getReliableQueuedAggregation(windowSnapshots, opts = {}) {
    if (!Array.isArray(windowSnapshots) || windowSnapshots.length === 0)
        return null;

    const outlierToleranceRatio = Number.isFinite(opts.outlierToleranceRatio)
        ? opts.outlierToleranceRatio
        : 0.7;

    const minSamples = Number.isFinite(opts.minSamples) ? opts.minSamples : 3;

    // Buckets of queued totals per direction across time
    const buckets = { N: [], S: [], E: [], W: [] };

    let latestTsMs = 0;

    for (const snap of windowSnapshots) {
        if (!snap) continue;
        const tsMs = Number(snap.tsMs || 0);
        if (tsMs > latestTsMs) latestTsMs = tsMs;

        const totals = totalsByDirectionFromPayload(snap.payload);
        buckets.N.push(totals.N);
        buckets.S.push(totals.S);
        buckets.E.push(totals.E);
        buckets.W.push(totals.W);
    }

    const laneTotals = { N: 0, S: 0, E: 0, W: 0 };
    const perDirSamples = {
        N: buckets.N.length,
        S: buckets.S.length,
        E: buckets.E.length,
        W: buckets.W.length,
    };

    for (const dir of ['N', 'S', 'E', 'W']) {
        const values = buckets[dir];

        // If not enough samples, fall back to the last value (still better than nothing)
        if (values.length < minSamples) {
            laneTotals[dir] = Math.round(
                Number(values[values.length - 1] || 0),
            );
            continue;
        }

        const cleaned = removeOutliersByMedian(values, outlierToleranceRatio);
        laneTotals[dir] = Math.round(median(cleaned));
    }

    return {
        latestTsMs,
        laneTotals,
        perDirSamples,
        sampleCount: windowSnapshots.length,
        reliability: 'QUEUED_MEDIAN_FILTERED',
    };
}

/**
 * Per-lane median across the window, then max per direction.
 * Used by max-pressure controllers that need the busiest lane in each direction.
 */
function getReliableLaneMaxByDirection(windowSnapshots, opts = {}) {
    if (!Array.isArray(windowSnapshots) || windowSnapshots.length === 0)
        return null;

    const outlierToleranceRatio = Number.isFinite(opts.outlierToleranceRatio)
        ? opts.outlierToleranceRatio
        : 0.7;
    const minSamples = Number.isFinite(opts.minSamples) ? opts.minSamples : 3;

    // laneId -> { direction, counts: [] }
    const laneBuckets = new Map();

    for (const snap of windowSnapshots) {
        const lanes = snap?.payload?.lanes;
        if (!Array.isArray(lanes)) continue;
        for (const lane of lanes) {
            const id = lane?.laneId;
            const dir = lane?.direction;
            if (!id || !['N', 'S', 'E', 'W'].includes(dir)) continue;
            if (!laneBuckets.has(id)) {
                laneBuckets.set(id, { direction: dir, counts: [] });
            }
            laneBuckets.get(id).counts.push(Number(lane.count || 0));
        }
    }

    const laneMedians = {};
    const maxByDir = { N: 0, S: 0, E: 0, W: 0 };

    for (const [laneId, { direction, counts }] of laneBuckets.entries()) {
        let value;
        if (counts.length < minSamples) {
            value = Math.round(Number(counts[counts.length - 1] || 0));
        } else {
            const cleaned = removeOutliersByMedian(counts, outlierToleranceRatio);
            value = Math.round(median(cleaned));
        }
        laneMedians[laneId] = value;
        if (value > maxByDir[direction]) maxByDir[direction] = value;
    }

    return { laneMedians, maxByDir };
}

module.exports = {
    getReliableQueuedAggregation,
    getReliableLaneMaxByDirection,
    totalsByDirectionFromPayload,
};
