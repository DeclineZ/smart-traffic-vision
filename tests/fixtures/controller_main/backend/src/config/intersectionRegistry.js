/**
 * The set of intersections the backend controls, and the per-intersection
 * parameters it controls them with.
 *
 * Loaded once at boot from `intersections` / `intersection_phases` and held in
 * memory, because the decision loop reads it on every tick and config changes
 * are rare. ADR-004: the five Sriracha intersections have different topologies,
 * so timing belongs to the intersection rather than to a global env var.
 *
 * A database that is down must not stop the lights from changing (ADR-005), so
 * a failed load falls back to the single intersection in
 * src/mock/intersection.config.json plus the env values that used to drive it.
 * That is the old behaviour exactly — degraded, not stopped.
 */

const pool = require('./db.config');
const fallbackConfig = require('../mock/intersection.config.json');
const { logEvent } = require('../repositories/systemEvent.repo');

const INTERSECTIONS_SQL = `
    SELECT intersection_id, name, lat, lon, topology,
           controller_name, agg_window_sec, freshness_max_age_ms, max_red_sec
    FROM intersections
    WHERE is_active
    ORDER BY intersection_id
`;

const PHASES_SQL = `
    SELECT intersection_id, phase, display_order, min_green_sec, max_green_sec,
           base_green_sec, yellow_sec, all_red_sec, avg_car_passed
    FROM intersection_phases
    ORDER BY intersection_id, display_order
`;

/** @type {Map<string, object>} keyed by intersection_id, insertion-ordered by id */
let registry = new Map();
let loadedFrom = 'none';

function buildFromRows(intersectionRows, phaseRows) {
    const phasesById = new Map();
    for (const row of phaseRows) {
        if (!phasesById.has(row.intersection_id)) phasesById.set(row.intersection_id, []);
        phasesById.get(row.intersection_id).push({
            phase: row.phase,
            displayOrder: Number(row.display_order),
            minGreenSec: Number(row.min_green_sec),
            maxGreenSec: Number(row.max_green_sec),
            baseGreenSec: Number(row.base_green_sec),
            yellowSec: Number(row.yellow_sec),
            allRedSec: Number(row.all_red_sec),
            avgCarPassed: Number(row.avg_car_passed ?? 0),
        });
    }

    const built = new Map();
    for (const row of intersectionRows) {
        const phases = phasesById.get(row.intersection_id) ?? [];
        // An intersection with no phases cannot be controlled at all; serving it
        // would mean a tab that can only ever show a crash.
        if (phases.length === 0) {
            console.error(`[REGISTRY] ${row.intersection_id} has no phases — skipped`);
            logEvent('intersection_misconfigured', 'error', row.intersection_id, {
                reason: 'no_phases',
            });
            continue;
        }

        built.set(row.intersection_id, {
            id: row.intersection_id,
            name: row.name,
            lat: row.lat,
            lon: row.lon,
            topology: row.topology ?? {},
            controllerName: row.controller_name,
            aggWindowSec: Number(row.agg_window_sec),
            freshnessMaxAgeMs: Number(row.freshness_max_age_ms),
            maxRedSec: row.max_red_sec === null ? null : Number(row.max_red_sec),
            phases,
        });
    }

    return built;
}

/** The pre-database behaviour: one intersection, configured by env vars. */
function buildFallback() {
    const phases = (fallbackConfig.phases ?? []).map((phase, index) => ({
        phase: phase.phase_id,
        displayOrder: index + 1,
        minGreenSec: Number(process.env.MIN_GREEN_SEC || 5),
        maxGreenSec: Number(process.env.MP_GREEN_MAX_SEC || 90),
        baseGreenSec: Number(phase.green_time_sec || 30),
        yellowSec: Number(process.env.YELLOW_TIME_SEC || 3),
        allRedSec: Number(process.env.ALL_RED_TIME_SEC || 2),
        avgCarPassed: Number(phase.avg_car_passed || 0),
    }));

    return new Map([[
        fallbackConfig.intersection_id,
        {
            id: fallbackConfig.intersection_id,
            name: fallbackConfig.intersection_id,
            lat: null,
            lon: null,
            topology: { lanes: fallbackConfig.lanes ?? [] },
            controllerName: process.env.CONTROLLER_NAME || 'QUEUE_BASED',
            aggWindowSec: Number(process.env.AGG_WINDOW_SEC || 10),
            freshnessMaxAgeMs: Number(process.env.FRESHNESS_MAX_AGE_MS || 2000),
            maxRedSec: Number(process.env.MAX_RED_SEC) || null,
            phases,
        },
    ]]);
}

/**
 * Replaces the in-memory registry. Never throws: the caller is the boot
 * sequence, and refusing to start is worse than starting on the fallback.
 *
 * @returns {Promise<{count: number, source: 'database'|'fallback'}>}
 */
async function loadIntersections() {
    try {
        const [intersectionRes, phaseRes] = await Promise.all([
            pool.query(INTERSECTIONS_SQL),
            pool.query(PHASES_SQL),
        ]);

        const built = buildFromRows(intersectionRes.rows, phaseRes.rows);
        if (built.size === 0) throw new Error('no active intersections with phases');

        registry = built;
        loadedFrom = 'database';
    } catch (err) {
        registry = buildFallback();
        loadedFrom = 'fallback';
        console.error(
            `[REGISTRY] could not load intersections from the database (${err.message}) — ` +
            `falling back to ${[...registry.keys()].join(', ')} from intersection.config.json`,
        );
        logEvent('registry_fallback', 'error', null, { reason: err.message });
    }

    console.log(
        `[REGISTRY] ${registry.size} intersection(s) from ${loadedFrom}: ` +
        [...registry.values()].map((i) => `${i.id}(${i.phases.length}p/${i.controllerName})`).join(' '),
    );

    return { count: registry.size, source: loadedFrom };
}

function getIntersections() {
    return [...registry.values()];
}

function getIntersectionIds() {
    return [...registry.keys()];
}

function getIntersection(intersectionId) {
    return registry.get(intersectionId) ?? null;
}

function hasIntersection(intersectionId) {
    return registry.has(intersectionId);
}

/** The tab the dashboard opens on when no intersection is named. */
function getDefaultIntersectionId() {
    return registry.keys().next().value ?? null;
}

function getPhase(intersectionId, phaseId) {
    const intersection = registry.get(intersectionId);
    if (!intersection) return null;
    return intersection.phases.find((p) => p.phase === phaseId) ?? null;
}

/**
 * The shape backend/src/brain/* has always been given. Kept as an adapter so
 * moving config into the database does not touch four controllers at once —
 * `green_time_sec` there is this table's base_green_sec.
 */
function toControllerConfig(intersection) {
    return {
        intersection_id: intersection.id,
        phases: intersection.phases.map((phase) => ({
            phase_id: phase.phase,
            green_time_sec: phase.baseGreenSec,
            avg_car_passed: phase.avgCarPassed,
        })),
        lanes: intersection.topology?.lanes ?? [],
    };
}

module.exports = {
    loadIntersections,
    getIntersections,
    getIntersectionIds,
    getIntersection,
    hasIntersection,
    getDefaultIntersectionId,
    getPhase,
    toControllerConfig,
};
