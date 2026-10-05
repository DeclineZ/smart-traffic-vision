// Runs pinned, unmodified main code with only its I/O and clock replaced.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const root = path.join(__dirname, 'fixtures/controller_main');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
let now = Date.parse(input.payload.timestamp) + 100;
class TestDate extends Date {
    constructor(...args) { super(...(args.length ? args : [now])); }
    static now() { return now; }
}
const timers = [];
const decisions = [];
const audits = [];
const writes = [];
const events = [];
const errors = [];
const cache = new Map();
const noop = () => {};
const src = 'backend/src/';
const stubs = {
    [src + 'config/mqtt.config.js']: { getMqttClient: () => ({
        publish: (topic, body) => { assert.equal(topic, 'traffic/decision/v1'); decisions.push(JSON.parse(body)); },
    }) },
    [src + 'utils/logger.js']: { logTraffic: noop },
    [src + 'repositories/writeQueue.js']: { enqueue: row => writes.push(row), waitForIdle: async () => true },
    [src + 'repositories/systemEvent.repo.js']: {
        logEvent: (...args) => events.push(args), logEventThrottled: (...args) => events.push(args), resetThrottle: noop,
    },
    [src + 'repositories/decision.repo.js']: { insertDecision: row => audits.push(row) },
    [src + 'repositories/phaseExecution.repo.js']: { closeOpenPhase: noop, openPhase: noop },
    [src + 'repositories/manualOverride.repo.js']: {},
    [src + 'config/db.config.js']: { query: async sql => ({ rows: sql.includes('FROM intersection_phases')
        ? ['N', 'E', 'S', 'W'].map((dir, i) => ({ intersection_id: 'INT-001', phase: dir + '_GO',
            display_order: i + 1, min_green_sec: 5, max_green_sec: 90, base_green_sec: 30,
            yellow_sec: 3, all_red_sec: 2, avg_car_passed: 15 }))
        : [{ intersection_id: 'INT-001', name: 'INT-001', topology: { lanes: ['N1', 'N2', 'E1', 'E2', 'S1', 'S2', 'W1', 'W2'] },
            controller_name: input.controller || 'MAXPRESSURE_SWITCHING_LOSS',
            agg_window_sec: 10, freshness_max_age_ms: 2000, max_red_sec: 120 }],
    }) },
};

function resolve(from, name) {
    const candidate = path.posix.normalize(path.posix.join(path.posix.dirname(from), name));
    return [candidate, candidate + '.js', candidate + '/index.js'].find(p => manifest.files[p] || stubs[p]);
}
function load(file) {
    if (stubs[file]) return stubs[file];
    if (cache.has(file)) return cache.get(file).exports;
    const bytes = fs.readFileSync(path.join(root, file));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), manifest.files[file], file + ' snapshot changed');
    const module = { exports: {} };
    cache.set(file, module);
    if (file.endsWith('.json')) module.exports = JSON.parse(bytes);
    else {
        const localRequire = name => {
            if (name === 'node:crypto') return crypto;
            const resolved = resolve(file, name);
            assert.ok(resolved, 'Unexpected dependency ' + name + ' from ' + file);
            return load(resolved);
        };
        const context = { module, exports: module.exports, require: localRequire, Date: TestDate,
            console: { log: noop, warn: noop, error: (...args) => errors.push(args.join(' ')) },
            process: { env: { DECISION_TOPIC: 'traffic/decision/v1', INITIAL_DECISION_DELAY_MS: '0' } },
            setTimeout: (fn, ms) => { const timer = { fn, ms, cleared: false }; timers.push(timer); return timer; },
            clearTimeout: timer => { timer.cleared = true; },
        };
        vm.runInNewContext(bytes.toString('utf8'), context, { filename: file });
    }
    return module.exports;
}

async function tick(timer) {
    assert.ok(timer && !timer.cleared, 'Missing scheduled decision');
    timer.cleared = true;
    now += timer.ms;
    timer.fn();
    await new Promise(setImmediate);
}

async function main() {
    const registry = load(src + 'config/intersectionRegistry.js');
    const loaded = await registry.loadIntersections();
    assert.equal(loaded.source, 'database');
    const intake = load(src + 'services/incomingTraffic.service.js');
    const memory = load(src + 'models/trafficMemory.model.js');
    const service = load(src + 'services/decision.service.js');
    const skip = load(src + 'services/skipPhase.service.js');
    let received;
    // Three real ingests also exercise the median/outlier branch.
    for (let i = 0; i < 3; i++) {
        received = await intake.handleIncomingCounts(input.payload);
        skip.onIncomingPayload('INT-001');
    }
    const camera = intake.getLatestCameraData('INT-001');
    const aggregate = service.aggregateWindow(memory.getSnapshots('INT-001'), 10, 2000);
    assert.ok(aggregate);
    service.startDecisionLoop();
    await tick(timers.find(timer => !timer.cleared));
    assert.equal(decisions.length, 1);
    const adaptive = decisions[0];
    now += 2001;
    assert.equal(service.aggregateWindow(memory.getSnapshots('INT-001'), 10, 2000), null);
    // Suppressed vision output produces no ingest. Main changes strategy at its next scheduled decision.
    await tick(timers.find(timer => !timer.cleared));
    assert.equal(decisions.length, 2);
    assert.equal(decisions[1].decision.strategy, 'FIXED_CYCLE');
    assert.equal(audits[1].fallbackReason, 'stale_data');
    assert.equal(audits[1].source, 'fallback');
    assert.equal(errors.length, 0, errors.join('\n'));
    const row = writes[0];
    assert.equal(row.params[2], 'CAM-01');
    process.stdout.write(JSON.stringify({ revision: manifest.revision, received, camera, aggregate,
        storedLaneCounts: JSON.parse(row.params[4]), storedDirectionTotals: JSON.parse(row.params[5]),
        adaptive, fallback: decisions[1], queuedWrites: writes.length }));
}
main().catch(err => { process.stderr.write(err.stack + '\n'); process.exitCode = 1; });
