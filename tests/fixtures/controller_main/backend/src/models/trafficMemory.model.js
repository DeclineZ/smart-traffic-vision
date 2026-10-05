// Stores snapshots per intersectionId
// memory[intersectionId] = [{ tsMs, payload }, ...]
const memory = new Map();

function addSnapshot(payload) {
    const id = payload.intersectionId;
    const tsMs = Date.parse(payload.timestamp);

    if (!memory.has(id)) memory.set(id, []);

    const arr = memory.get(id);
    arr.push({ tsMs, payload });

    // keep memory bounded (e.g., last 5 minutes)
    const cutoff = Date.now() - 5 * 60 * 1000;
    while (arr.length && arr[0].tsMs < cutoff) arr.shift();
}

function getSnapshots(intersectionId) {
    return memory.get(intersectionId) || [];
}

function getLatestSnapshot(intersectionId) {
    const arr = memory.get(intersectionId);
    if (!arr || arr.length === 0) return null;
    return arr[arr.length - 1];
}

function getAllIntersectionIds() {
    return Array.from(memory.keys());
}

module.exports = { addSnapshot, getSnapshots, getLatestSnapshot, getAllIntersectionIds };
