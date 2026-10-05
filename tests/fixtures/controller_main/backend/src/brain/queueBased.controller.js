function decidePhaseFromTotals(laneTotals) {
  let bestDir = 'N';
  let bestVal = -Infinity;
  for (const dir of ['N', 'S', 'E', 'W']) {
    if (laneTotals[dir] > bestVal) {
      bestVal = laneTotals[dir];
      bestDir = dir;
    }
  }
  const map = { N: 'N_GO', S: 'S_GO', E: 'E_GO', W: 'W_GO' };
  return map[bestDir];
}

function getChosenDirFromPhase(phase) {
  if (phase.startsWith('N_')) return 'N';
  if (phase.startsWith('S_')) return 'S';
  if (phase.startsWith('E_')) return 'E';
  return 'W';
}

function computeGreenTimeSec(laneTotals, chosenDir) {
  const base = 15;
  const perCar = 0.5;
  const count = laneTotals[chosenDir] || 0;
  const raw = base + count * perCar;
  return Math.max(15, Math.min(45, Math.round(raw)));
}

module.exports = {
  name: 'QUEUE_BASED',
  decide({ agg }) {
    console.log('[QUEUE_BASED] === Decision Start ===');
    console.log('[QUEUE_BASED] Input laneTotals:', JSON.stringify(agg.laneTotals));
    console.log('[QUEUE_BASED] Sample count:', agg.sampleCount);

    const phase = decidePhaseFromTotals(agg.laneTotals);
    const dir = getChosenDirFromPhase(phase);
    console.log('[QUEUE_BASED] Highest queue direction:', dir, '| vehicles:', agg.laneTotals[dir]);

    const greenTimeSec = computeGreenTimeSec(agg.laneTotals, dir);
    console.log('[QUEUE_BASED] Green time: base=15 + %d * 0.5 = %ds (clamped 15-45)', agg.laneTotals[dir], greenTimeSec);

    console.log('[QUEUE_BASED] Output: phase=%s greenTimeSec=%d', phase, greenTimeSec);
    console.log('[QUEUE_BASED] === Decision End ===');

    return {
      strategy: 'QUEUE_BASED',
      phase,
      greenTimeSec,
      meta: {
        laneTotals: agg.laneTotals,
        chosenDir: dir,
        sampleCount: agg.sampleCount,
      },
    };
  },
};
