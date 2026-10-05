function getNextPhaseInCycle(phases, currentPhase) {
  if (!currentPhase) return phases[0].phase_id;

  const ids = phases.map((p) => p.phase_id);
  const idx = ids.indexOf(currentPhase);
  if (idx === -1) return ids[0];
  return ids[(idx + 1) % ids.length];
}

function getDirFromPhase(phase) {
  if (phase.startsWith('N_')) return 'N';
  if (phase.startsWith('S_')) return 'S';
  if (phase.startsWith('E_')) return 'E';
  return 'W';
}

function computeGreenTimeSec(phaseCfg, actualCars) {
  const reactionTime = Number(process.env.REACTION_TIME_SEC) || 3;
  const perCarRate = phaseCfg.green_time_sec / phaseCfg.avg_car_passed;
  const raw = reactionTime + perCarRate * actualCars;
  return Math.max(reactionTime, Math.min(phaseCfg.green_time_sec, Math.round(raw)));
}

module.exports = {
  name: 'SIMPLE_CYCLE_BASED',
  getDirFromPhase,
  decide({ agg, intersectionConfig, state }) {
    const phases = intersectionConfig.phases;
    const nextPhase = getNextPhaseInCycle(phases, state.lastPhase);
    const dir = getDirFromPhase(nextPhase);
    const phaseCfg = phases.find((p) => p.phase_id === nextPhase);

    console.log('[SIMPLE_CYCLE_BASED] === Decision Start ===');
    console.log('[SIMPLE_CYCLE_BASED] Next phase: %s (dir=%s)', nextPhase, dir);

    // No traffic data => fall back to config green_time_sec
    if (!agg) {
      const greenTimeSec = Number(phaseCfg.green_time_sec);
      console.log('[SIMPLE_CYCLE_BASED] No agg data, fallback green=%ds', greenTimeSec);
      console.log('[SIMPLE_CYCLE_BASED] === Decision End ===');
      return {
        strategy: 'SIMPLE_CYCLE_BASED',
        phase: nextPhase,
        greenTimeSec,
        meta: { from: state.lastPhase || null, fallback: true },
      };
    }

    const actualCars = agg.laneTotals[dir] || 0;
    const greenTimeSec = computeGreenTimeSec(phaseCfg, actualCars);
    const perCarRate = phaseCfg.green_time_sec / phaseCfg.avg_car_passed;

    console.log('[SIMPLE_CYCLE_BASED] laneTotals:', JSON.stringify(agg.laneTotals));
    console.log('[SIMPLE_CYCLE_BASED] dir=%s | cars=%d | perCarRate = %d/%d = %s s/car',
      dir, actualCars, phaseCfg.green_time_sec, phaseCfg.avg_car_passed, perCarRate.toFixed(1));
    console.log('[SIMPLE_CYCLE_BASED] green = 3 + (%s * %d) = 3 + %s = %ds',
      perCarRate.toFixed(1), actualCars, (perCarRate * actualCars).toFixed(1), greenTimeSec);
    console.log('[SIMPLE_CYCLE_BASED] === Decision End ===');

    return {
      strategy: 'SIMPLE_CYCLE_BASED',
      phase: nextPhase,
      greenTimeSec,
      meta: {
        from: state.lastPhase || null,
        laneTotals: agg.laneTotals,
        chosenDir: dir,
        actualCars,
        perCarRate: Math.round(perCarRate * 100) / 100,
        sampleCount: agg.sampleCount,
      },
    };
  },
};
