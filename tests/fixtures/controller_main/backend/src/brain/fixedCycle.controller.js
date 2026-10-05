  function getNextPhaseInCycle(phases, currentPhase) {
    if (!currentPhase) return phases[0].phase_id;

    const ids = phases.map(p => p.phase_id);
    const idx = ids.indexOf(currentPhase);
    if (idx === -1) return ids[0];
    return ids[(idx + 1) % ids.length];
  }

  module.exports = {
    name: 'FIXED_CYCLE',
    decide({ intersectionConfig, state }) {
      const phases = intersectionConfig.phases;
      const nextPhase = getNextPhaseInCycle(phases, state.lastPhase);

      const cfg = phases.find(p => p.phase_id === nextPhase);
      const greenTimeSec = Number(cfg?.green_time_sec || 15);

      return {
        strategy: 'FIXED_CYCLE',
        phase: nextPhase,
        greenTimeSec,
        meta: { from: state.lastPhase || null },
      };
    },
  };
