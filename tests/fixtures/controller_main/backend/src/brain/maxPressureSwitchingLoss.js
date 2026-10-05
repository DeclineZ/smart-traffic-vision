const moduleLastServedMs = {};

function getDirFromPhase(phase) {
  if (phase.startsWith('N_')) return 'N';
  if (phase.startsWith('S_')) return 'S';
  if (phase.startsWith('E_')) return 'E';
  return 'W';
}

function computePhasePressures(phases, maxByDir) {
  const flowRates = {};
  const pressures = {};
  for (const p of phases) {
    const dir = getDirFromPhase(p.phase_id);
    const green = Number(p.green_time_sec);
    const avg = Number(p.avg_car_passed);
    const flow = green > 0 ? avg / green : 0;
    flowRates[p.phase_id] = flow;
    pressures[p.phase_id] = flow * (maxByDir[dir] || 0);
  }
  return { flowRates, pressures };
}

function pickNextPhase(phases, biasedPressures, lastPhase) {
  // Find the maximum biased pressure across all phases
  let bestVal = -Infinity;
  for (const p of phases) {
    const v = biasedPressures[p.phase_id] || 0;
    if (v > bestVal) bestVal = v;
  }

  // Count how many phases share that maximum
  const tied = phases.filter((p) => (biasedPressures[p.phase_id] || 0) === bestVal);

  // No clear winner (all-zero traffic or genuine tie) → advance to next in cycle
  // instead of always defaulting to phases[0], which would cause an infinite skip loop
  if (tied.length > 1 || bestVal <= 0) {
    const ids = phases.map((p) => p.phase_id);
    const idx = lastPhase ? ids.indexOf(lastPhase) : -1;
    return ids[(idx + 1) % ids.length];
  }

  return tied[0].phase_id;
}

module.exports = {
  name: 'MAXPRESSURE_SWITCHING_LOSS',
  getDirFromPhase,
  decide({ agg, intersectionConfig, state, nowMs = Date.now() }) {
    if (!state.lastServedMs) {
      state.lastServedMs = moduleLastServedMs;
    }
    const lastServedMs = state.lastServedMs;

    const reactionTime = Number(process.env.REACTION_TIME_SEC) || 3;
    const switchingBias = Number(process.env.MP_SWITCHING_BIAS) || 1.2;
    const maxGreenSec = Number(process.env.MP_GREEN_MAX_SEC) || 90;
    const maxRedSec = Number(process.env.MAX_RED_SEC) || 120;

    const phases = intersectionConfig.phases;
    const ids = phases.map((p) => p.phase_id);
    const maxByDir = agg?.laneMaxByDirection || { N: 0, S: 0, E: 0, W: 0 };

    console.log('[MAXPRESSURE_SWITCHING_LOSS] === Decision Start ===');
    console.log('[MAXPRESSURE_SWITCHING_LOSS] maxByDir:', JSON.stringify(maxByDir));
    console.log('[MAXPRESSURE_SWITCHING_LOSS] laneMedians:', JSON.stringify(agg?.laneMedians || {}));

    // Step 1: Max Red - track red light wait time for every phase
    for (const id of ids) {
      if (!lastServedMs[id]) lastServedMs[id] = nowMs;
    }

    const redWaitTimes = {};
    const triggeredPhases = [];
    for (const id of ids) {
      const waitSec = Math.max(0, Math.floor((nowMs - lastServedMs[id]) / 1000));
      redWaitTimes[id] = waitSec;
      if (waitSec >= maxRedSec) {
        triggeredPhases.push(id);
      }
    }
    console.log('[MAXPRESSURE_SWITCHING_LOSS] redWaitTimes:', JSON.stringify(redWaitTimes));

    // Step 2: pressure per phase
    const { flowRates, pressures } = computePhasePressures(phases, maxByDir);
    console.log('[MAXPRESSURE_SWITCHING_LOSS] flowRates:', JSON.stringify(
      Object.fromEntries(Object.entries(flowRates).map(([k, v]) => [k, Number(v.toFixed(3))]))
    ));
    console.log('[MAXPRESSURE_SWITCHING_LOSS] pressures:', JSON.stringify(
      Object.fromEntries(Object.entries(pressures).map(([k, v]) => [k, Number(v.toFixed(3))]))
    ));

    // Step 3: bias the current phase to model switching loss
    const biasedPressures = { ...pressures };
    if (state.lastPhase && biasedPressures[state.lastPhase] !== undefined) {
      biasedPressures[state.lastPhase] *= switchingBias;
    }
    console.log(
      '[MAXPRESSURE_SWITCHING_LOSS] biased (x%s on %s):',
      switchingBias,
      state.lastPhase || 'none',
      JSON.stringify(
        Object.fromEntries(
          Object.entries(biasedPressures).map(([k, v]) => [k, Number(v.toFixed(3))])
        )
      )
    );

    let nextPhase;
    let isMaxRedOverride = false;

    // Override with round-robin if Max Red threshold exceeded, otherwise pick by Max Pressure
    if (triggeredPhases.length > 0) {
      isMaxRedOverride = true;
      console.log('[MAXPRESSURE_SWITCHING_LOSS] Max Red triggered on:', JSON.stringify(triggeredPhases));
      const lastIdx = state.lastPhase ? ids.indexOf(state.lastPhase) : -1;
      for (let i = 1; i <= ids.length; i++) {
        const candidate = ids[(lastIdx + i) % ids.length];
        if (triggeredPhases.includes(candidate)) {
          nextPhase = candidate;
          break;
        }
      }
      if (!nextPhase) nextPhase = triggeredPhases[0];
    } else {
      nextPhase = pickNextPhase(phases, biasedPressures, state.lastPhase);
    }

    const chosenDir = getDirFromPhase(nextPhase);
    const flow = flowRates[nextPhase];
    const cars = maxByDir[chosenDir] || 0;

    const phaseCfg = phases.find((p) => p.phase_id === nextPhase);
    const perCarRate = phaseCfg && Number(phaseCfg.avg_car_passed) > 0
      ? Math.round((Number(phaseCfg.green_time_sec) / Number(phaseCfg.avg_car_passed)) * 100) / 100
      : 0;

    // Step 4: green time = cars / flow + reaction, clamped
    let greenTimeSec;
    if (flow > 0) {
      greenTimeSec = Math.round(cars / flow + reactionTime);
    } else {
      greenTimeSec = reactionTime;
    }
    greenTimeSec = Math.max(reactionTime, Math.min(maxGreenSec, greenTimeSec));

    const yellowTimeSec = Number(process.env.YELLOW_TIME_SEC || 3);
    const allRedTimeSec = Number(process.env.ALL_RED_TIME_SEC || 2);
    lastServedMs[nextPhase] = nowMs + (greenTimeSec + yellowTimeSec + allRedTimeSec) * 1000;

    console.log(
      '[MAXPRESSURE_SWITCHING_LOSS] chosen=%s | dir=%s | cars=%d | flow=%s | maxRedOverride=%s | perCarRate=%s',
      nextPhase, chosenDir, cars, flow.toFixed(3), isMaxRedOverride, perCarRate
    );
    console.log(
      '[MAXPRESSURE_SWITCHING_LOSS] green = round(%d / %s + %d) = %ds (clamped %d-%d)',
      cars, flow.toFixed(3), reactionTime, greenTimeSec, reactionTime, maxGreenSec
    );
    console.log('[MAXPRESSURE_SWITCHING_LOSS] === Decision End ===');

    return {
      strategy: 'MAXPRESSURE_SWITCHING_LOSS',
      phase: nextPhase,
      greenTimeSec,
      meta: {
        from: state.lastPhase || null,
        laneTotals: agg?.laneTotals || { N: 0, S: 0, E: 0, W: 0 },
        chosenDir,
        actualCars: cars,
        perCarRate,
        sampleCount: agg?.sampleCount,
        maxByDir,
        flowRates,
        pressures,
        biasedPressures,
        switchingBias,
        redWaitTimes,
        maxRedSec,
        isMaxRedOverride,
        triggeredPhases,
      },
    };
  },
};
