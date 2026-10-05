const queueBased = require('./queueBased.controller');
const fixedCycle = require('./fixedCycle.controller');
const simpleCycleBased = require('./simpleCycleBased.controller');
const maxPressureSwitchingLoss = require('./maxPressureSwitchingLoss');

const controllers = {
  QUEUE_BASED: queueBased,
  FIXED_CYCLE: fixedCycle,
  SIMPLE_CYCLE_BASED: simpleCycleBased,
  MAXPRESSURE_SWITCHING_LOSS: maxPressureSwitchingLoss,
};

function getController(name) {
  const key = String(name).toUpperCase();
  return controllers[key] || controllers.QUEUE_BASED; // Default to queue-based if not found
}

module.exports = { getController };
