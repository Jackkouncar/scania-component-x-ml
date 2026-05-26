const state = {
  data: null,
  model: null,
  lightgbmModel: null,
  selectedVehicle: null,
  counterFeature: null,
  histogramVariable: null,
  predictionMode: 'lightgbm_cost',
  telemetryMode: 'replay',
  streamIndex: 0,
  isPlaying: false,
  playbackTimer: null,
  eventLog: [],
  lastMlSmoothing: null
};

const els = {
  componentAnswer: document.getElementById('component-answer'),
  articleFindings: document.getElementById('article-findings'),
  metricVehicles: document.getElementById('metric-vehicles'),
  metricReadouts: document.getElementById('metric-readouts'),
  metricCounters: document.getElementById('metric-counters'),
  metricHistograms: document.getElementById('metric-histograms'),
  modelName: document.getElementById('model-name'),
  modelValidationCost: document.getElementById('model-validation-cost'),
  modelTestCost: document.getElementById('model-test-cost'),
  modelTopFeature: document.getElementById('model-top-feature'),
  modelComparisonTable: document.getElementById('model-comparison-table'),
  search: document.getElementById('vehicle-search'),
  splitFilter: document.getElementById('split-filter'),
  riskFilter: document.getElementById('risk-filter'),
  counterSelect: document.getElementById('counter-select'),
  histogramSelect: document.getElementById('histogram-select'),
  calendarUnit: document.getElementById('calendar-unit'),
  timeWarning: document.getElementById('time-warning'),
  vehicleList: document.getElementById('vehicle-list'),
  vehicleTitle: document.getElementById('vehicle-title'),
  vehicleSubtitle: document.getElementById('vehicle-subtitle'),
  riskPill: document.getElementById('risk-pill'),
  vehicleFacts: document.getElementById('vehicle-facts'),
  resetStream: document.getElementById('reset-stream'),
  sendTelemetry: document.getElementById('send-telemetry'),
  toggleAutoplay: document.getElementById('toggle-autoplay'),
  telemetryMode: document.getElementById('telemetry-mode'),
  playbackSpeed: document.getElementById('playback-speed'),
  predictionMode: document.getElementById('prediction-mode'),
  jumpTime: document.getElementById('jump-time'),
  jumpTimeButton: document.getElementById('jump-time-button'),
  jumpDate: document.getElementById('jump-date'),
  jumpDateButton: document.getElementById('jump-date-button'),
  jumpFailureWindow: document.getElementById('jump-failure-window'),
  liveIntake: document.getElementById('live-intake'),
  liveTimeStep: document.getElementById('live-time-step'),
  liveCounterForm: document.getElementById('live-counter-form'),
  seedLiveRecord: document.getElementById('seed-live-record'),
  clearLiveRecords: document.getElementById('clear-live-records'),
  insertLiveRecord: document.getElementById('insert-live-record'),
  simulatorCaption: document.getElementById('simulator-caption'),
  predictionCard: document.getElementById('prediction-card'),
  healthStatus: document.getElementById('health-status'),
  predictionText: document.getElementById('prediction-text'),
  telemetryTable: document.getElementById('telemetry-table'),
  eventLog: document.getElementById('event-log'),
  timelineCaption: document.getElementById('timeline-caption'),
  histogramCaption: document.getElementById('histogram-caption'),
  timelineCanvas: document.getElementById('timeline-chart'),
  histogramCanvas: document.getElementById('histogram-chart'),
  failureExplanation: document.getElementById('failure-explanation')
};

const RISK_COLORS = {
  0: '#52616b',
  1: '#2e6fbb',
  2: '#b66a19',
  3: '#d05631',
  4: '#bd3f47',
  unknown: '#52616b'
};

const HEALTH_LABELS = {
  0: 'Healthy',
  1: 'Watch',
  2: 'Elevated',
  3: 'Warning',
  4: 'Critical',
  unknown: 'Unknown'
};

const CLASS_WINDOWS = {
  0: 'No imminent failure',
  1: '48 to 24 relative time units before failure',
  2: '24 to 12 relative time units before failure',
  3: '12 to 6 relative time units before failure',
  4: '6 to 0 relative time units before failure'
};

const COST_MATRIX = [
  [0, 7, 8, 9, 10],
  [200, 0, 7, 8, 9],
  [300, 200, 0, 7, 8],
  [400, 300, 200, 0, 7],
  [500, 400, 300, 200, 0]
];

const PREDICTION_MODE_LABELS = {
  lightgbm_trend: 'LightGBM probability trend',
  lightgbm_cost: 'LightGBM cost-sensitive',
  lightgbm_accuracy: 'LightGBM 75% accuracy floor',
  lightgbm_expected: 'LightGBM expected-cost',
  knn: 'kNN live scorer',
  centroid: 'Nearest-centroid live scorer',
  baseline: 'Baseline timeline'
};

const LOWER_RISK_CONFIRMATIONS = 3;
const API_TOKEN_KEY = 'scania_component_x_api_token_v1';

function number(value) {
  return new Intl.NumberFormat('en-US').format(value);
}

function fixed(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits).replace(/\.0$/, '');
}

function round(value, digits = 3) {
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function log1pNonNegative(value) {
  return Math.log1p(Math.max(0, Number(value) || 0));
}

function classText(cls) {
  return cls === null || cls === undefined ? 'Unknown' : `Class ${cls}`;
}

function healthText(cls) {
  return HEALTH_LABELS[cls ?? 'unknown'] || HEALTH_LABELS.unknown;
}

function classifyTimeToFailure(timeToFailure) {
  if (timeToFailure === null || timeToFailure === undefined || !Number.isFinite(timeToFailure)) return 0;
  if (timeToFailure > 48) return 0;
  if (timeToFailure > 24) return 1;
  if (timeToFailure > 12) return 2;
  if (timeToFailure > 6) return 3;
  return 4;
}

function confidenceForClass(cls, mode) {
  const base = { 0: 0.72, 1: 0.64, 2: 0.72, 3: 0.82, 4: 0.92 }[cls] || 0.55;
  return mode === 'estimated-window' ? Math.max(0.5, base - 0.08) : base;
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const expScores = scores.map(score => Math.exp(score - maxScore));
  const total = expScores.reduce((sum, value) => sum + value, 0);
  return expScores.map(value => value / total);
}

function squaredDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return sum / Math.max(1, a.length);
}

function formatPseudoDate(timeStep) {
  const mode = els.calendarUnit.value;
  if (mode === 'unit') return `t=${fixed(timeStep, 1)} units`;

  const base = new Date(state.data.baseDate);
  const msPerUnit = mode === 'hour' ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
  const date = new Date(base.getTime() + Number(timeStep) * msPerUnit);
  const label = mode === 'hour' ? 'hour assumption' : 'day assumption';
  return `${date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })} (${label})`;
}

function timeStepToIsoDate(timeStep) {
  const mode = els.calendarUnit.value;
  if (mode === 'unit') return '';

  const base = new Date(state.data.baseDate);
  const msPerUnit = mode === 'hour' ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
  const date = new Date(base.getTime() + Number(timeStep) * msPerUnit);
  return date.toISOString().slice(0, 10);
}

function dateToTimeStep(dateValue) {
  if (!dateValue || els.calendarUnit.value === 'unit') return null;

  const target = new Date(`${dateValue}T00:00:00.000Z`);
  const base = new Date(state.data.baseDate);
  const msPerUnit = els.calendarUnit.value === 'hour' ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
  return (target.getTime() - base.getTime()) / msPerUnit;
}

function playbackDelay() {
  return Number(els.playbackSpeed.value || 350);
}

function nearestSeriesIndex(vehicle, targetTimeStep) {
  if (!vehicle || !vehicle.series.length || !Number.isFinite(targetTimeStep)) return 0;
  let bestIndex = 0;
  let bestDistance = Infinity;
  vehicle.series.forEach((point, index) => {
    const distance = Math.abs(point.t - targetTimeStep);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function currentTelemetry() {
  const vehicle = state.selectedVehicle;
  if (!vehicle || !vehicle.series.length) return null;
  return vehicle.series[Math.min(state.streamIndex, vehicle.series.length - 1)];
}

function streamedSeries(vehicle = state.selectedVehicle) {
  if (!vehicle) return [];
  return vehicle.series.slice(0, Math.min(state.streamIndex + 1, vehicle.series.length));
}

function latestSeriesPoint(vehicle = state.selectedVehicle) {
  if (!vehicle || !vehicle.series.length) return null;
  return vehicle.series[vehicle.series.length - 1];
}

function typicalTimeStepDelta(vehicle = state.selectedVehicle) {
  if (!vehicle || vehicle.series.length < 2) return 1;
  const deltas = [];
  for (let i = Math.max(1, vehicle.series.length - 12); i < vehicle.series.length; i++) {
    const delta = vehicle.series[i].t - vehicle.series[i - 1].t;
    if (delta > 0) deltas.push(delta);
  }
  if (!deltas.length) return 1;
  deltas.sort((a, b) => a - b);
  return deltas[Math.floor(deltas.length / 2)];
}

function nextLiveTimeStep(vehicle = state.selectedVehicle) {
  const latest = latestSeriesPoint(vehicle);
  if (!latest) return 0;
  return round(latest.t + typicalTimeStepDelta(vehicle), 3);
}

function liveRecordsForVehicle(vehicle) {
  return vehicle?.series?.filter(point => point.source === 'live_insert') || [];
}

function addLog(message) {
  const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  state.eventLog.unshift(`[${stamp}] ${message}`);
  state.eventLog = state.eventLog.slice(0, 30);
  renderEventLog();
}

function renderEventLog() {
  els.eventLog.innerHTML = state.eventLog
    .map(entry => `<div class="log-entry">${entry}</div>`)
    .join('');
}

async function apiRequest(path, { method = 'GET', body, authenticated = false } = {}) {
  const headers = {};
  if (body) headers['Content-Type'] = 'application/json';
  if (authenticated) {
    const token = window.localStorage.getItem(API_TOKEN_KEY);
    if (!token) {
      throw new Error('Sign in as admin in Operations Console before changing telemetry.');
    }
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || 'The API request failed.');
  }
  return payload;
}

function getFinalFailureEstimate(vehicle) {
  const failure = vehicle.failure || {};
  if (failure.kind === 'observed_repair') {
    return {
      kind: 'exact',
      failureTime: failure.failureTime
    };
  }
  if (failure.kind === 'future_window') {
    return {
      kind: 'window',
      earliest: failure.earliestFailureTime,
      latest: failure.latestFailureTime,
      midpoint: (failure.earliestFailureTime + failure.latestFailureTime) / 2
    };
  }
  return null;
}

function predictBaselineFromTelemetry(vehicle, telemetry) {
  if (!vehicle || !telemetry) {
    return {
      classLabel: null,
      health: 'Waiting',
      confidence: 0,
      window: 'Waiting for telemetry',
      mode: 'No telemetry received yet',
      recommendation: 'Send a telemetry packet to update health.'
    };
  }

  const failureEstimate = getFinalFailureEstimate(vehicle);
  const finalFailure = vehicle.failure || {};

  if (failureEstimate?.kind === 'exact') {
    const timeToFailure = failureEstimate.failureTime - telemetry.t;
    const classLabel = classifyTimeToFailure(timeToFailure);
    return {
      classLabel,
      health: healthText(classLabel),
      confidence: confidenceForClass(classLabel, 'exact'),
      window: CLASS_WINDOWS[classLabel],
      mode: 'Baseline from observed training repair time',
      timeToFailure,
      recommendation: recommendationForClass(classLabel)
    };
  }

  if (failureEstimate?.kind === 'window') {
    const timeToFailure = failureEstimate.midpoint - telemetry.t;
    const classLabel = classifyTimeToFailure(timeToFailure);
    return {
      classLabel,
      health: healthText(classLabel),
      confidence: confidenceForClass(classLabel, 'estimated-window'),
      window: CLASS_WINDOWS[classLabel],
      mode: 'Baseline from validation/test labeled failure window',
      timeToFailure,
      recommendation: recommendationForClass(classLabel),
      estimatedWindow: failureEstimate
    };
  }

  if (finalFailure.kind === 'censored') {
    return {
      classLabel: 0,
      health: healthText(0),
      confidence: 0.62,
      window: CLASS_WINDOWS[0],
      mode: 'Baseline from censored training record',
      recommendation: 'Continue monitoring. No repair was observed in the study window.'
    };
  }

  return {
    classLabel: 0,
    health: healthText(0),
    confidence: 0.7,
    window: CLASS_WINDOWS[0],
    mode: 'Baseline from class 0 label',
    recommendation: 'Continue monitoring.'
  };
}

function vectorFromTelemetry(vehicle, telemetry) {
  if (!state.model || !vehicle || !telemetry) return null;
  const previous = vehicle.series[Math.max(0, state.streamIndex - 1)] || telemetry;
  const dt = Math.max(0, telemetry.t - previous.t);

  return state.model.featureNames.map(featureName => {
    if (featureName === 'time_step_log') return log1pNonNegative(telemetry.t);

    const counterMatch = featureName.match(/^counter:(.+):log_(value|delta|rate)$/);
    if (counterMatch) {
      const counterName = counterMatch[1];
      const kind = counterMatch[2];
      const currentValue = telemetry.counters[counterName] ?? 0;
      const previousValue = previous.counters[counterName] ?? currentValue;
      const delta = Math.max(0, currentValue - previousValue);
      if (kind === 'value') return log1pNonNegative(currentValue);
      if (kind === 'delta') return log1pNonNegative(delta);
      return log1pNonNegative(dt > 0 ? delta / dt : 0);
    }

    const specMatch = featureName.match(/^spec:(Spec_\d+)=(.+)$/);
    if (specMatch) {
      return vehicle.specs?.[specMatch[1]] === specMatch[2] ? 1 : 0;
    }

    return 0;
  });
}

function predictCentroidFromTelemetry(vehicle, telemetry) {
  if (!state.model) return predictBaselineFromTelemetry(vehicle, telemetry);
  const vector = vectorFromTelemetry(vehicle, telemetry);
  const scaled = vector.map((value, index) => (value - state.model.scaler.mean[index]) / state.model.scaler.std[index]);
  const classes = [0, 1, 2, 3, 4];
  const distances = classes.map(label => squaredDistance(scaled, state.model.centroids[label]));
  const scores = distances.map(distance => -Math.sqrt(distance) * 2.2);
  const probabilities = softmax(scores);
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }

  const rawClass = classes[bestIndex];
  const confidence = probabilities[bestIndex];
  const thresholdApplied = rawClass !== 0 && rawClass !== 4 && confidence < state.model.alertThreshold;
  const classLabel = thresholdApplied ? 0 : rawClass;

  return {
    classLabel,
    rawClass,
    health: healthText(classLabel),
    confidence,
    window: CLASS_WINDOWS[classLabel],
    mode: 'ML live scorer',
    recommendation: recommendationForClass(classLabel),
    probabilities: Object.fromEntries(classes.map((label, index) => [label, probabilities[index]])),
    thresholdApplied,
    topSignal: state.model.topFeaturesClass0VsClass4?.[0]?.featureName
  };
}

function pushNeighbor(neighbors, candidate, k) {
  if (neighbors.length < k) {
    neighbors.push(candidate);
    neighbors.sort((a, b) => b.distance - a.distance);
    return;
  }
  if (candidate.distance < neighbors[0].distance) {
    neighbors[0] = candidate;
    neighbors.sort((a, b) => b.distance - a.distance);
  }
}

function predictKnnFromTelemetry(vehicle, telemetry) {
  if (!state.model?.knn?.exemplars?.length) return predictCentroidFromTelemetry(vehicle, telemetry);
  const vector = vectorFromTelemetry(vehicle, telemetry);
  const scaled = vector.map((value, index) => (value - state.model.scaler.mean[index]) / state.model.scaler.std[index]);
  const k = state.model.knn?.k || 25;
  const distancePower = state.model.knn?.distancePower || 2;
  const neighbors = [];

  for (const exemplar of state.model.knn.exemplars) {
    const distance = squaredDistance(scaled, exemplar.vector);
    pushNeighbor(neighbors, { distance, label: exemplar.label, vehicle_id: exemplar.vehicle_id, time_step: exemplar.time_step }, k);
  }

  const votes = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const neighbor of neighbors) {
    const weight = 1 / ((neighbor.distance + 0.000001) ** distancePower);
    votes[neighbor.label] += weight;
  }

  const totalVote = Object.values(votes).reduce((sum, value) => sum + value, 0);
  let classLabel = 0;
  for (const label of [1, 2, 3, 4]) {
    if (votes[label] > votes[classLabel]) classLabel = label;
  }

  const confidence = totalVote === 0 ? 0 : votes[classLabel] / totalVote;
  const nearest = [...neighbors].sort((a, b) => a.distance - b.distance)[0];
  return {
    classLabel,
    rawClass: classLabel,
    health: healthText(classLabel),
    confidence,
    window: CLASS_WINDOWS[classLabel],
    mode: 'ML kNN experiment',
    recommendation: recommendationForClass(classLabel),
    probabilities: Object.fromEntries(Object.entries(votes).map(([label, vote]) => [label, totalVote === 0 ? 0 : vote / totalVote])),
    thresholdApplied: false,
    topSignal: state.model.topFeaturesClass0VsClass4?.[0]?.featureName,
    nearestNeighbor: nearest
  };
}

function evaluateLightgbmTree(node, vector) {
  if (!node) return 0;
  if (Object.prototype.hasOwnProperty.call(node, 'leaf_value')) return Number(node.leaf_value) || 0;
  const value = Number(vector[node.split_feature] ?? 0);
  const threshold = Number(node.threshold);
  const goLeft = Number.isNaN(value)
    ? Boolean(node.default_left)
    : value <= threshold;
  return evaluateLightgbmTree(goLeft ? node.left_child : node.right_child, vector);
}

function lightgbmProbabilities(vector) {
  const model = state.lightgbmModel;
  const numClass = model?.numClass || 5;
  const scores = Array(numClass).fill(0);
  for (const tree of model.treeInfo || []) {
    const classIndex = Number(tree.tree_index) % numClass;
    scores[classIndex] += evaluateLightgbmTree(tree.tree_structure, vector);
  }
  return softmax(scores);
}

function expectedCostsFromProbabilities(probabilities) {
  return [0, 1, 2, 3, 4].map(predicted =>
    probabilities.reduce((sum, probability, actual) => sum + probability * COST_MATRIX[actual][predicted], 0)
  );
}

function classFromLightgbmDecision(probabilities, decisionConfig) {
  const mode = decisionConfig?.mode || 'argmax';
  if (mode === 'expected_cost') {
    const expectedCosts = expectedCostsFromProbabilities(probabilities);
    return expectedCosts.indexOf(Math.min(...expectedCosts));
  }
  if (mode === 'cost_margin_threshold') {
    const expectedCosts = expectedCostsFromProbabilities(probabilities);
    const nonZero = expectedCosts.slice(1);
    const bestNonZeroCost = Math.min(...nonZero);
    const bestNonZeroClass = nonZero.indexOf(bestNonZeroCost) + 1;
    const savings = expectedCosts[0] - bestNonZeroCost;
    return savings >= Number(decisionConfig.threshold || 0) ? bestNonZeroClass : 0;
  }
  return probabilities.indexOf(Math.max(...probabilities));
}

function predictLightgbmFromTelemetry(vehicle, telemetry, modeKey = state.predictionMode) {
  if (!state.lightgbmModel?.treeInfo?.length) return predictCentroidFromTelemetry(vehicle, telemetry);
  const vector = vectorFromTelemetry(vehicle, telemetry);
  const probabilities = lightgbmProbabilities(vector);
  const decision = state.lightgbmModel.decisionModes?.[modeKey] || state.lightgbmModel.decisionModes?.lightgbm_cost;
  const riskScore = probabilities.reduce((sum, probability, label) => sum + probability * label, 0);
  const classLabel = modeKey === 'lightgbm_trend'
    ? Math.max(0, Math.min(4, Math.round(riskScore)))
    : classFromLightgbmDecision(probabilities, decision?.decisionConfig);
  const rawClass = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[classLabel] ?? Math.max(...probabilities);
  const decisionLabel = modeKey === 'lightgbm_trend'
    ? ` Probability risk score: ${riskScore.toFixed(2)} / 4.`
    : (decision?.decisionMode ? ` Decision: ${decision.decisionMode}.` : '');

  return {
    classLabel,
    rawClass,
    health: healthText(classLabel),
    confidence,
    window: CLASS_WINDOWS[classLabel],
    mode: PREDICTION_MODE_LABELS[modeKey] || 'LightGBM browser scorer',
    recommendation: `${recommendationForClass(classLabel)}${decisionLabel}`,
    probabilities: Object.fromEntries([0, 1, 2, 3, 4].map((label, index) => [label, probabilities[index]])),
    thresholdApplied: modeKey !== 'lightgbm_trend' && decision?.decisionConfig?.mode === 'cost_margin_threshold' && classLabel === 0 && rawClass !== 0,
    thresholdLabel: decision?.decisionMode,
    riskScore,
    topSignal: state.model?.topFeaturesClass0VsClass4?.[0]?.featureName
  };
}

function isMlPredictionMode(mode = state.predictionMode) {
  return mode !== 'baseline';
}

function shouldSmoothPredictionMode(mode = state.predictionMode) {
  return mode === 'lightgbm_trend' || mode === 'centroid';
}

function predictFromTelemetry(vehicle, telemetry) {
  if (state.predictionMode === 'baseline') return predictBaselineFromTelemetry(vehicle, telemetry);
  if (state.predictionMode === 'knn') return predictKnnFromTelemetry(vehicle, telemetry);
  if (state.predictionMode === 'centroid') return predictCentroidFromTelemetry(vehicle, telemetry);
  return predictLightgbmFromTelemetry(vehicle, telemetry, state.predictionMode);
}

function resetMlSmoothing() {
  state.lastMlSmoothing = null;
}

function applyMlTemporalSmoothing(vehicle, prediction) {
  if (!isMlPredictionMode() || !shouldSmoothPredictionMode() || !vehicle || prediction.classLabel === null || prediction.classLabel === undefined) {
    return prediction;
  }

  const candidateClass = Number(prediction.classLabel);
  if (!Number.isFinite(candidateClass)) return prediction;

  const previous = state.lastMlSmoothing;
  const sameVehicle = previous?.vehicleKey === vehicle.key;
  const samePacket = sameVehicle && previous.streamIndex === state.streamIndex;
  const sequentialPacket = sameVehicle && previous.streamIndex + 1 === state.streamIndex;
  let classLabel = candidateClass;
  let lowerReadingStreak = 0;
  let smoothingNote = '';

  if (samePacket) {
    classLabel = previous.classLabel;
    lowerReadingStreak = previous.lowerReadingStreak || 0;
    smoothingNote = previous.smoothingNote || '';
  } else if (sequentialPacket) {
    const previousClass = Number(previous.classLabel);
    if (candidateClass >= previousClass) {
      classLabel = candidateClass;
      lowerReadingStreak = 0;
    } else {
      lowerReadingStreak = (previous.lowerReadingStreak || 0) + 1;
      if (lowerReadingStreak >= LOWER_RISK_CONFIRMATIONS) {
        classLabel = Math.max(candidateClass, previousClass - 1);
        lowerReadingStreak = 0;
        if (classLabel !== candidateClass) {
          smoothingNote = `Raw ML read ${classText(candidateClass)}, so the alert stepped down gradually to avoid a sudden risk collapse.`;
        }
      } else {
        classLabel = previousClass;
        smoothingNote = `Raw ML read ${classText(candidateClass)}, but the dashboard is waiting for ${LOWER_RISK_CONFIRMATIONS} lower-risk packets before lowering the alert.`;
      }
    }
  }

  state.lastMlSmoothing = {
    vehicleKey: vehicle.key,
    streamIndex: state.streamIndex,
    classLabel,
    lowerReadingStreak,
    smoothingNote
  };

  if (classLabel === candidateClass) {
    return {
      ...prediction,
      unsmoothedClass: candidateClass,
      smoothed: false
    };
  }

  return {
    ...prediction,
    unsmoothedClass: candidateClass,
    classLabel,
    health: healthText(classLabel),
    window: CLASS_WINDOWS[classLabel],
    recommendation: recommendationForClass(classLabel),
    smoothed: true,
    smoothingNote
  };
}

function recommendationForClass(cls) {
  if (cls === 4) return 'Immediate inspection recommended.';
  if (cls === 3) return 'Schedule inspection soon.';
  if (cls === 2) return 'Monitor closely and plan maintenance.';
  if (cls === 1) return 'Watch trend; maintenance window is approaching.';
  return 'Normal monitoring.';
}

function getChartContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function loadSummary() {
  const data = state.data;
  const totalVehicles = data.vehicles.length;
  const totalReadouts = Object.values(data.rawDataset.operationalRows).reduce((sum, count) => sum + count, 0);
  els.metricVehicles.textContent = number(totalVehicles);
  els.metricReadouts.textContent = number(totalReadouts);
  els.metricCounters.textContent = data.featureMetadata.counters.length;
  els.metricHistograms.textContent = data.featureMetadata.histograms.length;

  els.componentAnswer.textContent = data.componentAnswer.explanation;
  els.timeWarning.textContent = data.timeInterpretation.warning;
  els.articleFindings.innerHTML = data.articleFindings
    .map(item => `<div class="truth-item">${item}</div>`)
    .join('');
}

function loadModelSummary() {
  if (!state.model) {
    els.modelName.textContent = 'Model file not loaded';
    els.modelValidationCost.textContent = '-';
    els.modelTestCost.textContent = '-';
    els.modelTopFeature.textContent = '-';
    els.modelComparisonTable.innerHTML = '';
    return;
  }

  const recommendedName = state.model.modelSelection?.recommendedModel;
  const comparisonRows = state.model.modelComparison || [];
  const recommendedRow = comparisonRows.find(row => row.name === recommendedName);
  const displayedValidation = recommendedRow?.validation || state.model.evaluation.validation;
  const displayedTest = recommendedRow?.test || state.model.evaluation.test;

  els.modelName.textContent = recommendedName ? `${recommendedName} (fair comparison)` : `${state.model.modelName} (${state.model.modelType})`;
  els.modelValidationCost.textContent = `${number(displayedValidation.totalCost)} vs ${number(displayedValidation.allZeroBaselineCost)}`;
  els.modelTestCost.textContent = `${number(displayedTest.totalCost)} vs ${number(displayedTest.allZeroBaselineCost)}`;
  els.modelTopFeature.textContent = state.model.topFeaturesClass0VsClass4?.[0]?.featureName || '-';
  renderModelComparison();
}

function renderModelComparison() {
  const rows = state.model?.modelComparison || [];
  if (!rows.length) {
    els.modelComparisonTable.innerHTML = '<div>No comparison results exported.</div>';
    return;
  }

  const cells = [
    ['Model', 'comparison-heading'],
    ['Val accuracy', 'comparison-heading'],
    ['Validation cost', 'comparison-heading'],
    ['Test accuracy', 'comparison-heading'],
    ['Test cost', 'comparison-heading'],
    ['Scope', 'comparison-heading']
  ];

  const recommendedName = state.model?.modelSelection?.recommendedModel;
  rows.forEach(row => {
    const isSelected = recommendedName ? row.name === recommendedName : row.type === state.model.modelType;
    cells.push([row.name, isSelected ? 'comparison-selected' : '']);
    cells.push([`${Math.round((row.validation?.accuracy ?? 0) * 1000) / 10}%`, isSelected ? 'comparison-selected' : '']);
    cells.push([number(row.validation?.totalCost ?? 0), isSelected ? 'comparison-selected' : '']);
    cells.push([`${Math.round((row.test?.accuracy ?? 0) * 1000) / 10}%`, '']);
    cells.push([number(row.test?.totalCost ?? 0), '']);
    cells.push([row.validation?.scope || '-', '']);
  });

  els.modelComparisonTable.innerHTML = cells
    .map(([value, className]) => `<div class="${className}">${value}</div>`)
    .join('');
}

function renderLiveCounterInputs() {
  els.liveCounterForm.innerHTML = state.data.featureMetadata.counters
    .map(name => `
      <label>
        <span>${name}</span>
        <input type="number" min="0" step="0.001" data-counter="${name}">
      </label>
    `)
    .join('');
}

function populateLiveForm() {
  const vehicle = state.selectedVehicle;
  const latest = latestSeriesPoint(vehicle);
  if (!vehicle || !latest) return;

  els.liveTimeStep.value = fixed(nextLiveTimeStep(vehicle), 3);
  els.liveCounterForm.querySelectorAll('input[data-counter]').forEach(input => {
    const counterName = input.dataset.counter;
    input.value = fixed(latest.counters[counterName] ?? 0, 3);
  });
}

function updateTelemetryModeUi() {
  const liveMode = state.telemetryMode === 'live';
  els.liveIntake.hidden = !liveMode;
  els.sendTelemetry.disabled = liveMode;
  els.toggleAutoplay.disabled = liveMode;
  els.playbackSpeed.disabled = liveMode;
  els.jumpTime.disabled = liveMode;
  els.jumpTimeButton.disabled = liveMode;
  els.jumpDate.disabled = liveMode || els.calendarUnit.value === 'unit';
  els.jumpDateButton.disabled = liveMode || els.calendarUnit.value === 'unit';
  els.jumpFailureWindow.disabled = liveMode;
  if (liveMode) stopAutoplay();
  if (liveMode) populateLiveForm();
}

function loadControls() {
  els.counterSelect.innerHTML = state.data.featureMetadata.counters
    .map(name => `<option value="${name}">${name}</option>`)
    .join('');
  state.counterFeature = state.data.featureMetadata.counters.includes('427_0')
    ? '427_0'
    : state.data.featureMetadata.counters[0];
  els.counterSelect.value = state.counterFeature;

  els.histogramSelect.innerHTML = state.data.featureMetadata.histograms
    .map(item => `<option value="${item.variableCode}">Variable ${item.variableCode} (${item.binCount} bins)</option>`)
    .join('');
  state.histogramVariable = state.data.featureMetadata.histograms.find(item => item.variableCode === '397')?.variableCode
    || state.data.featureMetadata.histograms[0].variableCode;
  els.histogramSelect.value = state.histogramVariable;
  renderLiveCounterInputs();
  updateTelemetryModeUi();
}

function filteredVehicles() {
  const search = els.search.value.trim();
  const split = els.splitFilter.value;
  const risk = els.riskFilter.value;

  return state.data.vehicles.filter(vehicle => {
    if (search && !String(vehicle.vehicle_id).includes(search)) return false;
    if (split !== 'all' && vehicle.split !== split) return false;
    if (risk !== 'all' && String(vehicle.riskClass) !== risk) return false;
    return true;
  });
}

function renderVehicleList() {
  const vehicles = filteredVehicles().slice(0, 300);
  els.vehicleList.innerHTML = vehicles.map(vehicle => {
    const active = state.selectedVehicle && state.selectedVehicle.key === vehicle.key ? ' active' : '';
    const cls = vehicle.riskClass ?? 'unknown';
    return `
      <button class="vehicle-button${active}" type="button" data-key="${vehicle.key}">
        <span>
          <strong>${vehicle.vehicle_id}</strong>
          <span class="meta">${vehicle.split} | latest t=${fixed(vehicle.latest?.t, 1)} | ${vehicle.timelineKind.replaceAll('_', ' ')}</span>
        </span>
        <span class="class-badge" style="background:${RISK_COLORS[cls]}">${classText(vehicle.riskClass)}</span>
      </button>
    `;
  }).join('');

  els.vehicleList.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
      selectVehicle(button.dataset.key);
    });
  });
}

function selectVehicle(key) {
  stopAutoplay();
  state.selectedVehicle = state.data.vehicles.find(vehicle => vehicle.key === key) || state.data.vehicles[0];
  state.streamIndex = 0;
  state.eventLog = [];
  resetMlSmoothing();
  const first = currentTelemetry();
  addLog(`Vehicle ${state.selectedVehicle.vehicle_id} selected. Stream reset to t=${fixed(first?.t, 1)}.`);
  populateLiveForm();
  updateTelemetryModeUi();
  renderVehicle();
  renderVehicleList();
}

function renderVehicle() {
  const vehicle = state.selectedVehicle;
  if (!vehicle) return;
  const telemetry = currentTelemetry();
  let prediction = predictFromTelemetry(vehicle, telemetry);
  prediction = applyMlTemporalSmoothing(vehicle, prediction);
  const visibleSeries = streamedSeries(vehicle);

  els.vehicleTitle.textContent = `Vehicle ${vehicle.vehicle_id}`;
  els.vehicleSubtitle.textContent = `${vehicle.split} split | packet ${state.streamIndex + 1} of ${vehicle.series.length} | current ${formatPseudoDate(telemetry.t)}`;
  const cls = prediction.classLabel ?? 'unknown';
  els.riskPill.textContent = classText(prediction.classLabel);
  els.riskPill.style.background = RISK_COLORS[cls];
  const telemetrySource = telemetry.source === 'live_insert' ? 'Real-time insert' : 'Historical replay';

  els.vehicleFacts.innerHTML = [
    ['Current time_step', fixed(telemetry.t, 1)],
    ['Calendar display', formatPseudoDate(telemetry.t)],
    ['Timeline type', vehicle.timelineKind.replaceAll('_', ' ')],
    ['Packets received', `${visibleSeries.length} / ${vehicle.series.length}`],
    ['Current source', telemetrySource],
    ['Live inserts', liveRecordsForVehicle(vehicle).length]
  ].map(([label, value]) => `
    <div class="fact">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join('');

  updateJumpControlHints(telemetry);
  renderPrediction(prediction, telemetry);
  renderTelemetryTable(telemetry, prediction);
  renderFailureExplanation(vehicle, prediction, telemetry);
  drawTimeline(vehicle, visibleSeries, prediction);
  drawHistogram(vehicle);
}

function updateJumpControlHints(telemetry) {
  els.jumpTime.placeholder = fixed(telemetry.t, 1);
  const isoDate = timeStepToIsoDate(telemetry.t);
  els.jumpDate.disabled = state.telemetryMode === 'live' || els.calendarUnit.value === 'unit';
  els.jumpDateButton.disabled = state.telemetryMode === 'live' || els.calendarUnit.value === 'unit';
  if (isoDate) els.jumpDate.placeholder = isoDate;
}

function renderPrediction(prediction, telemetry) {
  const cls = prediction.classLabel ?? 'unknown';
  els.predictionCard.style.background = RISK_COLORS[cls];
  els.healthStatus.textContent = prediction.health;
  const confidenceLabel = prediction.smoothed ? 'raw ML confidence' : 'confidence';
  const thresholdLine = prediction.thresholdApplied
    ? `<br>Raw ML class ${prediction.rawClass} was suppressed by ${prediction.thresholdLabel || `threshold ${state.model?.alertThreshold}`}.`
    : '';
  const smoothingLine = prediction.smoothed
    ? `<br>Temporal smoothing: ${prediction.smoothingNote}`
    : '';
  const topSignalLine = prediction.topSignal ? `<br>Top learned class-4 signal: ${prediction.topSignal}` : '';
  const neighborLine = prediction.nearestNeighbor
    ? `<br>Nearest training example: vehicle ${prediction.nearestNeighbor.vehicle_id}, t=${fixed(prediction.nearestNeighbor.time_step, 1)}, class ${prediction.nearestNeighbor.label}.`
    : '';
  els.predictionText.innerHTML = `
    ${classText(prediction.classLabel)} | ${Math.round(prediction.confidence * 100)}% ${confidenceLabel}<br>
    ${prediction.window}<br>
    ${prediction.recommendation}${thresholdLine}${smoothingLine}${topSignalLine}${neighborLine}
  `;
  const telemetryMode = state.telemetryMode === 'live' ? 'real-time insert' : 'historical replay';
  els.simulatorCaption.textContent = `Current telemetry t=${fixed(telemetry.t, 1)}. ${telemetryMode}; ${prediction.mode}.`;
}

function renderTelemetryTable(telemetry, prediction) {
  const counterRows = Object.entries(telemetry.counters).slice(0, 8);
  const predictionLabel = prediction.smoothed && prediction.unsmoothedClass !== prediction.classLabel
    ? `${classText(prediction.classLabel)} / ${prediction.health} (raw ${classText(prediction.unsmoothedClass)})`
    : `${classText(prediction.classLabel)} / ${prediction.health}`;
  const baseRows = [
    ['vehicle_id', state.selectedVehicle.vehicle_id],
    ['time_step', fixed(telemetry.t, 1)],
    ['display_time', formatPseudoDate(telemetry.t)],
    ['packet_source', telemetry.source === 'live_insert' ? 'Real-time insert' : 'Historical replay'],
    ['prediction', predictionLabel],
    ['model_mode', PREDICTION_MODE_LABELS[state.predictionMode] || state.predictionMode]
  ];

  els.telemetryTable.innerHTML = [...baseRows, ...counterRows]
    .map(([label, value]) => `
      <div class="telemetry-cell">
        <span>${label}</span>
        <strong>${typeof value === 'number' ? number(Math.round(value)) : value}</strong>
      </div>
    `)
    .join('');
}

function resetStream() {
  if (!state.selectedVehicle) return;
  stopAutoplay();
  state.streamIndex = 0;
  resetMlSmoothing();
  addLog(`Stream reset for vehicle ${state.selectedVehicle.vehicle_id}. Prediction refreshed.`);
  populateLiveForm();
  renderVehicle();
  renderVehicleList();
}

function jumpToTimeStep(targetTimeStep, sourceLabel = 'manual jump') {
  if (!state.selectedVehicle || !Number.isFinite(targetTimeStep)) {
    addLog('Jump ignored: enter a valid time_step or display date.');
    return;
  }

  stopAutoplay();
  const previousIndex = state.streamIndex;
  state.streamIndex = nearestSeriesIndex(state.selectedVehicle, targetTimeStep);
  resetMlSmoothing();
  const telemetry = currentTelemetry();
  addLog(`${sourceLabel}: requested t=${fixed(targetTimeStep, 1)}, moved packet ${previousIndex + 1} -> ${state.streamIndex + 1} at t=${fixed(telemetry.t, 1)}. Prediction refreshed.`);
  renderVehicle();
  renderVehicleList();
}

function jumpToDisplayDate() {
  const targetTimeStep = dateToTimeStep(els.jumpDate.value);
  if (targetTimeStep === null) {
    addLog('Date jump needs day or hour display mode. Dataset-unit mode has no calendar date.');
    return;
  }
  jumpToTimeStep(targetTimeStep, `Display date jump (${els.jumpDate.value})`);
}

function jumpNearFailureWindow() {
  const vehicle = state.selectedVehicle;
  if (!vehicle) return;

  const failure = vehicle.failure || {};
  let targetTimeStep = null;
  if (failure.kind === 'observed_repair') {
    targetTimeStep = Math.max(0, failure.failureTime - 3);
  } else if (failure.kind === 'future_window') {
    targetTimeStep = failure.earliestFailureTime;
  }

  if (targetTimeStep === null) {
    addLog('This vehicle does not have a known failure marker/window to jump to.');
    return;
  }
  jumpToTimeStep(targetTimeStep, 'Failure-window jump');
}

function readLiveCounters(vehicle, latest) {
  const counters = {};
  els.liveCounterForm.querySelectorAll('input[data-counter]').forEach(input => {
    const counterName = input.dataset.counter;
    const parsed = Number(input.value);
    counters[counterName] = Number.isFinite(parsed)
      ? round(Math.max(0, parsed), 3)
      : (latest?.counters?.[counterName] ?? 0);
  });
  return counters;
}

async function insertLiveRecord() {
  const vehicle = state.selectedVehicle;
  if (!vehicle) return;

  stopAutoplay();
  const latest = latestSeriesPoint(vehicle);
  const timeStep = round(Number(els.liveTimeStep.value), 3);
  if (!Number.isFinite(timeStep)) {
    addLog('Live insert ignored: enter a valid time_step.');
    return;
  }
  if (latest && timeStep <= latest.t) {
    addLog(`Live insert ignored: time_step must be greater than latest t=${fixed(latest.t, 1)}.`);
    return;
  }

  const counters = readLiveCounters(vehicle, latest);
  try {
    const payload = await apiRequest('/api/telemetry-records', {
      method: 'POST',
      authenticated: true,
      body: {
        vehicle_db_id: vehicle.dbId,
        time_step: timeStep,
        source: 'live_insert',
        ...Object.fromEntries(
          Object.entries(counters).map(([counterName, value]) => [`counter_${counterName}`, value])
        )
      }
    });
    const saved = payload.telemetry;
    const liveRecord = {
      id: Number(saved.id),
      t: Number(saved.time_step),
      counters,
      source: saved.source,
      receivedAt: saved.created_at
    };

    vehicle.series.push(liveRecord);
    vehicle.latest = { t: liveRecord.t, counters: liveRecord.counters, source: liveRecord.source };
    state.streamIndex = vehicle.series.length - 1;
    resetMlSmoothing();
    addLog(`Telemetry saved to database for vehicle ${vehicle.vehicle_id} at t=${fixed(timeStep, 1)}. LightGBM prediction refreshed.`);
    renderVehicle();
    renderVehicleList();
    populateLiveForm();
  } catch (error) {
    addLog(`Live insert failed: ${error.message}`);
  }
}

async function clearSelectedLiveRecords() {
  const vehicle = state.selectedVehicle;
  if (!vehicle) return;

  const liveRecords = liveRecordsForVehicle(vehicle);
  const liveCount = liveRecords.length;
  if (!liveCount) {
    addLog(`No live records to clear for vehicle ${vehicle.vehicle_id}.`);
    return;
  }

  stopAutoplay();
  try {
    await Promise.all(
      liveRecords.map(record => apiRequest(`/api/telemetry-records/${record.id}`, {
        method: 'DELETE',
        authenticated: true
      }))
    );
    vehicle.series = vehicle.series.filter(point => point.source !== 'live_insert');
    vehicle.latest = latestSeriesPoint(vehicle);
    state.streamIndex = Math.min(state.streamIndex, vehicle.series.length - 1);
    resetMlSmoothing();
    addLog(`Soft-deleted ${liveCount} database telemetry record${liveCount === 1 ? '' : 's'} for vehicle ${vehicle.vehicle_id}.`);
    renderVehicle();
    renderVehicleList();
    populateLiveForm();
  } catch (error) {
    addLog(`Clear failed: ${error.message}`);
  }
}

function sendNextTelemetry() {
  if (!state.selectedVehicle) return;
  if (state.streamIndex < state.selectedVehicle.series.length - 1) {
    state.streamIndex++;
  } else {
    addLog(`End of stream reached for vehicle ${state.selectedVehicle.vehicle_id}.`);
    stopAutoplay();
    renderVehicle();
    return;
  }

  const telemetry = currentTelemetry();
  addLog(`Telemetry received: vehicle ${state.selectedVehicle.vehicle_id}, t=${fixed(telemetry.t, 1)}. Prediction refreshed.`);
  renderVehicle();
  renderVehicleList();
}

function toggleAutoplay() {
  if (state.isPlaying) {
    stopAutoplay();
    return;
  }
  state.isPlaying = true;
  els.toggleAutoplay.textContent = 'Pause';
  addLog(`Auto play started at ${playbackDelay()}ms per packet.`);
  state.playbackTimer = window.setInterval(sendNextTelemetry, playbackDelay());
}

function stopAutoplay() {
  if (state.playbackTimer) {
    window.clearInterval(state.playbackTimer);
    state.playbackTimer = null;
  }
  state.isPlaying = false;
  if (els.toggleAutoplay) els.toggleAutoplay.textContent = 'Auto Play';
}

function renderFailureExplanation(vehicle, prediction, telemetry) {
  const failure = vehicle.failure || {};
  let body = '';

  if (failure.kind === 'observed_repair') {
    const currentTimeToFailure = failure.failureTime - telemetry.t;
    body = `
      <p><strong>Observed training repair:</strong> Component X was repaired at time_step ${fixed(failure.failureTime, 1)}.</p>
      <p>From the current streamed telemetry packet, the repair is ${fixed(currentTimeToFailure, 1)} dataset time units away.</p>
    `;
  } else if (failure.kind === 'future_window') {
    body = `
      <p><strong>Validation/test failure window:</strong> this vehicle has ${classText(failure.classLabel)}.</p>
      <p>The failure is expected between time_step ${fixed(failure.earliestFailureTime, 1)} and ${fixed(failure.latestFailureTime, 1)} based on the class window.</p>
    `;
  } else if (failure.kind === 'censored') {
    body = `
      <p><strong>Censored training record:</strong> no Component X repair was observed before time_step ${fixed(failure.censorTime, 1)}.</p>
      <p>This does not prove the component never fails. It means no failure event was recorded during the study window.</p>
    `;
  } else {
    body = `
      <p><strong>No imminent failure label:</strong> the latest readout is outside the 48-time-unit warning window.</p>
      <p>For class 0, the data does not provide a specific future failure time.</p>
    `;
  }

  const scoringLine = state.predictionMode === 'baseline'
    ? '<p><strong>Dashboard scoring:</strong> baseline timeline mode uses the known repair/failure-window labels to explain the timeline.</p>'
    : `<p><strong>Dashboard scoring:</strong> live playback is currently using ${PREDICTION_MODE_LABELS[state.predictionMode] || state.predictionMode}. The comparison above still recommends LightGBM cost-sensitive by full-validation SCANIA cost.</p>`;
  const liveLine = telemetry.source === 'live_insert'
    ? '<p><strong>Inserted telemetry:</strong> this manual record was added to the selected vehicle and re-scored from its counter values, deltas, rates, time_step, and hidden vehicle specs.</p>'
    : '';

  els.failureExplanation.innerHTML = `
    ${body}
    <p><strong>Current health output:</strong> ${prediction.health}, ${classText(prediction.classLabel)}. ${prediction.recommendation}</p>
    ${scoringLine}
    ${liveLine}
    <p><strong>Data note:</strong> counters, histograms, and specs are anonymized signals from the same vehicle. The target is one anonymized part: Component X.</p>
  `;
}

function drawAxes(ctx, plot, xTicks, yTicks, formatX, formatY) {
  ctx.strokeStyle = '#d6dee4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y);
  ctx.lineTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.stroke();

  ctx.fillStyle = '#53616b';
  ctx.font = '12px Aptos, Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  xTicks.forEach(tick => {
    const x = plot.xScale(tick);
    ctx.strokeStyle = '#edf1f4';
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.fillText(formatX(tick), x, plot.y + plot.h + 8);
  });

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  yTicks.forEach(tick => {
    const y = plot.yScale(tick);
    ctx.strokeStyle = '#edf1f4';
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillStyle = '#53616b';
    ctx.fillText(formatY(tick), plot.x - 8, y);
  });
}

function drawTimeline(vehicle, visibleSeries, prediction) {
  const { ctx, width, height } = getChartContext(els.timelineCanvas);
  ctx.clearRect(0, 0, width, height);

  const feature = state.counterFeature;
  const points = visibleSeries
    .map(point => ({ t: point.t, y: point.counters[feature], source: point.source }))
    .filter(point => point.y !== null && point.y !== undefined);

  if (points.length < 2) {
    ctx.fillStyle = '#53616b';
    ctx.fillText('Not enough data for this vehicle.', 20, 40);
    return;
  }

  let minT = Math.min(...points.map(point => point.t));
  let maxT = Math.max(...points.map(point => point.t));
  const failure = vehicle.failure || {};
  const currentPoint = currentTelemetry();
  const showFutureWindow = failure.kind === 'future_window' && currentPoint && currentPoint.t >= failure.earliestFailureTime;
  const observedFailureLead = failure.kind === 'observed_repair' && Number.isFinite(failure.failureTime)
    ? failure.failureTime - (currentPoint?.t ?? 0)
    : Infinity;
  const showObservedFailure = failure.kind === 'observed_repair' && currentPoint && observedFailureLead <= 6;
  if (showObservedFailure) {
    maxT = Math.max(maxT, failure.failureTime);
    maxT += Math.max(1, (maxT - minT) * 0.02);
  }
  if (showFutureWindow) maxT = Math.max(maxT, failure.latestFailureTime);
  if (failure.censorTime) maxT = Math.max(maxT, failure.censorTime);
  if (maxT === minT) maxT = minT + 1;

  let minY = Math.min(...points.map(point => point.y));
  let maxY = Math.max(...points.map(point => point.y));
  if (maxY === minY) maxY = minY + 1;
  const paddingY = (maxY - minY) * 0.08;
  minY -= paddingY;
  maxY += paddingY;

  const plot = {
    x: 64,
    y: 24,
    w: width - 96,
    h: height - 78
  };
  plot.xScale = t => plot.x + ((t - minT) / (maxT - minT)) * plot.w;
  plot.yScale = y => plot.y + plot.h - ((y - minY) / (maxY - minY)) * plot.h;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map(frac => minT + (maxT - minT) * frac);
  const yTicks = [0, 0.5, 1].map(frac => minY + (maxY - minY) * frac);
  drawAxes(ctx, plot, ticks, yTicks, tick => fixed(tick, 0), tick => number(Math.round(tick)));

  if (showFutureWindow) {
    const x1 = plot.xScale(failure.earliestFailureTime);
    const x2 = plot.xScale(failure.latestFailureTime);
    ctx.fillStyle = 'rgba(189, 63, 71, 0.16)';
    ctx.fillRect(x1, plot.y, Math.max(2, x2 - x1), plot.h);
    ctx.fillStyle = '#bd3f47';
    ctx.font = '12px Aptos, Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('failure window', (x1 + x2) / 2, plot.y + 8);
  }

  if (failure.kind === 'censored') {
    const x = plot.xScale(failure.censorTime);
    ctx.strokeStyle = '#52616b';
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#52616b';
    ctx.textAlign = 'center';
    ctx.font = '12px Aptos, Segoe UI, sans-serif';
    ctx.fillText('study end', x, plot.y + 8);
  }

  ctx.strokeStyle = '#2e6fbb';
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = plot.xScale(point.t);
    const y = plot.yScale(point.y);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  points.forEach(point => {
    const x = plot.xScale(point.t);
    const y = plot.yScale(point.y);
    ctx.fillStyle = point.source === 'live_insert' ? '#13714d' : '#2e6fbb';
    if (point.source === 'live_insert') {
      ctx.fillRect(x - 3, y - 3, 6, 6);
    } else {
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  if (currentPoint) {
    const currentValue = currentPoint.counters[feature];
    const x = plot.xScale(currentPoint.t);
    const y = plot.yScale(currentValue);
    ctx.fillStyle = '#11181d';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (showObservedFailure) {
    const x = plot.xScale(failure.failureTime);
    ctx.strokeStyle = '#bd3f47';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.fillStyle = '#bd3f47';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '12px Aptos, Segoe UI, sans-serif';
    ctx.fillText('repair/failure', x, plot.y + 8);
  }

  const liveCount = liveRecordsForVehicle(vehicle).length;
  const liveText = liveCount ? ` ${liveCount} live inserted record${liveCount === 1 ? '' : 's'} shown in green.` : '';
  const hiddenMarkerText = (failure.kind === 'observed_repair' && !showObservedFailure)
    ? ' Failure marker will appear when the streamed readout enters the final class-4 repair window.'
    : '';
  const shownMarkerText = showObservedFailure
    ? ` Red repair marker shown at labeled Component X repair t=${fixed(failure.failureTime, 1)}.`
    : '';
  els.timelineCaption.textContent = `${feature} streamed over dataset time. Current readout: t=${fixed(currentPoint?.t, 1)} / ${formatPseudoDate(currentPoint?.t)}. Health: ${prediction.health}.${liveText}${hiddenMarkerText}${shownMarkerText}`;
}

function drawHistogram(vehicle) {
  const { ctx, width, height } = getChartContext(els.histogramCanvas);
  ctx.clearRect(0, 0, width, height);

  const variableCode = state.histogramVariable;
  const values = vehicle.latestHistograms[variableCode] || [];
  if (!values.length) {
    ctx.fillStyle = '#53616b';
    ctx.fillText('No histogram data for this vehicle.', 20, 40);
    return;
  }

  const plot = { x: 48, y: 28, w: width - 70, h: height - 80 };
  const maxValue = Math.max(...values.filter(value => value !== null), 1);
  const barGap = 3;
  const barWidth = Math.max(4, (plot.w - barGap * (values.length - 1)) / values.length);

  ctx.strokeStyle = '#d6dee4';
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y);
  ctx.lineTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.stroke();

  values.forEach((value, index) => {
    const heightRatio = value === null ? 0 : value / maxValue;
    const barH = heightRatio * plot.h;
    const x = plot.x + index * (barWidth + barGap);
    const y = plot.y + plot.h - barH;
    ctx.fillStyle = value === null ? '#d6dee4' : '#13714d';
    ctx.fillRect(x, y, barWidth, barH);
  });

  ctx.fillStyle = '#53616b';
  ctx.font = '12px Aptos, Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const labelEvery = values.length > 20 ? 5 : 2;
  values.forEach((_, index) => {
    if (index % labelEvery === 0 || index === values.length - 1) {
      const x = plot.x + index * (barWidth + barGap) + barWidth / 2;
      ctx.fillText(index, x, plot.y + plot.h + 8);
    }
  });

  els.histogramCaption.textContent = `Variable ${variableCode} at latest readout. Bins are anonymized operating-condition buckets.`;
}

function wireEvents() {
  [els.search, els.splitFilter, els.riskFilter].forEach(element => {
    element.addEventListener('input', renderVehicleList);
  });

  els.resetStream.addEventListener('click', resetStream);
  els.sendTelemetry.addEventListener('click', sendNextTelemetry);
  els.toggleAutoplay.addEventListener('click', toggleAutoplay);
  els.telemetryMode.addEventListener('change', () => {
    state.telemetryMode = els.telemetryMode.value;
    updateTelemetryModeUi();
    addLog(`Telemetry mode changed to ${state.telemetryMode === 'live' ? 'real-time insert' : 'historical replay'}.`);
    renderVehicle();
  });
  els.playbackSpeed.addEventListener('change', () => {
    if (state.isPlaying) {
      stopAutoplay();
      toggleAutoplay();
    }
  });
  els.predictionMode.addEventListener('change', () => {
    state.predictionMode = els.predictionMode.value;
    resetMlSmoothing();
    addLog(`Prediction mode changed to ${PREDICTION_MODE_LABELS[state.predictionMode] || state.predictionMode}. Prediction refreshed.`);
    renderVehicle();
  });
  els.jumpTimeButton.addEventListener('click', () => jumpToTimeStep(Number(els.jumpTime.value), 'time_step jump'));
  els.jumpTime.addEventListener('keydown', event => {
    if (event.key === 'Enter') jumpToTimeStep(Number(els.jumpTime.value), 'time_step jump');
  });
  els.jumpDateButton.addEventListener('click', jumpToDisplayDate);
  els.jumpDate.addEventListener('keydown', event => {
    if (event.key === 'Enter') jumpToDisplayDate();
  });
  els.jumpFailureWindow.addEventListener('click', jumpNearFailureWindow);
  els.seedLiveRecord.addEventListener('click', () => {
    populateLiveForm();
    addLog('Live form seeded from the latest available packet.');
  });
  els.clearLiveRecords.addEventListener('click', clearSelectedLiveRecords);
  els.insertLiveRecord.addEventListener('click', insertLiveRecord);

  els.counterSelect.addEventListener('change', () => {
    state.counterFeature = els.counterSelect.value;
    renderVehicle();
  });

  els.histogramSelect.addEventListener('change', () => {
    state.histogramVariable = els.histogramSelect.value;
    renderVehicle();
  });

  els.calendarUnit.addEventListener('change', () => {
    updateTelemetryModeUi();
    renderVehicle();
    renderVehicleList();
  });
  window.addEventListener('resize', () => {
    renderVehicle();
  });
}

async function init() {
  const response = await fetch('/api/dashboard/data');
  if (!response.ok) throw new Error('The database-backed dashboard API did not respond.');
  state.data = await response.json();
  try {
    const modelResponse = await fetch('/api/dashboard/model-output');
    state.model = modelResponse.ok ? await modelResponse.json() : null;
  } catch (error) {
    state.model = null;
    console.warn('Model output not loaded:', error);
  }
  try {
    const lightgbmResponse = await fetch('/api/dashboard/lightgbm-model');
    state.lightgbmModel = lightgbmResponse.ok ? await lightgbmResponse.json() : null;
  } catch (error) {
    state.lightgbmModel = null;
    console.warn('LightGBM browser model not loaded:', error);
  }
  loadSummary();
  loadModelSummary();
  loadControls();
  wireEvents();
  const firstClass4 = state.data.vehicles.find(vehicle => vehicle.split === 'train' && vehicle.riskClass === 4)
    || state.data.vehicles[0];
  selectVehicle(firstClass4.key);
}

init().catch(error => {
  document.body.innerHTML = `<main class="truth-band"><h1>Dashboard failed to load</h1><p>${error.message}</p></main>`;
  console.error(error);
});
