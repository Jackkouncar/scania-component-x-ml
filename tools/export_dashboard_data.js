const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, '2024-34-2', '2024-34-2', 'data');
const OUT_DIR = path.join(ROOT, 'site', 'data');
const OUT_FILE = path.join(OUT_DIR, 'dashboard_data.json');

const BASE_DATE = '2024-01-01T00:00:00.000Z';
const TRAIN_REPAIRED_LIMIT = Number(process.env.TRAIN_REPAIRED_LIMIT || 400);
const TRAIN_HEALTHY_LIMIT = Number(process.env.TRAIN_HEALTHY_LIMIT || 120);
const EVAL_CLASS0_LIMIT = Number(process.env.EVAL_CLASS0_LIMIT || 120);
const EVAL_POSITIVE_LIMIT = Number(process.env.EVAL_POSITIVE_LIMIT || 9999);
const MAX_POINTS_PER_VEHICLE = Number(process.env.MAX_POINTS_PER_VEHICLE || 180);

const SPLITS = {
  train: {
    specs: 'train_specifications.csv',
    operational: 'train_operational_readouts.csv',
    target: 'train_tte.csv'
  },
  validation: {
    specs: 'validation_specifications.csv',
    operational: 'validation_operational_readouts.csv',
    labels: 'validation_labels.csv'
  },
  test: {
    specs: 'test_specifications.csv',
    operational: 'test_operational_readouts.csv',
    labels: 'test_labels.csv'
  }
};

const CLASS_WINDOWS = {
  0: { label: 'No imminent failure', minAfter: 48, maxAfter: null },
  1: { label: '48 to 24 time units before failure', minAfter: 24, maxAfter: 48 },
  2: { label: '24 to 12 time units before failure', minAfter: 12, maxAfter: 24 },
  3: { label: '12 to 6 time units before failure', minAfter: 6, maxAfter: 12 },
  4: { label: '6 to 0 time units before failure', minAfter: 0, maxAfter: 6 }
};

function csvPath(file) {
  return path.join(DATA_DIR, file);
}

function parseCsvLine(line) {
  return line.split(',');
}

function round(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function toNumber(value) {
  if (value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function readFirstLine(file) {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(file, { encoding: 'utf8', highWaterMark: 65536 });
    let data = '';
    stream.on('data', chunk => {
      data += chunk;
      const newlineIndex = data.indexOf('\n');
      if (newlineIndex >= 0) {
        stream.destroy();
        resolve(data.slice(0, newlineIndex).replace(/\r$/, ''));
      }
    });
    stream.on('error', reject);
    stream.on('end', () => resolve(data.replace(/\r$/, '')));
  });
}

async function readSmallCsv(file) {
  const rows = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(csvPath(file)),
    crlfDelay: Infinity
  });

  let header = null;
  for await (const line of rl) {
    if (!header) {
      header = parseCsvLine(line);
      continue;
    }
    if (!line) continue;
    const cells = parseCsvLine(line);
    const row = {};
    for (let i = 0; i < header.length; i++) row[header[i]] = cells[i] ?? '';
    rows.push(row);
  }
  return rows;
}

function vehicleKey(split, vehicleId) {
  return `${split}:${vehicleId}`;
}

function classifyTimeToFailure(timeToFailure) {
  if (timeToFailure === null || timeToFailure === undefined || timeToFailure > 48) return 0;
  if (timeToFailure > 24 && timeToFailure <= 48) return 1;
  if (timeToFailure > 12 && timeToFailure <= 24) return 2;
  if (timeToFailure > 6 && timeToFailure <= 12) return 3;
  if (timeToFailure >= 0 && timeToFailure <= 6) return 4;
  return 4;
}

function makeVehicle(vehicles, split, vehicleId) {
  const key = vehicleKey(split, vehicleId);
  if (!vehicles.has(key)) {
    vehicles.set(key, {
      key,
      vehicle_id: Number(vehicleId),
      split,
      specs: {},
      label: null,
      repair: null,
      riskClass: null,
      timelineKind: 'unknown',
      series: [],
      latest: null,
      latestHistograms: {}
    });
  }
  return vehicles.get(key);
}

function thinSeries(series, maxPoints) {
  if (series.length <= maxPoints) return series;
  const thinned = [];
  const lastIndex = series.length - 1;
  for (let i = 0; i < maxPoints; i++) {
    const index = Math.round((i / (maxPoints - 1)) * lastIndex);
    if (thinned.length === 0 || thinned[thinned.length - 1] !== series[index]) {
      thinned.push(series[index]);
    }
  }
  return thinned;
}

async function getFeatureMetadata() {
  const header = parseCsvLine(await readFirstLine(csvPath('train_operational_readouts.csv')));
  const featureNames = header.slice(2);
  const groups = new Map();
  featureNames.forEach((featureName, featureOffset) => {
    const [variableCode, binIndexText] = featureName.split('_');
    if (!groups.has(variableCode)) groups.set(variableCode, []);
    groups.get(variableCode).push({
      featureName,
      variableCode,
      binIndex: Number(binIndexText),
      cellIndex: featureOffset + 2
    });
  });

  const variables = [...groups.entries()]
    .map(([variableCode, features]) => ({
      variableCode,
      kind: features.length === 1 ? 'counter' : 'histogram',
      binCount: features.length,
      features: features.sort((a, b) => a.binIndex - b.binIndex)
    }))
    .sort((a, b) => Number(a.variableCode) - Number(b.variableCode));

  return {
    idColumns: header.slice(0, 2),
    variables,
    counterFeatures: variables.filter(v => v.kind === 'counter').map(v => v.features[0]),
    histogramVariables: variables.filter(v => v.kind === 'histogram')
  };
}

async function selectVehicles(vehicles) {
  const trainRows = await readSmallCsv(SPLITS.train.target);
  const fullTrainRepairCounts = { 0: 0, 1: 0 };
  let selectedHealthy = 0;
  let selectedRepaired = 0;

  for (const row of trainRows) {
    const repaired = Number(row.in_study_repair);
    fullTrainRepairCounts[repaired]++;

    const shouldPick =
      (repaired === 1 && selectedRepaired < TRAIN_REPAIRED_LIMIT) ||
      (repaired === 0 && selectedHealthy < TRAIN_HEALTHY_LIMIT);

    if (!shouldPick) continue;

    const vehicle = makeVehicle(vehicles, 'train', row.vehicle_id);
    vehicle.repair = {
      lengthOfStudyTimeStep: Number(row.length_of_study_time_step),
      inStudyRepair: repaired
    };
    vehicle.timelineKind = repaired === 1 ? 'observed_repair' : 'censored';
    vehicle.riskClass = repaired === 1 ? null : 0;

    if (repaired === 1) selectedRepaired++;
    else selectedHealthy++;
  }

  const fullLabelCounts = {};
  for (const split of ['validation', 'test']) {
    const rows = await readSmallCsv(SPLITS[split].labels);
    fullLabelCounts[split] = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    const pickedByClass = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };

    for (const row of rows) {
      const classLabel = Number(row.class_label);
      fullLabelCounts[split][classLabel]++;
      const limit = classLabel === 0 ? EVAL_CLASS0_LIMIT : EVAL_POSITIVE_LIMIT;
      if (pickedByClass[classLabel] >= limit) continue;

      const vehicle = makeVehicle(vehicles, split, row.vehicle_id);
      vehicle.label = { classLabel, window: CLASS_WINDOWS[classLabel] };
      vehicle.riskClass = classLabel;
      vehicle.timelineKind = classLabel === 0 ? 'no_imminent_failure' : 'future_window';
      pickedByClass[classLabel]++;
    }
  }

  return { fullTrainRepairCounts, fullLabelCounts };
}

async function attachSpecs(vehicles) {
  for (const [split, config] of Object.entries(SPLITS)) {
    const rl = readline.createInterface({
      input: fs.createReadStream(csvPath(config.specs)),
      crlfDelay: Infinity
    });

    let header = null;
    for await (const line of rl) {
      if (!header) {
        header = parseCsvLine(line);
        continue;
      }
      if (!line) continue;

      const cells = parseCsvLine(line);
      const key = vehicleKey(split, cells[0]);
      if (!vehicles.has(key)) continue;

      const vehicle = vehicles.get(key);
      for (let i = 1; i < header.length; i++) vehicle.specs[header[i]] = cells[i];
    }
  }
}

async function attachOperationalReadouts(vehicles, featureMetadata) {
  const rawOperationalRows = {};
  const selectedBySplit = new Map();
  for (const vehicle of vehicles.values()) {
    if (!selectedBySplit.has(vehicle.split)) selectedBySplit.set(vehicle.split, new Set());
    selectedBySplit.get(vehicle.split).add(vehicle.vehicle_id);
  }

  for (const [split, config] of Object.entries(SPLITS)) {
    const selectedIds = selectedBySplit.get(split) || new Set();
    rawOperationalRows[split] = 0;

    const rl = readline.createInterface({
      input: fs.createReadStream(csvPath(config.operational)),
      crlfDelay: Infinity
    });

    let header = null;
    for await (const line of rl) {
      if (!header) {
        header = parseCsvLine(line);
        continue;
      }
      if (!line) continue;
      rawOperationalRows[split]++;

      const cells = parseCsvLine(line);
      const vehicleId = Number(cells[0]);
      if (!selectedIds.has(vehicleId)) continue;

      const vehicle = vehicles.get(vehicleKey(split, vehicleId));
      const timeStep = Number(cells[1]);
      const counters = {};
      for (const feature of featureMetadata.counterFeatures) {
        counters[feature.featureName] = round(toNumber(cells[feature.cellIndex]), 3);
      }
      vehicle.series.push({ t: round(timeStep, 3), counters });

      if (!vehicle.latest || timeStep >= vehicle.latest.t) {
        vehicle.latest = { t: round(timeStep, 3), counters };
        const histograms = {};
        for (const variable of featureMetadata.histogramVariables) {
          histograms[variable.variableCode] = variable.features.map(feature => round(toNumber(cells[feature.cellIndex]), 3));
        }
        vehicle.latestHistograms = histograms;
      }
    }
  }

  for (const vehicle of vehicles.values()) {
    vehicle.series.sort((a, b) => a.t - b.t);
    vehicle.series = thinSeries(vehicle.series, MAX_POINTS_PER_VEHICLE);
    deriveFailureInfo(vehicle);
  }

  return rawOperationalRows;
}

function deriveFailureInfo(vehicle) {
  if (!vehicle.latest) return;

  if (vehicle.split === 'train' && vehicle.repair) {
    if (vehicle.repair.inStudyRepair === 1) {
      const failureTime = vehicle.repair.lengthOfStudyTimeStep;
      const timeToFailure = failureTime - vehicle.latest.t;
      vehicle.riskClass = classifyTimeToFailure(timeToFailure);
      vehicle.failure = {
        kind: 'observed_repair',
        failureTime: round(failureTime, 3),
        timeToFailureFromLatest: round(timeToFailure, 3),
        explanation: 'Observed Component X repair/replacement time from train_tte.csv.'
      };
    } else {
      vehicle.failure = {
        kind: 'censored',
        censorTime: round(vehicle.repair.lengthOfStudyTimeStep, 3),
        explanation: 'No Component X repair was observed during this study window.'
      };
    }
    return;
  }

  if ((vehicle.split === 'validation' || vehicle.split === 'test') && vehicle.label) {
    const classLabel = vehicle.label.classLabel;
    if (classLabel === 0) {
      vehicle.failure = {
        kind: 'no_imminent_failure',
        explanation: 'The last readout is not inside the 48-time-unit failure warning window.'
      };
      return;
    }

    const window = CLASS_WINDOWS[classLabel];
    vehicle.failure = {
      kind: 'future_window',
      classLabel,
      earliestFailureTime: round(vehicle.latest.t + window.minAfter, 3),
      latestFailureTime: round(vehicle.latest.t + window.maxAfter, 3),
      minAfterLatest: window.minAfter,
      maxAfterLatest: window.maxAfter,
      explanation: `Class ${classLabel}: failure occurs ${window.label.toLowerCase()}.`
    };
  }
}

function summarizeSelectedVehicles(vehicles) {
  const summary = {};
  for (const vehicle of vehicles.values()) {
    if (!summary[vehicle.split]) summary[vehicle.split] = { total: 0, riskClasses: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 } };
    summary[vehicle.split].total++;
    const cls = vehicle.riskClass ?? 'unknown';
    if (summary[vehicle.split].riskClasses[cls] === undefined) summary[vehicle.split].riskClasses[cls] = 0;
    summary[vehicle.split].riskClasses[cls]++;
  }
  return summary;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const vehicles = new Map();
  const featureMetadata = await getFeatureMetadata();
  const selectionSummaries = await selectVehicles(vehicles);
  await attachSpecs(vehicles);
  const rawOperationalRows = await attachOperationalReadouts(vehicles, featureMetadata);

  const vehicleList = [...vehicles.values()]
    .filter(vehicle => vehicle.latest && vehicle.series.length > 0)
    .sort((a, b) => {
      const riskDelta = (b.riskClass ?? -1) - (a.riskClass ?? -1);
      if (riskDelta !== 0) return riskDelta;
      return b.latest.t - a.latest.t;
    });

  const output = {
    generatedAt: new Date().toISOString(),
    baseDate: BASE_DATE,
    timeInterpretation: {
      trueUnit: 'anonymized relative dataset time unit',
      calendarDisplay: 'The site maps time_step 0 to 2024-01-01 for visualization only.',
      warning: 'The papers do not reveal whether one time unit is an hour, day, mile, duty cycle, or another business-defined unit. Real timestamps were replaced with relative times, and frequencies may have been modified.'
    },
    articleFindings: [
      'The failing part is intentionally named only Component X.',
      'The exact component name was anonymized for proprietary reasons.',
      'Operational variable names were anonymized and omitted for privacy/proprietary reasons.',
      'The operational data contains 14 anonymized variables: 8 counters and 6 histogram variables.',
      'Relative time_step values are used instead of original timestamps.',
      'Repair and readout frequencies may have been modified and are not necessarily representative of actual truck usage.'
    ],
    componentAnswer: {
      canNamePhysicalComponent: false,
      physicalComponentName: 'Unknown / anonymized Component X',
      explanation: 'The dataset supports predicting failure of one anonymized engine component, not naming fuel pump, tire, battery, or other physical parts.'
    },
    rawDataset: {
      operationalRows: rawOperationalRows,
      trainRepairCounts: selectionSummaries.fullTrainRepairCounts,
      labelCounts: selectionSummaries.fullLabelCounts
    },
    dashboardSubset: summarizeSelectedVehicles(new Map(vehicleList.map(vehicle => [vehicle.key, vehicle]))),
    featureMetadata: {
      counters: featureMetadata.counterFeatures.map(f => f.featureName),
      histograms: featureMetadata.histogramVariables.map(v => ({
        variableCode: v.variableCode,
        binCount: v.binCount,
        features: v.features.map(f => f.featureName)
      })),
      variables: featureMetadata.variables.map(v => ({
        variableCode: v.variableCode,
        kind: v.kind,
        binCount: v.binCount
      }))
    },
    classWindows: CLASS_WINDOWS,
    vehicles: vehicleList
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(output));
  console.log(`Wrote ${OUT_FILE}`);
  console.log(`Dashboard vehicles: ${vehicleList.length}`);
  console.log(`Raw operational rows scanned: train=${rawOperationalRows.train}, validation=${rawOperationalRows.validation}, test=${rawOperationalRows.test}`);
  console.log(`Counter features: ${output.featureMetadata.counters.join(', ')}`);
  console.log(`Histogram variables: ${output.featureMetadata.histograms.map(h => `${h.variableCode}(${h.binCount})`).join(', ')}`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
