const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, '2024-34-2', '2024-34-2', 'data');
const OUT_DIR = path.join(ROOT, 'ml');
const OUT_FILE = path.join(OUT_DIR, 'ml_dataset.json');

const TRAIN_CLASS0_LIMIT = Number(process.env.TRAIN_CLASS0_LIMIT || 20000);
const TRAIN_POSITIVE_LIMIT = Number(process.env.TRAIN_POSITIVE_LIMIT || 8000);

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

function csvPath(file) {
  return path.join(DATA_DIR, file);
}

function parseCsvLine(line) {
  return line.split(',');
}

function toNumber(value) {
  if (value === '' || value === undefined) return 0;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function log1pNonNegative(value) {
  return Math.log1p(Math.max(0, value));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
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

async function getOperationalMetadata() {
  const header = parseCsvLine(await readFirstLine(csvPath(SPLITS.train.operational)));
  const featureNames = header.slice(2);
  const groups = new Map();

  featureNames.forEach((featureName, featureOffset) => {
    const [variableCode, binText] = featureName.split('_');
    if (!groups.has(variableCode)) groups.set(variableCode, []);
    groups.get(variableCode).push({
      featureName,
      variableCode,
      binIndex: Number(binText),
      cellIndex: featureOffset + 2
    });
  });

  const variables = [...groups.entries()].map(([variableCode, features]) => ({
    variableCode,
    features: features.sort((a, b) => a.binIndex - b.binIndex),
    kind: features.length === 1 ? 'counter' : 'histogram'
  }));

  return {
    header,
    counters: variables
      .filter(variable => variable.kind === 'counter')
      .map(variable => variable.features[0])
      .sort((a, b) => Number(a.variableCode) - Number(b.variableCode))
  };
}

async function readSpecMapsAndCategories() {
  const specMaps = {};
  const specCategorySets = {};

  for (const [split, config] of Object.entries(SPLITS)) {
    specMaps[split] = new Map();
    const rows = await readSmallCsv(config.specs);
    for (const row of rows) {
      const specs = {};
      for (const [key, value] of Object.entries(row)) {
        if (!key.startsWith('Spec_')) continue;
        specs[key] = value;
        if (!specCategorySets[key]) specCategorySets[key] = new Set();
        specCategorySets[key].add(value);
      }
      specMaps[split].set(Number(row.vehicle_id), specs);
    }
  }

  const specFeatures = [];
  for (const specName of Object.keys(specCategorySets).sort()) {
    for (const category of [...specCategorySets[specName]].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))) {
      specFeatures.push({ specName, category, featureName: `spec:${specName}=${category}` });
    }
  }

  return { specMaps, specFeatures };
}

async function readTrainTargets() {
  const targets = new Map();
  const rows = await readSmallCsv(SPLITS.train.target);
  for (const row of rows) {
    targets.set(Number(row.vehicle_id), {
      lengthOfStudyTimeStep: Number(row.length_of_study_time_step),
      inStudyRepair: Number(row.in_study_repair)
    });
  }
  return targets;
}

async function readLabels(file) {
  const labels = new Map();
  const rows = await readSmallCsv(file);
  for (const row of rows) labels.set(Number(row.vehicle_id), Number(row.class_label));
  return labels;
}

function classFromTimeToFailure(timeToFailure) {
  if (timeToFailure > 48) return 0;
  if (timeToFailure > 24) return 1;
  if (timeToFailure > 12) return 2;
  if (timeToFailure > 6) return 3;
  if (timeToFailure >= 0) return 4;
  return null;
}

function makeFeatureNames(counters, specFeatures) {
  return [
    'time_step_log',
    ...counters.map(counter => `counter:${counter.featureName}:log_value`),
    ...counters.map(counter => `counter:${counter.featureName}:log_delta`),
    ...counters.map(counter => `counter:${counter.featureName}:log_rate`),
    ...specFeatures.map(feature => feature.featureName)
  ];
}

function makeVector({ timeStep, cells, counters, previousState, specs, specFeatures }) {
  const vector = [round(log1pNonNegative(timeStep))];
  const previousCounters = previousState?.counters || {};
  const previousTimeStep = previousState?.timeStep;
  const dt = previousTimeStep === undefined ? 0 : Math.max(0, timeStep - previousTimeStep);
  const currentCounters = {};
  const deltas = {};

  for (const counter of counters) {
    const value = toNumber(cells[counter.cellIndex]);
    currentCounters[counter.featureName] = value;
    vector.push(round(log1pNonNegative(value)));
  }

  for (const counter of counters) {
    const value = currentCounters[counter.featureName];
    const previousValue = previousCounters[counter.featureName] ?? value;
    const delta = Math.max(0, value - previousValue);
    deltas[counter.featureName] = delta;
    vector.push(round(log1pNonNegative(delta)));
  }

  for (const counter of counters) {
    const delta = deltas[counter.featureName];
    const rate = dt > 0 ? delta / dt : 0;
    vector.push(round(log1pNonNegative(rate)));
  }

  for (const feature of specFeatures) {
    vector.push(specs?.[feature.specName] === feature.category ? 1 : 0);
  }

  return { vector, currentState: { timeStep, counters: currentCounters } };
}

async function buildTrainRows(metadata, specMaps, specFeatures, targets) {
  const trainRows = [];
  const rawClassCounts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, skipped_after_repair: 0 };
  const sampledClassCounts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
  const previousByVehicle = new Map();
  const limits = { 0: TRAIN_CLASS0_LIMIT, 1: TRAIN_POSITIVE_LIMIT, 2: TRAIN_POSITIVE_LIMIT, 3: TRAIN_POSITIVE_LIMIT, 4: TRAIN_POSITIVE_LIMIT };

  const rl = readline.createInterface({
    input: fs.createReadStream(csvPath(SPLITS.train.operational)),
    crlfDelay: Infinity
  });

  let header = null;
  let scanned = 0;

  for await (const line of rl) {
    if (!header) {
      header = parseCsvLine(line);
      continue;
    }
    if (!line) continue;
    scanned++;

    const cells = parseCsvLine(line);
    const vehicleId = Number(cells[0]);
    const timeStep = Number(cells[1]);
    const target = targets.get(vehicleId);
    if (!target) continue;

    let label = 0;
    if (target.inStudyRepair === 1) {
      const timeToFailure = target.lengthOfStudyTimeStep - timeStep;
      label = classFromTimeToFailure(timeToFailure);
      if (label === null) {
        rawClassCounts.skipped_after_repair++;
        continue;
      }
    }

    rawClassCounts[label]++;
    const previousState = previousByVehicle.get(vehicleId);
    const { vector, currentState } = makeVector({
      timeStep,
      cells,
      counters: metadata.counters,
      previousState,
      specs: specMaps.train.get(vehicleId),
      specFeatures
    });
    previousByVehicle.set(vehicleId, currentState);

    if (sampledClassCounts[label] >= limits[label]) continue;
    sampledClassCounts[label]++;
    trainRows.push({
      split: 'train',
      vehicle_id: vehicleId,
      time_step: round(timeStep, 3),
      label,
      vector
    });
  }

  return { trainRows, rawClassCounts, sampledClassCounts, scanned };
}

async function buildEvalRows(split, metadata, specMaps, specFeatures, labels) {
  const latestByVehicle = new Map();
  const previousByVehicle = new Map();
  let scanned = 0;

  const rl = readline.createInterface({
    input: fs.createReadStream(csvPath(SPLITS[split].operational)),
    crlfDelay: Infinity
  });

  let header = null;
  for await (const line of rl) {
    if (!header) {
      header = parseCsvLine(line);
      continue;
    }
    if (!line) continue;
    scanned++;

    const cells = parseCsvLine(line);
    const vehicleId = Number(cells[0]);
    if (!labels.has(vehicleId)) continue;

    const timeStep = Number(cells[1]);
    const previousState = previousByVehicle.get(vehicleId);
    const { vector, currentState } = makeVector({
      timeStep,
      cells,
      counters: metadata.counters,
      previousState,
      specs: specMaps[split].get(vehicleId),
      specFeatures
    });
    previousByVehicle.set(vehicleId, currentState);

    latestByVehicle.set(vehicleId, {
      split,
      vehicle_id: vehicleId,
      time_step: round(timeStep, 3),
      label: labels.get(vehicleId),
      vector
    });
  }

  return { rows: [...latestByVehicle.values()].sort((a, b) => a.vehicle_id - b.vehicle_id), scanned };
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const metadata = await getOperationalMetadata();
  const { specMaps, specFeatures } = await readSpecMapsAndCategories();
  const targets = await readTrainTargets();
  const validationLabels = await readLabels(SPLITS.validation.labels);
  const testLabels = await readLabels(SPLITS.test.labels);
  const featureNames = makeFeatureNames(metadata.counters, specFeatures);

  console.log('Building training rows...');
  const train = await buildTrainRows(metadata, specMaps, specFeatures, targets);
  console.log('Building validation rows...');
  const validation = await buildEvalRows('validation', metadata, specMaps, specFeatures, validationLabels);
  console.log('Building test rows...');
  const test = await buildEvalRows('test', metadata, specMaps, specFeatures, testLabels);

  const output = {
    generatedAt: new Date().toISOString(),
    modelTask: 'Predict Component X risk class 0-4 from current telemetry counters, counter deltas/rates, relative time, and vehicle specs.',
    featureNames,
    counterFeatures: metadata.counters.map(counter => counter.featureName),
    specFeatureCount: specFeatures.length,
    classWindows: {
      0: 'No imminent failure or more than 48 relative time units before failure',
      1: '48 to 24 relative time units before failure',
      2: '24 to 12 relative time units before failure',
      3: '12 to 6 relative time units before failure',
      4: '6 to 0 relative time units before failure'
    },
    train: train.trainRows,
    validation: validation.rows,
    test: test.rows,
    summary: {
      trainRawRowsScanned: train.scanned,
      validationRowsScanned: validation.scanned,
      testRowsScanned: test.scanned,
      trainRawClassCounts: train.rawClassCounts,
      trainSampledClassCounts: train.sampledClassCounts,
      validationRows: validation.rows.length,
      testRows: test.rows.length
    }
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(output));
  console.log(`Wrote ${OUT_FILE}`);
  console.log(`Features: ${featureNames.length}`);
  console.log(`Train rows: ${output.train.length}`);
  console.log(`Validation rows: ${output.validation.length}`);
  console.log(`Test rows: ${output.test.length}`);
  console.log(`Sampled class counts: ${JSON.stringify(train.sampledClassCounts)}`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
