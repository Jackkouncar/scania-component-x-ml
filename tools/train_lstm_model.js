const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

const ROOT = path.resolve(__dirname, '..');
const DASHBOARD_DATA_FILE = path.join(ROOT, 'site', 'data', 'dashboard_data.json');
const OUT_FILE = path.join(ROOT, 'site', 'data', 'lstm_model_output.json');

const CLASSES = [0, 1, 2, 3, 4];
const COST = [
  [0, 7, 8, 9, 10],
  [200, 0, 7, 8, 9],
  [300, 200, 0, 7, 8],
  [400, 300, 200, 0, 7],
  [500, 400, 300, 200, 0]
];

function argValue(name) {
  const prefix = `--${name}=`;
  const match = process.argv.find(arg => arg.startsWith(prefix));
  return match ? match.slice(prefix.length) : null;
}

const TRAINING_PRESET = argValue('preset') || (process.argv.includes('--deep') ? 'deep' : 'standard');
const USE_DEEP_PRESET = TRAINING_PRESET === 'deep';

function configNumber(envName, standardValue, deepValue) {
  return Number(process.env[envName] || (USE_DEEP_PRESET ? deepValue : standardValue));
}

const SEQUENCE_LENGTH = configNumber('LSTM_SEQUENCE_LENGTH', 8, 16);
const EPOCHS = configNumber('LSTM_EPOCHS', 12, 60);
const BATCH_SIZE = configNumber('LSTM_BATCH_SIZE', 64, 64);
const TRAIN_CLASS0_LIMIT = configNumber('LSTM_TRAIN_CLASS0_LIMIT', 1200, 3000);
const TRAIN_POSITIVE_LIMIT = configNumber('LSTM_TRAIN_POSITIVE_LIMIT', 700, 1400);
const LSTM_UNITS = configNumber('LSTM_UNITS', 16, 48);
const DENSE_UNITS = configNumber('LSTM_DENSE_UNITS', 16, 32);

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function standardDeviation(values, mu) {
  const variance = values.reduce((sum, value) => sum + (value - mu) ** 2, 0) / Math.max(1, values.length);
  return Math.sqrt(variance) || 1;
}

function log1pNonNegative(value) {
  return Math.log1p(Math.max(0, Number(value) || 0));
}

function classFromTimeToFailure(timeToFailure) {
  if (timeToFailure > 48) return 0;
  if (timeToFailure > 24) return 1;
  if (timeToFailure > 12) return 2;
  if (timeToFailure > 6) return 3;
  if (timeToFailure >= 0) return 4;
  return null;
}

function classCounts(samples) {
  const counts = Object.fromEntries(CLASSES.map(label => [label, 0]));
  for (const sample of samples) counts[sample.label]++;
  return counts;
}

function seededRandom(seed = 4567) {
  let value = seed;
  return () => {
    value |= 0;
    value = (value + 0x6d2b79f5) | 0;
    let next = Math.imul(value ^ (value >>> 15), 1 | value);
    next = (next + Math.imul(next ^ (next >>> 7), 61 | next)) ^ next;
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, random = seededRandom()) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function featureNames(counterNames) {
  return [
    'time_step_log',
    ...counterNames.map(name => `counter:${name}:log_value`),
    ...counterNames.map(name => `counter:${name}:log_delta`),
    ...counterNames.map(name => `counter:${name}:log_rate`)
  ];
}

function vectorFromPoint(point, previousPoint, counterNames) {
  const dt = previousPoint ? Math.max(0, point.t - previousPoint.t) : 0;
  const vector = [log1pNonNegative(point.t)];

  for (const name of counterNames) {
    vector.push(log1pNonNegative(point.counters[name]));
  }

  const deltas = {};
  for (const name of counterNames) {
    const currentValue = point.counters[name] ?? 0;
    const previousValue = previousPoint?.counters?.[name] ?? currentValue;
    const delta = Math.max(0, currentValue - previousValue);
    deltas[name] = delta;
    vector.push(log1pNonNegative(delta));
  }

  for (const name of counterNames) {
    const rate = dt > 0 ? deltas[name] / dt : 0;
    vector.push(log1pNonNegative(rate));
  }

  return vector.map(value => round(value));
}

function paddedSequence(sequence, featureCount) {
  const paddingRows = Math.max(0, SEQUENCE_LENGTH - sequence.length);
  return [
    ...Array.from({ length: paddingRows }, () => Array(featureCount).fill(0)),
    ...sequence.slice(-SEQUENCE_LENGTH)
  ];
}

function labelForTrainPoint(vehicle, point) {
  const failure = vehicle.failure || {};
  if (failure.kind === 'observed_repair') {
    return classFromTimeToFailure(failure.failureTime - point.t);
  }
  if (failure.kind === 'censored') return 0;
  return null;
}

function buildTrainingSamples(vehicles, counterNames, featureCount) {
  const byClass = Object.fromEntries(CLASSES.map(label => [label, []]));

  for (const vehicle of vehicles.filter(item => item.split === 'train')) {
    const rolling = [];
    let previousPoint = null;
    for (const point of vehicle.series) {
      const vector = vectorFromPoint(point, previousPoint, counterNames);
      previousPoint = point;
      rolling.push(vector);
      const label = labelForTrainPoint(vehicle, point);
      if (label === null || label === undefined) continue;
      byClass[label].push({
        vehicle_id: vehicle.vehicle_id,
        time_step: point.t,
        label,
        sequence: paddedSequence(rolling, featureCount)
      });
    }
  }

  const random = seededRandom(7301);
  const samples = [];
  for (const label of CLASSES) {
    shuffle(byClass[label], random);
    const limit = label === 0 ? TRAIN_CLASS0_LIMIT : TRAIN_POSITIVE_LIMIT;
    samples.push(...byClass[label].slice(0, limit));
  }
  return shuffle(samples, random);
}

function buildEvaluationSamples(vehicles, split, counterNames, featureCount) {
  const samples = [];
  for (const vehicle of vehicles.filter(item => item.split === split && Number.isFinite(item.riskClass))) {
    const rolling = [];
    let previousPoint = null;
    for (const point of vehicle.series) {
      const vector = vectorFromPoint(point, previousPoint, counterNames);
      previousPoint = point;
      rolling.push(vector);
    }
    samples.push({
      vehicle_id: vehicle.vehicle_id,
      time_step: vehicle.latest?.t,
      label: vehicle.riskClass,
      sequence: paddedSequence(rolling, featureCount)
    });
  }
  return samples;
}

function fitSequenceScaler(samples, featureCount) {
  const columns = Array.from({ length: featureCount }, () => []);
  for (const sample of samples) {
    for (const row of sample.sequence) {
      row.forEach((value, index) => columns[index].push(value));
    }
  }
  const meanValues = columns.map(values => mean(values));
  const stdValues = columns.map((values, index) => standardDeviation(values, meanValues[index]));
  return { mean: meanValues, std: stdValues };
}

function scaleSequence(sequence, scaler) {
  return sequence.map(row => row.map((value, index) => (value - scaler.mean[index]) / scaler.std[index]));
}

function tensorFromSamples(samples, scaler) {
  return tf.tensor3d(samples.map(sample => scaleSequence(sample.sequence, scaler)));
}

function labelsTensor(samples) {
  const labels = tf.tensor1d(samples.map(sample => sample.label), 'int32');
  const encoded = tf.oneHot(labels, CLASSES.length);
  labels.dispose();
  return encoded;
}

function summarizeEvaluation(samples, probabilities) {
  const matrix = CLASSES.map(() => CLASSES.map(() => 0));
  let correct = 0;
  let totalCost = 0;

  samples.forEach((sample, index) => {
    const row = probabilities[index];
    let predicted = 0;
    for (let i = 1; i < row.length; i++) {
      if (row[i] > row[predicted]) predicted = i;
    }
    matrix[sample.label][predicted]++;
    if (predicted === sample.label) correct++;
    totalCost += COST[sample.label][predicted];
  });

  const f1s = [];
  for (const label of CLASSES) {
    const tp = matrix[label][label];
    let fp = 0;
    let fn = 0;
    for (const other of CLASSES) {
      if (other !== label) {
        fp += matrix[other][label];
        fn += matrix[label][other];
      }
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    f1s.push(precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall));
  }

  const allZeroBaselineCost = samples.reduce((sum, sample) => sum + COST[sample.label][0], 0);
  return {
    scope: 'dashboard vehicle subset latest sequence',
    rows: samples.length,
    classCounts: classCounts(samples),
    accuracy: round(correct / Math.max(1, samples.length), 4),
    macroF1: round(mean(f1s), 4),
    totalCost,
    allZeroBaselineCost,
    costImprovementVsAllZero: round((allZeroBaselineCost - totalCost) / Math.max(1, allZeroBaselineCost), 4),
    confusionMatrix: matrix
  };
}

function makeModel(featureCount) {
  const model = tf.sequential();
  model.add(tf.layers.lstm({
    units: LSTM_UNITS,
    inputShape: [SEQUENCE_LENGTH, featureCount],
    recurrentInitializer: 'glorotUniform'
  }));
  model.add(tf.layers.dense({ units: DENSE_UNITS, activation: 'relu' }));
  model.add(tf.layers.dense({ units: CLASSES.length, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(0.003),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

async function predictProbabilities(model, samples, scaler) {
  const xs = tensorFromSamples(samples, scaler);
  const predictions = model.predict(xs);
  const probabilities = await predictions.array();
  xs.dispose();
  predictions.dispose();
  return probabilities;
}

async function main() {
  if (!fs.existsSync(DASHBOARD_DATA_FILE)) {
    throw new Error(`Missing ${DASHBOARD_DATA_FILE}. Run npm run export:dashboard first.`);
  }

  await tf.setBackend('cpu');
  await tf.ready();

  const dashboardData = JSON.parse(fs.readFileSync(DASHBOARD_DATA_FILE, 'utf8'));
  const counters = dashboardData.featureMetadata.counters;
  const features = featureNames(counters);
  const featureCount = features.length;
  const trainSamples = buildTrainingSamples(dashboardData.vehicles, counters, featureCount);
  const validationSamples = buildEvaluationSamples(dashboardData.vehicles, 'validation', counters, featureCount);
  const testSamples = buildEvaluationSamples(dashboardData.vehicles, 'test', counters, featureCount);
  const scaler = fitSequenceScaler(trainSamples, featureCount);

  const trainXs = tensorFromSamples(trainSamples, scaler);
  const trainYs = labelsTensor(trainSamples);
  const model = makeModel(featureCount);
  const history = await model.fit(trainXs, trainYs, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    shuffle: true,
    validationSplit: 0.1,
    verbose: 0
  });

  const validationProbabilities = await predictProbabilities(model, validationSamples, scaler);
  const testProbabilities = await predictProbabilities(model, testSamples, scaler);
  const validation = summarizeEvaluation(validationSamples, validationProbabilities);
  const test = summarizeEvaluation(testSamples, testProbabilities);

  const output = {
    generatedAt: new Date().toISOString(),
    modelName: 'compact_lstm_sequence_experiment',
    modelType: 'lstm_sequence_classifier',
    status: 'experimental_sequence_model',
    description: 'Small TensorFlow.js LSTM trained on compact rolling telemetry sequences from the dashboard vehicle subset. It is a comparison experiment, not the live dashboard model.',
    trainingPreset: TRAINING_PRESET,
    sequenceLength: SEQUENCE_LENGTH,
    featureNames: features,
    training: {
      rows: trainSamples.length,
      classCounts: classCounts(trainSamples),
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      lstmUnits: LSTM_UNITS,
      denseUnits: DENSE_UNITS,
      finalLoss: round(history.history.loss.at(-1), 4),
      finalAccuracy: round(history.history.acc?.at(-1) ?? history.history.accuracy?.at(-1) ?? 0, 4),
      finalValidationLoss: round(history.history.val_loss?.at(-1) ?? 0, 4),
      finalValidationAccuracy: round(history.history.val_acc?.at(-1) ?? history.history.val_accuracy?.at(-1) ?? 0, 4)
    },
    evaluation: { validation, test },
    comparisonEntry: {
      name: 'LSTM sequence model',
      type: 'lstm_sequence_classifier',
      validation,
      test,
      tunedHyperparameters: {
        sequenceLength: SEQUENCE_LENGTH,
        epochs: EPOCHS,
        units: LSTM_UNITS,
        batchSize: BATCH_SIZE
      }
    }
  };

  trainXs.dispose();
  trainYs.dispose();
  model.dispose();

  fs.writeFileSync(OUT_FILE, JSON.stringify(output));
  console.log(`Wrote ${OUT_FILE}`);
  console.log(`Preset: ${TRAINING_PRESET}, sequence length: ${SEQUENCE_LENGTH}, epochs: ${EPOCHS}, LSTM units: ${LSTM_UNITS}`);
  console.log(`Training rows: ${trainSamples.length} ${JSON.stringify(output.training.classCounts)}`);
  console.log(`Validation cost: ${validation.totalCost}, macro F1: ${validation.macroF1}, all-zero cost: ${validation.allZeroBaselineCost}`);
  console.log(`Test cost: ${test.totalCost}, macro F1: ${test.macroF1}, all-zero cost: ${test.allZeroBaselineCost}`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
