const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DATASET_FILE = path.join(ROOT, 'ml', 'ml_dataset.json');
const OUT_MODEL_FILE = path.join(ROOT, 'site', 'data', 'model_output.json');
const OUT_REPORT_FILE = path.join(ROOT, 'ml', 'model_report.md');
const LSTM_MODEL_FILE = path.join(ROOT, 'site', 'data', 'lstm_model_output.json');

function argValue(name) {
  const prefix = `--${name}=`;
  const match = process.argv.find(arg => arg.startsWith(prefix));
  return match ? match.slice(prefix.length) : null;
}

const CLASSES = [0, 1, 2, 3, 4];
const DEFAULT_K_VALUES = [1, 3, 5, 9, 15, 25, 35, 51, 75, 101, 151, 251, 351, 501];
const K_VALUES = (process.env.KNN_K_VALUES || DEFAULT_K_VALUES.join(','))
  .split(',')
  .map(value => Number(value.trim()))
  .filter(value => Number.isInteger(value) && value > 0)
  .sort((a, b) => a - b);
const KNN_DISTANCE_POWER = Number(process.env.KNN_DISTANCE_POWER || 2);
const KNN_SWEEP_TRAIN_PER_CLASS = Number(process.env.KNN_SWEEP_TRAIN_PER_CLASS || 700);
const KNN_SWEEP_EVAL_CLASS0_LIMIT = Number(process.env.KNN_SWEEP_EVAL_CLASS0_LIMIT || 800);
const KNN_SWEEP_EVAL_POSITIVE_LIMIT = Number(process.env.KNN_SWEEP_EVAL_POSITIVE_LIMIT || 9999);
const KNN_SELECTION_METRIC = (argValue('k-selection') || process.env.KNN_SELECTION_METRIC || 'macroF1').toLowerCase();
const COST = [
  [0, 7, 8, 9, 10],
  [200, 0, 7, 8, 9],
  [300, 200, 0, 7, 8],
  [400, 300, 200, 0, 7],
  [500, 400, 300, 200, 0]
];

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function standardDeviation(values, mu) {
  const variance = values.reduce((sum, value) => sum + (value - mu) ** 2, 0) / Math.max(1, values.length);
  return Math.sqrt(variance) || 1;
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function classCounts(rows) {
  const counts = Object.fromEntries(CLASSES.map(label => [label, 0]));
  for (const row of rows) counts[row.label]++;
  return counts;
}

function stratifiedSample(rows, limitsByClass) {
  const counts = Object.fromEntries(CLASSES.map(label => [label, 0]));
  return rows.filter(row => {
    const limit = limitsByClass[row.label] ?? 0;
    if (counts[row.label] >= limit) return false;
    counts[row.label]++;
    return true;
  });
}

function sampleLimits(defaultLimit, overrides = {}) {
  return Object.fromEntries(CLASSES.map(label => [label, overrides[label] ?? defaultLimit]));
}

function fitScaler(rows, featureCount) {
  const columns = Array.from({ length: featureCount }, () => []);
  for (const row of rows) {
    row.vector.forEach((value, index) => columns[index].push(value));
  }
  const meanValues = columns.map(values => mean(values));
  const stdValues = columns.map((values, index) => standardDeviation(values, meanValues[index]));
  return { mean: meanValues, std: stdValues };
}

function scaleVector(vector, scaler) {
  return vector.map((value, index) => (value - scaler.mean[index]) / scaler.std[index]);
}

function fitNearestCentroid(rows, featureCount) {
  const scaler = fitScaler(rows, featureCount);
  const sums = Object.fromEntries(CLASSES.map(label => [label, Array(featureCount).fill(0)]));
  const counts = Object.fromEntries(CLASSES.map(label => [label, 0]));

  for (const row of rows) {
    const scaled = scaleVector(row.vector, scaler);
    counts[row.label]++;
    for (let i = 0; i < featureCount; i++) sums[row.label][i] += scaled[i];
  }

  const centroids = {};
  for (const label of CLASSES) {
    const denominator = Math.max(1, counts[label]);
    centroids[label] = sums[label].map(value => round(value / denominator));
  }

  return { scaler, centroids, classCounts: counts };
}

function squaredDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return sum / a.length;
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const expScores = scores.map(score => Math.exp(score - maxScore));
  const total = expScores.reduce((sum, value) => sum + value, 0);
  return expScores.map(value => value / total);
}

function predictCentroid(model, vector) {
  const scaled = scaleVector(vector, model.scaler);
  const distances = CLASSES.map(label => squaredDistance(scaled, model.centroids[label]));
  const scores = distances.map(distance => -Math.sqrt(distance) * 2.2);
  const probabilities = softmax(scores);
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }
  return {
    predictedClass: CLASSES[bestIndex],
    rawPredictedClass: CLASSES[bestIndex],
    confidence: round(probabilities[bestIndex], 4),
    probabilities: Object.fromEntries(CLASSES.map((label, index) => [label, round(probabilities[index], 4)])),
    distances: Object.fromEntries(CLASSES.map((label, index) => [label, round(distances[index], 4)]))
  };
}

function fitGaussianNaiveBayes(rows, featureCount) {
  const scaler = fitScaler(rows, featureCount);
  const sums = Object.fromEntries(CLASSES.map(label => [label, Array(featureCount).fill(0)]));
  const sumsSq = Object.fromEntries(CLASSES.map(label => [label, Array(featureCount).fill(0)]));
  const counts = Object.fromEntries(CLASSES.map(label => [label, 0]));

  for (const row of rows) {
    const scaled = scaleVector(row.vector, scaler);
    counts[row.label]++;
    for (let i = 0; i < featureCount; i++) {
      sums[row.label][i] += scaled[i];
      sumsSq[row.label][i] += scaled[i] * scaled[i];
    }
  }

  const means = {};
  const variances = {};
  for (const label of CLASSES) {
    const denominator = Math.max(1, counts[label]);
    means[label] = sums[label].map(value => value / denominator);
    variances[label] = sumsSq[label].map((value, index) => {
      const variance = value / denominator - means[label][index] ** 2;
      return Math.max(0.05, variance);
    });
  }

  const totalRows = rows.length;
  const priors = Object.fromEntries(CLASSES.map(label => [
    label,
    (counts[label] + 1) / (totalRows + CLASSES.length)
  ]));

  return { scaler, means, variances, priors, classCounts: counts };
}

function predictGaussianNaiveBayes(model, vector) {
  const scaled = scaleVector(vector, model.scaler);
  const scores = CLASSES.map(label => {
    let score = Math.log(model.priors[label]);
    for (let i = 0; i < scaled.length; i++) {
      const variance = model.variances[label][i];
      const delta = scaled[i] - model.means[label][i];
      score += -0.5 * Math.log(2 * Math.PI * variance) - (delta * delta) / (2 * variance);
    }
    return score;
  });
  const probabilities = softmax(scores);
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }
  return {
    predictedClass: CLASSES[bestIndex],
    rawPredictedClass: CLASSES[bestIndex],
    confidence: round(probabilities[bestIndex], 4),
    probabilities: Object.fromEntries(CLASSES.map((label, index) => [label, round(probabilities[index], 4)]))
  };
}

function applyAlertThreshold(prediction, threshold) {
  if (prediction.rawPredictedClass !== 0 && prediction.confidence < threshold) {
    return {
      ...prediction,
      predictedClass: 0,
      thresholdApplied: true
    };
  }
  return {
    ...prediction,
    thresholdApplied: false
  };
}

function confusionMatrix(rows, predictions) {
  const matrix = CLASSES.map(() => CLASSES.map(() => 0));
  rows.forEach((row, index) => {
    matrix[row.label][predictions[index].predictedClass]++;
  });
  return matrix;
}

function summarizeEvaluation(rows, predictions) {
  const matrix = confusionMatrix(rows, predictions);
  const total = rows.length;
  let correct = 0;
  let totalCost = 0;

  rows.forEach((row, index) => {
    const predicted = predictions[index].predictedClass;
    if (predicted === row.label) correct++;
    totalCost += COST[row.label][predicted];
  });

  const perClass = {};
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
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    f1s.push(f1);
    perClass[label] = {
      precision: round(precision, 4),
      recall: round(recall, 4),
      f1: round(f1, 4),
      support: matrix[label].reduce((sum, value) => sum + value, 0)
    };
  }

  const allZeroCost = rows.reduce((sum, row) => sum + COST[row.label][0], 0);

  return {
    rows: total,
    classCounts: classCounts(rows),
    accuracy: round(correct / Math.max(1, total), 4),
    macroF1: round(mean(f1s), 4),
    totalCost,
    allZeroBaselineCost: allZeroCost,
    costImprovementVsAllZero: round((allZeroCost - totalCost) / Math.max(1, allZeroCost), 4),
    confusionMatrix: matrix,
    perClass,
    predictions
  };
}

function evaluateRows(model, rows, alertThreshold = 0, predictRow = predictCentroid) {
  const predictions = rows.map(row => applyAlertThreshold(predictRow(model, row.vector), alertThreshold));
  return summarizeEvaluation(rows, predictions);
}

function evaluateConstantClass(rows, predictedClass = 0) {
  const probabilities = Object.fromEntries(CLASSES.map(label => [label, label === predictedClass ? 1 : 0]));
  const predictions = rows.map(() => ({
    predictedClass,
    rawPredictedClass: predictedClass,
    confidence: 1,
    probabilities,
    thresholdApplied: false
  }));
  return summarizeEvaluation(rows, predictions);
}

function predictionCost(rows, predictions) {
  return rows.reduce((sum, row, index) => sum + COST[row.label][predictions[index].predictedClass], 0);
}

function tuneAlertThreshold(model, rows, predictRow = predictCentroid) {
  let best = {
    threshold: 0,
    totalCost: Infinity
  };

  for (let threshold = 0; threshold <= 1.0001; threshold += 0.01) {
    const roundedThreshold = round(threshold, 2);
    const predictions = rows.map(row => applyAlertThreshold(predictRow(model, row.vector), roundedThreshold));
    const totalCost = predictionCost(rows, predictions);
    if (totalCost < best.totalCost) {
      best = { threshold: roundedThreshold, totalCost };
    }
  }

  return best;
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

function fitKnnExemplars(rows, scaler) {
  return rows.map(row => ({
    vehicle_id: row.vehicle_id,
    time_step: row.time_step,
    label: row.label,
    vector: scaleVector(row.vector, scaler).map(value => round(value, 4))
  }));
}

function predictKnnFromSortedNeighbors(sortedNeighbors, k, distancePower) {
  const votes = Object.fromEntries(CLASSES.map(label => [label, 0]));
  for (const neighbor of sortedNeighbors.slice(0, k)) {
    const weight = 1 / ((neighbor.distance + 0.000001) ** distancePower);
    votes[neighbor.label] += weight;
  }

  const totalVote = Object.values(votes).reduce((sum, value) => sum + value, 0);
  let bestLabel = 0;
  for (const label of CLASSES.slice(1)) {
    if (votes[label] > votes[bestLabel]) bestLabel = label;
  }

  const probabilities = Object.fromEntries(CLASSES.map(label => [
    label,
    totalVote === 0 ? 0 : round(votes[label] / totalVote, 4)
  ]));

  return {
    predictedClass: bestLabel,
    rawPredictedClass: bestLabel,
    confidence: probabilities[bestLabel],
    probabilities,
    thresholdApplied: false
  };
}

function predictKnn(model, vector) {
  const scaled = scaleVector(vector, model.scaler);
  const neighbors = [];
  for (const exemplar of model.exemplars) {
    const distance = squaredDistance(scaled, exemplar.vector);
    pushNeighbor(neighbors, { distance, label: exemplar.label }, model.k);
  }
  const sortedNeighbors = neighbors.sort((a, b) => a.distance - b.distance);
  return predictKnnFromSortedNeighbors(sortedNeighbors, model.k, model.distancePower);
}

function evaluateKnnSweep({ trainRows, evaluationRows, scaler, kValues, distancePower }) {
  const maxK = Math.max(...kValues);
  const exemplars = fitKnnExemplars(trainRows, scaler);
  const predictionsByK = Object.fromEntries(kValues.map(k => [k, []]));

  for (const row of evaluationRows) {
    const scaled = scaleVector(row.vector, scaler);
    const neighbors = [];
    for (const exemplar of exemplars) {
      const distance = squaredDistance(scaled, exemplar.vector);
      pushNeighbor(neighbors, { distance, label: exemplar.label }, maxK);
    }
    const sortedNeighbors = neighbors.sort((a, b) => a.distance - b.distance);
    for (const k of kValues) {
      predictionsByK[k].push(predictKnnFromSortedNeighbors(sortedNeighbors, k, distancePower));
    }
  }

  return kValues.map(k => {
    const evaluation = summarizeEvaluation(evaluationRows, predictionsByK[k]);
    return {
      k,
      rows: evaluation.rows,
      accuracy: evaluation.accuracy,
      macroF1: evaluation.macroF1,
      totalCost: evaluation.totalCost,
      allZeroBaselineCost: evaluation.allZeroBaselineCost,
      costImprovementVsAllZero: evaluation.costImprovementVsAllZero
    };
  });
}

function selectBestK(kResults, selectionMetric) {
  if (selectionMetric === 'cost') {
    return [...kResults].sort((a, b) =>
      a.totalCost - b.totalCost ||
      b.macroF1 - a.macroF1 ||
      b.accuracy - a.accuracy ||
      a.k - b.k
    )[0];
  }

  if (selectionMetric === 'macrof1' || selectionMetric === 'macro-f1' || selectionMetric === 'f1') {
    return [...kResults].sort((a, b) =>
      b.macroF1 - a.macroF1 ||
      b.accuracy - a.accuracy ||
      a.totalCost - b.totalCost ||
      a.k - b.k
    )[0];
  }

  return [...kResults].sort((a, b) =>
    b.accuracy - a.accuracy ||
    b.macroF1 - a.macroF1 ||
    a.totalCost - b.totalCost ||
    a.k - b.k
  )[0];
}

function kSelectionMetricLabel(selectionMetric) {
  if (selectionMetric === 'cost') return 'lowest stratified validation cost';
  if (selectionMetric === 'macrof1' || selectionMetric === 'macro-f1' || selectionMetric === 'f1') {
    return 'highest stratified validation macro F1';
  }
  return 'highest stratified validation accuracy';
}

function kSelectionMetricKey(selectionMetric) {
  if (selectionMetric === 'cost') return 'cost';
  if (selectionMetric === 'macrof1' || selectionMetric === 'macro-f1' || selectionMetric === 'f1') return 'macroF1';
  return 'accuracy';
}

function compactEvaluation(evaluation, scope) {
  return {
    scope,
    rows: evaluation.rows,
    classCounts: evaluation.classCounts,
    accuracy: evaluation.accuracy,
    macroF1: evaluation.macroF1,
    totalCost: evaluation.totalCost,
    allZeroBaselineCost: evaluation.allZeroBaselineCost,
    costImprovementVsAllZero: evaluation.costImprovementVsAllZero
  };
}

function readLstmExperiment() {
  if (!fs.existsSync(LSTM_MODEL_FILE)) return null;
  try {
    return JSON.parse(fs.readFileSync(LSTM_MODEL_FILE, 'utf8'));
  } catch (error) {
    console.warn(`Could not read ${LSTM_MODEL_FILE}: ${error.message}`);
    return null;
  }
}

function serializePrediction(row, prediction) {
  return {
    key: `${row.split}:${row.vehicle_id}:${row.time_step}`,
    split: row.split,
    vehicle_id: row.vehicle_id,
    time_step: row.time_step,
    actual_class: row.label,
    predicted_class: prediction.predictedClass,
    raw_predicted_class: prediction.rawPredictedClass,
    confidence: prediction.confidence,
    threshold_applied: Boolean(prediction.thresholdApplied),
    probabilities: prediction.probabilities
  };
}

function topFeatureDifferences(dataset, model, topN = 12) {
  const healthy = model.centroids[0];
  const critical = model.centroids[4];
  return dataset.featureNames
    .map((featureName, index) => ({
      featureName,
      difference: round(Math.abs((critical?.[index] ?? 0) - (healthy?.[index] ?? 0)), 4)
    }))
    .sort((a, b) => b.difference - a.difference)
    .slice(0, topN);
}

function matrixMarkdown(matrix) {
  const header = ['Actual \\ Pred', ...CLASSES.map(String)].join(' | ');
  const divider = ['---', ...CLASSES.map(() => '---:')].join(' | ');
  const rows = matrix.map((row, index) => [String(index), ...row.map(String)].join(' | '));
  return [header, divider, ...rows].join('\n');
}

function modelComparisonMarkdown(modelComparison) {
  const header = 'Model | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1';
  const divider = '--- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:';
  const rows = modelComparison.map(item => [
    item.name,
    item.validation.scope,
    item.validation.accuracy,
    item.validation.totalCost,
    item.validation.macroF1,
    item.test.scope,
    item.test.accuracy,
    item.test.totalCost,
    item.test.macroF1
  ].join(' | '));
  return [header, divider, ...rows].join('\n');
}

function kTuningMarkdown(kResults) {
  const header = 'k | Validation rows | Validation accuracy | Validation cost | Macro F1';
  const divider = '---: | ---: | ---: | ---: | ---:';
  const rows = kResults.map(item => [
    item.k,
    item.rows,
    item.accuracy,
    item.totalCost,
    item.macroF1
  ].join(' | '));
  return [header, divider, ...rows].join('\n');
}

function writeReport(dataset, model, validation, test, topFeatures, alertThreshold, modelComparison, knnTuning) {
  const lstmIncluded = modelComparison.some(item => item.type === 'lstm_sequence_classifier');
  const lstmRationale = lstmIncluded
    ? '- LSTM sequence model: trained as a compact TensorFlow.js sequence experiment on rolling dashboard telemetry windows.'
    : '- LSTM sequence model: prepared as the neural sequence-model next step because it can learn ordered readout patterns; the current Node dashboard keeps kNN as the reliable live model.';
  const report = `# Baseline ML Model Report

Generated: ${new Date().toISOString()}

## Model

Dashboard streaming model: supervised distance-weighted nearest-neighbor classifier.

For each incoming telemetry packet, the dashboard compares the engineered feature vector against labeled historical training telemetry examples and predicts from the nearest neighbors.

The exported file also keeps a conservative centroid summary/evaluation so the project has a fast reportable baseline metric.

This is an explainable baseline model, not the final production model.

Alert threshold: ${alertThreshold}

Non-zero class predictions below this confidence are converted to class 0. The threshold is tuned on validation cost to reduce noisy false alarms.

## Model Selection Rationale

- Naive all-class-0 baseline: included because the dataset is highly imbalanced and most vehicles are not near failure.
- Nearest-centroid classifier: used as a fast reportable baseline and for class-separation summaries.
- Gaussian Naive Bayes: added as a second trained comparator. It is fast, but it makes a stronger feature-independence assumption.
- Distance-weighted kNN: selected for the dashboard because it is supervised, explainable, works with live telemetry records, and does not assume telemetry features are independent.
${lstmRationale}

The current project therefore uses kNN as the live dashboard model and compares it against multiple trained baselines.

## Model Comparison

${modelComparisonMarkdown(modelComparison)}

Raw accuracy is included for the class presentation, but it is not the only useful metric. Because most examples are class 0, the all-class-0 baseline can look strong on accuracy while missing every failure. The dashboard currently highlights the kNN k value selected by ${knnTuning.selectionMetric}; run \`npm run train:model:cost\` if you want the stricter maintenance-cost selection instead.

## Hyperparameter Tuning

- Feature scaling: z-score standardization is fit on training rows only.
- Centroid alert threshold: grid searched from 0.00 to 1.00 in 0.01 steps against validation cost; selected threshold ${alertThreshold}.
- Dashboard kNN: k = ${knnTuning.selectedK} and distance weighting power = ${knnTuning.distancePower}. The selected k is chosen by ${knnTuning.selectionMetric}.
- Temporal smoothing: the dashboard requires ${3} repeated lower-risk packets before lowering an alert, while higher-risk packets update immediately.

### kNN k Sweep

The sweep uses a stratified training/evaluation sample so it can run quickly in the project repo while still preserving all positive validation examples.

The exported dashboard chart shows validation accuracy by k. The selected k follows the configured selection metric so the highlighted bar matches the presentation story.

${kTuningMarkdown(knnTuning.results)}

## Feature Dependence

The operational inputs are anonymized telemetry signals from one vehicle, not independent physical components. The model treats them as correlated inputs by using standardized distances across the full feature vector; it does not use a naive independence assumption.

## Features

- Feature count: ${dataset.featureNames.length}
- Counter features: ${dataset.counterFeatures.join(', ')}
- Spec one-hot features: ${dataset.specFeatureCount}
- Inputs: current counter values, counter deltas, counter rates, relative time_step, vehicle spec categories

## Training Rows

${JSON.stringify(dataset.summary.trainSampledClassCounts, null, 2)}

## Validation

- Rows: ${validation.rows}
- Accuracy: ${validation.accuracy}
- Macro F1: ${validation.macroF1}
- Cost: ${validation.totalCost}
- All-zero baseline cost: ${validation.allZeroBaselineCost}
- Cost improvement vs all-zero: ${validation.costImprovementVsAllZero}

${matrixMarkdown(validation.confusionMatrix)}

## Test

- Rows: ${test.rows}
- Accuracy: ${test.accuracy}
- Macro F1: ${test.macroF1}
- Cost: ${test.totalCost}
- All-zero baseline cost: ${test.allZeroBaselineCost}
- Cost improvement vs all-zero: ${test.costImprovementVsAllZero}

${matrixMarkdown(test.confusionMatrix)}

## Most Different Features Between Class 0 And Class 4 Centroids

${topFeatures.map(item => `- ${item.featureName}: ${item.difference}`).join('\n')}
`;

  fs.writeFileSync(OUT_REPORT_FILE, report);
}

function main() {
  if (!fs.existsSync(DATASET_FILE)) {
    throw new Error(`Missing ${DATASET_FILE}. Run npm run build:ml-data first.`);
  }

  const dataset = JSON.parse(fs.readFileSync(DATASET_FILE, 'utf8'));
  const featureCount = dataset.featureNames.length;
  const model = fitNearestCentroid(dataset.train, featureCount);
  const tunedThreshold = tuneAlertThreshold(model, dataset.validation);
  const trainEval = evaluateRows(model, dataset.train, tunedThreshold.threshold);
  const validationEval = evaluateRows(model, dataset.validation, tunedThreshold.threshold);
  const testEval = evaluateRows(model, dataset.test, tunedThreshold.threshold);
  const allZeroValidationEval = evaluateConstantClass(dataset.validation, 0);
  const allZeroTestEval = evaluateConstantClass(dataset.test, 0);
  const naiveBayesModel = fitGaussianNaiveBayes(dataset.train, featureCount);
  const naiveBayesThreshold = tuneAlertThreshold(naiveBayesModel, dataset.validation, predictGaussianNaiveBayes);
  const naiveBayesValidationEval = evaluateRows(naiveBayesModel, dataset.validation, naiveBayesThreshold.threshold, predictGaussianNaiveBayes);
  const naiveBayesTestEval = evaluateRows(naiveBayesModel, dataset.test, naiveBayesThreshold.threshold, predictGaussianNaiveBayes);
  const knnTrainSample = stratifiedSample(dataset.train, sampleLimits(KNN_SWEEP_TRAIN_PER_CLASS));
  const knnValidationSample = stratifiedSample(dataset.validation, sampleLimits(KNN_SWEEP_EVAL_POSITIVE_LIMIT, { 0: KNN_SWEEP_EVAL_CLASS0_LIMIT }));
  const knnTestSample = stratifiedSample(dataset.test, sampleLimits(KNN_SWEEP_EVAL_POSITIVE_LIMIT, { 0: KNN_SWEEP_EVAL_CLASS0_LIMIT }));
  const knnSweepResults = evaluateKnnSweep({
    trainRows: knnTrainSample,
    evaluationRows: knnValidationSample,
    scaler: model.scaler,
    kValues: K_VALUES,
    distancePower: KNN_DISTANCE_POWER
  });
  const bestK = selectBestK(knnSweepResults, KNN_SELECTION_METRIC);
  const selectedKnnModel = {
    scaler: model.scaler,
    k: bestK.k,
    distancePower: KNN_DISTANCE_POWER,
    exemplars: fitKnnExemplars(knnTrainSample, model.scaler)
  };
  const selectedKnnValidationEval = summarizeEvaluation(
    knnValidationSample,
    knnValidationSample.map(row => predictKnn(selectedKnnModel, row.vector))
  );
  const selectedKnnTestEval = summarizeEvaluation(
    knnTestSample,
    knnTestSample.map(row => predictKnn(selectedKnnModel, row.vector))
  );
  const modelComparison = [
    {
      name: 'All-class-0 baseline',
      type: 'constant_classifier',
      validation: compactEvaluation(allZeroValidationEval, 'full validation set'),
      test: compactEvaluation(allZeroTestEval, 'full test set')
    },
    {
      name: 'Nearest-centroid classifier',
      type: 'nearest_centroid',
      validation: compactEvaluation(validationEval, 'full validation set'),
      test: compactEvaluation(testEval, 'full test set'),
      tunedHyperparameters: { alertThreshold: tunedThreshold.threshold }
    },
    {
      name: 'Gaussian Naive Bayes',
      type: 'gaussian_naive_bayes',
      validation: compactEvaluation(naiveBayesValidationEval, 'full validation set'),
      test: compactEvaluation(naiveBayesTestEval, 'full test set'),
      tunedHyperparameters: { alertThreshold: naiveBayesThreshold.threshold }
    },
    {
      name: `Distance-weighted kNN (k=${bestK.k})`,
      type: 'distance_weighted_knn',
      validation: compactEvaluation(selectedKnnValidationEval, 'stratified validation sample'),
      test: compactEvaluation(selectedKnnTestEval, 'stratified test sample'),
      tunedHyperparameters: { k: bestK.k, distancePower: KNN_DISTANCE_POWER }
    }
  ];
  const lstmExperiment = readLstmExperiment();
  if (lstmExperiment?.comparisonEntry) {
    modelComparison.push(lstmExperiment.comparisonEntry);
  }
  const knnTuning = {
    kValues: K_VALUES,
    selectedK: bestK.k,
    distancePower: KNN_DISTANCE_POWER,
    selectionMetric: kSelectionMetricLabel(KNN_SELECTION_METRIC),
    selectionMetricKey: kSelectionMetricKey(KNN_SELECTION_METRIC),
    trainRows: knnTrainSample.length,
    trainClassCounts: classCounts(knnTrainSample),
    validationRows: knnValidationSample.length,
    validationClassCounts: classCounts(knnValidationSample),
    results: knnSweepResults
  };
  const topFeatures = topFeatureDifferences(dataset, model);

  const output = {
    generatedAt: new Date().toISOString(),
    modelName: 'distance_weighted_knn_component_x_v2',
    modelType: 'supervised_distance_weighted_knn',
    status: 'baseline_ml_model',
    description: 'Predicts Component X risk class from engineered telemetry counters, counter deltas/rates, relative time_step, and vehicle spec categories. The dashboard uses distance-weighted nearest neighbors over labeled training telemetry examples.',
    realTimeMode: {
      supported: true,
      behavior: 'The dashboard can append a newly entered telemetry record for the selected vehicle and immediately score it with the same feature pipeline used for historical replay.'
    },
    modelSelection: {
      selectedDashboardModel: 'supervised distance-weighted k-nearest-neighbor classifier',
      rationale: `kNN is explainable, works directly with standardized live telemetry vectors, avoids assuming anonymized telemetry signals are independent physical components, and selected k=${bestK.k} by ${knnTuning.selectionMetric}.`,
      candidateModels: [
        {
          name: 'all-class-0 baseline',
          purpose: 'sanity baseline for the imbalanced dataset',
          validationCost: allZeroValidationEval.totalCost,
          testCost: allZeroTestEval.totalCost
        },
        {
          name: 'nearest-centroid classifier',
          purpose: 'fast reportable baseline and feature-separation summary',
          tunedHyperparameters: { alertThreshold: tunedThreshold.threshold },
          validationCost: validationEval.totalCost,
          testCost: testEval.totalCost
        },
        {
          name: 'Gaussian Naive Bayes',
          purpose: 'trained comparator model that tests a simple independence-assumption approach',
          tunedHyperparameters: { alertThreshold: naiveBayesThreshold.threshold },
          validationCost: naiveBayesValidationEval.totalCost,
          testCost: naiveBayesTestEval.totalCost
        },
        {
          name: 'distance-weighted kNN',
          purpose: 'live dashboard scoring from nearest labeled telemetry examples',
          tunedHyperparameters: { k: bestK.k, distancePower: KNN_DISTANCE_POWER },
          validationCost: selectedKnnValidationEval.totalCost,
          validationScope: 'stratified validation sample',
          testCost: selectedKnnTestEval.totalCost,
          testScope: 'stratified test sample'
        },
        {
          name: 'LSTM sequence model',
          purpose: 'next neural model for ordered telemetry sequences',
          status: lstmExperiment
            ? 'trained as a TensorFlow.js sequence experiment on compact rolling dashboard telemetry windows'
            : 'prepared as the recommended sequence-model extension; run npm run train:lstm to generate the optional experiment'
        }
      ]
    },
    modelComparison,
    knnTuning,
    lstmExperiment: lstmExperiment ? {
      modelName: lstmExperiment.modelName,
      modelType: lstmExperiment.modelType,
      status: lstmExperiment.status,
      sequenceLength: lstmExperiment.sequenceLength,
      training: lstmExperiment.training,
      evaluation: lstmExperiment.evaluation
    } : null,
    featureDependence: 'The anonymized counters, rates, histograms, and specs are correlated inputs from the same vehicle. They are not separate known components, and the selected model does not make a naive feature-independence assumption.',
    knn: {
      k: bestK.k,
      distancePower: KNN_DISTANCE_POWER,
      selection: {
        metric: knnTuning.selectionMetric,
        validationRows: knnTuning.validationRows,
        selectedValidationAccuracy: bestK.accuracy,
        selectedValidationCost: bestK.totalCost
      },
      exemplars: fitKnnExemplars(dataset.train, model.scaler)
    },
    alertThreshold: tunedThreshold.threshold,
    thresholdPurpose: 'Conservative threshold tuned on validation cost. Non-zero risk predictions below this confidence are converted to class 0 to reduce false alarms.',
    featureNames: dataset.featureNames,
    counterFeatures: dataset.counterFeatures,
    scaler: {
      mean: model.scaler.mean.map(value => round(value)),
      std: model.scaler.std.map(value => round(value))
    },
    centroids: model.centroids,
    trainClassCounts: model.classCounts,
    topFeaturesClass0VsClass4: topFeatures,
    evaluation: {
      train: {
        rows: trainEval.rows,
        classCounts: trainEval.classCounts,
        accuracy: trainEval.accuracy,
        macroF1: trainEval.macroF1,
        totalCost: trainEval.totalCost,
        confusionMatrix: trainEval.confusionMatrix
      },
      validation: {
        rows: validationEval.rows,
        classCounts: validationEval.classCounts,
        accuracy: validationEval.accuracy,
        macroF1: validationEval.macroF1,
        totalCost: validationEval.totalCost,
        allZeroBaselineCost: validationEval.allZeroBaselineCost,
        costImprovementVsAllZero: validationEval.costImprovementVsAllZero,
        confusionMatrix: validationEval.confusionMatrix,
        perClass: validationEval.perClass
      },
      test: {
        rows: testEval.rows,
        classCounts: testEval.classCounts,
        accuracy: testEval.accuracy,
        macroF1: testEval.macroF1,
        totalCost: testEval.totalCost,
        allZeroBaselineCost: testEval.allZeroBaselineCost,
        costImprovementVsAllZero: testEval.costImprovementVsAllZero,
        confusionMatrix: testEval.confusionMatrix,
        perClass: testEval.perClass
      }
    },
    predictions: {
      validation: dataset.validation.map((row, index) => serializePrediction(row, validationEval.predictions[index])),
      test: dataset.test.map((row, index) => serializePrediction(row, testEval.predictions[index]))
    }
  };

  fs.mkdirSync(path.dirname(OUT_MODEL_FILE), { recursive: true });
  fs.writeFileSync(OUT_MODEL_FILE, JSON.stringify(output));
  writeReport(dataset, model, validationEval, testEval, topFeatures, tunedThreshold.threshold, modelComparison, knnTuning);

  console.log(`Wrote ${OUT_MODEL_FILE}`);
  console.log(`Wrote ${OUT_REPORT_FILE}`);
  console.log(`Tuned alert threshold: ${tunedThreshold.threshold} (validation cost ${tunedThreshold.totalCost})`);
  console.log(`Gaussian NB threshold: ${naiveBayesThreshold.threshold} (validation cost ${naiveBayesThreshold.totalCost})`);
  console.log(`Selected kNN k: ${bestK.k} by ${knnTuning.selectionMetric} (accuracy ${bestK.accuracy}, cost ${bestK.totalCost}, rows ${knnValidationSample.length})`);
  if (lstmExperiment) {
    console.log(`Included LSTM experiment: validation cost ${lstmExperiment.evaluation.validation.totalCost}, test cost ${lstmExperiment.evaluation.test.totalCost}`);
  }
  console.log(`Validation accuracy: ${validationEval.accuracy}, macro F1: ${validationEval.macroF1}, cost: ${validationEval.totalCost}, all-zero cost: ${validationEval.allZeroBaselineCost}`);
  console.log(`Test accuracy: ${testEval.accuracy}, macro F1: ${testEval.macroF1}, cost: ${testEval.totalCost}, all-zero cost: ${testEval.allZeroBaselineCost}`);
  console.log('Model comparison:');
  modelComparison.forEach(item => console.log(`- ${item.name}: validation cost ${item.validation.totalCost} (${item.validation.scope}), test cost ${item.test.totalCost} (${item.test.scope})`));
  console.log('Top class 0 vs class 4 feature differences:');
  topFeatures.slice(0, 6).forEach(item => console.log(`- ${item.featureName}: ${item.difference}`));
}

try {
  main();
} catch (error) {
  console.error(error);
  process.exit(1);
}
