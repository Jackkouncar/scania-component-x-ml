# ML Model Notes

## What The Model Does

The current model predicts Component X risk class from incoming telemetry.

Input:

```text
current telemetry packet
previous telemetry packet
vehicle specifications
```

Output:

```text
class 0 = no imminent failure
class 1 = 48 to 24 relative time units before failure
class 2 = 24 to 12 relative time units before failure
class 3 = 12 to 6 relative time units before failure
class 4 = 6 to 0 relative time units before failure
```

## Feature Engineering

Script:

```text
tools/build_ml_dataset.js
```

Command:

```bash
npm run build:ml-data
```

Features:

- `time_step_log`
- log-scaled current values for the 8 counter features
- log-scaled counter deltas from the previous readout
- log-scaled counter rates from the previous readout
- one-hot encoded vehicle specs

The current model intentionally uses only features available during live dashboard streaming.

## Training Labels

Training labels are derived from `train_tte.csv`.

For repaired vehicles:

```text
time_to_failure = length_of_study_time_step - current_time_step
```

Then classes are assigned from the failure windows.

For unrepaired/censored vehicles:

```text
class 0
```

Validation/test labels come from:

```text
validation_labels.csv
test_labels.csv
```

## Model Type

Script:

```text
tools/train_baseline_model.js
```

Command:

```bash
npm run train:model
```

The default command trains the JavaScript comparison models, imports the Python tabular results, and recommends the lowest-cost full-set model that still clears the validation-accuracy operating floor. The kNN sweep is kept as comparison evidence and uses validation macro F1 for its own k choice, so it does not just pick the largest k value by raw accuracy.

```bash
npm run train:model:accuracy
```

```bash
npm run train:model:cost
```

Optional sequence experiment:

```bash
npm run train:lstm
```

Longer sequence-training preset:

```bash
npm run train:lstm:deep
```

Model:

```text
supervised nearest-centroid classifier
```

Additional trained comparisons:

```text
nearest-centroid baseline
Gaussian Naive Bayes
Random Forest
Logistic Regression
LightGBM
TensorFlow.js LSTM sequence experiment
```

Why this model first:

- easy to explain
- fast to train in Node
- works without Python/scikit-learn
- produces a real trained classifier for the dashboard
- does not assume the telemetry features are independent of each other

The dashboard model standardizes the incoming telemetry packet, compares it with the learned class centroid for each risk level, and predicts from the closest class pattern. The exported file also keeps the kNN sweep so we can explain why kNN was tested but not selected.

The dashboard adds temporal smoothing on top of raw packet predictions:

- risk can increase immediately
- risk only decreases after repeated lower-risk packets
- this prevents unrealistic one-packet jumps from `Critical` back to `Healthy`
- jumps to a new vehicle, time_step, or display date reset the smoothing state

## Model Selection, Bagging, And Boosting

Current selected dashboard model:

```text
nearest-centroid classifier, alert threshold = 0.21
```

Current recommended fair model:

```text
LightGBM regularized
```

Models considered:

| Model | Role | Decision |
|---|---|---|
| All-class-0 baseline | Sanity check for the imbalanced dataset | Kept as baseline cost comparator |
| Nearest centroid | Fast reportable baseline and feature-separation summary | Selected for current dashboard |
| Gaussian Naive Bayes | Fast second trained comparator | Trained and compared; worse than centroid on cost |
| Random Forest | Tree-based comparator for nonlinear telemetry patterns | Trained and compared; worse than centroid here |
| Logistic Regression | Regularized linear comparator | Trained and compared; worse than LightGBM on cost |
| LightGBM | Regularized boosted-tree comparator | Recommended because it has the lowest full-validation SCANIA cost among operating points with validation accuracy above 75% |
| Distance-weighted kNN | k-value sweep comparator | Not selected because the best k was high and stratified accuracy stayed low |
| LSTM sequence model | Neural model over rolling telemetry windows | Trained as an optional TensorFlow.js experiment |

The kNN sweep is still exported over `k = 1, 3, 5, 9, 15, 25, 35, 51, 75, 101, 151, 251, 351, 501`. The high selected k is evidence that kNN is not the best main model for this version, so the dashboard no longer presents the k sweep as the headline chart.

## Feature Dependence

The operational fields are anonymized telemetry signals, not named physical components. They should be treated as correlated measurements from the same vehicle readout. The current model uses standardized distances across the full feature vector and therefore does not rely on a naive independence assumption.

In plain class-demo language:

```text
We are not predicting that many separate parts fail together.
We are predicting the risk state of one target, Component X, from multiple correlated telemetry inputs.
```

## Real-Time Mode

The website now has two telemetry modes:

```text
Historical replay
Real-time insert
```

Historical replay steps through existing SCANIA readouts. Real-time insert lets the user append a new telemetry record for the selected vehicle, then immediately re-runs the same ML prediction pipeline with the new time_step and counter values. Inserted records are stored in browser local storage for the demo machine and can be cleared from the selected vehicle.

## Current Metrics

Current model file:

```text
site/data/model_output.json
```

Report:

```text
ml/model_report.md
```

Current tuned results:

```text
Training rows: 41,237

Recommended fair model: LightGBM regularized
LightGBM validation cost: 50,070
LightGBM validation accuracy: 88.94%
LightGBM test cost: 49,004
LightGBM test accuracy: 87.61%

Nearest-centroid validation cost: 56,884
Nearest-centroid validation accuracy: 87.06%
Validation all-zero baseline cost: 57,400

Nearest-centroid test cost: 54,627
Nearest-centroid test accuracy: 88.23%
Test all-zero baseline cost: 56,100

kNN selected k: 351
kNN stratified validation-sample accuracy: 23.40%
kNN stratified validation-sample macro F1: 12.96%
kNN stratified validation-sample cost: 33,736

LSTM dashboard-subset validation cost: 41,396
LSTM dashboard-subset test cost: 40,886
```

The cost comparison matters because the challenge metric penalizes missed failures heavily. LightGBM is the recommended fair model because it has the lowest full-validation SCANIA cost among comparable full-set models while staying above the 75% validation-accuracy operating floor. The centroid baseline remains the selected browser-live dashboard scorer because it is simple, explainable, and runs directly in JavaScript.

## Dashboard Integration

The dashboard now loads:

```text
site/data/model_output.json
```

The telemetry controls include:

```text
Prediction mode: ML model / Baseline demo
```

ML model mode predicts from the streamed telemetry packet using the nearest-centroid model, then applies the dashboard's temporal smoothing layer so health alerts behave like a monitoring system.

Baseline demo mode uses known repair times/failure windows to demonstrate the flow.

The model evidence panel shows both raw accuracy and cost. Accuracy is useful for comparison, but this dataset is very imbalanced, so the all-class-0 baseline can look accurate while missing every failure. Cost remains the better selection metric for maintenance risk.

## Next ML Improvements

Good next improvements:

- include histogram features in streamed telemetry
- add rolling averages over several recent packets
- tune a stronger tree-based or boosted model such as XGBoost as another comparator
- tune cost-sensitive thresholds by class
- show model probability by class in the UI
- add confusion matrix visualization to the webapp
