# Baseline ML Model Report

Generated: 2026-05-17T14:59:02.968Z

## Model

Dashboard streaming model: exported LightGBM browser scorer, with nearest-centroid, kNN, and baseline modes retained as comparison options.

For each incoming telemetry packet, the dashboard builds the same engineered feature vector used during tabular training, evaluates the exported LightGBM trees, converts class scores into probabilities, and applies the selected decision rule.

The exported file also keeps the kNN sweep and nearest-centroid outputs so the project can explain why those models were tested but not chosen as the final recommendation.

The final recommendation is the LightGBM cost-sensitive operating point because it has the lowest complete-validation SCANIA cost.

Alert threshold: 0

Non-zero class predictions below this confidence are converted to class 0. The threshold is tuned on validation cost to reduce noisy false alarms.

## Model Selection Rationale

- Naive all-class-0 baseline: included because the dataset is highly imbalanced and most vehicles are not near failure.
- Nearest-centroid classifier: used as a fast reportable baseline and for class-separation summaries.
- Gaussian Naive Bayes: added as a second trained comparator. It is fast, but it makes a stronger feature-independence assumption.
- Random Forest: added as a tree-based comparator for nonlinear tabular patterns.
- Logistic Regression and LightGBM: added as Python tabular comparators with regularization and full validation/test scoring.
- Distance-weighted kNN: evaluated with a k sweep, but not selected because the best k was high and the stratified accuracy stayed low.
- LSTM sequence model: trained as a compact TensorFlow.js sequence experiment and included with its scope clearly marked because it is not the complete validation/test set.

The current submission uses full validation-set SCANIA cost to choose the recommended trained model. Current recommended model by validation rule: LightGBM cost-sensitive.

## Model Comparison

Model | Train accuracy | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1
--- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:
All-class-0 baseline | 0.9771 | full validation set | 0.973 | 57400 | 0.1973 | full test set | 0.9719 | 56100 | 0.1971
Nearest-centroid classifier | 0.5576 | full validation set | 0.4322 | 55883 | 0.1379 | full test set | 0.4244 | 56752 | 0.132
Gaussian Naive Bayes | 0.7602 | full validation set | 0.824 | 53032 | 0.197 | full test set | 0.8196 | 55387 | 0.1883
Random Forest | 0.5442 | full validation set | 0.2632 | 68243 | 0.0868 | full test set | 0.2636 | 65629 | 0.0881
Distance-weighted kNN (k=351) | 1 | full validation set | 0.2866 | 57124 | 0.1025 | full test set | 0.2876 | 57785 | 0.1011
LightGBM cost-sensitive | 0.7327 | full validation set | 0.5438 | 35599 | 0.1508 | full test set | 0.6573 | 47930 | 0.1647
LightGBM 75% accuracy floor | 0.8685 | full validation set | 0.7501 | 44668 | 0.1822 | full test set | 0.8527 | 50094 | 0.1907
LightGBM balanced expected-cost | 0.24 | full validation set | 0.1056 | 45063 | 0.0409 | full test set | 0.1255 | 44896 | 0.0505
LightGBM expected-cost | 0.2319 | full validation set | 0.1021 | 45519 | 0.0396 | full test set | 0.1247 | 44768 | 0.0472
Logistic Regression (L2 balanced) | 0.4991 | full validation set | 0.2418 | 48292 | 0.0825 | full test set | 0.2115 | 49767 | 0.0733
LSTM sequence model (subset) | - | dashboard vehicle subset latest sequence | 0.1836 | 40891 | 0.1456 | dashboard vehicle subset latest sequence | 0.1985 | 40108 | 0.1485

The full-set tabular models are scored on the complete validation and test sets. The LSTM row, when present, shows its own scope in the table and is excluded from recommended-model selection until it is rebuilt on the identical full validation/test scope. Raw accuracy is included for context, but SCANIA cost drives model selection because most examples are class 0 and a high-accuracy all-class-0 model misses every failure.

## Supplemental Sequence Experiment

Model | Train accuracy | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1
--- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:
LSTM sequence model (subset) | - | dashboard vehicle subset latest sequence | 0.1836 | 40891 | 0.1456 | dashboard vehicle subset latest sequence | 0.1985 | 40108 | 0.1485

The LSTM result is kept as supplemental until it is rebuilt against the same complete validation/test scope as the tabular models. This avoids repeating the earlier unfair-comparison problem where some models were scored on sampled data and others were scored on full data.

## Hyperparameter Tuning

- Feature scaling: z-score standardization is fit on training rows only.
- Centroid alert threshold: grid searched from 0.00 to 1.00 in 0.01 steps against validation cost; selected threshold 0.
- kNN experiment: k = 351 and distance weighting power = 2. The sweep uses a balanced validation sample for speed, but the selected kNN model is now scored on the complete validation/test sets in the main comparison table.
- Temporal smoothing: the dashboard requires 3 repeated lower-risk packets before lowering an alert, while higher-risk packets update immediately.

### kNN k Sweep

The sweep uses a stratified training/evaluation sample so it can run quickly in the project repo while still preserving all positive validation examples.

The exported dashboard chart shows validation macro F1 by k. The high selected k is evidence that kNN is not the strongest final choice here; it is included to satisfy the hyperparameter comparison requirement. The headline kNN cost and accuracy use the complete validation/test sets.

k | Validation rows | Validation accuracy | Validation cost | Macro F1
---: | ---: | ---: | ---: | ---:
1 | 936 | 0.1015 | 30232 | 0.0714
3 | 936 | 0.1079 | 30236 | 0.075
5 | 936 | 0.1143 | 31710 | 0.0757
9 | 936 | 0.125 | 31568 | 0.0802
15 | 936 | 0.1325 | 32884 | 0.0829
25 | 936 | 0.1464 | 34056 | 0.0921
35 | 936 | 0.1549 | 34945 | 0.0977
51 | 936 | 0.1656 | 34654 | 0.1004
75 | 936 | 0.1731 | 34870 | 0.1064
101 | 936 | 0.1859 | 33981 | 0.111
151 | 936 | 0.1987 | 34084 | 0.1183
251 | 936 | 0.2094 | 34806 | 0.1186
351 | 936 | 0.219 | 32672 | 0.1254
501 | 936 | 0.2233 | 33744 | 0.1238

## Feature Dependence

The operational inputs are anonymized telemetry signals from one vehicle, not independent physical components. The model treats them as correlated inputs by using standardized distances across the full feature vector; it does not use a naive independence assumption.

## Features

- Feature count: 118
- Counter features: 100_0, 171_0, 309_0, 370_0, 427_0, 666_0, 835_0, 837_0
- Spec one-hot features: 93
- Inputs: current counter values, counter deltas, counter rates, relative time_step, vehicle spec categories

## Training Rows

{
  "0": 1096712,
  "1": 12503,
  "2": 6179,
  "3": 3200,
  "4": 3858
}

## Validation

- Rows: 5046
- Accuracy: 0.4322
- Macro F1: 0.1379
- Cost: 55883
- All-zero baseline cost: 57400
- Cost improvement vs all-zero: 0.0264

Actual \ Pred | 0 | 1 | 2 | 3 | 4
--- | ---: | ---: | ---: | ---: | ---:
0 | 2153 | 1302 | 465 | 436 | 554
1 | 7 | 6 | 1 | 0 | 2
2 | 3 | 5 | 2 | 0 | 4
3 | 6 | 11 | 5 | 4 | 4
4 | 15 | 28 | 14 | 3 | 16

## Test

- Rows: 5045
- Accuracy: 0.4244
- Macro F1: 0.132
- Cost: 56752
- All-zero baseline cost: 56100
- Cost improvement vs all-zero: -0.0116

Actual \ Pred | 0 | 1 | 2 | 3 | 4
--- | ---: | ---: | ---: | ---: | ---:
0 | 2118 | 1390 | 453 | 457 | 485
1 | 5 | 11 | 5 | 3 | 2
2 | 4 | 7 | 1 | 1 | 2
3 | 13 | 13 | 6 | 4 | 5
4 | 12 | 25 | 12 | 4 | 7

## Most Different Features Between Class 0 And Class 4 Centroids

- counter:835_0:log_value: 0.74
- counter:171_0:log_value: 0.7316
- counter:666_0:log_value: 0.7215
- counter:309_0:log_value: 0.699
- time_step_log: 0.6684
- counter:837_0:log_value: 0.6482
- counter:427_0:log_value: 0.4877
- counter:100_0:log_value: 0.4596
- counter:309_0:log_rate: 0.3567
- counter:309_0:log_delta: 0.2907
- spec:Spec_7=Cat6: 0.2692
- spec:Spec_2=Cat5: 0.2554
