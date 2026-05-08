# Baseline ML Model Report

Generated: 2026-05-08T19:21:57.627Z

## Model

Dashboard streaming model: nearest-centroid classifier.

For each incoming telemetry packet, the dashboard compares the engineered feature vector against one learned centroid per risk class and predicts from the closest class pattern.

The exported file also keeps the kNN sweep so the project can explain why kNN was tested but not chosen as the final dashboard model.

This is an explainable baseline model, not the final production model.

Alert threshold: 0.21

Non-zero class predictions below this confidence are converted to class 0. The threshold is tuned on validation cost to reduce noisy false alarms.

## Model Selection Rationale

- Naive all-class-0 baseline: included because the dataset is highly imbalanced and most vehicles are not near failure.
- Nearest-centroid classifier: used as a fast reportable baseline and for class-separation summaries.
- Gaussian Naive Bayes: added as a second trained comparator. It is fast, but it makes a stronger feature-independence assumption.
- Random Forest: added as a tree-based comparator because the TA recommended testing a tree method for nonlinear tabular patterns.
- Distance-weighted kNN: evaluated with a k sweep, but not selected because the best k was high and the stratified accuracy stayed low.
- LSTM sequence model: trained as a compact TensorFlow.js sequence experiment on rolling dashboard telemetry windows.

The current submission therefore uses the nearest-centroid classifier as the live dashboard model and compares it against multiple trained baselines.

## Model Comparison

Model | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1
--- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:
All-class-0 baseline | full validation set | 0.973 | 57400 | 0.1973 | full test set | 0.9719 | 56100 | 0.1971
Nearest-centroid classifier | full validation set | 0.8706 | 56884 | 0.2006 | full test set | 0.8823 | 54627 | 0.2096
Gaussian Naive Bayes | full validation set | 0.7463 | 61507 | 0.1773 | full test set | 0.7483 | 60594 | 0.1746
Random Forest | full validation set | 0.2632 | 68243 | 0.0868 | full test set | 0.2636 | 65629 | 0.0881
Distance-weighted kNN (k=351) | stratified validation sample | 0.234 | 33736 | 0.1296 | stratified test sample | 0.2197 | 32324 | 0.1276
LSTM sequence model | dashboard vehicle subset latest sequence | 0.1758 | 41396 | 0.1378 | dashboard vehicle subset latest sequence | 0.187 | 40886 | 0.1365

Raw accuracy is included for the class presentation, but it is not the only useful metric. Because most examples are class 0, the all-class-0 baseline can look strong on accuracy while missing every failure. The nearest-centroid model is the main dashboard model because it keeps full-set validation/test accuracy above 75% while also lowering maintenance cost compared with the all-class-0 baseline.

## Hyperparameter Tuning

- Feature scaling: z-score standardization is fit on training rows only.
- Centroid alert threshold: grid searched from 0.00 to 1.00 in 0.01 steps against validation cost; selected threshold 0.21.
- kNN experiment: k = 351 and distance weighting power = 2. This sweep is kept as validation evidence, not as the final selected model.
- Temporal smoothing: the dashboard requires 3 repeated lower-risk packets before lowering an alert, while higher-risk packets update immediately.

### kNN k Sweep

The sweep uses a stratified training/evaluation sample so it can run quickly in the project repo while still preserving all positive validation examples.

The exported dashboard chart shows validation macro F1 by k. The high selected k is evidence that kNN is not the strongest final choice here; it is included to satisfy the hyperparameter comparison requirement.

k | Validation rows | Validation accuracy | Validation cost | Macro F1
---: | ---: | ---: | ---: | ---:
1 | 936 | 0.1058 | 28847 | 0.0768
3 | 936 | 0.1036 | 29998 | 0.0727
5 | 936 | 0.11 | 31631 | 0.0744
9 | 936 | 0.1186 | 32504 | 0.0772
15 | 936 | 0.1282 | 32097 | 0.0826
25 | 936 | 0.1346 | 33826 | 0.0866
35 | 936 | 0.1432 | 33850 | 0.0901
51 | 936 | 0.1603 | 34404 | 0.0994
75 | 936 | 0.1731 | 34097 | 0.107
101 | 936 | 0.1795 | 35210 | 0.1047
151 | 936 | 0.1966 | 34808 | 0.1131
251 | 936 | 0.2158 | 34058 | 0.1196
351 | 936 | 0.234 | 33736 | 0.1296
501 | 936 | 0.2361 | 35184 | 0.1258

## Feature Dependence

The operational inputs are anonymized telemetry signals from one vehicle, not independent physical components. The model treats them as correlated inputs by using standardized distances across the full feature vector; it does not use a naive independence assumption.

## Features

- Feature count: 118
- Counter features: 100_0, 171_0, 309_0, 370_0, 427_0, 666_0, 835_0, 837_0
- Spec one-hot features: 93
- Inputs: current counter values, counter deltas, counter rates, relative time_step, vehicle spec categories

## Training Rows

{
  "0": 20000,
  "1": 8000,
  "2": 6179,
  "3": 3200,
  "4": 3858
}

## Validation

- Rows: 5046
- Accuracy: 0.8706
- Macro F1: 0.2006
- Cost: 56884
- All-zero baseline cost: 57400
- Cost improvement vs all-zero: 0.009

Actual \ Pred | 0 | 1 | 2 | 3 | 4
--- | ---: | ---: | ---: | ---: | ---:
0 | 4389 | 66 | 354 | 42 | 59
1 | 14 | 1 | 1 | 0 | 0
2 | 12 | 0 | 1 | 0 | 1
3 | 24 | 1 | 3 | 1 | 1
4 | 65 | 2 | 8 | 0 | 1

## Test

- Rows: 5045
- Accuracy: 0.8823
- Macro F1: 0.2096
- Cost: 54627
- All-zero baseline cost: 56100
- Cost improvement vs all-zero: 0.0263

Actual \ Pred | 0 | 1 | 2 | 3 | 4
--- | ---: | ---: | ---: | ---: | ---:
0 | 4444 | 60 | 337 | 30 | 32
1 | 22 | 2 | 2 | 0 | 0
2 | 12 | 0 | 3 | 0 | 0
3 | 33 | 1 | 5 | 1 | 1
4 | 53 | 2 | 3 | 1 | 1

## Most Different Features Between Class 0 And Class 4 Centroids

- counter:835_0:log_value: 0.7901
- counter:171_0:log_value: 0.7538
- time_step_log: 0.6962
- counter:666_0:log_value: 0.6626
- spec:Spec_7=Cat4: 0.65
- spec:Spec_7=Cat1: 0.6404
- counter:309_0:log_value: 0.5638
- counter:837_0:log_value: 0.5273
- spec:Spec_7=Cat7: 0.4926
- counter:100_0:log_value: 0.4731
- counter:427_0:log_value: 0.4716
- spec:Spec_7=Cat6: 0.4494
