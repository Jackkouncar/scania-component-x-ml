# Baseline ML Model Report

Generated: 2026-05-11T01:10:28.869Z

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
- Logistic Regression and LightGBM: added as Python tabular comparators with regularization and full validation/test scoring.
- Distance-weighted kNN: evaluated with a k sweep, but not selected because the best k was high and the stratified accuracy stayed low.
- LSTM sequence model: trained as a compact TensorFlow.js sequence experiment, but listed separately because its current exported scope is not the complete validation/test set.

The current submission keeps nearest-centroid as the browser-live model and uses the fair full-set table to choose the best trained tabular model. Current recommended model by validation rule: LightGBM regularized.

## Model Comparison

Model | Train accuracy | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1
--- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:
All-class-0 baseline | 0.485 | full validation set | 0.973 | 57400 | 0.1973 | full test set | 0.9719 | 56100 | 0.1971
Nearest-centroid classifier | 0.4988 | full validation set | 0.8706 | 56884 | 0.2006 | full test set | 0.8823 | 54627 | 0.2096
Gaussian Naive Bayes | 0.5387 | full validation set | 0.7463 | 61507 | 0.1773 | full test set | 0.7483 | 60594 | 0.1746
Random Forest | 0.5442 | full validation set | 0.2632 | 68243 | 0.0868 | full test set | 0.2636 | 65629 | 0.0881
Distance-weighted kNN (k=351) | 1 | full validation set | 0.3177 | 57049 | 0.1101 | full test set | 0.3146 | 55516 | 0.1102
LightGBM regularized | 0.5126 | full validation set | 0.8894 | 50070 | 0.1996 | full test set | 0.8761 | 49004 | 0.1947
Logistic Regression (L2 balanced) | 0.5053 | full validation set | 0.844 | 51874 | 0.1921 | full test set | 0.8333 | 52144 | 0.1867

All models in the main comparison above are scored on the complete validation and test sets. Raw accuracy is included for context, but it is not the only selection metric. Because most examples are class 0, the all-class-0 baseline can look strong on accuracy while missing every failure. The recommended-model rule prioritizes full-validation SCANIA cost among models with a reasonable validation-accuracy operating point, then reports train/validation/test accuracy to watch for overfitting and class imbalance effects.

## Supplemental Sequence Experiment

Model | Train accuracy | Validation scope | Validation accuracy | Validation cost | Validation macro F1 | Test scope | Test accuracy | Test cost | Test macro F1
--- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---:
LSTM sequence model | - | dashboard vehicle subset latest sequence | 0.1758 | 41396 | 0.1378 | dashboard vehicle subset latest sequence | 0.187 | 40886 | 0.1365

The LSTM result is kept as supplemental until it is rebuilt against the same complete validation/test scope as the tabular models. This avoids repeating the unfair-comparison problem called out by the TA.

## Hyperparameter Tuning

- Feature scaling: z-score standardization is fit on training rows only.
- Centroid alert threshold: grid searched from 0.00 to 1.00 in 0.01 steps against validation cost; selected threshold 0.21.
- kNN experiment: k = 351 and distance weighting power = 2. The sweep uses a balanced validation sample for speed, but the selected kNN model is now scored on the complete validation/test sets in the main comparison table.
- Temporal smoothing: the dashboard requires 3 repeated lower-risk packets before lowering an alert, while higher-risk packets update immediately.

### kNN k Sweep

The sweep uses a stratified training/evaluation sample so it can run quickly in the project repo while still preserving all positive validation examples.

The exported dashboard chart shows validation macro F1 by k. The high selected k is evidence that kNN is not the strongest final choice here; it is included to satisfy the hyperparameter comparison requirement. The headline kNN cost and accuracy use the complete validation/test sets.

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
