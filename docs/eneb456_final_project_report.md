# ENEB456 Final Project Report

Source code repository:

```text
https://github.com/Jackkouncar/scania-component-x-ml
```

## Project Title

Predictive Maintenance Dashboard for SCANIA Component X Using Cost-Sensitive Machine Learning

## Team Members

- Jack
- Talia

## Abstract

This project applies supervised machine learning to the SCANIA Component X predictive maintenance dataset. The goal is to classify vehicle risk into five classes: class 0 for no imminent failure and classes 1-4 for increasingly urgent failure windows. Because the dataset is highly imbalanced, raw accuracy can be misleading. The final model recommendation prioritizes the SCANIA asymmetric cost matrix while still reporting accuracy, macro F1, confusion matrices, and test performance. After expanding the training set to 1,122,452 readout rows and applying tuned LightGBM parameters, the selected LightGBM model achieved a validation cost of 35,599, compared with the all-class-0 baseline cost of 57,400.

## Data Preparation

The dataset comes from the SCANIA Component X project files. The raw data includes operational readouts, vehicle specifications, training time-to-event outcomes, and validation/test labels. The final training table contains 1,122,452 training readout rows, 5,046 validation vehicles, and 5,045 test vehicles.

Missing numeric values were treated as zero for this baseline because the anonymized operational counters are sparse telemetry signals. Counter values were transformed with log1p so very large counter values would not dominate distance-based or tree-based models. Categorical specification fields were one-hot encoded. The validation and test sets were kept at the complete vehicle level rather than being downsampled.

## Feature Engineering

The feature vector contains 118 engineered features:

- relative time_step transformed with log1p
- current values for the eight anonymized counters
- counter deltas from the previous readout for the same vehicle
- counter rates based on delta divided by elapsed time_step
- one-hot encoded vehicle specification categories

These features are streaming-safe: they can be computed when a new telemetry packet arrives.

## Model Development

The project compares the following models:

- all-class-0 baseline
- nearest-centroid classifier
- Gaussian Naive Bayes
- Random Forest
- distance-weighted kNN with a k sweep
- Logistic Regression
- LightGBM
- LSTM sequence experiment

LightGBM is the recommended model because it produced the lowest full-validation SCANIA cost among the comparable full-set models. The tuned LightGBM parameters were implemented:

```text
objective = multiclass
num_class = 5
class_weight = balanced
n_estimators = 500
learning_rate = 0.05
num_leaves = 63
min_child_samples = 5
random_state = 42
n_jobs = -1
```

The direct expected-cost rule is also included as a separate row. It computes predicted probabilities, multiplies them by the SCANIA cost matrix, and chooses the class with the lowest expected cost. A validation-tuned LightGBM decision rule was then selected because it reduced validation cost even further by applying a cost-savings margin threshold.

As an additional check, the notebook-style balanced-sample-weight method was rerun on the project's engineered 118-feature table. That added the **LightGBM balanced expected-cost** row. It improved the direct expected-cost validation result slightly, from 45,519 to 45,063, but it did not beat the validation-tuned cost-sensitive LightGBM row.

## Results And Analysis

| Model | Validation scope | Val accuracy | Val cost | Test accuracy | Test cost |
|---|---|---:|---:|---:|---:|
| All-class-0 baseline | full validation set | 97.3% | 57,400 | 97.2% | 56,100 |
| Nearest-centroid classifier | full validation set | 43.2% | 55,883 | 42.4% | 56,752 |
| Gaussian Naive Bayes | full validation set | 82.4% | 53,032 | 82.0% | 55,387 |
| Random Forest | full validation set | 26.3% | 68,243 | 26.4% | 65,629 |
| Distance-weighted kNN (k=351) | full validation set | 28.7% | 57,124 | 28.8% | 57,785 |
| LightGBM cost-sensitive | full validation set | 54.4% | 35,599 | 65.7% | 47,930 |
| LightGBM 75% accuracy floor | full validation set | 75.0% | 44,668 | 85.3% | 50,094 |
| LightGBM balanced expected-cost | full validation set | 10.6% | 45,063 | 12.6% | 44,896 |
| LightGBM expected-cost | full validation set | 10.2% | 45,519 | 12.5% | 44,768 |
| Logistic Regression (L2 balanced) | full validation set | 24.2% | 48,292 | 21.1% | 49,767 |
| LSTM sequence model (subset) | dashboard vehicle subset latest sequence | 18.4% | 40,891 | 19.9% | 40,108 |

The recommended model is **LightGBM cost-sensitive**. Its validation cost is 35,599, which is 21,801 lower than the all-class-0 validation baseline. The notebook-style balanced expected-cost LightGBM row produced validation cost 45,063 and test cost 44,896. The original direct expected-cost LightGBM row produced validation cost 45,519 and test cost 44,768. The cost-sensitive LightGBM row produced the lowest validation cost, while the expected-cost rows remain useful because they are the simplest probability-times-cost-matrix decision rules.

Accuracy is reported, but cost drives selection. The all-class-0 baseline has very high accuracy because class 0 dominates the dataset, but it misses every failure case and therefore has a high SCANIA cost. The selected LightGBM model intentionally predicts more positive failure-risk classes, which lowers raw accuracy but reduces the more important missed-failure cost. For context, the LightGBM 75% accuracy floor row reaches 75.0% validation accuracy, but its validation cost is 44,668, so it is not the best maintenance decision rule. This is why a lower-accuracy model can be the better maintenance model.

The LSTM sequence experiment is included as a comparison model, but its row is marked as a dashboard-subset scope. It should not be used to select the final model until it is rebuilt on the identical full validation/test scope as LightGBM.

## Visualization

The dashboard and presentation visualize model comparison, validation cost, test cost, kNN k-sweep behavior, class imbalance, confusion matrices, and selected feature-signal differences. The dashboard also demonstrates historical telemetry replay and real-time inserted telemetry scoring.

## Limitations And Future Work

- The physical component name and feature meanings are anonymized.
- The cost-sensitive LightGBM decision threshold is selected on validation data and should be monitored for test-set generalization.
- The LSTM experiment should be rebuilt on the same full validation/test vehicle scope before treating it as a fair final competitor.
- Future work could add XGBoost, better probability calibration, survival analysis for censored vehicles, and more complete sequence features.

## Source Code And Dataset

The source code includes all scripts used to build features and train models:

- `tools/build_ml_dataset.js`
- `tools/train_tabular_models.py`
- `tools/train_baseline_model.js`
- `tools/train_lstm_model.js`

The generated model report is `ml/model_report.md`. The expanded ML feature table is `ml/ml_dataset.json`. Because that uncompressed JSON file exceeds GitHub's direct-file limit, the GitHub submission provides it as compressed split files and includes `npm run restore:ml-data` to reconstruct the complete processed table before retraining. The raw SCANIA files may still be shared separately when a full raw-data rebuild is needed.
