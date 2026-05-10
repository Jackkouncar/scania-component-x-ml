# SCANIA Component X ML Code

This repository contains the machine-learning scripts used for the SCANIA Component X predictive-maintenance project.

The task is a 5-class classification problem:

| Class | Meaning |
|---|---|
| 0 | No imminent failure |
| 1 | 48 to 24 relative time units before failure |
| 2 | 24 to 12 relative time units before failure |
| 3 | 12 to 6 relative time units before failure |
| 4 | 6 to 0 relative time units before failure |

## What Is Included

- `tools/build_ml_dataset.js`: builds the feature table from the raw SCANIA CSV files.
- `tools/train_baseline_model.js`: trains/evaluates the centroid, Gaussian Naive Bayes, Random Forest, distance-weighted kNN, and imported Python tabular models.
- `tools/train_tabular_models.py`: trains/evaluates Logistic Regression and LightGBM on the complete validation/test sets.
- `tools/run_tabular_models.js`: Node wrapper for the Python tabular training script.
- `tools/train_lstm_model.js`: optional TensorFlow.js LSTM sequence experiment.
- `ml/ml_dataset.json`: generated feature table used by the training script.
- `ml/model_report.md`: current model report.
- `site/data/model_output.json`: current exported model metrics and dashboard model data.
- `site/data/tabular_model_output.json`: current Logistic Regression and LightGBM metrics.
- `site/data/lstm_model_output.json`: current LSTM experiment metrics.

The raw SCANIA dataset is not included because it is large. If rebuilding from raw CSVs, place the extracted dataset at:

```text
2024-34-2/2024-34-2/data/
```

## Install

```bash
npm install
```

## Run The Current Model Training

The generated feature table is already included, so the main model can be retrained with:

```bash
npm run train:model
```

This runs the model comparison and k-value sweep.

To refresh the TA-requested LightGBM and Logistic Regression comparison first:

```bash
py -m pip install -r requirements-ml.txt
npm run train:tabular
npm run train:model
```

Useful variants:

```bash
npm run train:model:accuracy
npm run train:model:cost
npm run train:lstm
```

## Current Model Choice

The recommended fair model is LightGBM regularized.

Reason:

- It is evaluated on the complete validation and test sets, not a stratified shortcut sample.
- It clears the 75% validation-accuracy target.
- It has the lowest full-validation SCANIA cost among the tested trained models.
- It uses regularization and a validation-tuned cost margin threshold to balance accuracy against expensive missed failures.
- The kNN sweep is still included, but kNN was not selected because its best k was high and its full-set accuracy stayed low.

The browser-live dashboard scorer is still a nearest-centroid classifier because it is small, explainable, and easy to run interactively in JavaScript.

The report also includes:

- all-class-0 baseline
- nearest-centroid baseline
- Gaussian Naive Bayes
- Random Forest
- distance-weighted kNN
- Logistic Regression
- LightGBM
- LSTM sequence experiment, listed separately until it is rebuilt on the same full validation/test scope

See:

```text
ml/model_report.md
```

## Notes On Metrics

The dataset is highly imbalanced, with many more healthy examples than near-failure examples. Raw accuracy can therefore be misleading because a model can look accurate by predicting class 0 too often. The report includes accuracy, macro F1, and failure-cost metrics.
