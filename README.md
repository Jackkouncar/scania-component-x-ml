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

- `tools/train_baseline_model.js`: trains/evaluates all-class-0, nearest-centroid, Gaussian Naive Bayes, Random Forest, distance-weighted kNN, and imported Python tabular models.
- `tools/train_tabular_models.py`: trains/evaluates Logistic Regression and LightGBM on the complete validation/test sets.
- `tools/run_tabular_models.js`: Node wrapper for the Python tabular training script.
- `tools/train_lstm_model.js`: optional TensorFlow.js LSTM sequence experiment.
- `ml/ml_dataset.json`: generated feature table used by the training scripts.
- `ml/model_report.md`: current model report.
- `site/data/model_output.json`: exported model metrics.
- `site/data/tabular_model_output.json`: current Logistic Regression and LightGBM metrics.
- `site/data/lstm_model_output.json`: current LSTM experiment metrics.

The processed feature table is included, so the current model comparison can be reproduced without the raw SCANIA CSV files. If rebuilding from raw CSVs, place the extracted dataset at:

```text
2024-34-2/2024-34-2/data/
```

## Install

```bash
npm install
py -m pip install -r requirements-ml.txt
```

## Reproduce Current Results

```bash
npm run train:tabular
npm run train:model
```

Useful variants:

```bash
npm run train:model:accuracy
npm run train:model:cost
npm run train:lstm
npm run train:lstm:deep
```

## Current Model Choice

The recommended fair model is `LightGBM regularized`.

Reason:

- Every main model is scored on the complete validation and test sets.
- Selection is driven by SCANIA cost, not raw accuracy.
- Accuracy is still shown beside cost for context because the dataset is heavily imbalanced.
- LightGBM has the lowest full-validation SCANIA cost among the comparable trained models.

Current LightGBM results:

| Split | Accuracy | SCANIA cost |
|---|---:|---:|
| Validation | 0.8894 | 50,070 |
| Test | 0.8761 | 49,004 |

The browser-live dashboard scorer is still a nearest-centroid classifier because it is small, explainable, and easy to run interactively in JavaScript. The LSTM result is kept supplemental until it is rebuilt using the identical evaluation rows and label assignment as the tabular models.

See:

```text
ml/model_report.md
```
