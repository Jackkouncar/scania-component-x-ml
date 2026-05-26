# IEEE-Style Paper Draft

Working title: Predictive Maintenance Dashboard for SCANIA Component X Using Streaming Telemetry and Supervised Risk Classification

Authors: Jack and Talia

Course: ENEB456 Machine Learning Tools

## Abstract

Predictive maintenance uses operational data to identify equipment that is approaching failure before a costly breakdown occurs. This project develops a dashboard and machine learning workflow for the SCANIA Component X predictive maintenance dataset. The dataset contains anonymized heavy-duty vehicle readouts, categorical specifications, repair outcomes, and failure-window labels. We build a normalized database design, prepare model features from telemetry counters and vehicle specifications, train several supervised classifiers, and deploy a live scorer inside an interactive web dashboard. The recommended fair model is a cost-sensitive LightGBM classifier selected by SCANIA validation cost while also reporting accuracy, macro F1, confusion matrices, and train-validation behavior. The browser demo exports the trained LightGBM trees for live playback and keeps simpler centroid/kNN modes as comparison options. Results show that LightGBM improves validation cost compared with an all-class-0 model and the simpler baselines.

## I. Introduction

Unexpected equipment failure can create safety risks, downtime, and unnecessary operating cost. In heavy-duty truck fleets, maintenance teams need early warning signals that identify vehicles moving toward failure while avoiding excessive false alarms. The SCANIA Component X dataset provides a realistic predictive maintenance problem: given vehicle telemetry and specification data, predict whether an anonymized engine component is close to failure.

The goal of this project is twofold. First, we design a data and dashboard workflow that can ingest vehicle readouts, display vehicle health, and support maintenance decisions. Second, we create a supervised machine learning baseline that predicts the risk class of Component X from the data available at scoring time. The project emphasizes honest interpretation because the dataset anonymizes the physical component name, operational feature names, and true time units.

Our main contributions are:

- A normalized 3NF database schema for vehicles, specifications, readouts, measurements, labels, repair outcomes, and predictions.
- A feature engineering pipeline for streaming telemetry classification.
- A regularized LightGBM model selected by SCANIA cost, plus browser comparison modes for probability trend, centroid, kNN, and baseline playback.
- A dashboard that supports both historical replay and real-time insertion of new telemetry records.
- A final-project documentation package connecting the implementation to the data preparation, model development, analysis, and IEEE paper requirements.

## II. Related Work And Background

Predictive maintenance commonly uses sensor data, operational counters, maintenance history, and equipment metadata to predict remaining useful life or classify failure risk. Common methods include logistic regression, nearest-neighbor methods, decision trees, random forests, gradient boosting, survival analysis, and neural sequence models. In many industrial settings, interpretability and deployment simplicity matter because maintenance users need to understand why an alert appears.

The SCANIA Component X challenge is framed as a multi-class classification problem with asymmetric costs. Class 0 represents no imminent failure, while classes 1 through 4 represent increasingly urgent failure windows. The challenge cost matrix penalizes missed failures much more heavily than unnecessary maintenance checks. This means raw accuracy can be misleading because a model that predicts class 0 for every vehicle can appear accurate while failing the maintenance goal.

## III. Dataset

The dataset is stored locally under:

```text
2024-34-2/2024-34-2/data/
```

The main files are:

- `train_operational_readouts.csv`
- `train_specifications.csv`
- `train_tte.csv`
- `validation_operational_readouts.csv`
- `validation_specifications.csv`
- `validation_labels.csv`
- `test_operational_readouts.csv`
- `test_specifications.csv`
- `test_labels.csv`

The operational readouts contain `vehicle_id`, `time_step`, and 105 anonymized operational features. Eight features are single counter variables and six variables are histogram groups. The categorical specification files contain eight anonymized specification fields. The training target file provides whether Component X was repaired during the study window and the length of study. Validation and test target files provide class labels for the latest available readout.

The class labels are interpreted as:

| Class | Meaning |
|---:|---|
| 0 | No imminent failure or outside the 48-time-unit warning window |
| 1 | 48 to 24 relative time units before failure |
| 2 | 24 to 12 relative time units before failure |
| 3 | 12 to 6 relative time units before failure |
| 4 | 6 to 0 relative time units before failure |

The exact physical component and time units are not public. Therefore, the project describes the target as Component X and treats `time_step` as an anonymized relative time unit.

## IV. Data Preparation

The data preparation process is implemented in:

```text
tools/build_ml_dataset.js
```

The model uses only fields that would be available during dashboard scoring. Each training row is constructed from a vehicle readout, the previous readout for that vehicle, and vehicle specifications.

Preprocessing decisions:

- Missing numeric telemetry values are converted to zero for modeling.
- Counter values are transformed with `log1p` to reduce the effect of very large counters.
- Counter deltas are computed from the previous readout.
- Counter rates are computed as delta divided by the elapsed relative time.
- Vehicle specification categories are one-hot encoded.
- Features are standardized in the training script using training-set mean and standard deviation.

Training labels are derived from `train_tte.csv`. For repaired vehicles, the time to failure is computed as:

```text
time_to_failure = length_of_study_time_step - current_time_step
```

That value is converted into classes 0 through 4 using the official failure windows. For vehicles without an observed repair during the study period, readouts are treated as class 0 for this baseline. This is a simplification; future survival-analysis work could better represent censored examples.

Validation and test rows use the latest readout for each vehicle because the provided labels describe the final available readout.

## V. Database And Dashboard Design

The database schema is documented in:

```text
docs/erd_3nf.md
```

The design separates vehicles, dataset splits, specifications, operational readouts, individual measurements, training repair outcomes, validation/test labels, model runs, and predictions. This avoids repeating vehicle and feature metadata while still supporting dashboard and machine learning workflows.

The dashboard is implemented in:

```text
site/index.html
site/styles.css
site/app.js
```

The dashboard includes:

- Vehicle lookup and filtering.
- Dataset split and risk-class controls.
- Model performance summary.
- Component X health prediction.
- Latest telemetry packet table.
- Event log.
- Readout timeline with failure marker or failure window.
- Histogram snapshot.
- Real-time telemetry insertion mode.

Real-time mode appends a user-entered telemetry record to the selected vehicle, stores it in browser local storage for the demo machine, and immediately re-runs the prediction pipeline.

## VI. Model Development

The main tabular model training script is:

```text
tools/train_tabular_models.py
```

The selected final model is:

```text
LightGBM cost-sensitive
```

This is a regularized gradient-boosted tree classifier. For a new telemetry packet, the dashboard builds the same feature vector used during training, evaluates the exported LightGBM trees, converts the five class scores into probabilities, and then applies the selected decision rule. The cost-sensitive mode multiplies probabilities by the official SCANIA cost matrix and uses a validation-tuned cost-savings threshold before predicting a nonzero failure class. The probability-trend mode is included for live demonstration because it converts the same class probabilities into a smoother risk trajectory.

Model candidates considered:

| Model | Role | Decision |
|---|---|---|
| All-class-0 baseline | Imbalanced-data sanity check | Used for comparison |
| Nearest centroid | Simple browser-live comparator | Kept as an explainable baseline option |
| Distance-weighted kNN | k-value sweep comparator | Tested, not selected |
| Bagging / Random Forest | Nonlinear ensemble candidate | Tested, not selected |
| Logistic Regression | Regularized linear comparator | Tested, not selected |
| LightGBM | Regularized boosted-tree comparator | Recommended fair model |
| LightGBM balanced expected-cost | Notebook-style weighted LightGBM using engineered features | Tested, improved expected-cost validation result, not selected |

LightGBM is the recommended fair model because it has the lowest full-validation SCANIA cost among the comparable full-set trained models. Accuracy is still reported beside cost because the course rubric requires classification metrics, but cost drives the main model choice because the dataset is heavily imbalanced. The dashboard now supports the exported LightGBM scorer for live telemetry playback while also retaining simpler modes for comparison and demonstration.

Hyperparameter decisions:

- LightGBM settings: `objective = multiclass`, `num_class = 5`, `class_weight = balanced`, `n_estimators = 500`, `learning_rate = 0.05`, `num_leaves = 63`, `min_child_samples = 5`, `random_state = 42`
- The exact expected-cost LightGBM rule computes `predict_proba(X) @ cost_matrix` and selects the class with the lowest expected cost. The selected LightGBM row uses the same probabilities with a validation-tuned cost-savings margin threshold.
- The notebook-style balanced expected-cost row uses balanced sample weights, `n_estimators = 500`, `learning_rate = 0.05`, `num_leaves = 63`, `min_child_samples = 100`, and the same `predict_proba(X) @ cost_matrix` decision rule, but retrained on the project's engineered feature table.
- kNN sweep over `k = 1, 3, 5, 9, 15, 25, 35, 51, 75, 101, 151, 251, 351, 501`
- feature standardization from the training set
- centroid alert threshold grid search from 0.00 to 1.00 in 0.01 steps
- dashboard temporal smoothing requiring repeated lower-risk packets before lowering an alert

Bagging and boosting were both tested. Random Forest was added as a bagging/tree comparator, but it did not beat the stronger tabular models in this implementation. LightGBM was added as the boosted-tree model and is the recommended fair model. XGBoost remains a reasonable future comparison if more time is available.

## VII. Results

The current model report is stored in:

```text
ml/model_report.md
```

Current validation results:

| Metric | Value |
|---|---:|
| Rows | 5,046 |
| Recommended model | LightGBM cost-sensitive |
| Accuracy | 0.5438 |
| Macro F1 | 0.1508 |
| Cost | 35,599 |
| All-zero baseline cost | 57,400 |
| Cost improvement vs all-zero | 0.3798 |

Additional LightGBM expected-cost check:

| Model | Validation accuracy | Validation cost | Test accuracy | Test cost |
|---|---:|---:|---:|---:|
| LightGBM balanced expected-cost | 0.1056 | 45,063 | 0.1255 | 44,896 |
| LightGBM expected-cost | 0.1021 | 45,519 | 0.1247 | 44,768 |

Current test results:

| Metric | Value |
|---|---:|
| Rows | 5,045 |
| Recommended model | LightGBM cost-sensitive |
| Accuracy | 0.6573 |
| Macro F1 | 0.1647 |
| Cost | 47,930 |
| All-zero baseline cost | 56,100 |
| Cost improvement vs all-zero | 0.1456 |

The results show that the boosted tree improves validation SCANIA cost over both the all-class-0 baseline and the simpler centroid scorer. As an additional check, the notebook-style balanced expected-cost method was rerun on the engineered feature table; this improved validation cost compared with the original expected-cost row but did not beat the validation-tuned cost-sensitive row. The exact expected-cost LightGBM rows have stronger test cost than the validation-tuned row, so the project reports both and treats threshold tuning as a validation-selected operating point. The macro F1 score remains low because classes 1 through 4 are rare, making minority-class detection difficult. This confirms that future model work should focus on full-scope sequence evaluation, stronger temporal features, and cost-sensitive calibration.

## VIII. Discussion

The most important strength of the project is that the dashboard is connected to real SCANIA readouts and a trained supervised model. It does not use invented component names, invented timestamps, or fabricated physical interpretations. The dashboard can also demonstrate a realistic streaming workflow by replaying historical telemetry or inserting a new live record.

The main limitation is that the live browser scorer uses the exported tabular feature vector rather than all possible histogram features or longer rolling histories. It also handles censored training vehicles conservatively by treating their observed readouts as class 0. Future work should rebuild the LSTM with identical evaluation rows and label assignment, test XGBoost, and tune class thresholds directly against the challenge cost matrix.

Another limitation is interpretability of the raw variables. Because operational feature names are anonymized, the project can identify which feature codes are useful but cannot truthfully claim that a feature represents a named physical part such as a fuel pump, battery, or tire.

## IX. Conclusion

This project demonstrates a complete predictive maintenance workflow for SCANIA Component X. It prepares raw telemetry and specification data, designs a normalized database, evaluates multiple supervised models, and deploys an interactive dashboard with historical replay and real-time telemetry insertion. The recommended LightGBM model improves cost compared with an all-class-0 model, and the documentation clearly identifies future work in full-scope LSTM evaluation, stronger temporal features, and cost-sensitive tuning.

## References

[1] SCANIA Component X predictive maintenance dataset documentation, local project files.

[2] ENEB456 Machine Learning Tools Final Project Guidelines, course handout.

[3] IEEE conference paper formatting template, course-provided Overleaf template.

[4] Predictive maintenance and imbalanced classification references to be added before final submission.
