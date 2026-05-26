import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - handled at runtime for setup docs
    LGBMClassifier = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "ml" / "ml_dataset.json"
OUT_FILE = ROOT / "site" / "data" / "tabular_model_output.json"
OUT_LGBM_FILE = ROOT / "site" / "data" / "lightgbm_model.json"
CLASSES = np.array([0, 1, 2, 3, 4])
MIN_OPERATING_ACCURACY = float(os.environ.get("MIN_OPERATING_ACCURACY", "0"))
DECISION_THRESHOLD_STEPS = int(os.environ.get("DECISION_THRESHOLD_STEPS", "2001"))
LOGISTIC_CLASS0_LIMIT = int(os.environ.get("LOGISTIC_CLASS0_LIMIT", "20000"))
LOGISTIC_POSITIVE_LIMIT = int(os.environ.get("LOGISTIC_POSITIVE_LIMIT", "8000"))
COST = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=float,
)


def load_split(dataset, split_name):
    rows = dataset[split_name]
    x = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return rows, x, y


def class_counts(y):
    return {str(label): int((y == label).sum()) for label in CLASSES}


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def macro_f1_from_matrix(matrix):
    f1s = []
    per_class = {}
    for label in CLASSES:
        label = int(label)
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        f1s.append(f1)
        per_class[str(label)] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(matrix[label, :].sum()),
        }
    return round(float(np.mean(f1s)), 4), per_class


def decision_from_probabilities(probabilities, decision):
    mode = decision["mode"] if isinstance(decision, dict) else decision
    if mode == "expected_cost":
        expected_cost = probabilities @ COST
        return expected_cost.argmin(axis=1).astype(np.int64)
    if mode == "cost_margin_threshold":
        expected_cost = probabilities @ COST
        best_nonzero_cost = expected_cost[:, 1:].min(axis=1)
        best_nonzero_class = expected_cost[:, 1:].argmin(axis=1) + 1
        savings = expected_cost[:, 0] - best_nonzero_cost
        threshold = decision["threshold"]
        return np.where(savings >= threshold, best_nonzero_class, 0).astype(np.int64)
    return probabilities.argmax(axis=1).astype(np.int64)


def prior_adjustment(y_train, y_target):
    train_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(float) + 1.0
    target_counts = np.bincount(y_target, minlength=len(CLASSES)).astype(float) + 1.0
    train_prior = train_counts / train_counts.sum()
    target_prior = target_counts / target_counts.sum()
    return target_prior / train_prior


def apply_prior_adjustment(probabilities, adjustment):
    adjusted = probabilities * adjustment.reshape(1, -1)
    denominator = adjusted.sum(axis=1, keepdims=True)
    return adjusted / np.maximum(denominator, 1e-12)


def evaluation_from_predictions(y_true, y_pred, scope):
    matrix = confusion_matrix(y_true, y_pred)
    total_cost = int(COST[y_true, y_pred].sum())
    all_zero_cost = int(COST[y_true, np.zeros_like(y_true)].sum())
    macro_f1, per_class = macro_f1_from_matrix(matrix)
    return {
        "scope": scope,
        "rows": int(len(y_true)),
        "classCounts": class_counts(y_true),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macroF1": macro_f1,
        "totalCost": total_cost,
        "allZeroBaselineCost": all_zero_cost,
        "costImprovementVsAllZero": round((all_zero_cost - total_cost) / max(1, all_zero_cost), 4),
        "confusionMatrix": matrix.tolist(),
        "perClass": per_class,
    }


def evaluation(y_true, probabilities, decision, scope):
    y_pred = decision_from_probabilities(probabilities, decision)
    return evaluation_from_predictions(y_true, y_pred, scope)


def positive_gap(train_eval, validation_eval):
    return {
        "accuracyGap": round(train_eval["accuracy"] - validation_eval["accuracy"], 4),
        "macroF1Gap": round(train_eval["macroF1"] - validation_eval["macroF1"], 4),
        "costGap": int(train_eval["totalCost"] - validation_eval["totalCost"]),
    }


def compact(entry):
    keep = [
        "scope",
        "rows",
        "classCounts",
        "accuracy",
        "macroF1",
        "totalCost",
        "allZeroBaselineCost",
        "costImprovementVsAllZero",
    ]
    return {key: entry[key] for key in keep}


def stratified_limit(x, y, class0_limit, positive_limit):
    limits = {0: class0_limit, 1: positive_limit, 2: positive_limit, 3: positive_limit, 4: positive_limit}
    counts = {label: 0 for label in CLASSES}
    keep = []
    for index, label in enumerate(y):
        label = int(label)
        if counts[label] >= limits[label]:
            continue
        counts[label] += 1
        keep.append(index)
    keep = np.asarray(keep, dtype=np.int64)
    return x[keep], y[keep]


def balanced_sample_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    weights = len(y) / (len(CLASSES) * np.maximum(counts, 1.0))
    return weights[y]


def decision_label(decision):
    if isinstance(decision, dict):
        if decision["mode"] == "cost_margin_threshold":
            return f"cost_margin_threshold >= {decision['threshold']:.4f}"
        return decision["mode"]
    return decision


def export_lightgbm_browser_model(dataset, estimator, entries, hyperparameters, training_seconds):
    decision_modes = {
        "lightgbm_cost": next(item for item in entries if item["name"] == "LightGBM cost-sensitive"),
        "lightgbm_accuracy": next(item for item in entries if item["name"] == "LightGBM 75% accuracy floor"),
        "lightgbm_expected": next(item for item in entries if item["name"] == "LightGBM expected-cost"),
    }
    booster_dump = estimator.booster_.dump_model()
    artifact = {
        "generatedAt": np.datetime64("now").astype(str),
        "modelName": "lightgbm_component_x_browser",
        "modelType": "lightgbm_multiclass",
        "sourceDataset": str(DATASET_FILE.relative_to(ROOT)),
        "featureNames": dataset["featureNames"],
        "numClass": 5,
        "hyperparameters": hyperparameters,
        "trainingSeconds": round(float(training_seconds), 2),
        "decisionModes": {
            key: {
                "name": entry["name"],
                "decisionMode": entry["decisionMode"],
                "decisionConfig": entry["decisionConfig"],
                "validation": entry["validation"],
                "test": entry["test"],
            }
            for key, entry in decision_modes.items()
        },
        "treeInfo": booster_dump["tree_info"],
    }
    OUT_LGBM_FILE.write_text(json.dumps(artifact))
    print(f"Wrote {OUT_LGBM_FILE}")


def model_entry(
    name,
    model_type,
    estimator,
    decision_mode,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    hyperparameters,
    adjustment=None,
    training_seconds=None,
):
    train_probs = estimator.predict_proba(x_train)
    val_probs = estimator.predict_proba(x_val)
    test_probs = estimator.predict_proba(x_test)
    if adjustment is not None:
        train_probs = apply_prior_adjustment(train_probs, adjustment)
        val_probs = apply_prior_adjustment(val_probs, adjustment)
        test_probs = apply_prior_adjustment(test_probs, adjustment)
    train_eval = evaluation(y_train, train_probs, decision_mode, "training sample")
    val_eval = evaluation(y_val, val_probs, decision_mode, "full validation set")
    test_eval = evaluation(y_test, test_probs, decision_mode, "full test set")
    entry = {
        "name": name,
        "type": model_type,
        "decisionMode": decision_label(decision_mode),
        "decisionConfig": decision_mode if isinstance(decision_mode, dict) else {"mode": decision_mode},
        "train": compact(train_eval),
        "validation": compact(val_eval),
        "test": compact(test_eval),
        "overfitCheck": positive_gap(train_eval, val_eval),
        "tunedHyperparameters": hyperparameters,
        "probabilityCalibration": "validation-prior correction for oversampled training data" if adjustment is not None else "none",
    }
    if training_seconds is not None:
        entry["trainingSeconds"] = round(float(training_seconds), 2)
    return entry


def threshold_candidates(probabilities):
    expected_cost = probabilities @ COST
    best_nonzero_cost = expected_cost[:, 1:].min(axis=1)
    savings = expected_cost[:, 0] - best_nonzero_cost
    quantiles = np.quantile(savings, np.linspace(0, 1, DECISION_THRESHOLD_STEPS))
    return np.unique(np.concatenate(([savings.min() - 1.0, 0.0, savings.max() + 1.0], quantiles)))


def choose_decision(estimator, x_val, y_val, adjustment=None, min_accuracy=None):
    probabilities = estimator.predict_proba(x_val)
    if adjustment is not None:
        probabilities = apply_prior_adjustment(probabilities, adjustment)
    argmax_eval = evaluation(y_val, probabilities, "argmax", "full validation set")
    cost_eval = evaluation(y_val, probabilities, "expected_cost", "full validation set")

    candidates = [("argmax", argmax_eval), ("expected_cost", cost_eval)]
    for threshold in threshold_candidates(probabilities):
        decision = {"mode": "cost_margin_threshold", "threshold": float(threshold)}
        candidates.append((decision, evaluation(y_val, probabilities, decision, "full validation set")))

    operating_candidates = candidates
    minimum_accuracy = MIN_OPERATING_ACCURACY if min_accuracy is None else min_accuracy
    if minimum_accuracy > 0:
        operating_candidates = [
            item for item in candidates if item[1]["accuracy"] >= minimum_accuracy
        ] or candidates

    operating_candidates.sort(
        key=lambda item: (
            item[1]["totalCost"],
            -item[1]["macroF1"],
            -item[1]["accuracy"],
        )
    )
    return operating_candidates[0][0]


def train_models():
    dataset = json.loads(DATASET_FILE.read_text())
    _, x_train, y_train = load_split(dataset, "train")
    _, x_val, y_val = load_split(dataset, "validation")
    _, x_test, y_test = load_split(dataset, "test")

    models = []

    x_logistic, y_logistic = stratified_limit(
        x_train,
        y_train,
        LOGISTIC_CLASS0_LIMIT,
        LOGISTIC_POSITIVE_LIMIT,
    )
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=700,
            solver="saga",
            random_state=2026,
        ),
    )
    start = time.perf_counter()
    logistic.fit(x_logistic, y_logistic)
    logistic_seconds = time.perf_counter() - start
    logistic_decision = choose_decision(logistic, x_val, y_val)
    models.append(
        model_entry(
            "Logistic Regression (L2 balanced)",
            "logistic_regression",
            logistic,
            logistic_decision,
            x_logistic,
            y_logistic,
            x_val,
            y_val,
            x_test,
            y_test,
            {"C": 0.2, "classWeight": "balanced", "solver": "saga", "regularization": "L2"},
            training_seconds=logistic_seconds,
        )
    )

    if LGBMClassifier is not None:
        notebook_hyperparameters = {
            "objective": "multiclass",
            "num_class": 5,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        weighted_estimator = LGBMClassifier(**notebook_hyperparameters)
        start = time.perf_counter()
        weighted_estimator.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
        weighted_seconds = time.perf_counter() - start
        models.append(
            model_entry(
                "LightGBM balanced expected-cost",
                "lightgbm_balanced_expected_cost",
                weighted_estimator,
                "expected_cost",
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                {
                    **notebook_hyperparameters,
                    "sample_weight": "balanced class weights",
                    "decision_rule": "predict_proba multiplied by SCANIA cost matrix",
                    "feature_engineering": "project engineered 118-feature vector",
                },
                training_seconds=weighted_seconds,
            )
        )

        lightgbm_hyperparameters = {
            "objective": "multiclass",
            "num_class": 5,
            "class_weight": "balanced",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 5,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        estimator = LGBMClassifier(**lightgbm_hyperparameters)
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        lightgbm_seconds = time.perf_counter() - start

        lightgbm_expected_entry = model_entry(
            "LightGBM expected-cost",
            "lightgbm_expected_cost",
            estimator,
            "expected_cost",
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            lightgbm_hyperparameters,
            training_seconds=lightgbm_seconds,
        )
        models.append(lightgbm_expected_entry)

        accuracy_floor_decision = choose_decision(estimator, x_val, y_val, min_accuracy=0.75)
        lightgbm_accuracy_entry = model_entry(
            "LightGBM 75% accuracy floor",
            "lightgbm_accuracy_floor",
            estimator,
            accuracy_floor_decision,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            {
                **lightgbm_hyperparameters,
                "decision_tuning": "validation sweep with a minimum 75% validation accuracy constraint",
                "decision_threshold_steps": DECISION_THRESHOLD_STEPS,
            },
            training_seconds=lightgbm_seconds,
        )
        models.append(lightgbm_accuracy_entry)

        tuned_decision = choose_decision(estimator, x_val, y_val)
        lightgbm_cost_entry = model_entry(
            "LightGBM cost-sensitive",
            "lightgbm",
            estimator,
            tuned_decision,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            {
                **lightgbm_hyperparameters,
                "decision_tuning": "validation sweep over expected-cost savings margin",
                "decision_threshold_steps": DECISION_THRESHOLD_STEPS,
            },
            training_seconds=lightgbm_seconds,
        )
        models.append(lightgbm_cost_entry)
        export_lightgbm_browser_model(
            dataset,
            estimator,
            [lightgbm_expected_entry, lightgbm_accuracy_entry, lightgbm_cost_entry],
            lightgbm_hyperparameters,
            lightgbm_seconds,
        )

    models.sort(
        key=lambda entry: (
            entry["validation"]["totalCost"],
            -entry["validation"]["accuracy"],
            entry["overfitCheck"]["accuracyGap"],
        )
    )

    output = {
        "generatedAt": np.datetime64("now").astype(str),
        "sourceDataset": str(DATASET_FILE.relative_to(ROOT)),
        "fairEvaluationRule": "All exported tabular models are evaluated on the complete validation and test sets.",
        "selectionRule": "Lowest full-validation SCANIA cost; accuracy and train-validation gap are reported for context because the data is heavily imbalanced.",
        "models": models,
        "bestModel": models[0] if models else None,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUT_FILE}")
    for model in models:
        val = model["validation"]
        test = model["test"]
        print(
            f"{model['name']}: val acc {val['accuracy']}, val cost {val['totalCost']}; "
            f"test acc {test['accuracy']}, test cost {test['totalCost']}; decision {model['decisionMode']}"
        )


if __name__ == "__main__":
    train_models()
