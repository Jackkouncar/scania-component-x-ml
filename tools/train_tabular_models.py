import json
import math
import os
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
CLASSES = np.array([0, 1, 2, 3, 4])
MIN_OPERATING_ACCURACY = float(os.environ.get("MIN_OPERATING_ACCURACY", "0.75"))
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


def decision_label(decision):
    if isinstance(decision, dict):
        if decision["mode"] == "cost_margin_threshold":
            return f"cost_margin_threshold >= {decision['threshold']:.4f}"
        return decision["mode"]
    return decision


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
):
    train_probs = estimator.predict_proba(x_train)
    val_probs = estimator.predict_proba(x_val)
    test_probs = estimator.predict_proba(x_test)
    if adjustment is not None:
        train_probs = apply_prior_adjustment(train_probs, adjustment)
        val_probs = apply_prior_adjustment(val_probs, adjustment)
        test_probs = apply_prior_adjustment(test_probs, adjustment)
    train_eval = evaluation(y_train, train_probs, decision_mode, "full training sample")
    val_eval = evaluation(y_val, val_probs, decision_mode, "full validation set")
    test_eval = evaluation(y_test, test_probs, decision_mode, "full test set")
    return {
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


def threshold_candidates(probabilities):
    expected_cost = probabilities @ COST
    best_nonzero_cost = expected_cost[:, 1:].min(axis=1)
    savings = expected_cost[:, 0] - best_nonzero_cost
    quantiles = np.quantile(savings, np.linspace(0, 1, 501))
    return np.unique(np.concatenate(([savings.min() - 1.0, 0.0, savings.max() + 1.0], quantiles)))


def choose_decision(estimator, x_val, y_val, adjustment=None):
    probabilities = estimator.predict_proba(x_val)
    if adjustment is not None:
        probabilities = apply_prior_adjustment(probabilities, adjustment)
    argmax_eval = evaluation(y_val, probabilities, "argmax", "full validation set")
    cost_eval = evaluation(y_val, probabilities, "expected_cost", "full validation set")

    candidates = [("argmax", argmax_eval), ("expected_cost", cost_eval)]
    for threshold in threshold_candidates(probabilities):
        decision = {"mode": "cost_margin_threshold", "threshold": float(threshold)}
        candidates.append((decision, evaluation(y_val, probabilities, decision, "full validation set")))

    operating_candidates = [
        item for item in candidates if item[1]["accuracy"] >= MIN_OPERATING_ACCURACY
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
    adjustment = prior_adjustment(y_train, y_val)

    models = []

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
    logistic.fit(x_train, y_train)
    logistic_decision = choose_decision(logistic, x_val, y_val, adjustment)
    models.append(
        model_entry(
            "Logistic Regression (L2 balanced)",
            "logistic_regression",
            logistic,
            logistic_decision,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            {"C": 0.2, "classWeight": "balanced", "solver": "saga", "regularization": "L2"},
            adjustment,
        )
    )

    if LGBMClassifier is not None:
        lightgbm_configs = [
            {
                "name": "LightGBM regularized",
                "class_weight": None,
                "n_estimators": 360,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 90,
                "subsample": 0.82,
                "colsample_bytree": 0.82,
                "reg_alpha": 0.15,
                "reg_lambda": 4.0,
            },
            {
                "name": "LightGBM mild cost regularized",
                "class_weight": {0: 1.0, 1: 1.3, 2: 1.8, 3: 2.4, 4: 3.2},
                "n_estimators": 380,
                "learning_rate": 0.03,
                "num_leaves": 28,
                "max_depth": 6,
                "min_child_samples": 110,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.25,
                "reg_lambda": 5.0,
            },
            {
                "name": "LightGBM balanced regularized",
                "class_weight": "balanced",
                "n_estimators": 420,
                "learning_rate": 0.03,
                "num_leaves": 24,
                "max_depth": 6,
                "min_child_samples": 120,
                "subsample": 0.78,
                "colsample_bytree": 0.78,
                "reg_alpha": 0.35,
                "reg_lambda": 6.0,
            },
        ]

        lightgbm_entries = []
        for config in lightgbm_configs:
            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=5,
                random_state=2026,
                n_jobs=-1,
                verbosity=-1,
                **{key: value for key, value in config.items() if key != "name"},
            )
            estimator.fit(x_train, y_train)
            decision_mode = choose_decision(estimator, x_val, y_val, adjustment)
            lightgbm_entries.append(
                model_entry(
                    config["name"],
                    "lightgbm",
                    estimator,
                    decision_mode,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    {key: value for key, value in config.items() if key != "name"},
                    adjustment,
                )
            )

        lightgbm_entries.sort(
            key=lambda entry: (
                entry["validation"]["totalCost"],
                -entry["validation"]["accuracy"],
                entry["overfitCheck"]["accuracyGap"],
            )
        )
        models.append(lightgbm_entries[0])

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
        "selectionRule": f"Lowest full-validation SCANIA cost among operating points with validation accuracy >= {MIN_OPERATING_ACCURACY:.2f}; accuracy and train-validation gap are reported for context.",
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
