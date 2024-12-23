"""
Classical ML models for the Quora Question Pair Similarity project.

Trains several models on the hand-crafted feature matrix (see features.py +
embeddings.py) and reports a full set of metrics (accuracy, precision,
recall, F1, ROC-AUC) rather than just accuracy, since the dataset is
imbalanced (real Quora data is ~63% non-duplicate / 37% duplicate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: dict = field(default_factory=dict)


def get_model_zoo(random_state: int = 42) -> dict:
    """Candidate models. Kept intentionally diverse: a linear baseline
    (Logistic Regression), two tree ensembles (RF, GBM) and XGBoost, which
    is usually the strongest of the four on this kind of tabular feature
    set (this mirrors what actually wins on this problem in practice)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }


def train_and_evaluate_all(X_train, y_train, X_test, y_test, random_state: int = 42):
    """Fit every model in the zoo, score it on the held-out test set, and
    return results sorted best-first by F1 score (a more meaningful metric
    than accuracy on an imbalanced binary task like this one)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name, model in get_model_zoo(random_state).items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        results.append(ModelResult(name=name, model=model, metrics=metrics))

    results.sort(key=lambda r: r.metrics["f1"], reverse=True)
    return results, scaler


def save_artifact(obj, path: str) -> None:
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)
