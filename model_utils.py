"""Fonctions de modélisation pour le dataset IEEE Fraud Detection."""

import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_curve, auc, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prépare X et y pour la modélisation.

    - Garde uniquement les colonnes numériques
    - Remplace les NaN par la médiane
    """
    target = df["isFraud"]
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["isFraud", "TransactionID"])
    numeric = numeric.fillna(numeric.median())
    return numeric, target


def train_naive_logistic(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Entraîne une régression logistique naïve et retourne le modèle, les données de test et l'accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    model = LogisticRegression(max_iter=1000, solver="saga")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, acc


def compute_fraud_metrics(model, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcule le recall sur la classe fraude et l'AUPRC."""
    recall = recall_score(y_test, y_pred)

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(rec_curve, precision)

    return {"recall": recall, "auprc": auprc, "precision_curve": precision, "recall_curve": rec_curve}


def evaluate_with_timeseries_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Évalue une régression logistique avec TimeSeriesSplit.

    Les données doivent être triées par ordre temporel AVANT l'appel.
    Retourne un DataFrame avec accuracy, recall et AUPRC par fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, solver="saga")),
    ])

    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(rec_curve, prec_curve)

        results.append({"fold": fold, "accuracy": acc, "recall": rec, "auprc": auprc})

    return pd.DataFrame(results)


def evaluate_xgb(X: pd.DataFrame, y: pd.Series, label: str, n_splits: int = 5) -> pd.DataFrame:
    """Évalue un XGBoost avec TimeSeriesSplit et scale_pos_weight automatique.

    Les données doivent être triées par ordre temporel AVANT l'appel.
    Retourne un DataFrame avec AUPRC, recall, F1 et temps par fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=ratio, eval_metric="aucpr",
            random_state=42, verbosity=0,
        )
        start = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - start

        y_proba = model.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        results.append({
            "fold": fold,
            "auprc": average_precision_score(y_te, y_proba),
            "recall": recall_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred),
            "temps (s)": round(train_time, 2),
        })

    return pd.DataFrame(results)

