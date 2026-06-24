"""Fonctions de modélisation pour le dataset IEEE Fraud Detection."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

