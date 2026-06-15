"""Entrainement du modele baseline - Classification maladie cardiaque."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data import load_data, split
from src.features import build_preprocessor


def build_model() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train() -> None:
    # Chargement et split
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # Entrainement
    model = build_model()
    model.fit(x_train, y_train)

    # Evaluation
    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"f1={f1:.3f}  roc_auc={roc_auc:.3f}")

    # Sauvegarde du modele
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("Modèle sauvegardé dans models/model.joblib")


if __name__ == "__main__":
    train()