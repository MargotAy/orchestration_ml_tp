"""Entrainement et comparaison de plusieurs modeles de classification.

Compare trois familles de modeles (Random Forest, XGBoost, LightGBM),
chacune optimisee par GridSearchCV, et persiste le meilleur dans models/model.joblib.

Lancement :
    PYTHONPATH=. python -m src.train_models
    PYTHONPATH=. python -m src.train_models --cv 3 --scoring roc_auc
    PYTHONPATH=. python -m src.train_models --no-mlflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from src.data import load_data, split
from src.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    precision: float
    recall: float
    preds: np.ndarray


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, 20],
                "clf__min_samples_leaf": [1, 5],
            }
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            }
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            }
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("clf", estimator),
    ])


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=f1_score(y_test, preds),
        roc_auc=roc_auc_score(y_test, proba),
        precision=precision_score(y_test, preds),
        recall=recall_score(y_test, preds),
        preds=preds,
    )


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        # Hyperparamètres
        mlflow.log_params(result.best_params)

        # Métriques — precision et recall ajoutés car contexte médical
        mlflow.log_metrics({
            f"cv_{scoring}": result.cv_score,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
            "precision": result.precision,
            "recall": result.recall,  # crucial : ne pas rater un malade
        })

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = classification_report(y_test, result.preds, output_dict=True)
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = classification_report(y_test, result.preds)
        mlflow.log_text(report_text, "classification_report.txt")

        # Sauvegarde du modèle
        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )


def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(
            "Suivi MLflow : %s (experience: %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
        )

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best.name, best.roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()