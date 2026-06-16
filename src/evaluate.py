"""Evaluation automatisee et validation du modele."""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from src.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.tracking import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Return URI for latest registered version of MODEL_NAME."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement (train_models/train_optuna)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Build quality gate thresholds for MLflow validation."""
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluate a registry model and optionally enforce quality thresholds."""
    df = load_data()
    _, x_test, _, y_test = split(df)
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    setup_experiment()
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        dataset = mlflow.data.from_pandas(
            eval_df,
            source=str(DATA_PATH),
            targets=TARGET,
            name="eval",
        )
        mlflow.log_input(dataset, context="evaluation")

        result = mlflow.models.evaluate(
            model=model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )
        logger.info(
            "f1_score=%.3f roc_auc=%.3f",
            result.metrics["f1_score"],
            result.metrics["roc_auc"],
        )

        if validate:
            mlflow.validate_evaluation_results(build_thresholds(), result)

        return result


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="URI du modele a evaluer (defaut: derniere version de MODEL_NAME)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Evalue sans appliquer la porte qualite (seuils)",
    )
    args = parser.parse_args()

    model_uri = args.model_uri or None
    try:
        evaluate_model(model_uri=model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
