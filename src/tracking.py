"""Configuration partagee du suivi MLflow."""

from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from src.config import DATA_PATH, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, TARGET

logger = logging.getLogger(__name__)

# Types autorises pour la serialisation skops (MLflow 3+) des pipelines sklearn + XGB/LGBM.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "numpy.ndarray",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.preprocessing._encoders.OneHotEncoder",
    "sklearn.impute._base.SimpleImputer",
    "sklearn.preprocessing._data.StandardScaler",
]


def setup_experiment() -> None:
    """Configure MLflow tracking URI and experiment metadata."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("Suivi MLflow : %s (experience: %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)

    # Optional metadata if defined in config.
    from src import config as cfg

    description = getattr(cfg, "MLFLOW_EXPERIMENT_DESCRIPTION", None)
    tags = getattr(cfg, "MLFLOW_EXPERIMENT_TAGS", {})
    client = mlflow.MlflowClient()
    if description:
        client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", description)
    for key, value in tags.items():
        client.set_experiment_tag(experiment.experiment_id, key, str(value))


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Log a pandas dataset lineage entry in the current MLflow run."""
    dataset = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name=name)  # type: ignore[attr-defined]
    mlflow.log_input(dataset, context=context)
