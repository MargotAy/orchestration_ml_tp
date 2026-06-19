"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline : preparation des donnees -> entrainement -> controle qualite.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def setup_project_context() -> Path:
    """Racine du projet montee dans le conteneur Airflow."""
    root = Path(os.environ.get("ML_PROJECT_ROOT", "/opt/airflow"))
    if not (root / "src").is_dir():
        root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root)
    return root


def task_prepare_data(**context) -> None:
    setup_project_context()
    from prepare_data import main as prepare

    prepare()
    logger.info("Donnees preparees dans data/")


def task_train(**context) -> None:
    setup_project_context()
    from src.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine : f1=%.3f", metrics["f1"])


def task_check_quality(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None:
        raise ValueError("Metrique f1 introuvable dans XCom (tache train)")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"f1={f1:.3f} < seuil {QUALITY_THRESHOLD}")
    logger.info("Qualite OK : f1=%.3f", f1)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
