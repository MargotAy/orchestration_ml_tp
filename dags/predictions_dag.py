"""DAG Airflow - trafic de previsions quotidien.

Seance 17 - TP Airflow (suite)
    Chaque jour a 10h, echantillonne N_PREDICTIONS lignes et les envoie en POST /predict.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def setup_project_context() -> Path:
    root = Path(os.environ.get("ML_PROJECT_ROOT", "/opt/airflow"))
    if not (root / "src").is_dir():
        root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root)
    return root


def task_send_predictions(**context) -> None:
    setup_project_context()
    import httpx

    from src.config import API_URL, RANDOM_STATE, TARGET
    from src.data import load_data

    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=N_PREDICTIONS, random_state=RANDOM_STATE)

    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, API_URL)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
