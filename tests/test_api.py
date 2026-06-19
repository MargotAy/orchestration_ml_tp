"""Tests de l'API FastAPI (seance 12 / client seance 15)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app

VALID_PAYLOAD = {
    "BMI": 27.4,
    "PhysicalHealth": 2,
    "MentalHealth": 0,
    "SleepTime": 7,
    "Smoking": "No",
    "AlcoholDrinking": "No",
    "Stroke": "No",
    "DiffWalking": "No",
    "Sex": "Female",
    "AgeCategory": "55-69",
    "Race": "White",
    "Diabetic": "No",
    "PhysicalActivity": "Yes",
    "GenHealth": "Very good",
    "Asthma": "No",
    "KidneyDisease": "No",
    "SkinCancer": "No",
}


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_prediction() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_age_category() -> None:
    invalid_payload = {**VALID_PAYLOAD, "AgeCategory": "55-59"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_model_info_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    assert "version" in response.json()


def test_predictions_endpoint_logs_predict_calls() -> None:
    from src.api import prediction_log

    prediction_log.clear()
    with TestClient(app) as client:
        empty = client.get("/predictions")
        assert empty.status_code == 200
        assert empty.json() == []

        predict = client.post("/predict", json=VALID_PAYLOAD)
        assert predict.status_code == 200

        history = client.get("/predictions")
        assert history.status_code == 200
        rows = history.json()
        assert len(rows) == 1
        assert rows[0]["prediction"] == predict.json()["prediction"]
        assert rows[0]["probability"] == predict.json()["probability"]
        assert rows[0]["BMI"] == VALID_PAYLOAD["BMI"]
        assert "timestamp" in rows[0]
