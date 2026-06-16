"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    Lancement : `uv run uvicorn src.api:app --reload`
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict[str, object] = {}


class Features(BaseModel):
    """Schema d'entree aligne avec NUMERIC_FEATURES + CATEGORICAL_FEATURES."""

    BMI: float = Field(..., ge=0, le=100)
    PhysicalHealth: float = Field(..., ge=0, le=30)
    MentalHealth: float = Field(..., ge=0, le=30)
    SleepTime: float = Field(..., ge=0, le=24)

    Smoking: Literal["Yes", "No"]
    AlcoholDrinking: Literal["Yes", "No"]
    Stroke: Literal["Yes", "No"]
    DiffWalking: Literal["Yes", "No"]
    Sex: Literal["Male", "Female"]
    AgeCategory: Literal["18-39", "40-54", "55-69", "70+"]
    Race: Literal[
        "White",
        "Black",
        "Asian",
        "American Indian/Alaskan Native",
        "Other",
        "Hispanic",
    ]
    Diabetic: Literal["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"]
    PhysicalActivity: Literal["Yes", "No"]
    GenHealth: Literal["Poor", "Fair", "Good", "Very good", "Excellent"]
    Asthma: Literal["Yes", "No"]
    KidneyDisease: Literal["Yes", "No"]
    SkinCancer: Literal["Yes", "No"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(MODEL_DIR / "model.joblib")
    logger.info("Modele charge depuis %s", MODEL_DIR / "model.joblib")
    yield
    ml.clear()


app = FastAPI(title="Classification API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])  # type: ignore[attr-defined]
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
