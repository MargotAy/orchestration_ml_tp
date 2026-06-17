"""Tests sur le pipeline d'entrainement baseline."""

from __future__ import annotations

from src.data import load_data, split
from src.train import build_model


def test_build_model_fits_on_training_sample() -> None:
    df = load_data()
    x_train, _, y_train, _ = split(df)
    model = build_model()
    model.fit(x_train, y_train)
    assert hasattr(model, "predict_proba")
