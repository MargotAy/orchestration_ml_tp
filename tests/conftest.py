"""Fixtures partagees pour les tests API."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evite de charger model.joblib (plateforme / chemin) pendant les tests API."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    monkeypatch.setattr("src.api.joblib.load", MagicMock(return_value=model))
