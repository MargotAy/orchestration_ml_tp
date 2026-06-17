"""Tests unitaires sur le chargement et le decoupage des donnees."""

from __future__ import annotations

from src.config import TARGET
from src.data import load_data, split


def test_load_data_returns_balanced_binary_target() -> None:
    df = load_data()
    assert TARGET in df.columns
    assert set(df[TARGET].unique()) == {0, 1}
    assert len(df[TARGET].value_counts()) == 2
    assert df[TARGET].value_counts().nunique() == 1


def test_split_preserves_feature_columns() -> None:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    assert TARGET not in x_train.columns
    assert TARGET not in x_test.columns
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
