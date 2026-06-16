"""Chargement et decoupage des donnees."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Suppression des doublons
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Doublons supprimés : {n_before - len(df)}")

    # Vérification des valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        print(f"Valeurs manquantes détectées :\n{missing[missing > 0]}")
        df = df.dropna()
    else:
        print("Aucune valeur manquante")

    # Modification de la colonne AgeCategory on réduit le nombre de groupes pour en avoir que 4
    AGE_GROUPS = {
        "18-24": "18-39",
        "25-29": "18-39",
        "30-34": "18-39",
        "35-39": "18-39",
        "40-44": "40-54",
        "45-49": "40-54",
        "50-54": "40-54",
        "55-59": "55-69",
        "60-64": "55-69",
        "65-69": "55-69",
        "70-74": "70+",
        "75-79": "70+",
        "80 or older": "70+",
    }
    df["AgeCategory"] = df["AgeCategory"].map(AGE_GROUPS)

    # Undersampling : équilibrage des classes
    df_minority = df[df[TARGET] == 1]  # ~27 000 malades
    df_majority = df[df[TARGET] == 0].sample(  # on garde ~27 000 sains
        n=len(df_minority), random_state=RANDOM_STATE
    )

    df_balanced = pd.concat([df_minority, df_majority])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Dataset final : {len(df_balanced)} lignes")
    print(f"Distribution :\n{df_balanced[TARGET].value_counts()}")

    return df_balanced


def split(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
