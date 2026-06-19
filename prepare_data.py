"""Script de preparation des donnees - S0.
Convertit HeartDisease Yes/No en 1/0 et ecrit le CSV final dans data/.
"""

from __future__ import annotations

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/heart_2020_cleaned.csv")
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    assert df["HeartDisease"].isin([0, 1]).all(), (
        "Erreur : la cible contient des valeurs inattendues"
    )
    print(f"Distribution de la cible :\n{df['HeartDisease'].value_counts()}")
    print(f"Lignes : {len(df)} | Colonnes : {df.columns.tolist()}")
    df.to_csv("data/heart_2020_prepared.csv", index=False)
    print("Fichier écrit : data/heart_2020_prepared.csv")


if __name__ == "__main__":
    main()