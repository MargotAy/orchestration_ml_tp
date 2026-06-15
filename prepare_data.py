"""Script de preparation des donnees - S0.
Convertit HeartDisease Yes/No en 1/0 et ecrit le CSV final dans data/.
"""
import pandas as pd

# Lecture du CSV brut
df = pd.read_csv("data/heart_2020_cleaned.csv")

# Mapping de la cible Yes/No -> 1/0
df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})

# Vérification
assert df["HeartDisease"].isin([0, 1]).all(), "Erreur : la cible contient des valeurs inattendues"
print(f"Distribution de la cible :\n{df['HeartDisease'].value_counts()}")
print(f"Lignes : {len(df)} | Colonnes : {df.columns.tolist()}")

# Ecriture du CSV final
df.to_csv("data/heart_2020_prepared.csv", index=False)
print("Fichier écrit : data/heart_2020_prepared.csv")