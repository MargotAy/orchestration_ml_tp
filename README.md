# orchestration_ml_tp

# Choix de la problématique et dataset 

Prédire si un adulte américain est atteint d'une maladie cardiaque (1) ou non (0) à partir d'indicateurs de santé personnels (IMC, tabagisme, activité physique, antécédents…), afin d'identifier les profils à risque élevé.

Le dataset a été trouvé sur Kaggle : 
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data

Le csv : heart_2020_cleaned.csv 
Il se trouve dans le dossier 'data' 

Les données sont déjà "propres" cependant on pourra y faire quelques améliorations/nettoyage dans le fichier features.py

# Dossier /src

## config.py
Ensuite, nous créons le premier fichier du dossier src qui est le 'config.py'. 
On déclare des variables d'environnement (on aurait pu faire un fichier .env également)

## data.py et features.py 
On fait un nettoyage de données. Elles sont relativement propres mais elles sont nombreuses (300 000 lignes). De plus la colonne cible est déséquilibrée (9% de 1 et 89% de 0), on garde les lignes à 1, donc environ 27 000 et on en garde autant avec la valeur 0 mais choisie aléatoirement
Si le dataset n'était pas préalablement nettoyé, c'est dans ces fichiers que nous aurions du faire des modificaitons/transformations.

- `data.py` : chargement, doublons, undersampling
- `features.py` : preprocessing sklearn (imputation, scaling, one-hot) — appliqué en mémoire à l'entraînement, pas dans le CSV

## train.py
On implémente un modèle de Régression Logistique en baseline pour répondre à la question : 
"Quelle est la probabilité d'un patient ait une maladie cardiaque en fonction de ses facteurs ?" 
On utilise la régression logistique car nous voulons une solution binaire (est-ce que le patient pourrait avoir une maladie cardiaque *oui* ou *non*)

```bash
uv run python -m src.train
```

## train_models.py
Permet de comparer plusieurs modèles avec des paramètres différents
on visualise les résultats sur MLFLOW 

Compare Random Forest, XGBoost et LightGBM avec **GridSearchCV**. Le meilleur modèle est enregistré dans le Model Registry et sauvegardé dans `models/model.joblib`.

```bash
uv run python -m src.train_models --cv 3 --scoring roc_auc
uv run python -m src.train_models --no-mlflow
```

## tracking.py 
Permet de rcentraliser le tracking mlflow pour le réutiliser dans différents fichiers comme train.py et train_models.py 

Fonctions principales : `setup_experiment()` et `log_dataset()` (traçabilité des données dans MLflow).

## evaluation.py
Outils d'évaluation partagés (graphiques loggés comme artefacts MLflow). Contient notamment `log_shap_summary()` utilisé par `train_models.py` et `train_optuna.py`.

## train_optuna.py 
Ce modèle compare Random Forest, XGBoost et LightGBM, teste plein de combinaisons d'hyperparamètres, garde le meilleur, et le sauvegarde dans models/model.joblib.
La différence avec train_models.py c'est qu'on utilisait GridSearchCV.
Optuna fait des essais ciblés pour converger plus rapdiement vers les bons paramètres tandis que GridSearch teste TOUTES les combinaisons

Commande pour lancer le test : 
```bash
uv run python -m src.train_optuna --n-trials 30
```

## evaluate.py 
Permet de faire une évaluation "standardisée" du modèle enregistré dans mlflow registry
Il récupère le modèle, l'évalue automatiquement sur un jeu de test et valide les seuils

Utilise `mlflow.models.evaluate` puis `mlflow.validate_evaluation_results` : si `roc_auc` ou `f1_score` est sous le seuil configuré dans `config.py`, le modèle est rejeté.

```bash
uv run python -m src.evaluate
uv run python -m src.evaluate --model-uri models:/heart-disease-model/1
uv run python -m src.evaluate --no-validate
```
