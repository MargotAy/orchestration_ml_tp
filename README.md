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


