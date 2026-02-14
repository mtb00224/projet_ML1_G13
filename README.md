# 📊 Customer Churn Prediction -- Machine Learning Project

## 📌 Description

Ce projet implémente un pipeline complet de Machine Learning pour
prédire le churn (résiliation) des clients bancaires.

Objectifs : - Prétraiter les données (encodage + standardisation) -
Entraîner un modèle de classification - Sauvegarder le modèle entraîné -
Effectuer des prédictions sur de nouvelles données - Structurer le
projet comme un projet ML professionnel

------------------------------------------------------------------------

## Structure du projet

    Groupe13/
    |__ api
    |______ main.py            # pour exposer le model via une api
    │
    ├── datasets/              # Données d'entraînement
    │   └── train.csv
    |___ model
    |_______ model.pkl         # le model entrainé et sauveagrder
    │
    ├── notebooks/             # Analyses exploratoires (EDA)
    │
    ├── scripts/
    │   ├── __init__.py
    │   ├── preprocessing.py   # Création du pipeline de preprocessing
    │   ├── train.py           # Entraînement du modèle
    │   ├── predict.py         # Chargement et prédiction
    │
    ├── tests/
    │   ├── __init__.py
    │   └── tests.py           # Simulation de tests sur le jeu de données test.csv
    │
    └── README.md

------------------------------------------------------------------------

## Installation

Cloner le projet :

``` bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

Créer un environnement virtuel :

``` bash
python3 -m venv .venv
source venv/bin/activate (sur windows : .venv\Scripts\activate)
```

Installer les dépendances :

``` bash
pip install -r requirements.txt

```

------------------------------------------------------------------------

Creer un dossier nommé : "model" à la racine du projet

------------------------------------------------------------------------
Lancer les tests

``` bash
python -m tests.tests

```

------------------------------------------------------------------------

Demarrer le server pour lancer l'api :
``` bash
uvicorn api.main:app --reload

```
ensuite acceder à la page : http://127.0.0.1:8000/docs pour test l'endpoint
------------------------------------------------------------------------

## Pipeline Machine Learning

### 1️⃣ Preprocessing

-   Encodage des variables catégorielles
-   Standardisation des variables numériques
-   Utilisation d'un ColumnTransformer

### 2️Entraînement

-   Séparation train/test
-   Création d'un Pipeline
-   Entraînement du modèle (XGBoost)
-   Sauvegarde avec joblib

### 3️⃣ Prédiction

-   Chargement du modèle sauvegardé
-   Prédiction sur nouvelles données
-   Retour : classe prédite + probabilité

------------------------------------------------------------------------

## 📦 Technologies utilisées

-   Python 3
-   pandas
-   scikit-learn
-   xgboost
-   joblib

------------------------------------------------------------------------

## 👨‍💻 Auteur

Projet réalisé par le groupe 13 de la classe Master 1 IA / G1 -- Machine Learning.
