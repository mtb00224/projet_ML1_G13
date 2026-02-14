import joblib
import pandas as pd


def load_model(model_path="model/model.pkl"):
    return joblib.load(model_path)


def predict(data, model_path="model/model.pkl"):
    """
    data : dictionnaire ou DataFrame
    """

    model = load_model(model_path)

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }


"""
Le modèle final est sauvegardé au format pickle via joblib afin de pouvoir être rechargé en production 
sans nécessiter un réentraînement. Le pipeline complet est sérialisé pour garantir la cohérence 
entre preprocessing et prédiction.
"""