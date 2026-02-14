from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle UNE seule fois
model = joblib.load("model/model.pkl")

app = FastAPI(
    title="Churn Prediction API", 
    description="API pour prédire le churn des clients de la banque Furneo", 
    version="1.0"
)

class Customer(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1][0]  # Probabilité de churn (classe 1)
    return {"prediction": int(prediction[0]), "probability": float(probability)}
    