from scripts.train import train_model
from scripts.predict import predict
import pandas as pd

# Entraîner
train_model("datasets/train.csv")

# Tester la prédiction
df_test = pd.read_csv("datasets/test.csv").drop(columns=["id", "Surname", "CustomerId"]).head(2500)  # Limiter à 2500 lignes pour les tests

resultats = []

for _, row in df_test.iterrows():
    resultat = predict(row.to_dict())
    resultats.append(resultat)

#Convertir en DataFrame
results_df = pd.DataFrame(resultats)

# Fusionner avec les données de test
final_df = pd.concat([df_test.reset_index(drop=True), results_df], axis=1)

final_df.to_csv("tests/test_result.csv", index=False)


print("✅ Fichier de test généré avec succès.")