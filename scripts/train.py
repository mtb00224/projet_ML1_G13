import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from scripts.preprocessing import create_preprocessor


def train_model(data_path, model_path="model/model.pkl"):

    # Chargement des données
    df = pd.read_csv(data_path)

    X = df.drop(["Exited", "id", "Surname", "CustomerId"], axis=1)
    y = df["Exited"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = create_preprocessor(X_train)

    # Modèle
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # Pipeline complet
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Sauvegarde
    joblib.dump(pipeline, model_path)

    print("✅ Modèle entraîné et sauvegardé avec succès.")


if __name__ == "__main__":
    train_model("datasets/train.csv")
