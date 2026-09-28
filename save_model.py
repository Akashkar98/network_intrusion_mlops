import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


dataset = pd.read_csv("clean_dataset.csv")

X = dataset.drop("Label", axis=1)
y = dataset["Label"]


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

categorical_features = ["Protocol"]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("Training final pipeline...")

pipeline.fit(X_train, y_train)
print("Training completed.")

os.makedirs("models", exist_ok=True)

model_path = "models/intrusion_detection_pipeline.joblib"

joblib.dump(
    pipeline,
    model_path
)

print(f"Model saved successfully: {model_path}")