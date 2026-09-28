import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


mlflow.set_experiment("Network_Intrusion_Detection")

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

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# with mlflow.start_run(run_name="Dummy_Classifier"):

#     dummy_model = DummyClassifier(
#         strategy="most_frequent"
#     )

#     dummy_model.fit(
#         X_train_processed,
#         y_train
#     )

#     y_val_pred = dummy_model.predict(
#         X_val_processed
#     )

#     accuracy = accuracy_score(y_val, y_val_pred)

#     precision = precision_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     recall = recall_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     f1 = f1_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     mlflow.log_param(
#         "strategy",
#         "most_frequent"
#     )

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1_score", f1)


#     print("\nDummy Classifier")
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1:", f1)

# from sklearn.linear_model import LogisticRegression

# with mlflow.start_run(run_name="Logistic_Regression"):

#     logistic_model = LogisticRegression(
#         max_iter=1000,
#         random_state=42
#     )

#     logistic_model.fit(
#         X_train_processed,
#         y_train
#     )

#     y_val_pred = logistic_model.predict(
#         X_val_processed
#     )

#     accuracy = accuracy_score(y_val, y_val_pred)

#     precision = precision_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     recall = recall_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     f1 = f1_score(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )

#     mlflow.log_param("max_iter", 1000)
#     mlflow.log_param("random_state", 42)

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1_score", f1)

#     print("\nLogistic Regression")
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1:", f1)

from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run(run_name="Random_Forest"):

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_processed,
        y_train
    )

    y_val_pred = rf_model.predict(
        X_val_processed
    )

    accuracy = accuracy_score(y_val, y_val_pred)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_param("random_state", 42)
#     mlflow.log_param("n_jobs", -1)

#     # Metrics
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1_score", f1)

#     print("\nRandom Forest")
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1:", f1)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Final evaluation on untouched test set
y_test_pred = rf_model.predict(X_test_processed)


print("Accuracy:",
      accuracy_score(y_test, y_test_pred))

print("Precision:",
      precision_score(y_test, y_test_pred))

print("Recall:",
      recall_score(y_test, y_test_pred))

print("F1:",
      f1_score(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))