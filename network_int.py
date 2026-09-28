import pandas as pd
from sklearn.model_selection import train_test_split
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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


categorical_features = [
    "Protocol"
]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
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
from sklearn.metrics import classification_report, confusion_matrix

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train_processed, y_train)

y_val_pred = dummy_model.predict(X_val_processed)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

y_val_pred = logistic_model.predict(
    X_val_processed
)

# print("Confusion Matrix:")
# print(confusion_matrix(y_val, y_val_pred))

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_val,
#         y_val_pred,
#         zero_division=0
#     )
# )

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_processed,
    y_train
)

y_val_pred_rf = rf_model.predict(
    X_val_processed
)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_rf))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_val_pred_rf,
        zero_division=0
    )
)
