import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv"
)

# =========================
# DROP UNNECESSARY COLUMNS
# =========================

drop_cols = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Label",
    "Churn Reason"
]

existing_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_cols)

# =========================
# ENCODE CATEGORICAL DATA
# =========================

encoders = {}

for col in df.columns:
    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

# =========================
# FEATURES & TARGET
# =========================

selected_features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
    "Payment Method",
    "Internet Service",
    "Tech Support",
    "Online Security",
    "Paperless Billing",
    "Senior Citizen"
]

available_features = [
    col for col in selected_features
    if col in df.columns
]

X = df[available_features]
y = df["Churn Value"]

feature_names = available_features
y = df["Churn Value"]

feature_names = list(X.columns)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

best_model = None
best_accuracy = 0

# =========================
# MODEL TRAINING
# =========================

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# =========================
# SAVE RANDOM FOREST MODEL
# =========================

os.makedirs("models", exist_ok=True)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# Save Model
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save Encoders
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Save Feature Names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# =========================
# RANDOM FOREST EVALUATION
# =========================

rf_pred = rf_model.predict(X_test)

print("\n")
print("=" * 60)
print("FINAL RANDOM FOREST RESULTS")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1_score(y_test, rf_pred) * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\n✅ random_forest_model.pkl saved")
print("✅ label_encoders.pkl saved")
print("✅ feature_names.pkl saved")
print("✅ Files stored inside models/ folder")