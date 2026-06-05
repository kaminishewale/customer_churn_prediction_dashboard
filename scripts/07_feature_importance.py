import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv"
)

# Drop columns
drop_cols = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Label",
    "Churn Reason"
]

df = df.drop(columns=drop_cols)

# Encode categorical variables
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10))

# Plot
plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:10],
    importance["Importance"][:10]
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.show()