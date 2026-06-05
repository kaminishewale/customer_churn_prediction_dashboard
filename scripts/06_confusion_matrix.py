import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load data
df = pd.read_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv"
)

# Drop unnecessary columns
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

# Split data
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

print(cm)

# Plot
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

plt.show()