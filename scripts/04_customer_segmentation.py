import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv"
)

# Select important numerical features
X = df[[
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV"
]]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create KMeans model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Check cluster counts
print(df["Cluster"].value_counts())

# Scatter Plot
plt.figure(figsize=(8,6))

plt.scatter(
    df["Monthly Charges"],
    df["CLTV"],
    c=df["Cluster"]
)

plt.xlabel("Monthly Charges")
plt.ylabel("CLTV")
plt.title("Customer Segmentation")

plt.show()