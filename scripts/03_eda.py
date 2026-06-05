import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv"
)

plt.figure(figsize=(6,4))

sns.countplot(
    x="Churn Label",
    data=df
)

plt.title("Customer Churn Distribution")
plt.show()

#second visualization:Contact vs Churn
plt.figure(figsize=(8,5))

sns.countplot(
    x="Contract",
    hue="Churn Label",
    data=df
)

plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)

plt.show()

#Third Visualization: Monthly Charges vs Churn
plt.figure(figsize=(8,5))

sns.boxplot(
    x="Churn Label",
    y="Monthly Charges",
    data=df
)

plt.title("Monthly Charges vs Churn")

plt.show()

#Fourth Visualization: Payment Method vs Churn
plt.figure(figsize=(10,5))

sns.countplot(
    x="Payment Method",
    hue="Churn Label",
    data=df
)

plt.xticks(rotation=45)

plt.title("Payment Method vs Churn")

plt.show()