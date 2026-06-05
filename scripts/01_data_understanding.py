import pandas as pd

df = pd.read_excel(r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\Telco_customer_churn.xlsx"
)

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())