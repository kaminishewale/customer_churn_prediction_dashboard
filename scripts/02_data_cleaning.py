import pandas as pd

df = pd.read_excel(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\Telco_customer_churn.xlsx"
)

print("Duplicate Records:", df.duplicated().sum())

# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Check missing values again
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing Total Charges if any
df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].mean()
)

# Save cleaned dataset
df.to_csv(
    r"C:\Users\Kamini Shewale\OneDrive\Desktop\customer_churn_project\dataset\cleaned_churn.csv",
    index=False
)

print("\nCleaning Completed Successfully!")