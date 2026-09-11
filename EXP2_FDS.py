import numpy as np
import pandas as pd

# =====================================================================
# TASK 1: Read dataset related to chosen AI use case
# =====================================================================
# AI Use Case: Customer Churn Prediction for Telecom/SaaS
# (Reading CSV; if file doesn't exist, we create dummy data for testing)
file_path = "customer_churn_ai_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully from file.")
except FileNotFoundError:
    print("File not found. Creating sample AI Use Case Dataset...")
    data = {
        "CustomerID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        "ContractType": ["Month-to-Month", "One-Year", "Month-to-Month", "Two-Year", "Month-to-Month", 
                        "One-Year", "Month-to-Month", "Two-Year", "Month-to-Month", "One-Year", "Two-Year", "Month-to-Month"],
        "Tenure_Months": [2, 24, 1, 36, 5, np.nan, 12, 48, 3, 18, 60, 8],
        "MonthlyCharges": [70.5, 55.0, 85.2, np.nan, 95.0, 40.0, 65.5, 110.0, 80.0, 50.0, 105.0, np.nan],
        "TotalCharges": [141.0, 1320.0, 85.2, 4100.0, 475.0, 960.0, np.nan, 5280.0, 240.0, 900.0, 6300.0, 640.0],
        "AI_Risk_Score": [0.88, 0.21, 0.95, 0.05, 0.78, 0.35, 0.62, 0.02, 0.81, 0.40, 0.01, 0.72],
        "Churn": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)

print("\n--- Initial Dataset Preview ---")
print(df.head())


# =====================================================================
# TASK 2 & 3: Get Information and Description of Dataset
# =====================================================================
print("\n" + "="*60)
print("TASK 2 & 3: DATASET INFO AND DESCRIPTIVE STATISTICS")
print("="*60)

print("\n--- DataFrame Information ---")
df.info()

print("\n--- Descriptive Statistics (Numerical Columns) ---")
print(df.describe())

print("\n--- Descriptive Statistics (Categorical Columns) ---")
print(df.describe(include=['object']))


# =====================================================================
# TASK 4 & 5: Check, Count, and Handle Null Values
# =====================================================================
print("\n" + "="*60)
print("TASK 4 & 5: NULL VALUE CHECK & IMPUTATION")
print("="*60)

# Check null count
null_counts = df.isnull().sum()
print("\n--- Count of Null Values per Column ---")
print(null_counts)

# Handle missing values using inplace imputation
# Impute numerical features with median to handle skewness
df["Tenure_Months"].fillna(df["Tenure_Months"].median(), inplace=True)
df["MonthlyCharges"].fillna(df["MonthlyCharges"].median(), inplace=True)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

print("\n--- Count of Null Values After Inplace Imputation ---")
print(df.isnull().sum())


# =====================================================================
# TASK 6: Sorting & Displaying Topmost 5 to 8 Records
# =====================================================================
print("\n" + "="*60)
print("TASK 6: TOPMOST RECORDS BASED ON CONDITIONAL SORTING")
print("="*60)

# Display top 6 records sorted by highest AI Risk Score
top_risk_customers = df.sort_values(by="AI_Risk_Score", ascending=False).head(6)
print("\n--- Top 6 Highest AI Risk Score Customers ---")
print(top_risk_customers[["CustomerID", "ContractType", "AI_Risk_Score", "Churn"]])


# =====================================================================
# TASK 7: Frequency Listing of Relevant Column (2 Cases)
# =====================================================================
print("\n" + "="*60)
print("TASK 7: FREQUENCY LISTINGS")
print("="*60)

print("\n--- Case 1: Frequency Distribution of Contract Types ---")
print(df["ContractType"].value_counts())

print("\n--- Case 2: Frequency Distribution of Churn Status (0 = Retained, 1 = Churned) ---")
print(df["Churn"].value_counts(normalize=True) * 100)  # Percentage distribution


# =====================================================================
# TASK 8: Sorting of Rows and Columns (Implicit and Explicit Indexing)
# =====================================================================
print("\n" + "="*60)
print("TASK 8: ROW & COLUMN SORTING (INDEXING)")
print("="*60)

# Sorting rows explicitly by index and columns implicitly by position
sorted_df = df.sort_values(by=["ContractType", "MonthlyCharges"], ascending=[True, False])

print("\n--- Explicit Indexing Access (loc - top 3 sorted records) ---")
print(sorted_df.loc[:, ["CustomerID", "ContractType", "MonthlyCharges"]].head(3))

print("\n--- Implicit Indexing Access (iloc - first 3 rows, first 4 columns) ---")
print(sorted_df.iloc[0:3, 0:4])


# =====================================================================
# TASK 9: Accessing Particular Rows based on Compound Conditions (3 Cases)
# =====================================================================
print("\n" + "="*60)
print("TASK 9: COMPOUND CONDITION FILTERING (3 CASES)")
print("="*60)

# Case 1: High AI Risk Score (> 0.70) AND Month-to-Month contract
case1 = df.loc[(df["AI_Risk_Score"] > 0.70) & (df["ContractType"] == "Month-to-Month"), ["CustomerID", "AI_Risk_Score", "ContractType"]]
print("\n--- Case 1: High Risk & Month-to-Month Customers ---")
print(case1)

# Case 2: Monthly Charges > 70 OR Total Charges > 2000, displaying selected columns
case2 = df.loc[(df["MonthlyCharges"] > 70.0) | (df["TotalCharges"] > 2000.0), ["CustomerID", "MonthlyCharges", "TotalCharges"]]
print("\n--- Case 2: High Spenders (Monthly > 70 OR Total > 2000) ---")
print(case2)

# Case 3: Churned customers (Churn == 1) with Tenure <= 12 months
case3 = df.loc[(df["Churn"] == 1) & (df["Tenure_Months"] <= 12), ["CustomerID", "Tenure_Months", "Churn", "AI_Risk_Score"]]
print("\n--- Case 3: Early Churned Customers (Tenure <= 12) ---")
print(case3)


# =====================================================================
# TASK 10: Minimum and Maximum Values Related Analysis
# =====================================================================
print("\n" + "="*60)
print("TASK 10: MINIMUM AND MAXIMUM VALUE ANALYSIS")
print("="*60)

max_risk_idx = df["AI_Risk_Score"].idxmax()
min_risk_idx = df["AI_Risk_Score"].idxmin()

print(f"Maximum AI Risk Score : {df['AI_Risk_Score'].max():.2f} (Customer ID: {df.loc[max_risk_idx, 'CustomerID']})")
print(f"Minimum AI Risk Score : {df['AI_Risk_Score'].min():.2f} (Customer ID: {df.loc[min_risk_idx, 'CustomerID']})")
print(f"Monthly Charges Range : Min = ${df['MonthlyCharges'].min():.2f}, Max = ${df['MonthlyCharges'].max():.2f}")


# =====================================================================
# TASK 11 & 13: GroupBy with Aggregate Functions (2 Cases)
# =====================================================================
print("\n" + "="*60)
print("TASK 11 & 13: GROUPBY AND AGGREGATIONS (2 CASES)")
print("="*60)

# Case 1: Group by ContractType with multiple aggregates
print("\n--- Case 1: Contract Type Aggregations ---")
grp1 = df.groupby("ContractType").agg({
    "MonthlyCharges": ["mean", "max"],
    "AI_Risk_Score": "mean",
    "CustomerID": "count"
})
print(grp1)

# Case 2: Group by Churn Status and ContractType with aggregates
print("\n--- Case 2: Churn Status & Contract Type Aggregations ---")
grp2 = df.groupby(["Churn", "ContractType"]).agg({
    "Tenure_Months": "mean",
    "TotalCharges": ["sum", "mean"]
})
print(grp2)


# =====================================================================
# TASK 12: Adding and Populating New Column using Existing Data
# =====================================================================
print("\n" + "="*60)
print("TASK 12: ADDING AND POPULATING NEW COLUMN")
print("="*60)

# Calculate Estimated Annual Cost using existing MonthlyCharges and Tenure_Months
df["Estimated_Annual_Cost"] = df["MonthlyCharges"] * 12

# Create a categorization column based on AI Risk Score using NumPy
df["Risk_Category"] = np.where(df["AI_Risk_Score"] >= 0.70, "High Risk", "Low/Medium Risk")

print("\n--- Updated DataFrame with New Columns ---")
print(df[["CustomerID", "MonthlyCharges", "Estimated_Annual_Cost", "AI_Risk_Score", "Risk_Category"]].head(6))
