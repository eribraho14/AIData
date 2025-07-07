import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load data
df = pd.read_csv("DataAI.csv")

# 1. Remove duplicate rows
print(f"Duplicates before: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Shape after duplicate removal: {df.shape}")

# 2. Drop rows with missing Client_ID or Loan_ID (critical fields)
df = df.dropna(subset=['Client_ID', 'Loan_ID'])

# 3. Handle missing categorical fields
cat_cols = ['Gender', 'CivilStatus', 'EducationLevel', 'UrbanRural', 'EmploymentStatus',
            'LoanStatus', 'LoanPurpose', 'LoanSegment', 'FirstApplYesNo', 'Is_GUA']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.lower()
        df[col] = df[col].astype("category")

# 4. Handle missing numeric fields with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Loan_Default' in num_cols:
    num_cols.remove('Loan_Default')  # Keep target unchanged

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Convert date columns to datetime
date_cols = ['ApplicationDate', 'MaturityDate', 'DateCreated']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"{col} has {df[col].isnull().sum()} invalid dates")
        df = df.dropna(subset=[col])


# 6. Filter unrealistic ages
if 'Age' in df.columns:
    print(f"Age min: {df['Age'].min()}, max: {df['Age'].max()}")
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

# 7. Feature engineering

# HasLateHistory
if 'TotNumInstLate' in df.columns:
    df['HasLateHistory'] = (df['TotNumInstLate'] > 0).astype(int)

# LoanAmountToIncome
if 'Amount' in df.columns and 'MonthlyIncome' in df.columns:
    df['LoanAmountToIncome'] = df['Amount'] / df['MonthlyIncome'].replace(0, np.nan)

# InstallmentRatio
if 'InstallmentValue' in df.columns and 'MonthlyIncome' in df.columns:
    df['InstallmentRatio'] = df['InstallmentValue'] / df['MonthlyIncome'].replace(0, np.nan)

# LoanAge in months
if 'ApplicationDate' in df.columns:
    df['LoanAge'] = (pd.to_datetime("today") - df['ApplicationDate']).dt.days // 30

# 8. Optional: Review LoanStatus vs Loan_Default
if 'LoanStatus' in df.columns and 'Loan_Default' in df.columns:
    print(df[['LoanStatus', 'Loan_Default']].drop_duplicates())

# 9. Reset index
df = df.reset_index(drop=True)

# 10. Final shape
print(f"Cleaned and engineered data shape: {df.shape}")
print(df[['HasLateHistory', 'LoanAmountToIncome', 'InstallmentRatio', 'LoanAge']].head())



################################################################# check class distirbution   ######################################################
# Check class distribution of the target
class_counts = df['Loan_Default'].value_counts()
class_percent = df['Loan_Default'].value_counts(normalize=True) * 100

print("🔍 Loan_Default Class Distribution:")
print(class_counts)
print("\n📊 Percentages:")
print(class_percent.round(2))

# Drop non-features
X = df.drop(columns=['Loan_Default', 'Client_ID', 'Loan_ID', 'ApplicationDate', 'MaturityDate', 'DateCreated'], errors='ignore')
y = df['Loan_Default']

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Impute missing values (median strategy)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Apply SMOTE to imputed data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

print("✅ After SMOTE:", Counter(y_resampled))

# Now scale the data (optional, but recommended)
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)


############################################################################################################
# Create DataFrame from features
df_features = pd.DataFrame(X_resampled, columns=X_imputed.columns)

# Create DataFrame from target
df_target = pd.DataFrame(y_resampled, columns=['Loan_Default'])

# Concatenate along columns
df_balanced = pd.concat([df_features, df_target], axis=1)

# Now save
df_balanced.to_csv('balanced_dataset.csv', index=False)