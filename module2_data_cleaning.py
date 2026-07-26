import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*50)
print("MODULE 2: Data Cleaning & Feature Engineering")
print("="*50)

# Load the data
print("\n1. Loading raw data...")
df = pd.read_excel('Ecopack_AI_Dataset_final.xlsx', sheet_name='EcoPackAI_Cleaned_Data')
print(f"   Original shape: {df.shape}")

# Check for missing values
print("\n2. Checking for missing values...")
missing = df.isnull().sum()
print(f"   Missing values:\n{missing[missing > 0]}")

# Handle missing values
print("\n3. Handling missing values...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   Filled missing in {col} with median")
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            print(f"   Filled missing in {col} with mode")

# Remove duplicates
print("\n4. Removing duplicates...")
before = len(df)
df = df.drop_duplicates(subset=['Material_Name', 'Product', 'Industry'])
after = len(df)
print(f"   Removed {before - after} duplicate records")

# Encode categorical variables
print("\n5. Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['Industry', 'Material_Type', 'Product', 'Data_Source']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} unique values")

# Create derived features
print("\n6. Creating derived features...")

# Ensure Weight_kg exists
if 'Weight_kg' not in df.columns:
    df['Weight_kg'] = df['Weight_or_Capacity'].str.extract('(\d+\.?\d*)').astype(float)

# Normalize weight
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Weight_Normalized'] = scaler.fit_transform(df[['Weight_kg']])

# Create additional scores
df['Eco_Score'] = (
    df['Biodegradability_Score'] * 0.4 +
    df['Recyclability_Percentage'] * 0.4 +
    (1 - df['CO2_Emission_Score']) * 0.2
)

df['Circular_Economy_Score'] = (
    df['Recyclability_Percentage'] * 0.6 +
    df['Biodegradability_Score'] * 0.4
)

print(f"   Created new features: Eco_Score, Circular_Economy_Score, Weight_Normalized")

# Save cleaned dataset
print("\n7. Saving cleaned dataset...")
df.to_csv('data/ecopack_ai_cleaned_final.csv', index=False)
print(f"   Saved to: data/ecopack_ai_cleaned_final.csv")
print(f"   Final shape: {df.shape}")

# Generate summary report
print("\n" + "="*50)
print("DATA QUALITY SUMMARY")
print("="*50)
print(f"Total records: {len(df)}")
print(f"Total features: {len(df.columns)}")
print(f"\nMaterial types distribution:")
print(df['Material_Type'].value_counts().head(10))
print(f"\nIndustry distribution:")
print(df['Industry'].value_counts().head(10))
print(f"\nAverage scores:")
print(f"  Biodegradability: {df['Biodegradability_Score'].mean():.3f}")
print(f"  Recyclability: {df['Recyclability_Percentage'].mean():.3f}")
print(f"  CO2 Emission: {df['CO2_Emission_Score'].mean():.3f}")
print(f"  Eco Score: {df['Eco_Score'].mean():.3f}")

print("\n" + "="*50)
print("✅ MODULE 2 COMPLETE")
print("="*50)