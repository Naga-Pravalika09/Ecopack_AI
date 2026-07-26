import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("="*50)
print("MODULE 3: Machine Learning Dataset Preparation")
print("="*50)

# Load cleaned data
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('data/ecopack_ai_cleaned_final.csv')
print(f"   Loaded {len(df)} records")

# Select features
print("\n2. Selecting features for ML...")

# Feature columns
feature_columns = [
    'Weight_kg', 'Biodegradability_Score', 'Recyclability_Percentage',
    'Industry_encoded', 'Material_Type_encoded'
]

# Ensure all features exist
available_features = [col for col in feature_columns if col in df.columns]

# Add missing features if needed
if 'Weight_kg' not in df.columns:
    df['Weight_kg'] = 10  # default value
    available_features.append('Weight_kg')

feature_columns = available_features
print(f"   Selected features: {feature_columns}")

# Target columns
target_columns = {
    'cost': 'Cost_Efficiency_Index',
    'co2': 'CO2_Emission_Score',
    'suitability': 'Material_Suitability_Score'
}

print(f"   Target variables: {list(target_columns.values())}")

# Prepare feature matrix X
print("\n3. Preparing feature matrix...")
X = df[feature_columns].copy()

# Handle any missing values
X = X.fillna(X.median())
print(f"   Feature matrix shape: {X.shape}")

# Prepare target variables
print("\n4. Preparing target variables...")
y_cost = df[target_columns['cost']].fillna(df[target_columns['cost']].median())
y_co2 = df[target_columns['co2']].fillna(df[target_columns['co2']].median())
y_suit = df[target_columns['suitability']].fillna(df[target_columns['suitability']].median())

print(f"   Cost target shape: {y_cost.shape}")
print(f"   CO2 target shape: {y_co2.shape}")
print(f"   Suitability target shape: {y_suit.shape}")

# Split data
print("\n5. Splitting data into train/test sets...")
X_train, X_test, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)
_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)
_, _, y_suit_train, y_suit_test = train_test_split(
    X, y_suit, test_size=0.2, random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# Scale features
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Features scaled successfully")

# Save prepared data
print("\n7. Saving prepared data...")
os.makedirs('models', exist_ok=True)

# Save scaler and feature info
joblib.dump(scaler, 'models/preprocessor.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')
joblib.dump(target_columns, 'models/target_columns.pkl')

# Save train/test data
joblib.dump(X_train_scaled, 'models/X_train_scaled.pkl')
joblib.dump(X_test_scaled, 'models/X_test_scaled.pkl')
joblib.dump(y_cost_train, 'models/y_cost_train.pkl')
joblib.dump(y_cost_test, 'models/y_cost_test.pkl')
joblib.dump(y_co2_train, 'models/y_co2_train.pkl')
joblib.dump(y_co2_test, 'models/y_co2_test.pkl')
joblib.dump(y_suit_train, 'models/y_suit_train.pkl')
joblib.dump(y_suit_test, 'models/y_suit_test.pkl')

print("   All prepared data saved to 'models/' directory")

# Summary
print("\n" + "="*50)
print("DATA PREPARATION SUMMARY")
print("="*50)
print(f"""
┌─────────────────────────────────────────────────────────┐
│              DATA PREPARATION PIPELINE                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Total samples: {len(X)}                                      │
│  Features: {len(feature_columns)}                                      │
│  Training samples: {len(X_train)}                                │
│  Testing samples: {len(X_test)}                                 │
│                                                         │
│  Features used:                                         │
│    • Weight_kg                                          │
│    • Biodegradability_Score                             │
│    • Recyclability_Percentage                           │
│    • Industry_encoded                                   │
│    • Material_Type_encoded                              │
│                                                         │
│  Targets:                                               │
│    • Cost_Efficiency_Index                              │
│    • CO2_Emission_Score                                 │
│    • Material_Suitability_Score                         │
└─────────────────────────────────────────────────────────┘
""")

print("✅ MODULE 3 COMPLETE")
print("="*50)