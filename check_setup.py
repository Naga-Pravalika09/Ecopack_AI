# check_setup.py - Run this to diagnose issues
import os
import sys

print("="*50)
print("ECOPACKAI SETUP CHECKER")
print("="*50)

# Check directories
print("\n1. Checking directories...")
for dir_name in ['models', 'data', 'templates']:
    if os.path.exists(dir_name):
        print(f"   ✅ {dir_name}/ exists")
    else:
        print(f"   ❌ {dir_name}/ MISSING")

# Check model files
print("\n2. Checking model files...")
model_files = ['best_cost_model.pkl', 'best_co2_model.pkl', 'best_suitability_model.pkl', 
               'preprocessor.pkl', 'feature_columns.pkl']
for f in model_files:
    path = os.path.join('models', f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   ✅ {f} ({size} bytes)")
    else:
        print(f"   ❌ {f} MISSING")

# Check data file
print("\n3. Checking data file...")
data_path = os.path.join('data', 'ecopack_ai_cleaned_final.csv')
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"   ✅ Data file loaded: {len(df)} rows, {len(df.columns)} columns")
else:
    print(f"   ❌ {data_path} MISSING")

# Check template
print("\n4. Checking template...")
template_path = os.path.join('templates', 'index.html')
if os.path.exists(template_path):
    size = os.path.getsize(template_path)
    print(f"   ✅ index.html ({size} bytes)")
else:
    print(f"   ❌ index.html MISSING")

print("\n" + "="*50)
print("If any items show ❌, run the corresponding module:")
print("  - Module 1 & 2: Create data files")
print("  - Module 3 & 4: Create model files")
print("  - Save index.html in templates/ folder")
print("="*50)