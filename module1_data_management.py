import pandas as pd
import sqlite3
import os
from datetime import datetime

print("="*50)
print("MODULE 1: Data Collection & Management")
print("="*50)

# Create data directory
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load the dataset
print("\n1. Loading dataset...")
df = pd.read_excel('Ecopack_AI_Dataset_final.xlsx', sheet_name='EcoPackAI_Cleaned_Data')

print(f"   Loaded {len(df)} records")
print(f"   Columns: {df.columns.tolist()}")

# Create SQLite database
print("\n2. Creating SQLite database...")
conn = sqlite3.connect('data/ecopackai.db')
cursor = conn.cursor()

# Create materials table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        material_type TEXT,
        industry TEXT,
        product_category TEXT,
        weight_kg REAL,
        biodegradability_score REAL,
        co2_emission_score REAL,
        recyclability_percentage REAL,
        co2_impact_index REAL,
        cost_efficiency_index REAL,
        material_suitability_score REAL,
        data_source TEXT,
        created_at TIMESTAMP
    )
''')

# Insert data
print("\n3. Inserting data into database...")
for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO materials (
            material_name, material_type, industry, product_category,
            weight_kg, biodegradability_score, co2_emission_score,
            recyclability_percentage, co2_impact_index,
            cost_efficiency_index, material_suitability_score,
            data_source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row.get('Material_Name', ''),
        row.get('Material_Type', ''),
        row.get('Industry', ''),
        row.get('Product', ''),
        float(row.get('Weight_kg', 0)) if pd.notna(row.get('Weight_kg')) else 0,
        float(row.get('Biodegradability_Score', 0)) if pd.notna(row.get('Biodegradability_Score')) else 0,
        float(row.get('CO2_Emission_Score', 0)) if pd.notna(row.get('CO2_Emission_Score')) else 0,
        float(row.get('Recyclability_Percentage', 0)) if pd.notna(row.get('Recyclability_Percentage')) else 0,
        float(row.get('CO2_Impact_Index', 0)) if pd.notna(row.get('CO2_Impact_Index')) else 0,
        float(row.get('Cost_Efficiency_Index', 0)) if pd.notna(row.get('Cost_Efficiency_Index')) else 0,
        float(row.get('Material_Suitability_Score', 0)) if pd.notna(row.get('Material_Suitability_Score')) else 0,
        row.get('Data_Source', ''),
        datetime.now()
    ))

conn.commit()

# Verify data
cursor.execute("SELECT COUNT(*) FROM materials")
count = cursor.fetchone()[0]
print(f"\n✅ Database created successfully!")
print(f"   Total materials in database: {count}")

# Show sample
print("\n4. Sample data from database:")
cursor.execute("SELECT material_name, material_type, industry FROM materials LIMIT 5")
for row in cursor.fetchall():
    print(f"   - {row[0]} ({row[1]}) - {row[2]}")

conn.close()

print("\n" + "="*50)
print("✅ MODULE 1 COMPLETE")
print("="*50)