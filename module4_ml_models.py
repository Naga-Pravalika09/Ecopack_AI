# module4_ml_models.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("="*60)
print("MODULE 4: AI Recommendation Model with Visualization")
print("="*60)

# Load prepared data
print("\n📊 Loading prepared data...")

# Check if files exist
if not os.path.exists('models/X_train_scaled.pkl'):
    print("❌ Error: Please run module3 first!")
    print("   Run: python module3_ml_dataset_preparation.py")
    exit(1)

X_train_scaled = joblib.load('models/X_train_scaled.pkl')
X_test_scaled = joblib.load('models/X_test_scaled.pkl')
y_cost_train = joblib.load('models/y_cost_train.pkl')
y_cost_test = joblib.load('models/y_cost_test.pkl')
y_co2_train = joblib.load('models/y_co2_train.pkl')
y_co2_test = joblib.load('models/y_co2_test.pkl')
y_suit_train = joblib.load('models/y_suit_train.pkl')
y_suit_test = joblib.load('models/y_suit_test.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

print(f"   Training data: {X_train_scaled.shape}")
print(f"   Testing data: {X_test_scaled.shape}")

# Create directory for graphs
os.makedirs('graphs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Dictionary to store models
models = {}
results = []

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, target):
    print(f"   Training {model_name} for {target}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"      RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return model, {'model': model_name, 'target': target, 'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_pred

# Train models
targets = [
    ('cost', y_cost_train, y_cost_test, 'Cost Efficiency Index'),
    ('co2', y_co2_train, y_co2_test, 'CO2 Emission Score'),
    ('suitability', y_suit_train, y_suit_test, 'Material Suitability')
]

all_predictions = {}

for target_key, y_train, y_test, target_name in targets:
    print(f"\n{'='*50}")
    print(f"Training for: {target_name}")
    print('='*50)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf, rf_results, rf_pred = train_and_evaluate(rf, 'Random Forest', X_train_scaled, y_train, X_test_scaled, y_test, target_key)
    models[f'rf_{target_key}'] = rf
    results.append(rf_results)
    all_predictions[f'rf_{target_key}_pred'] = rf_pred
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model, xgb_results, xgb_pred = train_and_evaluate(xgb_model, 'XGBoost', X_train_scaled, y_train, X_test_scaled, y_test, target_key)
    models[f'xgb_{target_key}'] = xgb_model
    results.append(xgb_results)
    all_predictions[f'xgb_{target_key}_pred'] = xgb_pred
    
    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge, ridge_results, ridge_pred = train_and_evaluate(ridge, 'Ridge', X_train_scaled, y_train, X_test_scaled, y_test, target_key)
    models[f'ridge_{target_key}'] = ridge
    results.append(ridge_results)
    all_predictions[f'ridge_{target_key}_pred'] = ridge_pred

results_df = pd.DataFrame(results)

# ============================================
# GRAPH 1: Model Performance Comparison
# ============================================
print("\n📈 Generating Graph 1: Model Performance Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
targets_display = ['Cost Efficiency', 'CO2 Emission', 'Material Suitability']
target_keys = ['cost', 'co2', 'suitability']
colors = ['#2ecc71', '#3498db', '#e74c3c']

for idx, (target_key, title, color) in enumerate(zip(target_keys, targets_display, colors)):
    target_results = results_df[results_df['target'] == target_key]
    ax = axes[idx]
    
    bars = ax.bar(target_results['model'], target_results['R2'], color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('R2 Score', fontsize=11)
    ax.set_title(f'{title}\nPrediction Performance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good Threshold')
    ax.legend()
    
    for bar, val in zip(bars, target_results['R2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('EcoPackAI - Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/model_performance_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/model_performance_comparison.png")

# ============================================
# GRAPH 2: Actual vs Predicted
# ============================================
print("\n📈 Generating Graph 2: Actual vs Predicted Values...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (target_key, title, color) in enumerate(zip(target_keys, targets_display, colors)):
    ax = axes[idx]
    
    best_model_name = results_df[results_df['target'] == target_key].sort_values('R2', ascending=False).iloc[0]['model'].lower()
    y_pred = all_predictions[f'{best_model_name}_{target_key}_pred']
    
    if target_key == 'cost':
        y_actual = y_cost_test
    elif target_key == 'co2':
        y_actual = y_co2_test
    else:
        y_actual = y_suit_test
    
    ax.scatter(y_actual, y_pred, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title(f'{title}\nBest Model: {best_model_name.title()}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('EcoPackAI - Prediction Accuracy', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/actual_vs_predicted.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/actual_vs_predicted.png")

# ============================================
# GRAPH 3: Feature Importance
# ============================================
print("\n📈 Generating Graph 3: Feature Importance Analysis...")

rf_cost_model = models['rf_cost']
feature_importance = rf_cost_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = plt.cm.Greens(np.linspace(0.3, 0.9, len(importance_df)))

bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_imp, edgecolor='black')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Feature Importance for Cost Prediction\n(Random Forest Model)', fontsize=14, fontweight='bold')

for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('graphs/feature_importance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/feature_importance.png")

# ============================================
# GRAPH 4: Error Distribution
# ============================================
print("\n📈 Generating Graph 4: Error Distribution Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (target_key, title, color) in enumerate(zip(target_keys, targets_display, colors)):
    ax = axes[idx]
    
    best_model_name = results_df[results_df['target'] == target_key].sort_values('R2', ascending=False).iloc[0]['model'].lower()
    y_pred = all_predictions[f'{best_model_name}_{target_key}_pred']
    
    if target_key == 'cost':
        y_actual = y_cost_test
    elif target_key == 'co2':
        y_actual = y_co2_test
    else:
        y_actual = y_suit_test
    
    residuals = y_actual - y_pred
    
    ax.hist(residuals, bins=30, color=color, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (Residuals)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{title}\nError Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('EcoPackAI - Prediction Error Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/error_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/error_distribution.png")

# ============================================
# GRAPH 5: Material Distribution
# ============================================
print("\n📈 Generating Graph 5: Material Type Distribution...")

df_materials = pd.read_csv('data/ecopack_ai_cleaned_final.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

material_counts = df_materials['Material_Type'].value_counts()
colors_pie = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
axes[0].pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%', 
            colors=colors_pie[:len(material_counts)], startangle=90, explode=[0.05]*len(material_counts))
axes[0].set_title('Material Type Distribution', fontsize=14, fontweight='bold')

industry_counts = df_materials['Industry'].value_counts().head(10)
bars = axes[1].bar(industry_counts.index, industry_counts.values, color='#3498db', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Industry', fontsize=12)
axes[1].set_ylabel('Number of Materials', fontsize=12)
axes[1].set_title('Materials by Industry (Top 10)', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=10)

for bar, val in zip(bars, industry_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/material_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/material_distribution.png")

# ============================================
# GRAPH 6: Sustainability Distribution
# ============================================
print("\n📈 Generating Graph 6: Sustainability Score Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df_materials['Biodegradability_Score'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Biodegradability Score Distribution', fontsize=12, fontweight='bold')
axes[0].axvline(x=df_materials['Biodegradability_Score'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {df_materials['Biodegradability_Score'].mean():.2f}")
axes[0].legend()

axes[1].hist(df_materials['Recyclability_Percentage'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Recyclability Distribution', fontsize=12, fontweight='bold')
axes[1].axvline(x=df_materials['Recyclability_Percentage'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {df_materials['Recyclability_Percentage'].mean():.2f}")
axes[1].legend()

axes[2].hist(df_materials['CO2_Emission_Score'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Score', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('CO2 Emission Score Distribution', fontsize=12, fontweight='bold')
axes[2].axvline(x=df_materials['CO2_Emission_Score'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {df_materials['CO2_Emission_Score'].mean():.2f}")
axes[2].legend()

plt.suptitle('EcoPackAI - Sustainability Metrics Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/sustainability_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/sustainability_distribution.png")

# ============================================
# GRAPH 7: Correlation Heatmap
# ============================================
print("\n📈 Generating Graph 7: Feature Correlation Heatmap...")

corr_columns = ['Biodegradability_Score', 'Recyclability_Percentage', 'CO2_Emission_Score',
                'Cost_Efficiency_Index', 'Material_Suitability_Score', 'Weight_kg']
corr_df = df_materials[corr_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('graphs/correlation_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: graphs/correlation_heatmap.png")

# ============================================
# Select and save best models
# ============================================
print("\n🎯 Selecting best models...")
best_models = {}

for target in ['cost', 'co2', 'suitability']:
    target_results = results_df[results_df['target'] == target]
    best_idx = target_results['R2'].idxmax()
    best_model_name = target_results.loc[best_idx, 'model'].lower()
    best_model = models[f'{best_model_name}_{target}']
    best_models[target] = best_model
    print(f"   Best for {target}: {best_model_name} (R2 = {target_results.loc[best_idx, 'R2']:.4f})")

# Save models
for target, model in best_models.items():
    joblib.dump(model, f'models/best_{target}_model.pkl')
    print(f"   Saved: models/best_{target}_model.pkl")

# ============================================
# Generate Summary Report (FIXED - No special characters)
# ============================================
print("\n📄 Generating Summary Report...")

# Create a simple text report without special characters
report_lines = []
report_lines.append("="*60)
report_lines.append("ECOPACKAI - MODEL REPORT")
report_lines.append("="*60)
report_lines.append("")
report_lines.append("Dataset Statistics:")
report_lines.append(f"  - Total Materials: {len(df_materials)}")
report_lines.append(f"  - Material Types: {df_materials['Material_Type'].nunique()}")
report_lines.append(f"  - Industries: {df_materials['Industry'].nunique()}")
report_lines.append(f"  - Features Used: {len(feature_columns)}")
report_lines.append("")
report_lines.append("Model Performance (Best Models):")
report_lines.append(f"  - Cost Prediction:     R2 = {results_df[results_df['target']=='cost']['R2'].max():.4f}")
report_lines.append(f"  - CO2 Prediction:      R2 = {results_df[results_df['target']=='co2']['R2'].max():.4f}")
report_lines.append(f"  - Suitability:         R2 = {results_df[results_df['target']=='suitability']['R2'].max():.4f}")
report_lines.append("")
report_lines.append("Average Sustainability Scores:")
report_lines.append(f"  - Biodegradability:    {df_materials['Biodegradability_Score'].mean():.2%}")
report_lines.append(f"  - Recyclability:       {df_materials['Recyclability_Percentage'].mean():.2%}")
report_lines.append(f"  - CO2 Emission:        {df_materials['CO2_Emission_Score'].mean():.2%}")
report_lines.append("")
report_lines.append("Generated Graphs (saved in 'graphs/' folder):")
report_lines.append("  - 1. model_performance_comparison.png")
report_lines.append("  - 2. actual_vs_predicted.png")
report_lines.append("  - 3. feature_importance.png")
report_lines.append("  - 4. error_distribution.png")
report_lines.append("  - 5. material_distribution.png")
report_lines.append("  - 6. sustainability_distribution.png")
report_lines.append("  - 7. correlation_heatmap.png")
report_lines.append("")
report_lines.append("="*60)

# Print to console
for line in report_lines:
    print(line)

# Save to file with proper UTF-8 encoding
with open('graphs/model_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("\n✅ Report saved to: graphs/model_report.txt")

print("\n" + "="*60)
print("✅ MODULE 4 COMPLETE - All graphs generated!")
print("="*60)
print("\n📁 Output files:")
print("   📊 Graphs saved in 'graphs/' folder")
print("   🤖 Models saved in 'models/' folder")
print("   📄 Report saved in 'graphs/model_report.txt'")