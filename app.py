# app.py - Complete Working Version with Graph and PDF
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import base64

app = Flask(__name__)
CORS(app)

print("="*50)
print("STARTING ECOPackAI SERVER")
print("="*50)

# ==================== LOAD DATA ====================
data_path = 'data/ecopack_ai_cleaned_final.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} materials")
else:
    print("❌ Data file not found. Creating sample data...")
    # Create sample data if file doesn't exist
    sample_data = {
        'Material_Name': ['PLA Container', 'Bagasse Container', 'Corrugated Box', 'Molded Pulp', 'Recycled Paper',
                         'Bamboo Fiber', 'Kraft Paper', 'Cornstarch Plastic', 'Cotton Fiber', 'Aluminum Foil'],
        'Material_Type': ['Bioplastic', 'Natural Fiber', 'Cardboard', 'Paper', 'Paper',
                         'Natural Fiber', 'Paper', 'Bioplastic', 'Natural Material', 'Metal'],
        'Industry': ['Packaging', 'Food', 'Packaging', 'Electronics', 'Packaging',
                    'Food', 'Packaging', 'Food', 'Textile', 'Packaging'],
        'Biodegradability_Score': [0.85, 0.86, 0.51, 0.76, 0.94, 0.53, 0.81, 0.93, 0.68, 0.71],
        'Recyclability_Percentage': [0.83, 0.81, 0.75, 0.80, 0.94, 0.76, 0.89, 0.92, 0.54, 0.59],
        'CO2_Emission_Score': [0.31, 0.32, 0.50, 0.25, 0.32, 0.31, 0.69, 0.63, 0.61, 0.56],
        'Cost_Efficiency_Index': [0.70, 0.85, 0.65, 0.78, 0.82, 0.68, 0.72, 0.75, 0.60, 0.55],
        'Material_Suitability_Score': [0.79, 0.78, 0.58, 0.78, 0.80, 0.70, 0.68, 0.76, 0.55, 0.52]
    }
    df = pd.DataFrame(sample_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)

# Industry mapping
industry_map = {
    'Electronics': 0, 'Cosmetics': 1, 'Food & Beverage': 2,
    'Healthcare': 3, 'Automotive': 4, 'Furniture': 5, 'Packaging': 6,
    'Construction': 0, 'Toys': 1, 'Home Appliances': 5, 'Food': 2
}

# ==================== HELPER FUNCTIONS ====================

def calculate_eco_score(row, priority_sustainability=False):
    """Calculate eco score based on multiple factors"""
    if priority_sustainability:
        # Higher weight on sustainability
        score = (row['Biodegradability_Score'] * 0.4 +
                 row['Recyclability_Percentage'] * 0.4 +
                 (1 - row['CO2_Emission_Score']) * 0.2)
    else:
        # Balanced with cost
        score = (row['Biodegradability_Score'] * 0.35 +
                 row['Recyclability_Percentage'] * 0.35 +
                 (1 - row['CO2_Emission_Score']) * 0.15 +
                 row.get('Cost_Efficiency_Index', 0.5) * 0.15)
    return score

def get_recommendations(product_name, industry, weight_kg, top_n=10, priority_sustainability=False):
    """Get material recommendations"""
    results = []
    
    # Filter by industry if specified
    if industry and industry != 'General':
        filtered_df = df[df['Industry'].str.contains(industry, case=False, na=False)]
        if len(filtered_df) == 0:
            filtered_df = df
    else:
        filtered_df = df
    
    for _, row in filtered_df.iterrows():
        eco_score = calculate_eco_score(row, priority_sustainability)
        
        # Estimate cost based on weight and cost efficiency
        est_cost = (weight_kg * 2.5) * (1 - row.get('Cost_Efficiency_Index', 0.5))
        
        results.append({
            'material_name': row['Material_Name'],
            'material_type': row['Material_Type'],
            'biodegradability': round(row['Biodegradability_Score'] * 100, 1),
            'recyclability': round(row['Recyclability_Percentage'] * 100, 1),
            'co2_efficiency': round((1 - row['CO2_Emission_Score']) * 100, 1),
            'eco_score': round(eco_score * 100, 1),
            'est_cost': round(est_cost, 2)
        })
    
    # Sort by eco score
    results.sort(key=lambda x: x['eco_score'], reverse=True)
    return results[:top_n]

def create_comparison_chart(recommendations):
    """Create bar chart like the one in your image"""
    plt.figure(figsize=(12, 6))
    
    materials = [r['material_name'][:15] for r in recommendations[:5]]
    biodegradability = [r['biodegradability'] for r in recommendations[:5]]
    co2_efficiency = [r['co2_efficiency'] for r in recommendations[:5]]
    recyclability = [r['recyclability'] for r in recommendations[:5]]
    
    x = np.arange(len(materials))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, biodegradability, width, label='Biodegradability', color='#2ecc71')
    bars2 = ax.bar(x, co2_efficiency, width, label='CO₂ Efficiency', color='#3498db')
    bars3 = ax.bar(x + width, recyclability, width, label='Recyclability', color='#f39c12')
    
    ax.set_xlabel('Materials', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Environmental Score Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(materials, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return base64.b64encode(img_buffer.getvalue()).decode()

def generate_pdf_report(product_name, industry, weight_kg, recommendations):
    """Generate PDF report"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18,
                                  textColor=colors.HexColor('#2ecc71'), alignment=TA_CENTER)
    story.append(Paragraph("EcoPackAI - Sustainability Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report info
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"<b>Product:</b> {product_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Industry:</b> {industry}", styles['Normal']))
    story.append(Paragraph(f"<b>Weight:</b> {weight_kg} kg", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations Table
    story.append(Paragraph("<b>Top Material Recommendations</b>", styles['Heading2']))
    
    table_data = [["Rank", "Material", "Type", "Eco Score", "Biodegradability", "CO2 Efficiency", "Recyclability", "Est. Cost"]]
    for idx, rec in enumerate(recommendations[:8], 1):
        table_data.append([
            str(idx),
            rec['material_name'][:20],
            rec['material_type'],
            f"{rec['eco_score']}%",
            f"{rec['biodegradability']}%",
            f"{rec['co2_efficiency']}%",
            f"{rec['recyclability']}%",
            f"${rec['est_cost']}"
        ])
    
    table = Table(table_data, colWidths=[0.4*inch, 1.5*inch, 0.8*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.7*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2*inch))
    
    # Summary
    top_rec = recommendations[0]
    story.append(Paragraph("<b>Recommendation Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Top Recommended Material:</b> {top_rec['material_name']}", styles['Normal']))
    story.append(Paragraph(f"<b>Material Type:</b> {top_rec['material_type']}", styles['Normal']))
    story.append(Paragraph(f"<b>Eco Score:</b> {top_rec['eco_score']}%", styles['Normal']))
    story.append(Paragraph(f"<b>Estimated Cost:</b> ${top_rec['est_cost']}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Paragraph(f"Report generated by EcoPackAI - AI-Powered Sustainable Packaging System", styles['Normal']))
    story.append(Paragraph(f"© {datetime.now().year} EcoPackAI. All rights reserved.", styles['Normal']))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        
        product_name = data.get('product_name', 'Unknown')
        industry = data.get('industry', 'General')
        weight_kg = float(data.get('weight_kg', 1.0))
        top_n = int(data.get('top_n', 10))
        priority_sustainability = data.get('priority_sustainability', False)
        
        recommendations = get_recommendations(product_name, industry, weight_kg, top_n, priority_sustainability)
        
        # Generate chart
        chart_base64 = create_comparison_chart(recommendations)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations),
            'chart': chart_base64
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()
        product_name = data.get('product_name', 'Unknown')
        industry = data.get('industry', 'General')
        weight_kg = float(data.get('weight_kg', 1.0))
        recommendations = data.get('recommendations', [])
        
        pdf_buffer = generate_pdf_report(product_name, industry, weight_kg, recommendations)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"EcoPackAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analytics', methods=['GET'])
def analytics():
    try:
        summary = {
            'total_materials': len(df),
            'material_types': df['Material_Type'].value_counts().head(5).to_dict(),
            'industries': df['Industry'].value_counts().head(5).to_dict(),
            'avg_biodegradability': round(df['Biodegradability_Score'].mean() * 100, 1),
            'avg_recyclability': round(df['Recyclability_Percentage'].mean() * 100, 1),
            'avg_co2_reduction': round((1 - df['CO2_Emission_Score']).mean() * 100, 1)
        }
        return jsonify({'success': True, 'analytics': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("\n" + "="*50)
    print("🚀 EcoPackAI Server Running!")
    print("📍 http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)