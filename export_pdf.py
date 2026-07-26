# export_pdf.py - Standalone PDF export script
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from datetime import datetime

def export_material_database_pdf():
    """Export entire material database as PDF"""
    
    df = pd.read_csv('data/ecopack_ai_cleaned_final.csv')
    
    pdf_file = f"EcoPackAI_Material_Database_{datetime.now().strftime('%Y%m%d')}.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter))
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#2ecc71'), alignment=TA_CENTER)
    story.append(Paragraph("EcoPackAI - Complete Material Database", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Create table from dataframe
    table_data = [df.columns.tolist()] + df.head(100).values.tolist()
    
    # Truncate long text
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            if isinstance(table_data[i][j], str) and len(table_data[i][j]) > 30:
                table_data[i][j] = table_data[i][j][:27] + "..."
    
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
    ]))
    
    story.append(table)
    doc.build(story)
    
    print(f"✅ Database exported to: {pdf_file}")

if __name__ == "__main__":
    export_material_database_pdf()