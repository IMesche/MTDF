#!/usr/bin/env python3
"""
Audit script to extract all data from DB_Workbook_STRICT_V18.xlsx
NO CHANGES - REPORTING ONLY

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""
import pandas as pd
import json
from pathlib import Path

def audit_workbook(path):
    """Extract all sheets and key information from workbook"""

    print("="*80)
    print("WORKBOOK AUDIT - DB_Workbook_STRICT_V18.xlsx")
    print("="*80)
    print()

    xl = pd.ExcelFile(path)

    print("📋 SHEETS FOUND:")
    for i, sheet in enumerate(xl.sheet_names, 1):
        print(f"  {i}. {sheet}")
    print()

    # Extract Pillar Targets
    print("="*80)
    print("🎯 PILLAR TARGETS (from Pillar_Targets sheet):")
    print("="*80)
    try:
        targets = xl.parse("Pillar_Targets", header=1)
        print(f"Shape: {targets.shape}")
        print(f"Columns: {list(targets.columns)}")
        print()

        # Show each pillar target
        for _, row in targets.iterrows():
            label = row.get('pillar_id', row.get('target_id', row.get('label', '?')))
            value = row.get('target', row.get('value', row.get('target_value', '?')))
            sigma = row.get('uncertainty', row.get('sigma', '?'))
            unit = row.get('unit', '')

            print(f"  {label}: {value} ± {sigma} {unit}")
        print()
    except Exception as e:
        print(f"  ❌ Error reading Pillar_Targets: {e}")
        print()

    # Extract Pillar Formulas
    print("="*80)
    print("📐 PILLAR FORMULAS (from Pillar_Formulas sheet):")
    print("="*80)
    try:
        formulas = xl.parse("Pillar_Formulas", header=1)
        print(f"Shape: {formulas.shape}")
        print(f"Columns: {list(formulas.columns)}")
        print()

        for _, row in formulas.iterrows():
            label = row.get('pillar_id', row.get('label', '?'))
            latex = row.get('latex', row.get('equation_latex', ''))
            python_expr = row.get('python_expr', '')

            print(f"  {label}:")
            if latex:
                print(f"    LaTeX: {latex[:80]}{'...' if len(str(latex)) > 80 else ''}")
            if python_expr:
                print(f"    Python: {python_expr[:80]}{'...' if len(str(python_expr)) > 80 else ''}")
            print()
    except Exception as e:
        print(f"  ❌ Error reading Pillar_Formulas: {e}")
        print()

    # Extract Parameters
    param_sheets = [
        "Params_Fundamental",
        "Params_Constants",
        "Params_Observational",
        "Params_Coefficients",
        "Params_Units"
    ]

    print("="*80)
    print("⚙️  PARAMETERS:")
    print("="*80)

    for sheet_name in param_sheets:
        try:
            df = xl.parse(sheet_name, header=1)
            print(f"\n{sheet_name} ({df.shape[0]} parameters):")
            print(f"  Columns: {list(df.columns)}")

            # Show first few parameters
            for _, row in df.head(5).iterrows():
                token = row.get('input_token', row.get('token', row.get('name', '?')))
                value = row.get('value_si', '?')
                unit = row.get('unit', '')
                print(f"    {token} = {value} {unit}")

            if len(df) > 5:
                print(f"    ... and {len(df) - 5} more")
            print()
        except Exception as e:
            print(f"  ❌ Error reading {sheet_name}: {e}")
            print()

    # Check Model Predictions Matrix
    print("="*80)
    print("📊 MODEL PREDICTIONS (from Model_Predictions_Matrix sheet):")
    print("="*80)
    try:
        preds = xl.parse("Model_Predictions_Matrix", header=1)
        print(f"Shape: {preds.shape}")
        print(f"Columns: {list(preds.columns)}")

        # Try to identify which columns are model predictions
        meta_cols = {'pillar_id', 'label', 'unit', 'notes', 'updated_at', 'updated_by',
                     'canonical_id', 'class', 'output_id', 'latex', 'python_expr'}
        model_cols = [c for c in preds.columns if str(c).lower() not in meta_cols]

        print(f"\nModel columns found: {model_cols}")
        print()
    except Exception as e:
        print(f"  ❌ Error reading Model_Predictions_Matrix: {e}")
        print()

    print("="*80)
    print("✅ AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    workbook_path = Path("../data/DB_Workbook_STRICT_V18.xlsx")
    if not workbook_path.exists():
        print(f"❌ Workbook not found: {workbook_path}")
        exit(1)

    audit_workbook(workbook_path)
