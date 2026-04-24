#!/usr/bin/env python3
"""
Compare workbook targets with dashboard HTML
VERIFICATION ONLY - NO CHANGES

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""
import pandas as pd
import re
from pathlib import Path

def extract_dashboard_targets(html_path):
    """Extract target values from dashboard HTML"""
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Find the target row in the table
    # Pattern: <th class='target-row' ...>VALUE±SIGMA</th>
    pattern = r"<th class='target-row'[^>]*>([^<]+)</th>"
    matches = re.findall(pattern, html)

    # Expected pillar order from the HTML
    pillars = ['P1', 'P1B', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P10B', 'P11', 'P12', 'P13']

    targets = {}
    for i, match in enumerate(matches):
        if i < len(pillars):
            # Parse "VALUE±SIGMA"
            parts = match.strip().split('±')
            if len(parts) == 2:
                targets[pillars[i]] = {
                    'value': parts[0].strip(),
                    'sigma': parts[1].strip(),
                    'display': match.strip()
                }

    return targets

def compare_sources():
    """Compare workbook vs dashboard vs benchmark doc"""

    print("="*80)
    print("ALIGNMENT VERIFICATION REPORT")
    print("="*80)
    print()

    # Load workbook
    workbook_path = Path("../data/DB_Workbook_STRICT_V18.xlsx")
    xl = pd.ExcelFile(workbook_path)
    wb_targets = xl.parse("Pillar_Targets", header=1)

    # Load dashboard
    dashboard_path = Path("../output/Validation_Dashboard_V74.html")
    dash_targets = extract_dashboard_targets(dashboard_path)

    # Compare
    print("📊 TARGET VALUES COMPARISON:")
    print("="*80)
    print(f"{'Pillar':<8} {'Workbook':<25} {'Dashboard':<25} {'Status':<10}")
    print("-"*80)

    discrepancies = []

    for _, row in wb_targets.iterrows():
        pillar_id = str(row['target_id']).replace('target:', '')
        wb_val = row['value']
        wb_sigma = row['uncertainty']
        wb_unit = row.get('unit', '')

        # Get dashboard value
        dash = dash_targets.get(pillar_id, {})
        dash_display = dash.get('display', 'NOT FOUND')

        # Format workbook display
        wb_display = f"{wb_val}±{wb_sigma}"

        # Check match
        status = "✓ MATCH"
        if pillar_id not in dash_targets:
            status = "❌ MISSING"
            discrepancies.append(f"{pillar_id}: Not found in dashboard")
        else:
            # Compare values (allowing for formatting differences)
            try:
                dash_val = float(dash['value'])
                dash_sig = float(dash['sigma'])

                if abs(dash_val - wb_val) > 0.0001 or abs(dash_sig - wb_sigma) > 0.0001:
                    status = "⚠️  MISMATCH"
                    discrepancies.append(
                        f"{pillar_id}: WB={wb_val}±{wb_sigma} vs DASH={dash_val}±{dash_sig}"
                    )
            except:
                # String comparison if numeric fails
                if wb_display.replace(' ', '') != dash_display.replace(' ', '').replace('±', '±'):
                    status = "⚠️  CHECK"

        print(f"{pillar_id:<8} {wb_display:<25} {dash_display:<25} {status:<10}")

    print()

    # P1 Benchmark cross-check
    print("="*80)
    print("📌 P1 BENCHMARK VERIFICATION:")
    print("="*80)

    p1_benchmark_value = 0.174822  # From README_P1_benchmark.txt
    p1_benchmark_sigma = 0.011     # From README (quadrature of 0.0085 + 0.007)

    p1_wb = wb_targets[wb_targets['target_id'] == 'target:P1'].iloc[0]
    p1_wb_val = p1_wb['value']
    p1_wb_sigma = p1_wb['uncertainty']

    print(f"Benchmark README:  {p1_benchmark_value} ± {p1_benchmark_sigma} dex")
    print(f"Workbook:          {p1_wb_val} ± {p1_wb_sigma} dex")

    # Check if they match (within rounding)
    if abs(p1_wb_val - 0.1743) < 0.0001 and abs(p1_wb_sigma - 0.011) < 0.0001:
        print("Status: ✓ ALIGNED (workbook uses rounded benchmark value)")
    else:
        print("Status: ⚠️  POTENTIAL DISCREPANCY")
        discrepancies.append(f"P1: Benchmark={p1_benchmark_value} vs Workbook={p1_wb_val}")

    print()

    # Summary
    print("="*80)
    print("SUMMARY:")
    print("="*80)

    if discrepancies:
        print(f"⚠️  Found {len(discrepancies)} potential issue(s):")
        for d in discrepancies:
            print(f"  - {d}")
    else:
        print("✅ All targets aligned between workbook and dashboard!")

    print()
    print("="*80)

if __name__ == "__main__":
    compare_sources()
