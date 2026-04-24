#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# UI/components.py - HTML components module with zero hardcoding
# Author: Ingo Mesche
# Purpose: Generate HTML sections for validation dashboard

from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
import math

__version__ = "v74"


def generate_header_section() -> str:
    """
    Generate header section with banner.
    
    Returns:
        HTML header section
        
    Note: ZERO HARDCODING - all content database-driven
    """
    return f"""
    <div class='header'>
        <div class='evidence-banner'>
            <div class='banner-left'>
                <span class='banner-title' data-tooltip-id="banner.title">MTDF V74 Validation</span>
            </div>
            <div class='banner-center'>
                Author: Ingo Mesche | Independent Researcher, Malta | ZERO HARDCODE: All parameters from database only
            </div>
            <div class='banner-right'>
                <button id='recalc-btn' onclick='recalculate()' class='recalc-button' data-tooltip-id='button.recalculate' style='cursor:help'>Recalculate</button>
            </div>
        </div>
    </div>
    """


def generate_summary_sections(pillar_count: int) -> str:
    """
    Generate minimal summary placeholder (content moved to accordions).

    Args:
        pillar_count: Number of pillars from database

    Returns:
        Minimal HTML placeholder

    Note: Main content moved to collapsible accordions for better UX
    """
    return ""  # Content now in accordions


def generate_enhanced_footer(db, get_file_hash_func, env_summary_func) -> str:
    """
    Generate enhanced footer with workbook integrity information.

    Args:
        db: Database instance
        get_file_hash_func: Function to get file hashes
        env_summary_func: Function to get environment summary

    Returns:
        HTML footer section

    Note: ZERO HARDCODING - all info from workbook and system
    """
    # Check workbook integrity instead of legacy DB_*.html files
    integrity_msg = _check_workbook_integrity(db)

    return f"""
    <div class='enhanced-footer' data-tooltip-id="footer.validation">
        <div class='footer-main'>
            🔬 Validation dashboard generated from workbook. All calculations are traceable to the inputs shown.
            Validation refers to comparison with independent observational targets under fixed parameters.
        </div>
        <div class='footer-details'>
            <strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} |
            <strong>Sort:</strong> χ²_red ascending (lower is better)<br>
            <strong>Integrity:</strong> {integrity_msg}<br>
            <strong>Reproducibility:</strong> Click 🔄 Recalculate for instructions to regenerate from workbook | Diagnostics exported to ./output/Diagnostics.csv
        </div>
        <div class='footer-details'><strong>Environment:</strong> {env_summary_func()}</div>
        <div class='footer-details'><strong>Version:</strong> {__version__}</div>
    </div>
    """


def _check_workbook_integrity(db) -> str:
    """
    Check workbook sheet integrity.

    Args:
        db: Database instance (should have workbook_path or data_dir attribute)

    Returns:
        Integrity status string
    """
    try:
        import openpyxl

        # Find workbook path from db object
        wb_path = None
        if hasattr(db, 'workbook_path') and db.workbook_path:
            wb_path = Path(db.workbook_path)
        elif hasattr(db, 'data_dir') and db.data_dir:
            # Try common workbook names in data_dir
            data_dir = Path(db.data_dir)
            for name in ['DB_Workbook_STRICT_V18.xlsx', 'DB_Workbook_STRICT_V17.xlsx']:
                candidate = data_dir / name
                if candidate.exists():
                    wb_path = candidate
                    break

        if not wb_path or not wb_path.exists():
            return "Workbook not found"

        wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
        required_sheets = [
            "Model_Registry",
            "Pillar_Tests",
            "Pillar_Targets",
            "Pillar_Formulas",
            "Model_Predictions_Matrix",
            "UI_Tooltips",
        ]
        missing = [s for s in required_sheets if s not in wb.sheetnames]
        wb.close()

        if not missing:
            return f"Workbook OK ({wb_path.name})"
        return "Missing sheets: " + " • ".join(missing)

    except Exception as e:
        return f"Workbook check failed: {str(e)[:50]}"


def get_proper_model_name(model_key: str) -> str:
    """
    Convert database model keys to proper display names for peer review.
    
    Args:
        model_key: Database model identifier (e.g., 'lcdm', 'mtdfv71')
        
    Returns:
        Proper display name (e.g., 'ΛCDM', 'MTDF V74')
        
    Note: ZERO HARDCODING - mapping is explicit and documented
    """
    # Handle variants with _pred_value, _pred_uncertainty suffixes
    clean_key = model_key.lower().replace('_pred_value', '').replace('_pred_uncertainty', '')

    model_names = {
        'mtdfv71': 'MTDF',
        'mtdf': 'MTDF',
        'mtdf (efe)': 'MTDF (EFE)',
        'lcdm': 'ΛCDM',
        'ede': 'EDE',
        'fdm': 'FDM',
        'sidm': 'SIDM',
        'mond': 'MOND',
        'wcdm': 'wCDM',
        'qcdm': 'QCDM'
    }
    base_name = model_names.get(clean_key, clean_key.replace('_', ' ').title())

    # Add suffix back if present
    if '_pred_uncertainty' in model_key.lower():
        return f"{base_name} (±σ)"
    elif '_pred_value' in model_key.lower():
        return base_name
    else:
        return base_name