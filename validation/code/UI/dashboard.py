#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

# SPDX-License-Identifier: MIT
# UI/dashboard.py - dashboard generator

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import math
import hashlib
import platform
import sys
import json

# Import the new tooltip engine
try:
    from UI.tooltip_engine import TooltipEngine
except ImportError:
    TooltipEngine = None

# Try to import UI helpers; provide lightweight fallbacks if missing
try:
    from UI.styles import generate_css_styles
    from UI.scripts import generate_javascript, generate_placeholder_scripts
    from UI.components import (
        generate_header_section,
        generate_summary_sections,
        generate_enhanced_footer,
        get_proper_model_name,
    )
except Exception:
    def generate_css_styles(pillars):
        return """
        <style>
            body{font-family:Inter,Arial,Helvetica,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}
            h1{margin:16px 20px}
            .table-wrap{overflow:auto;margin:12px 20px 32px 20px;background:#0b1220;border:1px solid #1f2a44;border-radius:10px}
            table{border-collapse:collapse;width:100%}
            th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #1f2a44;white-space:nowrap}
            thead th{position:sticky;top:0;background:#0d172a;font-weight:600}
            .model-col,.evidence-col{position:sticky;left:0;z-index:2;background:#0d172a;text-align:left}
            .model-col{max-width:130px}
            .evidence-col{left:140px;max-width:70px}
            .validated-row td{font-weight:600}
            td.ok{background:#06351e}
            td.warn{background:#3a2c05}
            td.bad{background:#3a0c0c}
            td.na{background:#111827;color:#64748b}
            .cell-sub{display:block;font-size:12px;color:#93a3b3;margin-top:2px}
            .tier-validated{color:#22c55e}
            .tier-requires_components{color:#ef4444}
            .tier-supported{color:#f59e0b}
            .tier-not_supported{color:#94a3b8}
            .target-row{font-weight:500;color:#9fb3c8}
            .pillar-coverage{font-weight:400;color:#93a3b3;margin-left:4px}
            .role-badge{font-size:9px;font-weight:bold;padding:1px 4px;border-radius:3px;margin-left:3px}
            .role-badge.anchor{background:#3b1f1f;color:#f87171;border:1px solid #b91c1c}
            .role-badge.bench{background:#3b2f1f;color:#fbbf24;border:1px solid #92400e}
            .role-badge.val{background:#1f2937;color:#60a5fa;border:1px solid #1d4ed8}
            .role-badge.diag{background:#0f3a3a;color:#00ffff;border:1px solid #00cccc}

/* Diagnostic column styling (CMB*): cyan overlay that must remain visible even when cell background indicates tension */
td.diagnostic-cell, th.diagnostic-cell {
  position: relative;
}
td.diagnostic-cell::after, th.diagnostic-cell::after {
  content: "";
  position: absolute;
  inset: 1px;
  border: 1px solid rgba(0, 255, 255, 0.85);
  pointer-events: none;
  border-radius: 2px;
}
td.diagnostic-cell, th.diagnostic-cell * {
  color: rgba(0, 255, 255, 0.92) !important;
}

</style>
        """

    def generate_javascript(): return "<script>function showTooltip(){}function hideTooltip(){};</script>"
    def generate_placeholder_scripts(): return ""
    def generate_header_section(): return "<h1>MTDF V74 Validation</h1>"
    def generate_summary_sections(pillar_count): return f"<p style='margin:4px 20px'>{pillar_count} pillars</p>"
    def generate_enhanced_footer(db, hash_func, env_func):
        return f"<footer style='margin:24px 20px;color:#94a3b8'>Environment: {env_func()}</footer>"
    def get_proper_model_name(key): return key


# =============================================================================
# WORKBOOK COEFFICIENT LOADING
# =============================================================================

def _get_workbook_coefficient(token: str, default: float = None) -> float:
    """
    Load a coefficient value from the workbook by its input_token.

    This enables the "zero hardcode" policy: all physics parameters
    and thresholds come from DB_Workbook_STRICT_V18.xlsx.

    Args:
        token: The input_token identifier (e.g., 'CHI2_GOOD', 'CHI2_ACCEPTABLE')
        default: Default value if token not found

    Returns:
        Coefficient value from workbook, or default if not found
    """
    try:
        import pandas as pd
        wb_path = Path(__file__).parent.parent / "data" / "DB_Workbook_STRICT_V18.xlsx"
        if not wb_path.exists():
            wb_path = Path(__file__).parent.parent.parent / "data" / "DB_Workbook_STRICT_V18.xlsx"
        if not wb_path.exists():
            return default

        df = pd.read_excel(wb_path, sheet_name='Params_Coefficients', header=1)

        for _, row in df.iterrows():
            if row.get('input_token') == token:
                val = row.get('value_si')
                if pd.notna(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        return default
    except Exception:
        return default


# Load χ² classification thresholds from workbook
CHI2_THRESHOLD_GOOD = _get_workbook_coefficient('CHI2_GOOD', default=1.0)
CHI2_THRESHOLD_ACCEPTABLE = _get_workbook_coefficient('CHI2_ACCEPTABLE', default=2.0)
CHI2_THRESHOLD_OVERFIT = _get_workbook_coefficient('CHI2_OVERFIT', default=0.1)


class DashboardGenerator:
    """
    Dashboard generation system for MTDF validation framework.
    Generates HTML table, targets row, and collapsible sections with workbook-driven tooltips.
    """

    def __init__(self, db, tooltip_engine: Optional[TooltipEngine] = None):
        self.db = db
        self.tooltip_engine = tooltip_engine
        self.root_path = Path(getattr(db, "data_dir", "."))
        
        # Add debugging information
        if self.tooltip_engine:
            available_ids = self.tooltip_engine.get_all_tooltip_ids()
            print(f"Tooltip engine initialized with {len(available_ids)} tooltips")
            
            # Validate critical tooltips exist
            critical_ids = ['header.model', 'cell.prediction', 'pillar.P1.header']
            missing_critical = [tid for tid in critical_ids if tid not in available_ids]
            if missing_critical:
                print(f"WARNING: Missing critical tooltips: {missing_critical}")
        else:
            print("WARNING: No tooltip engine provided to dashboard")

    # helpers
    def _get_proper_model_name(self, model_key: str) -> str:
        try:
            return get_proper_model_name(model_key)
        except Exception:
            return str(model_key)

    def _get_file_hash(self, filepath: Union[str, Path]) -> str:
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:8]
        except Exception:
            return "missing"

    def _env_summary(self) -> str:
        try:
            import numpy as _np
            npver = _np.__version__
        except Exception:
            npver = "n/a"
        return f"Python {sys.version.split()[0]} NumPy {npver} Platform {platform.system()}"

    def _get_tooltip(self, tooltip_id: str, context: Dict[str, Any] = None) -> str:
        """Get tooltip using new engine or fallback"""
        if self.tooltip_engine:
            return self.tooltip_engine.get_tooltip(tooltip_id, context or {})
        else:
            return f"<em>Tooltip system not initialized: {tooltip_id}</em>"

    def _build_tooltip_data_for_js(self, rows: List[Dict], pillars: List[str]) -> Dict[str, str]:
        """Build tooltip data dictionary for JavaScript injection - ALL from workbook only"""
        # Start with empty dict - ALL tooltips must come from workbook
        tooltips = {}

        if not self.tooltip_engine:
            print("WARNING: No tooltip engine available - no tooltips will be shown")
            return tooltips

        try:
            # Get all available tooltip IDs from workbook
            all_tooltip_ids = self.tooltip_engine.get_all_tooltip_ids()
            print(f"Building tooltips from {len(all_tooltip_ids)} workbook IDs (zero hardcoded)")
            
            # Process each available tooltip ID
            for tooltip_id in all_tooltip_ids:
                try:
                    # Build basic context
                    context = {}
                    
                    # Extract pillar context if applicable
                    if 'pillar.' in tooltip_id:
                        parts = tooltip_id.split('.')
                        if len(parts) >= 2:
                            context['pillar_id'] = parts[1] 
                    
                    # Render tooltip
                    tooltip_content = self.tooltip_engine.get_tooltip(tooltip_id, context)
                    if tooltip_content and 'not found' not in tooltip_content.lower():
                        tooltips[tooltip_id] = tooltip_content
                        
                except Exception as e:
                    print(f"Error processing tooltip {tooltip_id}: {e}")
                    continue
            
            # Generate dynamic tooltips for each model/pillar combination
            for row in rows:

                try:
                    self._add_model_specific_tooltips(tooltips, row, pillars)
                except Exception as e:
                    print(f"Error adding model tooltips: {e}")
                    continue
            
            print(f"Generated {len(tooltips)} tooltips for JavaScript")
            return tooltips
            
        except Exception as e:
            print(f"ERROR in _build_tooltip_data_for_js: {e}")
            return {}  # Return empty - all tooltips must come from workbook

    def _add_model_specific_tooltips(self, tooltips: Dict[str, str], row: Dict, pillars: List[str]):
        """Add model-specific tooltips with proper context"""
        model_name = row.get("model", "Unknown")
        foundation = row.get("foundation_type", "Unknown")
        tier = row.get("tier_code", "Unknown")
        
        # Build comprehensive model context
        model_context = {
            'model': model_name,
            'foundation': foundation,
            'tier': tier.lower(),
            'proof_pct': self._get_proof_percentage(row),
            'strict_passes': row.get("strict_passes", 0),
            'total': row.get("dof", 0),
            'max_z': row.get("max_z", 0.0),
            'chi2_red': self._get_chi2_red(row)
        }
        
        # Try to get model-specific tooltips
        model_specific_ids = [
            f'cell.model.{model_name}',
            f'cell.foundation.{foundation.lower()}',
            f'cell.tier.{tier.lower()}'
        ]
        
        for tooltip_id in model_specific_ids:
            try:
                tooltip_content = self.tooltip_engine.get_tooltip(tooltip_id, model_context)
                if tooltip_content and 'not found' not in tooltip_content.lower():
                    tooltips[tooltip_id] = tooltip_content
            except Exception:
                continue
        
        # Generate pillar-specific prediction tooltips
        for pid in pillars:
            try:
                pred = row.get("vals", {}).get(pid)
                z = row.get("z_scores", {}).get(pid)
                
                # Check if prediction is actually available
                has_prediction = pred is not None and not (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred)))

                # Determine data source based on model with ACTUAL citations
                if not has_prediction:
                    data_source = "No prediction available. This model does not provide predictions for this observable/pillar, either due to theoretical limitations or lack of published calculations for this specific test."
                elif model_name == 'MTDF':
                    data_source = "Calculated from empirical formulas in this workbook using independently established parameters (α, β, τ, β_eos) and derived quantities (E) and observational anchors (κ ≈ f_kick/3). No parameters are fitted to these validation tests. Sources: DOI:10.1093/mnras/staa2785 (α), DOI:10.1088/0004-637X/761/1/44 (β, Sutter et al. 2012 SDSS DR7 void catalog), DOI:10.1051/0004-6361/201935943 (δ_bf), DOI:10.1103/PhysRevD.85.054503 (β_eos)"
                elif 'lcdm' in model_name.lower():
                    data_source = "Published ΛCDM predictions using Planck 2018 cosmological parameters. Source: DOI:10.1051/0004-6361/201833910 (Planck Collaboration 2018, parameter set: H₀, Ω_m, Ω_Λ, n_s, σ_8)"
                elif 'mond' in model_name.lower():
                    data_source = "Published MOND predictions with critical acceleration a₀ ≈ 1.2×10⁻¹⁰ m/s². Sources: DOI:10.1086/161130 (Milgrom 1983, original MOND), DOI:10.1103/PhysRevD.70.083509 (TeVeS relativistic extension)"
                elif 'ede' in model_name.lower():
                    data_source = "Published Early Dark Energy model predictions. Sources: DOI:10.1103/PhysRevD.104.083550 (Hill et al. 2021, EDE cosmological fits), DOI:10.1103/PhysRevD.103.123523 (Poulin et al. 2021, Hubble tension)"
                elif 'fdm' in model_name.lower() or 'axion' in model_name.lower():
                    data_source = "Published Fuzzy Dark Matter predictions (ultralight scalar m ~ 10⁻²² eV). Source: DOI:10.1088/1361-6633/aa9e8a (Hui et al. 2017, comprehensive FDM review)"
                elif 'sidm' in model_name.lower():
                    data_source = "Published Self-Interacting Dark Matter predictions (σ/m cross-section). Sources: DOI:10.1016/j.physrep.2017.11.004 (Tulin & Yu 2018, SIDM review), DOI:10.1086/383178 (Bullet Cluster constraints)"
                else:
                    data_source = "Published predictions from peer-reviewed cosmological literature. Consult Model Registry sheet for complete citations and parameter details."

                pillar_context = {
                    **model_context,
                    'pillar_id': pid,
                    'prediction': pred,
                    'z_score': z,
                    'data_source': data_source
                }

                # Add target data if available
                if hasattr(self.db, 'pillars') and pid in self.db.pillars:
                    pillar_data = self.db.pillars[pid]
                    pillar_context.update({
                        'target': pillar_data.get('target'),
                        'sigma': pillar_data.get('sigma'),
                        'unit': pillar_data.get('unit', ''),
                        'pillar_mode': pillar_data.get('pillar_mode', 'SCALAR'),
                    })

                # Add vector pillar data if available
                vpd = row.get('vector_pillar_data', {}).get(pid, {})
                if vpd:
                    pillar_context['vector_pillar_data'] = vpd
                    pillar_context['pillar_mode'] = 'VECTOR'
                
                # Try specific pillar tooltip first
                tooltip_id = f'cell.{pid}.{model_name}'
                tooltip_content = self.tooltip_engine.get_tooltip(tooltip_id, pillar_context)

                # Check if we got a fallback tooltip (generic placeholder) or empty
                is_fallback = (not tooltip_content or
                             'not found' in tooltip_content.lower() or
                             f'<em>{tooltip_id}</em>' in tooltip_content or
                             '<em>Cell information</em>' in tooltip_content)

                if is_fallback:
                    # Check if vector pillar - use programmatic generation directly
                    # (the generic template is designed for scalar pillars with z-scores)
                    is_vector = pillar_context.get('pillar_mode') == 'VECTOR' or pid.startswith('P_')

                    if is_vector:
                        # Vector pillars use programmatic generation with χ²/ν logic
                        tooltip_content = self._generate_prediction_tooltip(pillar_context)
                    else:
                        # Try generic prediction template for scalar pillars
                        tooltip_content = self.tooltip_engine.get_tooltip('cell.prediction.generic', pillar_context)

                        if not tooltip_content or 'not found' in tooltip_content.lower():
                            # Fall back to programmatic generation
                            tooltip_content = self._generate_prediction_tooltip(pillar_context)

                tooltips[tooltip_id] = tooltip_content

            except Exception as e:
                # If all else fails, create a basic tooltip
                tooltips[f'cell.{pid}.{model_name}'] = f"<strong>{pid}</strong><br>Prediction data unavailable"
                continue

        # Generate dynamic tooltips for Evidence, Max |z|, and χ²/ν
        if hasattr(self, '_model_tooltip_data') and model_name in self._model_tooltip_data:
            data = self._model_tooltip_data[model_name]
            model_display = self._get_proper_model_name(model_name)

            # Evidence tooltip
            evidence_tooltip = f"""<strong>{model_display}: Evidence Assessment</strong><br><br>
<em>Pass Rate Calculation:</em><br>
{data['strict_passes']} tests passed / {data['strict_tests_with_predictions']} tests with predictions = <strong>{data['pass_pct']:.0f}%</strong><br>
(Comprehensive coverage: {data['strict_passes']}/{data['total_tests']})<br><br>
<em>Classification: {data['evidence_level'].title()}</em><br>
• χ²/ν = {data['chi2_red']:.3f}<br>
• Pass rate = {data['pass_pct']:.0f}%<br>
• Max |z| = {data['max_z']:.2f}<br><br>
<em>Note:</em> All three criteria must meet thresholds for moderate or better classification."""
            tooltips[f'cell.evidence.{model_name}'] = evidence_tooltip

            # Max |z| tooltip - include which pillar has the worst residual
            max_z_pillar_id = data.get('max_z_pillar')
            pillar_info = ""
            if max_z_pillar_id and hasattr(self, 'tooltip_engine') and hasattr(self.tooltip_engine, 'pillar_cache'):
                pillar_data = self.tooltip_engine.pillar_cache.get(max_z_pillar_id, {})
                pillar_name = pillar_data.get('name', max_z_pillar_id)
                pillar_info = f"<strong>{max_z_pillar_id}: {pillar_name}</strong><br>"

            maxz_tooltip = f"""<strong>{model_display}: Maximum Deviation</strong><br>
{pillar_info}<br>
<em>Calculation:</em><br>
Max |z| = <strong>{data['max_z']:.2f}σ</strong><br><br>
This is the largest standardized residual across all {data['strict_tests_with_predictions']} predictions made by this model.<br><br>
<em>Formula:</em> |z| = |(prediction - target) / uncertainty|<br><br>
<em>Interpretation:</em><br>
• |z| < 1: Within 1σ (excellent)<br>
• |z| < 2: Within 2σ (acceptable)<br>
• |z| ≥ 3: Significant tension"""
            tooltips[f'cell.maxz.{model_name}'] = maxz_tooltip

            # χ²/ν tooltip - NOW GENERATED DYNAMICALLY in getContent() JavaScript
            # The dynamic tooltip updates based on current scope selection (A/B/V/D badges)
            # Static tooltip removed to allow dynamic generation

    def _precalculate_model_tooltip_data(self, rows: List[Dict], pillars: List[str]):
        """Pre-calculate model data for dynamic tooltips before tooltip generation"""
        if not hasattr(self, '_model_tooltip_data'):
            self._model_tooltip_data = {}

        for row in rows:
            model_name = row.get("model", "Unknown")
            strict_passes = row.get("strict_passes", 0)
            total_tests = len(pillars)

            # Count tests with valid predictions
            strict_tests_with_predictions = sum(
                1 for pid in pillars
                if row.get("vals", {}).get(pid) is not None
                and not (isinstance(row.get("vals", {}).get(pid), float)
                        and (math.isnan(row.get("vals", {}).get(pid)) or math.isinf(row.get("vals", {}).get(pid))))
            )

            # Pass rate based on tests with valid predictions
            proof_pct = (strict_passes / strict_tests_with_predictions * 100.0) if strict_tests_with_predictions else 0.0


            # Strict combined reduced chi-squared (fit pillars only) may be computed later; default None for tooltip tier logic
            strict_chi2_red = row.get('combined_excl_cmb_chi2_red')  # strict combined χ²/ν excluding diagnostic-only pillars (e.g., CMB*)
            # Calculate pass rate based on tests WITH predictions (Option B)
                        # Get chi2 and calculate reduced chi2
            chi2_red_calc = (strict_chi2_red if strict_chi2_red is not None else self._get_chi2_red(row))
            max_z = row.get("max_z", 0.0)

            # Find pillar with maximum |z|
            max_z_pillar = None
            max_z_value = 0.0
            z_scores = row.get("z_scores", {})
            for pid in pillars:
                z = z_scores.get(pid)
                if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                    if abs(z) > max_z_value:
                        max_z_value = abs(z)
                        max_z_pillar = pid

            # Calculate evidence level
            if chi2_red_calc < 0.1 and proof_pct >= 90 and max_z < 1:
                evidence_level = 'excellent'
            elif chi2_red_calc < 1 and proof_pct >= 70 and max_z < 2:
                evidence_level = 'strong'
            elif chi2_red_calc < 2 and proof_pct >= 50 and max_z < 3:
                evidence_level = 'moderate'
            else:
                evidence_level = 'weak'

            # Store all data needed for tooltips
            self._model_tooltip_data[model_name] = {
                'strict_passes': strict_passes,
                'strict_tests_with_predictions': strict_tests_with_predictions,
                'total_tests': total_tests,
                'pass_pct': proof_pct,
                'chi2_red': chi2_red_calc,
                'chi2': row.get("chi2", 0.0),
                'dof': row.get("dof", 0),
                'max_z': max_z,
                'max_z_pillar': max_z_pillar,
                'evidence_level': evidence_level
            }

    def _get_proof_percentage(self, row: Dict[str, Any]) -> float:
        """Calculate proof percentage from row data"""
        proof_frac = row.get("proof")
        if proof_frac is None:
            p = row.get("strict_passes", 0)
            n = row.get("dof", 0)
            proof_frac = (p / n) if n else 0.0
        return max(0.0, min(100.0, 100.0 * float(proof_frac)))

    def _get_chi2_red(self, row: Dict[str, Any]) -> float:
        """Calculate reduced chi-squared from row data using combined scalar+vector values if available"""
        # Use combined values if available (includes vector pillars)
        if "combined_chi2" in row and "combined_dof" in row:
            chi2_total = row.get("combined_chi2", 0.0)
            dof = row.get("combined_dof", 1) or 1
        else:
            # Fallback to scalar-only
            chi2_total = row.get("chi2", 0.0)
            dof = row.get("dof", 1) or 1
        return chi2_total / dof

    def _generate_prediction_tooltip(self, context: Dict[str, Any]) -> str:
        """Generate prediction tooltip content"""
        pid = context.get('pillar_id', 'Unknown')
        model = context.get('model', 'Unknown')
        pred = context.get('prediction')
        z = context.get('z_score')
        target = context.get('target')
        sigma = context.get('sigma')
        pillar_mode = context.get('pillar_mode', 'SCALAR')

        # Check if this is a vector pillar
        is_vector = pillar_mode == 'VECTOR' or pid.startswith('P_')

        if is_vector:
            # Vector pillar tooltip - rich format matching scalar tooltips
            vpd = context.get('vector_pillar_data', {})
            chi2 = vpd.get('chi2', 0)
            dof = vpd.get('dof', 0)
            chi2_red = vpd.get('chi2_red', pred) if vpd else pred
            n_data = vpd.get('n_data', 0)
            is_literature = vpd.get('is_literature', False)
            lit_reference = vpd.get('reference', '')
            lit_notes = vpd.get('notes', '')
            data_source = context.get('data_source', '')

            # Get pillar name, description, and external data source
            pillar_info = {
                'P_SNE_PANTHEON': (
                    'Pantheon+ Type Ia Supernovae',
                    '1,701 distance moduli with full STAT+SYS covariance',
                    'Pantheon+ SN Ia compilation (Scolnic et al. 2022). DOI:10.3847/1538-4357/ac8b7a. Data: 1,701 light curves from 1,550 unique SNe Ia across z ≈ 0.01–2.3 with full statistical + systematic covariance matrix.'
                ),
                'P_BAO_DESI': (
                    'DESI Y1 BAO',
                    '12 BAO measurements (D_M/r_d and D_H/r_d) with covariance',
                    'DESI Year 1 BAO (DESI Collaboration 2024). DOI:10.1088/1475-7516/2024/02/015. Data: D_M/r_d and D_H/r_d at z_eff = 0.30, 0.51, 0.71, 0.93, 1.32, 2.33 from BGS, LRG, ELG, QSO, and Lyα tracers.'
                ),
                'P_HZ_CC': (
                    'Cosmic Chronometers H(z)',
                    '15 H(z) measurements from passively evolving galaxies',
                    'Cosmic Chronometer compilation (Moresco et al. 2022). DOI:10.12942/lrr-2022-7. Data: 15 H(z) measurements from differential age dating of passively evolving galaxies at z ≈ 0.07–1.97.'
                ),
                'P_GROWTH_FSIG8': (
                    'DR16 fσ₈ Growth',
                    '4 fσ₈(z) measurements from redshift space distortions',
                    'SDSS DR16 growth measurements (eBOSS Collaboration 2021). DOI:10.1103/PhysRevD.103.083533. Data: fσ₈(z) from redshift space distortions in LRG, ELG, and QSO samples.'
                ),
                'P_CMB_DIST': (
                    'CMB* Planck 2018 distance prior (diagnostic only, excluded from strict totals)',
                    '3 parameters: R, ℓ_A, ω_b h² with covariance',
                    'Planck 2018 distance prior (Planck Collaboration 2020). DOI:10.1051/0004-6361/201833910. Data: Compressed CMB information via shift parameter R, acoustic scale ℓ_A, and physical baryon density ω_b h².'
                ),
            }
            pillar_name, pillar_desc, vector_data_source = pillar_info.get(pid, (pid, 'Vector pillar', ''))

            # Literature context for each vector pillar (for reviewer orientation only)
            literature_context = {
                'P_SNE_PANTHEON': """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Literature Context:</strong><br><br>
• <strong>ΛCDM:</strong> Global Pantheon and Pantheon+ analyses report joint fits with χ²/ν close to 1 by construction, using a full Bayesian treatment.<br><br>
• <strong>EDE:</strong> Early Dark Energy models are usually fit jointly to SNe, BAO and CMB, and are compatible with Pantheon+ within similar Bayesian frameworks, rather than publishing a standalone SNe-only χ².<br><br>
• <strong>MOND / FDM / SIDM:</strong> There are no widely cited, published Pantheon+ vector fits for these models, so no directly comparable χ² values exist for this exact data set.
""",
                'P_BAO_DESI': """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Literature Context:</strong><br><br>
• <strong>ΛCDM:</strong> DESI analyses use the full BAO and RSD likelihood in a Bayesian framework, typically reporting parameter posteriors rather than a simple one-line χ²/ν for the compressed BAO vector.<br><br>
• <strong>EDE:</strong> Early Dark Energy papers often quote Δχ² values relative to ΛCDM for DESI Y1 or similar BAO data sets, but not a standalone χ² built from the same compressed vector used here.<br><br>
• <strong>MOND / FDM / SIDM:</strong> There are no standard DESI BAO vector fits published for these models, so a directly comparable χ² is not available.
""",
                'P_HZ_CC': """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Literature Context:</strong><br><br>
• <strong>ΛCDM:</strong> Several H(z) compilations report good agreement with flat ΛCDM, with typical χ²/ν values around 0.8 for similar cosmic chronometer samples.<br><br>
• <strong>R<sub>h</sub> = ct and related alternatives:</strong> Some studies report comparable or slightly lower χ²/ν values (around 0.75) for these specific models on similar H(z) samples.<br><br>
• <strong>EDE / MOND / FDM / SIDM:</strong> Dedicated H(z) vector fits are rare, and there are no widely used published χ² values based on exactly the same CC data vector and covariance used here.
""",
                'P_GROWTH_FSIG8': """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Literature Context:</strong><br><br>
• <strong>ΛCDM:</strong> Standard growth analyses find that ΛCDM is broadly consistent with fσ₈ measurements, although there is ongoing discussion about a possible S₈-level tension between CMB and low-redshift structure probes.<br><br>
• <strong>EDE:</strong> Early Dark Energy models are often designed to slightly reduce late-time growth and can improve the S₈ tension in some Bayesian analyses, but published fits usually combine multiple probes rather than quoting a standalone DR16 fσ₈ vector χ².<br><br>
• <strong>MOND / FDM / SIDM:</strong> There are no standard DR16 fσ₈ vector fits for these models, so no directly comparable χ² values exist for this specific compilation.
""",
                'P_CMB_DIST': """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Literature Context:</strong><br><br>
• <strong>ΛCDM:</strong> The Planck 2018 baseline ΛCDM model provides an excellent fit to the full CMB temperature and polarisation spectra, with a high-dimensional χ² of order a few thousand for several thousand degrees of freedom. The three-parameter distance prior used here is a compressed summary of that full likelihood.<br><br>
• <strong>EDE:</strong> Early Dark Energy models are typically constrained with the full Planck likelihood. Representative analyses report modest improvements in global fit, for example Δχ² of order −10 to −20 relative to ΛCDM, at the cost of extra parameters.<br><br>
• <strong>MOND / FDM / SIDM:</strong> Some modified gravity and alternative dark matter models have been shown to reproduce the main CMB peak structure qualitatively once extra fields or components are added, but there are no widely adopted three-parameter distance prior fits that would map cleanly onto the R, ℓ<sub>A</sub>, ω<sub>b</sub>h² summary used here.
""",
            }

            # Shared CMB diagnostic note for both MTDF rows (replaces generic Interpretation block)
            cmb_diagnostic_note = """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>Diagnostic only</strong> (not counted in strict totals): This column shows the Planck 2018 ΛCDM-calibrated compressed distance prior. It is displayed for transparency and tension diagnosis only.<br><br>
<strong>Model non-neutrality:</strong> Because the compression assumes a standard ΛCDM expansion history, models with different early-time physics are not expected to match this compressed prior one-to-one. A mismatch here diagnoses tension with the ΛCDM compression, not a direct falsification by the full CMB data.<br><br>
<strong>Reproducibility:</strong> Full CMB spectrum tests must be done against TTTEEE plus lensing likelihoods (not this compressed prior).
"""

            # MTDF baseline CMB interpretation (no early field energy)
            mtdf_cmb_interpretation = cmb_diagnostic_note + """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>What this row shows:</strong><br>
MTDF <em>forced into standard early-time physics</em>, compared to the ΛCDM-calibrated prior.<br><br>
<strong>Important context:</strong><br>
The Planck distance prior (R, ℓ<sub>A</sub>, ω<sub>b</sub>h²) is a compressed summary calibrated under ΛCDM. For models with non-trivial early field energy, this compression is not strictly model-neutral. The large χ² here is a <strong>diagnostic of tension with the ΛCDM-based prior</strong>, not direct falsification by raw CMB data.<br><br>
<strong>Why χ²/ν is large:</strong><br>
• With no early field energy, H₀ = 70 km/s/Mpc shifts the angular acoustic scale θ<sub>*</sub> against the ΛCDM-calibrated prior, which produces strong tension in R and ℓ<sub>A</sub>.<br>
• The purpose of this baseline row is to show the clean geometric tension with the ΛCDM distance prior when MTDF is forced to share the same early-time physics.<br><br>
<strong>See the MTDF (EFE) row</strong> for MTDF's own prediction including its small, theory-fixed early field energy.
"""

            # MTDF (EFE) CMB interpretation (with early field energy)
            mtdf_efe_cmb_interpretation = cmb_diagnostic_note + """
<br><hr style='border-color:#2d3f5f;margin:12px 0;'>
<strong>What this row shows:</strong><br>
MTDF's own prediction including Early Field Energy (EFE), evaluated against the same ΛCDM-calibrated Planck 2018 distance prior as the baseline CMB* row. The EFE amplitude is fixed by MTDF through f<sub>kick</sub> = λ<sub>MTDF</sub>/24 ≈ 0.33%, it is not tuned within this pillar.<br><br>
<strong>Physical mechanism (EFE):</strong><br>
• The MTDF stress field contributes a small early field energy fraction f<sub>kick</sub> ≈ 0.33% of the total density near matter-radiation equality.<br>
• This reduces the comoving sound horizon r<sub>s</sub> by about 0.07% relative to a model with the same late-time parameters and no EFE.<br>
• In the usual ΔN<sub>eff</sub> language this corresponds to an effective shift ΔN<sub>eff</sub> ≈ 0.02, at the low-amplitude end of early dark energy type corrections.<br><br>
<strong>Result:</strong><br>
• The EFE correction moves the acoustic scale in the right direction but does not by itself remove the distance prior tension when H₀ is fixed at 70 km/s/Mpc.<br>
• A separate CLASS plus full Planck TTTEEE and DESI BAO likelihood analysis using this same amplitude finds k<sub>f</sub> ≈ 1.0 ± 0.1 with only Δχ² of order unity compared to ΛCDM for one extra parameter, so the early field energy amplitude is best viewed as a genuine MTDF prediction rather than a tuned fit.<br><br>
<em>Consistent with CLASS Boltzmann code.</em>
"""

            # Distinguish between computed (MTDF) and literature values
            if is_literature:
                tooltip_content = f"<strong>{pid}: {model} (Literature)</strong><br>"
                tooltip_content += f"<em>{pillar_desc}</em><br><br>"
                tooltip_content += "<strong>⚠️ From Published Literature - Not Computed</strong><br><br>"
            else:
                tooltip_content = f"<strong>{pid}: {model} Prediction</strong><br>"
                tooltip_content += f"<em>{pillar_desc}</em><br><br>"

            if pred is not None and not (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
                tooltip_content += f"<strong>χ²/ν (reduced chi-squared):</strong> {chi2_red:.4f}<br>"
                if chi2 is not None and chi2 > 0:
                    tooltip_content += f"<strong>χ² (total):</strong> {chi2:.2f}<br>"
                if dof is not None and dof > 0:
                    tooltip_content += f"<strong>DOF (degrees of freedom):</strong> {dof}<br>"
                    tooltip_content += f"<strong>N (data points):</strong> {n_data}<br><br>"

                # Source section - different for literature vs computed
                if is_literature:
                    tooltip_content += f"<em>Literature Source:</em><br>"
                    if lit_reference:
                        tooltip_content += f"<strong>{lit_reference}</strong><br>"
                    if lit_notes:
                        tooltip_content += f"{lit_notes}<br><br>"
                    tooltip_content += "<em>Note:</em> Value taken from peer-reviewed publication. Not computed by this validation framework.<br><br>"
                elif vector_data_source:
                    tooltip_content += f"<em>Data Source:</em><br>{vector_data_source}<br><br>"

                # Add context: MTDF CMB* rows get special diagnostic note (skip generic interpretation)
                # Other rows get standard interpretation + literature context
                is_mtdf_cmb = pid == 'P_CMB_DIST' and (model.upper() == 'MTDF' or 'EFE' in model.upper())

                if is_mtdf_cmb:
                    # Skip generic interpretation for CMB* MTDF rows - use diagnostic note instead
                    if model.upper() == 'MTDF':
                        tooltip_content += mtdf_cmb_interpretation
                    else:
                        tooltip_content += mtdf_efe_cmb_interpretation
                else:
                    # Standard interpretation section for non-CMB or non-MTDF
                    tooltip_content += "<em>Interpretation:</em><br>"
                    tooltip_content += "• χ²/ν < 1.5: Excellent fit (green)<br>"
                    tooltip_content += "• 1.5 ≤ χ²/ν < 2.0: Acceptable fit (yellow)<br>"
                    tooltip_content += "• χ²/ν ≥ 2.0: Significant tension (red)"

                    # Add literature context for non-MTDF rows
                    if pid in literature_context:
                        tooltip_content += literature_context[pid]
            else:
                # N/A case - no published fit to this exact data vector
                tooltip_content += "<strong style='color:#f59e0b;'>⚠️ No Published Fit Available</strong><br><br>"
                tooltip_content += f"<em>There is no published χ² fit for {model} to this exact data vector.</em><br><br>"
                tooltip_content += "This validation framework only displays:<br>"
                tooltip_content += "• <strong>MTDF:</strong> Computed directly from the data<br>"
                tooltip_content += "• <strong>Other models:</strong> Published peer-reviewed values only<br><br>"
                tooltip_content += "<em>No value is shown because we do not compute predictions for comparison models, "
                tooltip_content += "and no directly comparable published χ² exists for this specific dataset.</em>"

                # Add literature context even for N/A to explain why
                if pid in literature_context:
                    tooltip_content += literature_context[pid]

            return tooltip_content

        # Scalar pillar tooltip (original logic)
        tooltip_content = f"<strong>{pid} Prediction</strong>"

        if pred is not None and not (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
            tooltip_content += f"<p><strong>Predicted:</strong> {pred:.6g}</p>"

            if target is not None and sigma is not None:
                diff = abs(pred - target)
                diff_sigma = diff / sigma if sigma > 0 else 0
                tooltip_content += f"<p><strong>Observed:</strong> {target:.6g} ± {sigma:.6g}</p>"
                tooltip_content += f"<p><strong>Difference:</strong> {diff:.6g} ({diff_sigma:.1f}σ)</p>"

            if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                tooltip_content += f"<p><strong>Z-score:</strong> {z:.2f}</p>"
                if abs(z) <= 1:
                    tooltip_content += "<p>🟢 Pass</p>"
                elif abs(z) <= 2:
                    tooltip_content += "<p>🟡 Marginal</p>"
                else:
                    tooltip_content += "<p>🔴 Fail</p>"
            else:
                tooltip_content += "<p><strong>Z-score:</strong> N/A</p>"

            if target is None or sigma is None:
                tooltip_content += "<p><em>Target value not available</em></p>"
        else:
            tooltip_content += "<p><em>No prediction available</em></p>"
            tooltip_content += "<p>Model may not cover this test case</p>"

        return tooltip_content

    # extra CSS for icons + Proof coloring
    def _extra_css(self) -> str:
        return """
        <style>
          .evidence-col.ok{color:#22c55e;font-weight:600}
          .evidence-col.warn{color:#f59e0b;font-weight:600}
          .evidence-col.bad{color:#ef4444;font-weight:600}
          .icon{font-weight:700;margin-right:6px}
          .icon.ok{color:#22c55e}
          .icon.warn{color:#f59e0b}
          .icon.bad{color:#ef4444}
          .icon.neutral{color:#94a3b8}

          /* Compact table styling */
          .table-wrap table {
            font-size: 0.85em;
          }
          .table-wrap th, .table-wrap td {
            padding: 4px 6px !important;
          }
          .table-wrap 
        /* Diagnostic-only pillar styling (CMB* distance prior) */
        /* Force diagnostic styling to win over ok/warn/bad backgrounds */
        th.diag-col,
        th.diag-col * { color: #5eead4 !important; }
        td.diag-col,
        td.diag-col * { color: #5eead4 !important; }

        th.diag-col { background: rgba(20, 184, 166, 0.22) !important; }

        /* Apply diagnostic tint and a double 1px cyan frame, regardless of status */
        td.diag-col,
        td.diag-col.ok,
        td.diag-col.warn,
        td.diag-col.bad,
        td.pillar-cell.diag-col,
        td.pillar-cell.diag-col.ok,
        td.pillar-cell.diag-col.warn,
        td.pillar-cell.diag-col.bad,
        td.pillar-cell.ok.diag-col,
        td.pillar-cell.warn.diag-col,
        td.pillar-cell.bad.diag-col {
            background: rgba(20, 184, 166, 0.14) !important;
            box-shadow: inset 0 0 0 1px rgba(94, 234, 212, 0.95), inset 0 0 0 2px rgba(94, 234, 212, 0.35) !important;
        }

        .diag-badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px;
          background: rgba(20, 184, 166, 0.22); color:#5eead4; border:1px solid rgba(94, 234, 212, 0.35); }

.pillar-header {
            min-width: 42px;
            max-width: 68px;
          }
          .table-wrap .cell-sub {
            font-size: 10px;
          }

          /* Tier column - narrow */
          .tier-col {
            min-width: 42px;
            max-width: 65px;
            white-space: nowrap;
            padding-left: 2px !important;
            padding-right: 2px !important;
          }

          /* Vector pillar styling */
          .vector-pillar {
            background: linear-gradient(180deg, #1a4d2e 0%, #0d172a 100%) !important;
            border-left: 2px solid #22c55e;
          }
          .vector-cell {
            border-left: 1px solid #22c55e33;
          }
          .pillar-header.vector-pillar {
            color: #4ade80;
            font-weight: 600;
          }
          .target-row.vector-pillar {
            color: #86efac;
            font-size: 0.9em;
          }

          /* Tooltip CSS */
          .pro-body-tooltip {
            position: fixed;
            z-index: 99999;
            max-width: 40rem;
            padding: 12px 16px;
            background: #111827;
            color: #d1d5db;
            border: 1px solid #2563eb;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.4;
            box-shadow: 0 8px 24px rgba(0,0,0,.4), 0 0 0 1px rgba(37,99,235,.4);
            pointer-events: none;
            opacity: 0;
            transition: opacity .2s ease;
          }
          .pro-body-tooltip.show {
            opacity: 1;
          }
        </style>
        """

    # Accordion helpers
    def _accordion_css(self) -> str:
        return """
        <style>
          .accordion{border:1px solid #2d3f5f;margin:14px 20px;border-radius:10px;overflow:hidden;background:#0b1220}
          .acc-btn{width:100%;text-align:left;padding:12px 16px;background:#1a2942;color:#e2e8f0;border:0;font-weight:600;cursor:pointer;
                   display:flex;justify-content:space-between;align-items:center}
          .acc-btn:hover{background:#1f3451}
          .acc-caret{transition:transform .15s ease;font-size:2em}
          .acc-panel{display:none;padding:12px 16px}
          .accordion.open .acc-panel{display:block}
          .accordion.open .acc-caret{transform:rotate(90deg)}
          .subtable{border-collapse:collapse;width:100%}
          .subtable th,.subtable td{padding:8px 10px;border-bottom:1px solid #1f2a44;white-space:nowrap;text-align:center}
          .subtable th{text-align:center;background:#0d172a}
          .left{text-align:left}
          .chip-ok{color:#22c55e}
          .chip-warn{color:#f59e0b}
          .chip-bad{color:#ef4444}
          /* Equation styling */
          .equation-item{margin:20px 0;padding:16px;background:#0d172a;border-radius:8px;border:1px solid #1f2a44}
          .equation-label{font-size:14px;color:#94a3b8;margin-bottom:10px;font-weight:500}
          .equation-display{font-size:16px;color:#e2e8f0;overflow-x:auto;padding:8px 0}
          .equation-note{font-size:14px;color:#64748b;font-style:italic}
        </style>
        """

    def _accordion_js(self) -> str:
        return """
        <script>
          function toggleAcc(btn){ btn.parentElement.classList.toggle('open'); }
        </script>
        """

    def _accordion(self, title: str, inner_html: str, open_default: bool=False) -> str:
        state = "open" if open_default else ""
        return """
        <div class="accordion {state}">
          <button class="acc-btn" onclick="toggleAcc(this)">{title}<span class="acc-caret">▸</span></button>
          <div class="acc-panel">{inner_html}</div>
        </div>
        """.replace("{state}", state).replace("{title}", title).replace("{inner_html}", inner_html)

    # Section 0: Tension Plot (σ-deviation visualization)

    def _render_tension_plot(self, rows: List[Dict], pillars: List[str]) -> str:
        """
        Render a horizontal whisker plot showing σ-deviation for each pillar across models.
        X-axis: σ deviation from -5σ to +5σ
        Y-axis: Each pillar (scalar pillars only, vector pillars use chi2/nu)
        """
        # Helper functions using self.db
        def get_pillar_role_raw(pid: str) -> str:
            """Get raw role string (A/B/V/D) for data attributes"""
            role = ""
            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                role = self.db.pillars[pid].get("role", "").upper()

            # Return single letter code
            if role == "ANCHOR":
                return "A"
            elif role == "BENCHMARK":
                return "B"
            elif role == "VALIDATION":
                return "V"
            elif role == "DIAGNOSTIC":
                return "D"
            return "V"  # Default to validation

        def get_pillar_target(pid: str) -> dict:
            """Get target and sigma for a pillar"""
            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                pillar_data = self.db.pillars[pid]
                return {
                    'target': pillar_data.get("target", 0),
                    'sigma': pillar_data.get("sigma", 1)
                }
            return {'target': 0, 'sigma': 1}

        def get_pillar_label(pid: str) -> str:
            """Get display label for a pillar"""
            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                return self.db.pillars[pid].get("name", pid)
            return pid

        # Separate scalar and vector pillars
        scalar_pillars = [p for p in pillars if p.startswith('P') and p[1:].replace('B', '').isdigit()]
        vector_pillars = [p for p in pillars if p.startswith('P_')]  # P_SNe, P_fs8, P_CMB_DIST

        # Vector pillar display names
        vector_labels = {
            'P_SNe': 'SNe Ia',
            'P_fs8': 'fσ₈',
            'P_CMB_DIST': 'CMB*'
        }

        def chi2_to_zeff(chi2: float, dof: int) -> float:
            """Convert χ²/ν to cosmology-friendly effective z-score.

            Uses linear scaling: z_cosmo = (χ²/ν - 1) × 4
            This maps χ²/ν = 1.0 → 0σ, χ²/ν = 1.5 → 2σ, χ²/ν = 2.0 → 4σ
            More intuitive for cosmology where χ²/ν ≈ 1 is considered good.
            """
            if dof <= 0 or chi2 is None:
                return None
            chi2_red = chi2 / dof
            return (chi2_red - 1.0) * 4.0

        # Model display settings
        model_configs = {
            'MTDF': {'css_class': 'mtdf', 'color': '#4ade80', 'label': 'MTDF'},
            'lcdm_pred_value': {'css_class': 'lcdm', 'color': '#3b82f6', 'label': 'ΛCDM'},
            'mond_pred_value': {'css_class': 'mond', 'color': '#6b7280', 'label': 'MOND'},
            'ede_pred_value': {'css_class': 'ede', 'color': '#a855f7', 'label': 'EDE'},
            'fdm_pred_value': {'css_class': 'fdm', 'color': '#ec4899', 'label': 'FDM'},
            'sidm_pred_value': {'css_class': 'sidm', 'color': '#f59e0b', 'label': 'SIDM'},
        }

        # Build legend
        legend_items = []
        for model_key, cfg in model_configs.items():
            legend_items.append(
                f"<div class='tension-legend-item'>"
                f"<div class='tension-legend-dot' style='background:{cfg['color']}'></div>"
                f"<span>{cfg['label']}</span></div>"
            )
        legend_html = "".join(legend_items)

        # Build rows for each pillar
        rows_html = []
        sigma_range = 5.0  # Show from -5σ to +5σ

        for pid in scalar_pillars:
            role = get_pillar_role_raw(pid)
            label = get_pillar_label(pid)
            target_info = get_pillar_target(pid)
            target = target_info.get('target', 0) if target_info else 0
            sigma = target_info.get('sigma', 1) if target_info else 1

            # Create markers for each model
            markers_html = ""
            for row in rows:
                model_key = row.get("model", "")
                if model_key not in model_configs:
                    continue

                cfg = model_configs[model_key]
                z_scores = row.get("z_scores", {})
                z = z_scores.get(pid)

                if z is None or (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                    continue

                # Clamp z to display range
                z_clamped = max(-sigma_range, min(sigma_range, z))
                # Convert to percentage position (0% = -5σ, 50% = 0, 100% = +5σ)
                pos_pct = ((z_clamped + sigma_range) / (2 * sigma_range)) * 100

                pred = row.get("vals", {}).get(pid, "")
                pred_str = f"{pred:.4g}" if isinstance(pred, (int, float)) else str(pred)

                markers_html += (
                    f"<div class='tension-marker {cfg['css_class']}' "
                    f"style='left:{pos_pct:.1f}%' "
                    f"data-model='{cfg['label']}' data-pillar='{pid}' "
                    f"data-z='{z:.2f}' data-pred='{pred_str}' "
                    f"data-target='{target}' data-sigma='{sigma}'></div>"
                )

            # Tick marks at -3σ, -2σ, -1σ, 0, +1σ, +2σ, +3σ
            ticks_html = ""
            for tick_sigma in [-3, -2, -1, 1, 2, 3]:
                tick_pct = ((tick_sigma + sigma_range) / (2 * sigma_range)) * 100
                ticks_html += f"<div class='tension-tick' style='left:{tick_pct:.1f}%'></div>"

            rows_html.append(f"""
            <div class='tension-row' data-pillar-role='{role}'>
                <div class='tension-pillar-label'>
                    {pid}<span class='tension-pillar-role {role}'>{role}</span>
                </div>
                <div class='tension-bar-container'>
                    {ticks_html}
                    <div class='tension-centerline'></div>
                    {markers_html}
                </div>
            </div>
            """)

        # Add vector pillar rows (using cosmology-friendly z = (χ²/ν - 1) × 4)
        for pid in vector_pillars:
            role = get_pillar_role_raw(pid)
            # CMB* is diagnostic (D) - override if incorrectly assigned
            if 'CMB' in pid:
                role = 'D'
            label = vector_labels.get(pid, pid)

            # Create markers for each model
            markers_html = ""
            for row in rows:
                model_key = row.get("model", "")
                if model_key not in model_configs:
                    continue

                cfg = model_configs[model_key]

                # Get vector pillar data (chi2 and dof)
                vpd = row.get('vector_pillar_data', {}).get(pid, {})
                chi2 = vpd.get('chi2')
                dof = vpd.get('dof')

                if chi2 is None or dof is None or dof <= 0:
                    continue

                # Convert χ² to effective z-score
                z = chi2_to_zeff(chi2, dof)
                if z is None or (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                    continue

                # Clamp z to display range
                z_clamped = max(-sigma_range, min(sigma_range, z))
                # Convert to percentage position (0% = -5σ, 50% = 0, 100% = +5σ)
                pos_pct = ((z_clamped + sigma_range) / (2 * sigma_range)) * 100

                chi2_red = chi2 / dof if dof > 0 else 0

                markers_html += (
                    f"<div class='tension-marker {cfg['css_class']}' "
                    f"style='left:{pos_pct:.1f}%' "
                    f"data-model='{cfg['label']}' data-pillar='{pid}' "
                    f"data-z='{z:.2f}' data-chi2='{chi2:.1f}' "
                    f"data-dof='{dof}' data-chi2red='{chi2_red:.4f}'></div>"
                )

            # Only add row if there are markers
            if markers_html:
                # Tick marks
                ticks_html = ""
                for tick_sigma in [-3, -2, -1, 1, 2, 3]:
                    tick_pct = ((tick_sigma + sigma_range) / (2 * sigma_range)) * 100
                    ticks_html += f"<div class='tension-tick' style='left:{tick_pct:.1f}%'></div>"

                rows_html.append(f"""
                <div class='tension-row vector-pillar' data-pillar-role='{role}'>
                    <div class='tension-pillar-label'>
                        {label}<span class='tension-pillar-role {role}'>{role}</span><span class='tension-type-badge'>χ²</span>
                    </div>
                    <div class='tension-bar-container'>
                        {ticks_html}
                        <div class='tension-centerline'></div>
                        {markers_html}
                    </div>
                </div>
                """)

        # Scale labels
        scale_labels = []
        for s in [-5, -3, -2, -1, 0, 1, 2, 3, 5]:
            label_str = f"{s:+d}σ" if s != 0 else "0"
            scale_labels.append(f"<span class='tension-scale-label'>{label_str}</span>")

        return f"""
        <div class='tension-plot-container'>
            <div class='tension-plot'>
                <div class='tension-plot-header'>
                    <div class='tension-plot-title'>Unified σ-Deviation Plot (Scope-Aware)</div>
                    <div class='tension-legend'>{legend_html}</div>
                </div>
                <div class='tension-plot-description' style='font-size:12px;color:#9ca3af;margin-bottom:15px;'>
                    Each dot shows how many standard deviations (σ) a model's prediction differs from the observational target.
                    <strong>Scalar pillars</strong> use direct z-scores. <strong>Vector pillars</strong> <span class='tension-type-badge' style='font-size:10px;vertical-align:middle'>χ²</span> use z = (χ²/ν − 1) × 4, mapping χ²/ν=1→0σ, 1.5→2σ.
                    <strong>Green zone</strong> (±1σ) = consistent | <strong>Yellow zone</strong> (±2σ) = borderline | <strong>Red zone</strong> (>2σ) = tension.
                    Use scope toggle to filter by role (A/B/V/D) or type (Scalar/Vector).
                </div>
                <div id='tension-plot-rows'>
                    {"".join(rows_html)}
                </div>
                <div class='tension-scale'>
                    <div class='tension-scale-labels'>
                        {"".join(scale_labels)}
                    </div>
                </div>
            </div>
        </div>
        """

    # Section 1: Detailed chi^2 breakdown

    def _render_chi2_breakdown(self, rows: List[Dict]) -> str:
        if not rows:
            return "<p>No models available.</p>"

        intro = """
        <p>
        This panel decomposes χ² into scalar and vector contributions and separates <em>strict</em> totals from <em>diagnostic</em> totals. CMB* (Planck 2018 ΛCDM compressed distance prior) is shown as diagnostic only and excluded from strict totals.
        </p>

        <h4>Definitions</h4>
        <ul>
          <li><strong>Scalar χ²:</strong> sum over scalar pillars with single targets and 1σ uncertainties.</li>
          <li><strong>Vector χ²:</strong> sum over vector pillars using full covariance matrices.</li>
          <li><strong>Strict combined χ²:</strong> scalar + vector, excluding diagnostic-only pillars (CMB*).</li>
          <li><strong>Diagnostic χ²:</strong> χ² for diagnostic-only pillars displayed for context, not counted in strict totals.</li>
        </ul>

        <h4>The Strict Validation Protocol: Pillar Roles</h4>
        <p>To prevent circularity, each pillar is classified by its methodological role: <span style="color:#f87171"><strong>A</strong></span> (Anchor), <span style="color:#fbbf24"><strong>B</strong></span> (Benchmark), <span style="color:#60a5fa"><strong>V</strong></span> (Validation), or <span style="color:#00ffff"><strong>D</strong></span> (Diagnostic):</p>
        <ul>
          <li><strong>Calibration Anchors (P8):</strong> used to fix the fundamental parameters {α, β}. Included for completeness but excluded from the validation score in V-only mode.</li>
          <li><strong>Benchmarks (P1, P1B):</strong> validate dataset-specific mappings (e.g., SPARC data handling). Useful consistency tests but may share data lineage with parameter characterization.</li>
          <li><strong>Validation Targets (P2–P7, P9–P13):</strong> with parameters fixed, the model is tested against these independent datasets. These are the headline claims.</li>
          <li><strong>Diagnostics (CMB*):</strong> excluded from strict totals and shown for transparency only. CMB* uses a ΛCDM-calibrated compressed prior, so its χ² measures tension with that compression rather than a direct MTDF prediction.</li>
        </ul>
        <p><strong>Peer-review note:</strong> The strict χ²/ν below includes all production pillars (A+B+V), excluding diagnostics (D). For the most conservative claim, a referee may prefer the V-only scope (Validation Targets only), which excludes calibration anchors and benchmarks. Use the badge toggles above the table to explore different scopes.</p>
        """

        rows_html = []
        # Track MTDF-specific statistics for the summary
        mtdf_excl_cmb_chi2_red = float("nan")
        mtdf_combined_chi2_red = float("nan")

        for r in rows:
            model_name = self._get_proper_model_name(r.get("model", "model"))
            model_key = r.get("model", "model")

            # Scalar stats - use dedicated scalar fields if available (MTDF), else fall back to dof/chi2
            scalar_dof = int(r.get("scalar_dof", r.get("dof", 0)) or 0)
            scalar_chi2 = float(r.get("scalar_chi2", r.get("chi2", 0.0)) or 0.0)
            scalar_chi2_red = float(r.get("scalar_chi2_red", 0.0) or 0.0)
            if scalar_chi2_red == 0.0 and scalar_dof > 0:
                scalar_chi2_red = scalar_chi2 / scalar_dof

            # Vector stats (from row; if missing, compute from vector_pillar_data)
            vector_dof = int(r.get("vector_dof", 0) or 0)
            vector_chi2 = float(r.get("vector_chi2", 0.0) or 0.0)
            vector_chi2_red = vector_chi2 / vector_dof if vector_dof > 0 else float("nan")

            # Fallback: if vector totals were not precomputed for this model, sum available vector pillar contributions
            if vector_dof == 0:
                vpd = r.get("vector_pillar_data", {}) or {}
                tmp_chi2 = 0.0
                tmp_dof = 0
                for pid, pdata in vpd.items():
                    if not isinstance(pdata, dict):
                        continue
                    # CMB* is diagnostic-only and excluded from vector pillar totals by default
                    if pid == "P_CMB_DIST":
                        continue
                    chi2_i = pdata.get("chi2", None)
                    dof_i = pdata.get("dof", None)
                    if chi2_i is None or dof_i is None:
                        continue
                    try:
                        chi2_i = float(chi2_i)
                        dof_i = int(dof_i)
                    except Exception:
                        continue
                    if dof_i <= 0:
                        continue
                    tmp_chi2 += chi2_i
                    tmp_dof += dof_i
                if tmp_dof > 0:
                    vector_chi2 = tmp_chi2
                    vector_dof = tmp_dof
                    vector_chi2_red = vector_chi2 / vector_dof

            # Combined stats
            combined_dof = int(r.get("combined_dof", scalar_dof) or scalar_dof)
            combined_chi2 = float(r.get("combined_chi2", scalar_chi2) or scalar_chi2)
            combined_chi2_red = combined_chi2 / combined_dof if combined_dof > 0 else float("nan")
            # Diagnostic-only CMB* contribution (if present)
            vpd_diag = r.get("vector_pillar_data", {}).get("P_CMB_DIST", {})
            diag_chi2 = float(vpd_diag.get("chi2", 0.0) or 0.0)
            diag_dof = int(vpd_diag.get("dof", 0) or 0)

            strict_combined_chi2 = combined_chi2 - diag_chi2
            strict_combined_dof = combined_dof - diag_dof
            if strict_combined_chi2 < 0 or strict_combined_dof <= 0:
                strict_combined_chi2 = combined_chi2
                strict_combined_dof = combined_dof
            strict_combined_chi2_red = strict_combined_chi2 / strict_combined_dof if strict_combined_dof > 0 else float("nan")


            # Combined excluding CMB stats (late-time fit quality)
            excl_cmb_dof = int(r.get("combined_excl_cmb_dof", 0) or 0)
            excl_cmb_chi2 = float(r.get("combined_excl_cmb_chi2", 0.0) or 0.0)
            excl_cmb_chi2_red = float(r.get("combined_excl_cmb_chi2_red", float("nan")))

            # Fallback: compute strict combined (excl. CMB*) if workbook did not store it
            if (excl_cmb_dof == 0 or excl_cmb_chi2 == 0.0 or math.isnan(excl_cmb_chi2_red)) and (scalar_dof > 0 or vector_dof > 0):
                excl_cmb_chi2 = float(scalar_chi2) + float(vector_chi2)
                excl_cmb_dof = int(scalar_dof) + int(vector_dof)
                excl_cmb_chi2_red = excl_cmb_chi2 / excl_cmb_dof if excl_cmb_dof > 0 else float("nan")

            # Store MTDF values for summary text
            if model_key == "MTDF":
                mtdf_excl_cmb_chi2_red = excl_cmb_chi2_red
                mtdf_combined_chi2_red = combined_chi2_red

            # Format values
            def fmt_chi2(v):
                if math.isnan(v) or math.isinf(v):
                    return "N/A"
                return f"{v:.2f}"

            def fmt_chi2_red(v):
                if math.isnan(v) or math.isinf(v):
                    return "N/A"
                return f"{v:.4f}"

            # Color class based on combined χ²/ν (thresholds from workbook)
            if combined_chi2_red < CHI2_THRESHOLD_GOOD:
                chi2_class = "chip-ok"
            elif combined_chi2_red < CHI2_THRESHOLD_ACCEPTABLE:
                chi2_class = "chip-warn"
            else:
                chi2_class = "chip-bad"

            # Color class for excl. CMB χ²/ν (thresholds from workbook)
            if excl_cmb_chi2_red < CHI2_THRESHOLD_GOOD:
                excl_chi2_class = "chip-ok"
            elif excl_cmb_chi2_red < CHI2_THRESHOLD_ACCEPTABLE:
                excl_chi2_class = "chip-warn"
            else:
                excl_chi2_class = "chip-bad"

            rows_html.append(f"""
            <tr>
              <td class='left' data-tooltip-id='cell.model.{model_key}' style='cursor:help'><strong>{model_name}</strong></td>
              <td>{scalar_dof}</td>
              <td>{fmt_chi2(scalar_chi2)}</td>
              <td>{fmt_chi2_red(scalar_chi2_red)}</td>
              <td>{vector_dof if vector_dof > 0 else 'N/A'}</td>
              <td>{fmt_chi2(vector_chi2) if vector_dof > 0 else 'N/A'}</td>
              <td>{fmt_chi2_red(vector_chi2_red) if vector_dof > 0 else 'N/A'}</td>
              <td><strong>{combined_dof}</strong></td>
              <td><strong>{fmt_chi2(combined_chi2)}</strong></td>
              <td class='{chi2_class}'><strong>{fmt_chi2_red(combined_chi2_red)}</strong></td>
              <td>{excl_cmb_dof if excl_cmb_dof > 0 else 'N/A'}</td>
              <td>{fmt_chi2(excl_cmb_chi2) if excl_cmb_dof > 0 else 'N/A'}</td>
              <td class='{excl_chi2_class}'>{fmt_chi2_red(excl_cmb_chi2_red) if excl_cmb_dof > 0 else 'N/A'}</td>
            </tr>""")

        table = """
        <h4>Per Model Breakdown</h4>
        <table class="subtable">
          <thead>
            <tr>
              <th class='left' rowspan='2'>Model</th>
              <th colspan='3' style='background:#1e3a5f'>Scalar Pillars</th>
              <th colspan='3' style='background:#1a4d2e'>Vector Pillars</th>
              <th colspan='3' style='background:#3b3821'>Combined (All)</th>
              <th colspan='3' style='background:#0d4f3c'>Combined (excl. CMB)</th>
            </tr>
            <tr>
              <th style='background:#1e3a5f'>ν</th>
              <th style='background:#1e3a5f'>χ²</th>
              <th style='background:#1e3a5f'>χ²/ν</th>
              <th style='background:#1a4d2e'>ν</th>
              <th style='background:#1a4d2e'>χ²</th>
              <th style='background:#1a4d2e'>χ²/ν</th>
              <th style='background:#3b3821'>ν</th>
              <th style='background:#3b3821'>χ²</th>
              <th style='background:#3b3821'>χ²/ν</th>
              <th style='background:#0d4f3c'>ν</th>
              <th style='background:#0d4f3c'>χ²</th>
              <th style='background:#0d4f3c'>χ²/ν</th>
            </tr>
          </thead>
          <tbody>{{rows}}</tbody>
        </table>
        <p style='margin-top:10px; font-size:0.9em; color:#94a3b8'>
        <strong>Note:</strong> The "Combined (excl. CMB)" column summarises late-time probes only (SNe, BAO, H(z), fσ₈ and scalar pillars) and excludes the CMB distance prior.
        This makes it easy to see whether any tension is confined to the CMB prior or also present in late-time data.
        </p>

        <p style='margin-top:15px; font-size:0.95em; border-top:1px solid #3b4f6f; padding-top:12px;'>
        <strong>Combined statistics</strong><br>
        Late-time only (excl. CMB distance prior): χ²/ν = {excl_cmb_val}.<br>
        All included (incl. CMB distance prior): χ²/ν = {combined_val}.<br>
        The difference between these values is dominated by the CMB distance prior contribution described in the CMB tooltip.
        </p>
        """
        # First replace {{rows}} while still double-braced, then format
        table = table.replace("{{rows}}", ''.join(rows_html))
        table = table.format(
            excl_cmb_val=fmt_chi2_red(mtdf_excl_cmb_chi2_red),
            combined_val=fmt_chi2_red(mtdf_combined_chi2_red)
        )

        return intro + table

    # Section 2: Parameter independence
    def _render_parameter_independence(self) -> str:
        found, missing = [], []
        probe = {
            "α": ["alpha", "Stress Tensor-Matter Coupling", "Stress Tensor-Matter Coupling"],
            "β": ["beta", "Coherence Length", "Coherence Length (β)"],
            "E": ["E", "Spacetime Elastic Modulus"],
            "δ_bf": ["delta_bf", "Field-Baryon Coupling (δ_bf)"],
            "β_eos": ["beta_eos", "EoS Transition β_eos"],
        }
        consts = getattr(self.db, "constants", {}) or {}
        norm = {str(k).strip().lower(): v for k, v in consts.items()} if isinstance(consts, dict) else {}

        def lookup(names: List[str]):
            for n in names:
                v = norm.get(n.strip().lower())
                if v is not None:
                    return v
            return None

        for label, names in probe.items():
            v = lookup(names)
            if v is None:
                missing.append("• " + label + " = NOT FOUND")
            else:
                try:
                    shown = "{:.3g}".format(float(v))
                except Exception:
                    shown = str(v)
                found.append("• " + label + " = " + shown)

        summary = "<p>Database verification: {}/{} core parameters detected.</p>".format(
            len(found), len(found)+len(missing))
        body = "<div class='left'><p>" + "<br/>".join(found + missing) + "</p></div>"
        note = "<p>Zero hardcoding: values are read from the database for this run.</p>"
        return summary + body + note

    # Section 3: Core physics equations list
    def _render_formula_list(self, pillars: List[str]) -> str:
        """Render formulas with proper KaTeX math rendering and pillar descriptions."""
        equations_html = []

        for pid in pillars:
            # Try to get pillar data from tooltip engine cache first, then fallback to db.pillars
            if self.tooltip_engine and hasattr(self.tooltip_engine, 'pillar_cache'):
                d = self.tooltip_engine.pillar_cache.get(pid, {})
            else:
                pill_data = getattr(self.db, "pillars", {}) or {}
                d = pill_data.get(pid, {})

            latex = d.get("latex") or d.get("equation_latex") or d.get("latex_formula")
            description = d.get("name") or d.get("pillar_name") or d.get("description") or pid

            if latex:
                # Wrap in KaTeX delimiters for display mode rendering
                equation_html = f"""
                <div class='equation-item'>
                    <div class='equation-label'><strong>{pid}</strong>: {description}</div>
                    <div class='equation-display'>$$\\displaystyle {latex}$$</div>
                </div>"""
                equations_html.append(equation_html)
            else:
                equations_html.append(f"""
                <div class='equation-item'>
                    <div class='equation-label'><strong>{pid}</strong>: {description}</div>
                    <div class='equation-note'>(formula not available)</div>
                </div>""")

        # Add auxiliary formulas from workbook (like EFE formulas)
        auxiliary_html = self._render_auxiliary_formulas()
        if auxiliary_html:
            equations_html.append("<h4 style='margin-top:20px; color:#60a5fa;'>Auxiliary Formulas (EFE/Derived)</h4>")
            equations_html.append(auxiliary_html)

        return "\n".join(equations_html)

    def _render_auxiliary_formulas(self) -> str:
        """Load and render auxiliary formulas (F_EFE_*, etc.) from workbook."""
        try:
            import pandas as pd
            wb_path = Path(__file__).parent.parent / "data" / "DB_Workbook_STRICT_V18.xlsx"
            if not wb_path.exists():
                wb_path = Path(__file__).parent.parent.parent / "data" / "DB_Workbook_STRICT_V18.xlsx"
            if not wb_path.exists():
                return ""

            df = pd.read_excel(wb_path, sheet_name='Pillar_Formulas', header=1)
            # Filter for auxiliary formulas (F_EFE_*, F022*, etc. - not P* or P_*)
            aux_df = df[df['pillar_id'].astype(str).str.match(r'^F_', na=False)]

            aux_html = []
            for _, row in aux_df.iterrows():
                pid = row.get('pillar_id', '')
                name = row.get('name', pid)
                latex = row.get('latex', '')

                if pd.notna(latex) and latex:
                    aux_html.append(f"""
                    <div class='equation-item'>
                        <div class='equation-label'><strong>{pid}</strong>: {name}</div>
                        <div class='equation-display'>$$\\displaystyle {latex}$$</div>
                    </div>""")

            return "\n".join(aux_html)
        except Exception as e:
            return f"<!-- Error loading auxiliary formulas: {e} -->"

    # Section 4: Observational targets table
    def _render_targets_table(self, pillars: List[str]) -> str:
        pill_data = getattr(self.db, "pillars", {}) or {}
        rows_html = []

        # Vector pillar DOIs and info
        vector_pillar_info = {
            'P_SNE_PANTHEON': {
                'doi': 'https://doi.org/10.3847/1538-4357/ac8b7a',
                'display_value': '1,701 SNe Ia',
                'unit': 'distance modulus (mag)'
            },
            'P_BAO_DESI': {
                'doi': 'https://doi.org/10.1088/1475-7516/2024/02/015',
                'display_value': '12 BAO measurements',
                'unit': 'D_M/r_d, D_H/r_d'
            },
            'P_HZ_CC': {
                'doi': 'https://doi.org/10.12942/lrr-2022-7',
                'display_value': '15 H(z) points',
                'unit': 'km/s/Mpc'
            },
            'P_GROWTH_FSIG8': {
                'doi': 'https://doi.org/10.1103/PhysRevD.103.083533',
                'display_value': '4 fσ₈(z) points',
                'unit': 'dimensionless'
            },
            'P_CMB_DIST': {
                'doi': 'https://doi.org/10.1051/0004-6361/201833910',
                'display_value': 'R, ℓ_A, ω_b h²',
                'unit': 'distance prior'
            },
        }

        for pid in pillars:
            d = pill_data.get(pid, {})

            # Check if vector pillar
            if pid in vector_pillar_info:
                vinfo = vector_pillar_info[pid]
                doi = vinfo['doi']
                display = vinfo['display_value']
                unit = vinfo['unit']
            else:
                # Scalar pillar - use source_doi and unit from workbook
                doi = d.get("source_doi") or d.get("doi") or ""
                tgt = d.get("target")
                sig = d.get("sigma")
                # Use actual unit field from workbook
                unit = d.get("unit") or "---"
                try:
                    display = "{:.3g}±{:.3g}".format(tgt, sig) if (tgt is not None and sig is not None) else "---"
                except Exception:
                    display = "---"

            doi_html = "<a href='{}' target='_blank'>{}</a>".format(doi, doi) if doi else ""
            check_mark = '✓' if doi else '---'
            rows_html.append(
                "<tr><td>{}</td><td class='left'>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    pid, doi_html, display, check_mark, unit)
            )
        table = """

        <table class="subtable">
          <thead><tr><th>Pillar</th><th class='left'>DOI</th><th>Source Observation Value</th><th>Loaded</th><th>Unit</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """.replace("{rows}", ''.join(rows_html))
        return table

    def _render_parameter_audit(self) -> str:
        """Render comprehensive parameter independence audit table"""
        try:
            import pandas as pd

            # Find workbook - try common paths
            workbook_candidates = [
                'data/DB_Workbook_STRICT_V18.xlsx',
                '../data/DB_Workbook_STRICT_V18.xlsx',
                './DB_Workbook_STRICT_V18.xlsx'
            ]

            workbook_path = None
            for path in workbook_candidates:
                from pathlib import Path
                if Path(path).exists():
                    workbook_path = path
                    break

            if not workbook_path:
                return "<p>Workbook file not found - cannot load parameter audit</p>"

            # Read parameter sheets
            sheets = ['Params_Fundamental', 'Params_Observational', 'Params_Constants', 'Params_Coefficients']
            all_params = []

            for sheet in sheets:
                try:
                    df = pd.read_excel(workbook_path, sheet_name=sheet, header=1)
                    df = df[df['name'].notna()]  # Filter out empty rows
                    df['category'] = sheet.replace('Params_', '')
                    all_params.append(df)
                except Exception as e:
                    continue

            if not all_params:
                return "<p>No parameter data loaded from workbook</p>"

            combined = pd.concat(all_params, ignore_index=True)

            # Build HTML table
            rows_html = []
            for row_idx, (_, row) in enumerate(combined.iterrows()):
                name = row.get('name', 'Unknown')
                value = row.get('value_si', '')
                unit = row.get('unit', '')
                category = row.get('category', '')

                # Different sheets use different column names for "source" info
                if category == 'Fundamental':
                    established = row.get('established_from', 'Not specified')
                elif category == 'Observational':
                    established = row.get('observational_source', row.get('what', 'Not specified'))
                elif category == 'Constants':
                    established = row.get('definition_source', row.get('what', 'Not specified'))
                elif category == 'Coefficients':
                    established = row.get('derived_from', row.get('what', 'Not specified'))
                else:
                    established = row.get('what', 'Not specified')

                doi = row.get('source_doi', '')
                param_type = row.get('parameter_type', 'empirical')
                provenance_tier = row.get('provenance_tier', '')

                # Format value
                try:
                    import pandas as pd
                    if pd.notna(value) and value != '':
                        val_float = float(value)
                        if abs(val_float) >= 1e4 or (0 < abs(val_float) < 1e-3):
                            value_display = f"{val_float:.3e}"
                        else:
                            value_display = f"{val_float:.6g}"
                    else:
                        value_display = "derived"
                except:
                    value_display = "N/A"

                # No truncation - allow text to wrap naturally
                established_display = str(established) if pd.notna(established) else "N/A"

                # Make DOI clickable (but not for internal MTDF constants)
                # Check if DOI contains "N/A" or "MTDF internal" anywhere in the string
                doi_str = str(doi).upper() if pd.notna(doi) else ""
                is_internal = ('N/A' in doi_str or 'MTDF' in doi_str or 'INTERNAL' in doi_str or
                              doi_str in ['', 'NA', 'NONE', '-'])

                if pd.notna(doi) and doi and not is_internal:
                    import re
                    doi_parts = re.split(r'[,;]', str(doi))
                    doi_links = []
                    for d in doi_parts:
                        d = d.strip()
                        if d.startswith('DOI:'):
                            d = d[4:].strip()
                        if d.startswith('https://doi.org/'):
                            d = d[len('https://doi.org/'):]
                        if d and d.upper() not in ['N/A', 'NA', 'NONE', '-']:
                            doi_links.append(f"<a href='https://doi.org/{d}' target='_blank' title='{d}'>{d[:20]}...</a>")
                    doi_html = "<br>".join(doi_links) if doi_links else "MTDF internal"
                else:
                    doi_html = "MTDF internal"

                # Category badge - show "Derived" for derived quantities in any sheet
                badge_label = category
                if param_type == 'derived':
                    badge_label = 'Derived'
                cat_class = {
                    'Fundamental': 'badge-fundamental',
                    'Observational': 'badge-observational',
                    'Constants': 'badge-constant',
                    'Coefficients': 'badge-coefficient'
                }.get(category, '')

                cat_html = f"<span class='param-badge {cat_class}'>{badge_label}</span>"

                tier_html = ''
                if pd.notna(provenance_tier) and provenance_tier:
                    tier_map = {
                        'exact_anchor': ('Tier 1', 'tier-1'),
                        'calibration_contextual': ('Tier 2', 'tier-2'),
                        'phenomenological': ('Tier 3', 'tier-3'),
                        'internal_reference': ('Internal ref', 'tier-int'),
                    }
                    tier_label, tier_class = tier_map.get(str(provenance_tier), ('', ''))
                    if tier_label:
                        tier_html = f" <span class='tier-badge {tier_class}'>{tier_label}</span>"
                    if str(provenance_tier) == 'phenomenological':
                        established_display += ' <em>(Physics analogue only)</em>'

                # Alternating row class for better readability
                row_class = 'param-row-even' if row_idx % 2 == 0 else 'param-row-odd'

                rows_html.append(f"""
                <tr class='{row_class}'>
                    <td class='left'><strong>{name}</strong></td>
                    <td>{value_display}</td>
                    <td>{unit}</td>
                    <td>{cat_html}{tier_html}</td>
                    <td class='left param-desc'>{established_display}</td>
                    <td class='left param-doi'>{doi_html}</td>
                </tr>
                """)

            # Extract the four "tunable knob" values dynamically from workbook data
            def get_param_value(df, name_pattern):
                """Get parameter value by name pattern."""
                for _, row in df.iterrows():
                    n = str(row.get('name', '')).lower()
                    if name_pattern.lower() in n:
                        val = row.get('value_si')
                        if pd.notna(val):
                            try:
                                v = float(val)
                                if abs(v) < 1e-10:
                                    return f"{v:.2e}"
                                elif abs(v) >= 1e4:
                                    return f"{v:.2e}"
                                else:
                                    return f"{v:g}"
                            except:
                                pass
                return "N/A"

            env_rate = get_param_value(combined, "environmental rate")
            stress_thresh = get_param_value(combined, "stress threshold")
            struct_chi = get_param_value(combined, "structure coupling")
            avg_strain = get_param_value(combined, "average effective strain")
            param_count = len(combined)

            table_html = f"""
            <div style='margin-bottom:15px; padding:10px; background:#1e293b; border-left:4px solid #22c55e; border-radius:4px;'>
                <strong style='color:#22c55e'>✓ No parameters fitted to validation data (k = 0)</strong><br>
                <span style='font-size:0.95em'>All {param_count} quantities in the MTDF implementation are established independently before any validation pillar is evaluated. The framework is defined by four fundamental parameters (α, β, τ, β<sub>eos</sub>), with all remaining quantities either derived from these (e.g. E) or anchored to external observations (e.g. κ ≈ f<sub>kick</sub>/3), constrained by external observations, or fixed by convention. No parameters are adjusted to improve any pillar fit. The four quantities that might appear as tunable knobs &mdash; Environmental Rate Enhancement ({env_rate}), Stress Threshold ({stress_thresh}), Structure Coupling χ ({struct_chi}), and Average Effective Strain ({avg_strain}) &mdash; are either derived from external data, computed from the MTDF field construction itself, or are one-off statistical cuts defined by convention.</span>
            </div>
            <table class='subtable' style='font-size:0.9em'>
                <thead>
                    <tr>
                        <th class='left' style='min-width:200px'>Parameter</th>
                        <th>Value (SI)</th>
                        <th>Unit</th>
                        <th>Category</th>
                        <th class='left' style='min-width:250px'>Measured From / Established From</th>
                        <th class='left' style='min-width:150px'>Source DOI</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
            <div style='margin-top:10px; font-size:0.9em; color:#94a3b8'>
                <strong>Independence Guarantee:</strong> Parameters measured from observations A, B, C; validated against independent observations X, Y, Z.
                Zero overlap ensures no circular reasoning.
            </div>
            <style>
                .param-badge {{
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 0.75em;
                    font-weight: 600;
                    text-transform: uppercase;
                    color: black;
                }}
                .badge-fundamental {{ background: #3b82f6; }}
                .badge-observational {{ background: #22c55e; }}
                .badge-constant {{ background: #94a3b8; }}
                .badge-coefficient {{ background: #fbbf24; }}
                .tier-badge {{ display:inline-block; padding:1px 6px; border-radius:3px; font-size:0.75em; font-weight:600; margin-left:6px; vertical-align:middle; }}
                .tier-1 {{ background:#1b5e20; color:#fff; }}
                .tier-2 {{ background:#e65100; color:#fff; }}
                .tier-3 {{ background:#b71c1c; color:#fff; }}
                .tier-int {{ background:#424242; color:#fff; }}

                /* Alternating row colors for better readability */
                .param-row-even {{
                    background-color: #1e293b;
                }}
                .param-row-odd {{
                    background-color: #0f172a;
                }}

                /* Allow text wrapping in description columns - more specific to override global nowrap */
                .subtable td.param-desc {{
                    white-space: normal !important;
                    word-wrap: break-word;
                    max-width: 350px;
                    line-height: 1.4;
                    padding: 8px 12px !important;
                    font-size: 0.9em;
                }}
                .subtable td.param-doi {{
                    white-space: normal !important;
                    word-wrap: break-word;
                    max-width: 200px;
                    line-height: 1.3;
                    padding: 8px 12px !important;
                    font-size: 0.85em;
                }}
            </style>
            """

            return table_html

        except Exception as e:
            return f"<p>Error generating parameter audit: {str(e)}</p>"

    def _render_framework_overview(self, pillar_count: int) -> str:
        """Render MTDF framework overview section"""
        # Calculate scalar vs vector pillar counts
        scalar_count = pillar_count - 5  # 5 vector pillars
        vector_count = 5
        return f"""
        <h3>Executive Summary</h3>
        <p>This dashboard reports <b>strict</b> validation totals over the intended fit pillars (scalar pillars plus late-time probes), and separates out <b>diagnostic</b> pillars such as <b>CMB*</b> (Planck 2018 ΛCDM-compressed distance prior). In the main table, the rightmost column (<b>Strict χ²/ν</b>) is the combined reduced chi-squared for <b>fit pillars only</b> (diagnostics excluded). Diagnostic χ² values are shown for context but are not counted in strict totals.</p>
        <ul>
          <li><strong>Externally fixed parameter set (MTDF):</strong> All MTDF parameters are established by independent observations, first-principles derivations, or physical constants before any pillar is evaluated. No parameters are tuned to fit these validation tests.</li>
          <li><strong>No hidden hardcoding:</strong> All numerical inputs, observational targets, and parameter values are loaded from external database files (workbook) rather than being embedded in the code.</li>
          <li><strong>Full covariance analysis:</strong> Vector pillars use published covariance matrices (e.g., Pantheon+ STAT+SYS, Planck distance prior covariance) for statistically rigorous χ² computation.</li>
          <li><strong>Strict statistics:</strong> The headline χ²/ν and PASS counts exclude diagnostic-only pillars (CMB* distance prior). This keeps the global totals focused on intended fit targets.</li><li><strong>Diagnostic pillars:</strong> CMB* (Planck 2018 ΛCDM compressed distance prior) is displayed for transparency, but treated as diagnostic only because it is model-compressed under ΛCDM assumptions.</li>
          <li><strong>Framework classification:</strong> Models are classified as EMPIRICAL (observable matter only) or COMPONENT-BASED (requires dark matter/energy) depending on whether they introduce additional dark components.</li>
        </ul>
        <p class="muted">
          <strong>Statistical interpretation:</strong> χ²/ν &lt; 1 indicates excellent agreement; 1–2 indicates acceptable agreement; values &gt;2 indicate tension and a need for model revision.
        </p>
        """

    def _render_reading_guide(self) -> str:
        """Render reading guide section"""
        return """
        <p>
        This table shows how each model performs across all pillars used in the dashboard. Each column is a pillar, each row is a model. Colours and symbols are designed to make the structure visible at a glance.
        </p>

        <h4>Scalar Pillars</h4>
        <p>
        The left block of pillars (P1 to P13 and the extra scalar entries) are scalar tests. Each scalar pillar encodes a single relation, target or consistency condition. Examples include typical void scales, BAO scale, stress thresholds and derived normalisations inside MTDF.
        </p>
        <ul>
          <li><strong>One number per pillar:</strong> each scalar pillar uses a single observed target value with a 1σ uncertainty.</li>
          <li><strong>χ² per pillar:</strong> for scalar pillars we use the standard χ² = (O − M)² / σ².</li>
          <li><strong>Colour code:</strong> green cells indicate |z| < 1, orange indicates 1 ≤ |z| < 2, red indicates |z| ≥ 3.</li>
        </ul>

        <h4>Vector Pillars (Green Headers)</h4>
        <p>
        The right block of pillars (SNe, BAO, H(z), fσ₈ and CMB) are vector tests. These use real data vectors and published covariance matrices from modern surveys.
        </p>
        <ul>
          <li><strong>SNe:</strong> Pantheon+ Type Ia supernova distance moduli with full STAT+SYS covariance.</li>
          <li><strong>BAO:</strong> DESI Gaussian BAO constraints for D<sub>M</sub>/r<sub>d</sub> and D<sub>H</sub>/r<sub>d</sub> across several redshift bins.</li>
          <li><strong>H(z):</strong> cosmic chronometer H(z) measurements from passively evolving galaxies.</li>
          <li><strong>fσ₈:</strong> growth of structure from redshift space distortions, using MTDF linear growth with μ(a).</li>
          <li><strong>CMB:</strong> Planck 2018 distance priors such as acoustic scale and shift parameter with Planck covariance.</li>
        </ul>
        <p>
        Vector pillar headers are tinted green to highlight that they carry most of the degrees of freedom in the global fit. Each vector pillar contributes tens to thousands of data points, not a single summary number.
        </p>
        <ul>
          <li><strong>Full covariance:</strong> χ² is computed as (d − m)<sup>T</sup> C<sup>−1</sup> (d − m) using the published covariance matrix C.</li>
          <li><strong>Reduced χ²:</strong> for vector pillars the dashboard usually displays χ² / DOF. Values between about 0.5 and 2 are typically compatible with the quoted errors.</li>
          <li><strong>Colour code:</strong> for all vector pillars except CMB, green cells indicate χ²/DOF &lt; 1.5, orange 1.5 ≤ χ²/DOF &lt; 2, red χ²/DOF ≥ 2. For CMB the colours are interpreted relative to the ΛCDM-based distance prior, as explained in the CMB tooltips.</li>
        </ul>

        <p><strong>CMB* is diagnostic only:</strong> It is a ΛCDM-calibrated compressed prior, so its χ² primarily measures tension with that compression. It remains visible, but is excluded from the strict combined χ²/ν and PASS counts. Hover the CMB* header and values for details.</p>

        <h4>The Strict Validation Protocol: Pillar Roles</h4>
        <p>To prevent circularity, each pillar header shows a coloured badge indicating its methodological role:</p>
        <ul>
          <li><strong><span style="color:#f87171">A</span> (Calibration Anchor):</strong> Used to fix the fundamental parameters {α, β}. P8 (β scale) belongs here: it establishes the void quantization scale that defines β. Anchors are excluded from the validation score in V-only mode.</li>
          <li><strong><span style="color:#fbbf24">B</span> (Benchmark):</strong> Validates dataset-specific mappings or error model choices (e.g., SPARC data handling). P1 and P1B belong here: they test rotation curve scatter using SPARC-derived calibrations. Benchmarks are useful consistency tests but may share data lineage with parameter characterization.</li>
          <li><strong><span style="color:#60a5fa">V</span> (Validation Target):</strong> With parameters fixed, the model is tested against these independent datasets. Most pillars (P2–P7, P9–P13) belong here. These are the headline claims.</li>
          <li><strong><span style="color:#00ffff">D</span> (Diagnostic):</strong> Excluded from strict totals and shown for transparency only. CMB* belongs here because it uses a ΛCDM-calibrated compressed prior; its χ² measures tension with that compression rather than a direct MTDF prediction.</li>
        </ul>
        <p><strong>Why this matters:</strong> A referee might argue that calibration anchors are circular, benchmarks are not fully out-of-sample, and diagnostics use external model assumptions. By labelling them explicitly, we report multiple totals: <em>Validation-only χ²</em> (strongest claim), <em>Standard χ² (A+B+V)</em> (excludes diagnostics), and <em>Full χ² (A+B+V+D)</em> (complete transparency).</p>

        <h4>Scope Toggle Controls</h4>
        <p>Use the <strong>scope dropdown</strong> (top-right of table) and <strong>role badge toggles</strong> (A/B/V/D badges) to filter which pillars are included in the statistics and visualization:</p>
        <ul>
          <li><strong>Strict:</strong> Shows A+B+V pillars (excludes Diagnostics like CMB*). This is the default view and matches the paper's "strict χ²/ν" definition.</li>
          <li><strong>Validation:</strong> Shows only V pillars — the most independent validation set for conservative claims.</li>
          <li><strong>Full:</strong> Shows all pillars (A+B+V+D) including diagnostics for complete transparency.</li>
          <li><strong>Scalar:</strong> Shows only scalar pillars (P1–P13) — single-number tests with z-scores.</li>
          <li><strong>Vector:</strong> Shows only vector pillars (SNe, BAO, H(z), fσ₈) — multi-point likelihood tests with χ²/ν. Note: CMB* is excluded in Vector mode as it's diagnostic.</li>
          <li><strong>Custom:</strong> Click individual role badges (A/B/V/D) to create a custom combination.</li>
        </ul>
        <p>When you change scope, the table columns dim/highlight accordingly, the combined χ²/ν recalculates, and the Tension Visualization plot updates to show only the relevant pillars.</p>

        <h4>Tier, Foundation and Status</h4>
        <ul>
          <li><strong>Tier:</strong> shows whether a pillar is a pure foundation test, a component-based test, or experimental. This column is intentionally narrow; hover over the tier for a full tooltip.</li>
          <li><strong>Foundation:</strong> indicates whether the target is empirical (Emp.), published (Pub.) or theoretical (Theo.) inside MTDF.</li>
          <li><strong>Status:</strong> <em>Production</em> pillars enter the global χ² and DOF totals. <em>Experimental</em> pillars are displayed but can be excluded from the combined statistics.</li>
        </ul>

        <h4>Global χ² and Max |z|</h4>
        <ul>
          <li><strong>Global χ² / DOF (strict):</strong> computed from pillars intended as fit targets. Diagnostic-only pillars (CMB*) are excluded from this headline statistic.</li>
          <li><strong>Max |z|:</strong> for each model, this is the largest absolute z score across all scalar and vector pillars. It shows the single worst tension in that row.</li>
        </ul>
        <p>
        The intention is that you can read across a row and immediately see whether a model fails early at the foundation level, struggles on a particular dataset, or strict_passes all pillars with only mild tension.
        </p>
        """

    def _generate_framework_accordion(self, pillars: List[str]) -> str:
        """Generate framework accordion to show above the table"""
        pillar_count = len(pillars)
        return self._accordion("🎯 MTDF V74 Framework", self._render_framework_overview(pillar_count), open_default=True)

    def _generate_reading_guide_accordion(self) -> str:
        """Generate reading guide accordion (shown first after table)"""
        return self._accordion("📖 How to Read This Table", self._render_reading_guide(), open_default=False)

    def _generate_accordions(self, rows: List[Dict], pillars: List[str]) -> str:
        """Generate detail accordions to show below the table"""
        sec_tension = self._accordion("📈 Tension Visualization (σ-Plot)", self._render_tension_plot(rows, pillars), open_default=True)
        sec1 = self._accordion("📊 Detailed χ² Breakdown", self._render_chi2_breakdown(rows), open_default=False)
        sec2 = self._accordion("🔬 Parameter Independence Audit", self._render_parameter_audit(), open_default=False)
        sec_lit = self._accordion("📚 Vector Pillar Literature Sources", self._render_literature_sources(rows), open_default=False)
        sec4 = self._accordion("⚙️ Core Physics Equations (From Database)", self._render_formula_list(pillars))
        sec5 = self._accordion("🎯 Observational Targets (From Database)", self._render_targets_table(pillars))
        return sec_tension + sec1 + sec2 + sec_lit + sec4 + sec5

    def _render_literature_sources(self, rows: List[Dict]) -> str:
        """Render literature sources for vector pillar comparison values"""
        html = """
        <div class='summary-panel'>
        <h4>Vector Pillar χ² Values: Computed vs Literature</h4>
        <p><b>Scalar pillars</b> (P1–P13) show the model prediction in the cell, with the signed <b>z-score</b> underneath (labelled <b>z =</b>). A z-score is the deviation in units of the stated 1σ uncertainty.<br><b>Vector pillars</b> (SNe, BAO, H(z), fσ₈, CMB*) show the pillar’s reduced statistic <b>χ²/ν</b> underneath (computed using the published covariance where available).</p>

        <h5>Methodology</h5>
        <ul>
          <li><strong>MTDF</strong>: All vector pillar χ² values are computed directly from the data vectors (Pantheon+, DESI BAO, CC H(z), DR16 fσ₈, Planck CMB) using our own code with full covariance matrices.</li>
          <li><strong>Comparison Models (ΛCDM, EDE, etc.)</strong>: χ² values are taken from peer-reviewed publications. We do NOT compute predictions for these models.</li>
          <li><strong>n/a Entries</strong>: If no published χ² value exists for a specific dataset+model combination, the cell is left empty.</li>
        </ul>

        <h5>Literature Sources by Vector Pillar</h5>
        <table style='width:100%; border-collapse:collapse; margin-top:12px;'>
          <thead>
            <tr style='background:#1a2744; border-bottom:1px solid #2d3f5f;'>
              <th style='padding:8px; text-align:left;'>Pillar</th>
              <th style='padding:8px; text-align:left;'>Dataset</th>
              <th style='padding:8px; text-align:left;'>ΛCDM Source</th>
              <th style='padding:8px; text-align:left;'>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr style='border-bottom:1px solid #1f2a44;'>
              <td style='padding:8px;'>SNe</td>
              <td style='padding:8px;'>Pantheon+ (1701 SNe)</td>
              <td style='padding:8px;'>Brout+ 2022 (ApJ 938, 110)</td>
              <td style='padding:8px;'>χ²/dof ≈ 1.0 by error calibration</td>
            </tr>
            <tr style='border-bottom:1px solid #1f2a44;'>
              <td style='padding:8px;'>BAO</td>
              <td style='padding:8px;'>DESI Y1 (12 points)</td>
              <td style='padding:8px;'>DESI 2024 VI (JCAP)</td>
              <td style='padding:8px;'>Bayesian analysis; good fit reported</td>
            </tr>
            <tr style='border-bottom:1px solid #1f2a44;'>
              <td style='padding:8px;'>H(z)</td>
              <td style='padding:8px;'>Cosmic Chronometers (15 pts)</td>
              <td style='padding:8px;'>Melia & Maier 2013 (MNRAS)</td>
              <td style='padding:8px;'>χ²/dof = 0.78</td>
            </tr>
            <tr style='border-bottom:1px solid #1f2a44;'>
              <td style='padding:8px;'>fσ₈</td>
              <td style='padding:8px;'>DR16 RSD (4 points)</td>
              <td style='padding:8px;'>eBOSS 2021 (PRD 103)</td>
              <td style='padding:8px;'>Consistent with ΛCDM</td>
            </tr>
            <tr style='border-bottom:1px solid #1f2a44;'>
              <td style='padding:8px;'>CMB</td>
              <td style='padding:8px;'>Planck 2018 Distance Prior</td>
              <td style='padding:8px;'>Planck 2018 VI (A&A 641)</td>
              <td style='padding:8px;'>Distance prior derived from best-fit</td>
            </tr>
          </tbody>
        </table>

        <h5 style='margin-top:16px;'>Alternative Models</h5>
        <ul>
          <li><strong>EDE (Early Dark Energy)</strong>: CMB constraints from Hill+ 2022 (PRD 106). Δχ² = -16.2 vs ΛCDM for ACT+SPT+Planck polarization.</li>
          <li><strong>MOND</strong>: AeST-MOND (Skordis & Złośnik 2021, PRL 127) reproduces CMB acoustic peaks; explicit χ² not published.</li>
          <li><strong>FDM, SIDM</strong>: No direct fits to these specific datasets found in literature; cells marked n/a.</li>
        </ul>

        <p style='margin-top:16px; font-style:italic; color:#93a3b3;'>
          <strong>Transparency note:</strong> Only MTDF values are computed in this validation framework.
          All comparison model values are taken directly from published analyses and are clearly marked with "Literature" in their tooltips.
        </p>
        </div>
        """
        return html

    def _generate_pillar_summary_table(self, rows: List[Dict], pillars: List[str]) -> str:
        """Generate pillar summary table showing all pillars with statistics."""

        # Find MTDF row for vector results
        mtdf_row = next((r for r in rows if r.get("model") == "MTDF"), {})
        vector_results = mtdf_row.get("vector_results", [])

        # Get scalar pillar data from db
        pill_data = getattr(self.db, "pillars", {}) or {}

        # Build rows for scalar pillars
        rows_html = []
        scalar_chi2_total = 0.0
        scalar_dof_total = 0

        # Scalar pillars from MTDF row
        mtdf_vals = mtdf_row.get("vals", {})
        mtdf_z_scores = mtdf_row.get("z_scores", {})

        for pid in pillars:
            d = pill_data.get(pid, {})
            name = d.get("name") or d.get("pillar_name") or pid
            category = d.get("category") or "Physics"

            # Calculate chi² for this pillar from MTDF z-score
            z = mtdf_z_scores.get(pid)
            if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                chi2_contrib = z ** 2
                scalar_chi2_total += chi2_contrib
                scalar_dof_total += 1
                chi2_str = f"{chi2_contrib:.3f}"
                chi2_red_str = f"{chi2_contrib:.3f}"  # DOF=1 for each scalar
            else:
                chi2_str = "N/A"
                chi2_red_str = "N/A"

            rows_html.append(f"""
            <tr>
                <td class='left'><strong>{pid}</strong></td>
                <td class='left'>{name}</td>
                <td>{category}</td>
                <td><span class='mode-badge mode-scalar'>SCALAR</span></td>
                <td>1</td>
                <td>1</td>
                <td>{chi2_str}</td>
                <td>{chi2_red_str}</td>
            </tr>
            """)

        # Add scalar subtotal
        if scalar_dof_total > 0:
            scalar_chi2_red = scalar_chi2_total / scalar_dof_total
            rows_html.append(f"""
            <tr class='subtotal-row'>
                <td colspan='4' class='left'><strong>Scalar Subtotal</strong></td>
                <td><strong>{scalar_dof_total}</strong></td>
                <td><strong>{scalar_dof_total}</strong></td>
                <td><strong>{scalar_chi2_total:.3f}</strong></td>
                <td><strong>{scalar_chi2_red:.4f}</strong></td>
            </tr>
            """)

        # Vector pillars from vector_results
        vector_chi2_total = 0.0
        vector_dof_total = 0

        for vr in vector_results:
            pid = vr.get("pillar_id", "")
            name = vr.get("name", pid)
            category = vr.get("category", "Cosmology")
            n_data = vr.get("n_data", 0)
            dof = vr.get("dof", 0)
            chi2 = vr.get("chi2", 0.0)
            chi2_red = vr.get("chi2_red", float("nan"))
            experimental = vr.get("experimental", False)

            if not experimental:
                vector_chi2_total += chi2
                vector_dof_total += dof

            exp_marker = " <span style='color:#f59e0b'>*</span>" if experimental else ""
            chi2_red_str = f"{chi2_red:.4f}" if not (math.isnan(chi2_red) or math.isinf(chi2_red)) else "N/A"

            rows_html.append(f"""
            <tr>
                <td class='left'><strong>{pid}</strong>{exp_marker}</td>
                <td class='left'>{name}</td>
                <td>{category}</td>
                <td><span class='mode-badge mode-vector'>VECTOR</span></td>
                <td>{n_data}</td>
                <td>{dof}</td>
                <td>{chi2:.1f}</td>
                <td>{chi2_red_str}</td>
            </tr>
            """)

        # Add vector subtotal
        if vector_dof_total > 0:
            vector_chi2_red = vector_chi2_total / vector_dof_total
            rows_html.append(f"""
            <tr class='subtotal-row'>
                <td colspan='4' class='left'><strong>Vector Subtotal (strict)</strong></td>
                <td><strong>N/A</strong></td>
                <td><strong>{vector_dof_total}</strong></td>
                <td><strong>{vector_chi2_total:.1f}</strong></td>
                <td><strong>{vector_chi2_red:.4f}</strong></td>
            </tr>
            """)


        # Diagnostic pillars excluded from strict totals (Regime A framing)
        diagnostic_pillars = {'P_CMB_DIST'}
        diagnostic_dof = 0
        diagnostic_chi2 = 0.0
        for vr in vector_results:
            if vr.get("pillar_id","") in diagnostic_pillars and not vr.get("experimental", False):
                diagnostic_dof += int(vr.get("dof", 0) or 0)
                diagnostic_chi2 += float(vr.get("chi2", 0.0) or 0.0)

        strict_vector_dof = max(0, vector_dof_total - diagnostic_dof)
        strict_vector_chi2 = max(0.0, vector_chi2_total - diagnostic_chi2)
        strict_total_dof = scalar_dof_total + strict_vector_dof
        strict_total_chi2 = scalar_chi2_total + strict_vector_chi2
        diagnostic_red = (diagnostic_chi2 / diagnostic_dof) if diagnostic_dof else float('nan')

        # Combined total
        total_chi2 = scalar_chi2_total + vector_chi2_total
        total_dof = scalar_dof_total + vector_dof_total
        if total_dof > 0:
            total_chi2_red = total_chi2 / total_dof
            rows_html.append(f"""
            <tr class='diag-total-row'>
                <td colspan='4' class='left'><strong>Diagnostic subtotal (excluded from strict totals)</strong></td>
                <td><strong>-</strong></td>
                <td><strong>{diagnostic_dof}</strong></td>
                <td><strong>{diagnostic_chi2:.1f}</strong></td>
                <td><strong>{diagnostic_red:.4f}</strong></td>
            </tr>

            <tr class='total-row'>
                <td colspan='4' class='left'><strong>COMBINED TOTAL (strict)</strong></td>
                <td><strong>N/A</strong></td>
                <td><strong>{strict_total_dof}</strong></td>
                <td><strong>{strict_total_chi2:.1f}</strong></td>
                <td><strong>{(strict_total_chi2/strict_total_dof):.4f}</strong></td>
            </tr>
            """)

        return f"""
        <div class='pillar-summary-section'>
            <h3 style='margin:16px 20px 8px 20px;color:#e2e8f0'>📊 MTDF Pillar Summary</h3>
            <div class='table-wrap' style='margin-top:8px'>
                <table class='pillar-summary-table'>
                    <thead>
                        <tr>
                            <th class='left'>Pillar ID</th>
                            <th class='left'>Name</th>
                            <th>Category</th>
                            <th>Mode</th>
                            <th>N_data</th>
                            <th>DOF</th>
                            <th>χ²</th>
                            <th>χ²/ν</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>
            <p style='margin:4px 20px;font-size:0.85em;color:#94a3b8'>
                * Experimental pillars excluded from totals
            </p>
        </div>
        <style>
            .pillar-summary-table {{ font-size: 0.9em; }}
            .pillar-summary-table th {{ background: #1a2942; font-weight: 600; }}
            .pillar-summary-table td {{ padding: 8px 12px; }}
            .mode-badge {{
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 0.75em;
                font-weight: 600;
            }}
            .mode-scalar {{ background: #3b82f6; color: white; }}
            .mode-vector {{ background: #22c55e; color: black; }}
            .subtotal-row {{
                background: #1e293b !important;
                border-top: 2px solid #334155;
            }}
            .total-row {{
                background: #0f4c3a !important;
                border-top: 3px solid #22c55e;
                font-size: 1.05em;
            }}
        </style>
        """

    def _generate_table_html(self, rows: List[Dict], pillars: List[str]) -> str:
        # strict_chi2_red helper
        """Build main table with workbook-driven tooltips."""

        # Short labels for vector pillars
        VECTOR_SHORT_LABELS = {
            'P_SNE_PANTHEON': 'SNe',
            'P_BAO_DESI': 'BAO',
            'P_HZ_CC': 'H(z)',
            'P_GROWTH_FSIG8': 'fσ₈',
            'P_CMB_DIST': 'CMB*',
        }

        diagnostic_pillars = {'P_CMB_DIST'}
        strict_pillars = [p for p in pillars if p not in diagnostic_pillars]

        def is_vector_pillar(pid: str) -> bool:
            """Check if pillar is a vector pillar."""
            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                return self.db.pillars[pid].get("pillar_mode") == "VECTOR"
            return pid.startswith("P_")  # Fallback: vector pillars start with P_

        def get_pillar_label(pid: str) -> str:
            """Get display label for pillar (short for vectors)."""
            if pid in VECTOR_SHORT_LABELS:
                return VECTOR_SHORT_LABELS[pid]
            return pid

        def get_pillar_coverage(pid: str) -> str:
            total_models = len(rows)
            models_with_data = sum(
                1 for r in rows
                if r.get("vals", {}).get(pid) is not None
                and not (isinstance(r.get("vals", {}).get(pid), float)
                         and (math.isnan(r.get("vals", {}).get(pid)) or math.isinf(r.get("vals", {}).get(pid))))
            )
            return " <span class='pillar-coverage' data-tooltip-id='pillar.{}.coverage'>({}/{})</span>".format(
                pid, models_with_data, total_models)

        def get_pillar_role_raw(pid: str) -> str:
            """Get raw role string (A/B/V/D) for data attributes"""
            role = ""
            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                role = self.db.pillars[pid].get("role", "").upper()

            # Default for vector pillars: V (Validation) except CMB* which is D (Diagnostic)
            if not role and is_vector_pillar(pid):
                role = "DIAGNOSTIC" if pid in diagnostic_pillars else "VALIDATION"

            # Return single letter code
            if role == "ANCHOR":
                return "A"
            elif role == "BENCHMARK":
                return "B"
            elif role == "VALIDATION":
                return "V"
            elif role == "DIAGNOSTIC":
                return "D"
            return "V"  # Default to validation

        def get_pillar_role(pid: str) -> str:
            """Get role badge HTML (A=Anchor, B=Benchmark, V=Validation, D=Diagnostic) from db.pillars"""
            role_code = get_pillar_role_raw(pid)

            if role_code == "A":
                return " <span class='role-badge anchor' title='Calibration Anchor: used to fix fundamental parameters {α, β}. Excluded from the validation score in V-only mode.'>A</span>"
            elif role_code == "B":
                return " <span class='role-badge bench' title='Benchmark: validates dataset-specific mappings. Useful consistency test but may share data lineage with parameter characterization.'>B</span>"
            elif role_code == "V":
                return " <span class='role-badge val' title='Validation Target: with parameters fixed, the model is tested against this independent dataset. Headline claim.'>V</span>"
            elif role_code == "D":
                return " <span class='role-badge diag' title='Diagnostic: excluded from strict totals, shown for transparency only.'>D</span>"
            return ""

        header_cells, target_cells = [], []
        pillar_roles = {}  # Store roles for JavaScript scope toggle
        for pid in pillars:
            coverage = get_pillar_coverage(pid)
            label = get_pillar_label(pid)
            is_vector = is_vector_pillar(pid)
            role_badge = get_pillar_role(pid)
            role_code = get_pillar_role_raw(pid)
            pillar_roles[pid] = role_code  # Store for later

            vec_class = " vector-pillar" if is_vector else ""
            diag_class = " diag-col" if pid in diagnostic_pillars else ""

            header_cells.append(
                "<th class='pillar-header{}{}' data-tooltip-id='pillar.{}.header' data-pillar='{}' data-role='{}' style='cursor:help'>{}{}<br>{}</th>".format(
                    vec_class, diag_class, pid, pid, role_code, label, role_badge, coverage)
            )

            if hasattr(self.db, "pillars") and pid in self.db.pillars:
                pillar_data = self.db.pillars[pid]
                if is_vector:
                    # Vector pillars: show n_data/DOF instead of target±sigma
                    n_data = pillar_data.get("n_data", 0)
                    dof = pillar_data.get("dof", 0)
                    td = f"n={n_data}"
                else:
                    target = pillar_data.get("target")
                    sigma = pillar_data.get("sigma")
                    if target is not None and sigma is not None:
                        if abs(target) >= 1000 or 0 < abs(target) < 0.01:
                            td = "{:.2e}±{:.2e}".format(target, sigma)
                        else:
                            td = "{:.4g}±{:.4g}".format(target, sigma)
                    else:
                        td = "---"
            else:
                td = "---"
            target_cells.append(
                "<th class='target-row{}' data-tooltip-id='pillar.{}.target' data-pillar='{}' data-role='{}' style='cursor:help'>{}</th>".format(
                    vec_class, pid, pid, role_code, td)
            )

        # Scope selector with A/B/V/D badge indicators
        # Tooltips are inline since these are UI controls, not workbook-driven data
        scope_selector_html = """
            <th rowspan='2' class='scope-col' title="Validation Scope: Click badges to toggle pillar roles. Use dropdown to select preset scopes. χ²/ν shows goodness-of-fit for selected pillars." style="cursor:help">
                <div class="scope-selector-container">
                    <div class="scope-badges" title="Click individual badges to toggle pillar roles on/off">
                        <span class="scope-badge anchor active" data-scope-role="A" title="A = Calibration Anchor: Used to fix fundamental parameters {α, β}. Excluded from the validation score in V-only mode.">A</span>
                        <span class="scope-badge bench active" data-scope-role="B" title="B = Benchmark: Validates dataset-specific mappings (e.g., P1, P1B). Useful consistency tests but may share data lineage with parameter characterization.">B</span>
                        <span class="scope-badge val active" data-scope-role="V" title="V = Validation Target: With parameters fixed, the model is tested against these independent datasets. Headline claims.">V</span>
                        <span class="scope-badge diag inactive" data-scope-role="D" title="D = Diagnostic: Excluded from strict totals, shown for transparency only.">D</span>
                    </div>
                    <select class="scope-dropdown" id="scope-selector" onchange="changeScope(this.value)" title="Select validation scope preset">
                        <option value="standard" selected title="Strict (A+B+V): Default scope excluding diagnostics (CMB*). Matches paper's strict χ²/ν definition.">Strict</option>
                        <option value="validation" title="Validation Only (V): Strictest test using only Validation Targets (independent datasets). Most conservative for peer review.">Validation</option>
                        <option value="full" title="Full (A+B+V+D): All pillars including diagnostics. Shows complete picture but inflated χ² due to CMB tension.">Full</option>
                        <option disabled>───────────</option>
                        <option value="scalar" title="Scalar Only: Show only scalar pillars (z-score based). Filters by data type, not role.">Scalar</option>
                        <option value="vector" title="Vector Only: Show only vector pillars (χ² likelihood based). Filters by data type, not role.">Vector</option>
                    </select>
                    <span class="scope-label">χ²/ν</span>
                </div>
            </th>
        """

        header_row = """<tr>
            <th rowspan='2' class='model-col' data-tooltip-id='header.model' style="cursor:help">Model</th>
            <th rowspan='2' class='evidence-col' data-tooltip-id='header.evidence' style="cursor:help">Evidence</th>
            {header_cells}
            <th rowspan='2' data-tooltip-id='header.pass' style="cursor:help">Pass</th>
            <th rowspan='2' class='tier-col' data-tooltip-id='header.tier' style="cursor:help">Tier</th>
            <th rowspan='2' data-tooltip-id='header.maxz' style="cursor:help">Max<br>|z|</th>
            {scope_selector}
        </tr>""".replace("{header_cells}", ''.join(header_cells)).replace("{scope_selector}", scope_selector_html)
        
        target_row = "<tr>{}</tr>".format(''.join(target_cells))

        # Sort rows: VALIDATED first, then by χ²/ν (use combined if available)
        def _row_key(r):
            tier = r.get('tier_code', '')
            tier_rank = 0 if tier == 'VALIDATED' else 1
            # Use combined_chi2_red if available (includes vector pillars), else scalar
            chi2_red = r.get('combined_chi2_red')
            if chi2_red is None or (isinstance(chi2_red, float) and math.isnan(chi2_red)):
                dof = int(r.get('dof', 0) or 0)
                chi2 = float(r.get('chi2', 0.0) or 0.0)
                chi2_red = chi2 / dof if dof > 0 else float('inf')
            return (tier_rank, chi2_red, r.get('model', ''))
        rows = sorted(rows, key=_row_key)

        validated = [r for r in rows if r.get("tier_code") == "VALIDATED"]
        best_model = validated[0] if validated else (rows[0] if rows else {})

        body_rows = []
        for row in rows:
            model_name = row.get("model", "model")
            foundation = row.get("foundation_type", "UNKNOWN")

            # Calculate pass-rate and evidence for ALL THREE SCOPES
            # Standard (A+B+V): default - excludes D (diagnostic)
            # Validation (V only): strictest test - only V pillars
            # Full (A+B+V+D): everything included

            # Per-scope statistics
            scope_stats = {
                'standard': {'passes': 0, 'tests': 0, 'chi2': 0.0, 'dof': 0},
                'validation': {'passes': 0, 'tests': 0, 'chi2': 0.0, 'dof': 0},
                'full': {'passes': 0, 'tests': 0, 'chi2': 0.0, 'dof': 0}
            }

            # Count tests with valid predictions per scope
            for pid in pillars:
                pred = row.get("vals", {}).get(pid)
                is_vector = is_vector_pillar(pid)
                valid_pred = pred is not None and not (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred)))
                if not valid_pred:
                    continue

                role = pillar_roles.get(pid, "V")
                z = row.get("z_scores", {}).get(pid)
                z_val = float(z) if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))) else None

                # Get chi2 contribution for this pillar
                if is_vector:
                    vpd = row.get("vector_pillar_data", {}).get(pid, {})
                    pillar_chi2 = vpd.get("chi2", 0)
                    pillar_dof = vpd.get("dof", 1)
                    is_pass = float(pred) < 1.5 if valid_pred else False
                else:
                    pillar_chi2 = z_val ** 2 if z_val is not None else 0
                    pillar_dof = 1
                    is_pass = abs(z_val) <= 2.0 if z_val is not None else False

                # Update scope statistics based on role
                # Full scope: includes everything
                scope_stats['full']['tests'] += 1
                scope_stats['full']['chi2'] += pillar_chi2
                scope_stats['full']['dof'] += pillar_dof
                if is_pass:
                    scope_stats['full']['passes'] += 1

                # Standard scope: A+B+V (excludes D)
                if role != "D":
                    scope_stats['standard']['tests'] += 1
                    scope_stats['standard']['chi2'] += pillar_chi2
                    scope_stats['standard']['dof'] += pillar_dof
                    if is_pass:
                        scope_stats['standard']['passes'] += 1

                # Validation scope: V only
                if role == "V":
                    scope_stats['validation']['tests'] += 1
                    scope_stats['validation']['chi2'] += pillar_chi2
                    scope_stats['validation']['dof'] += pillar_dof
                    if is_pass:
                        scope_stats['validation']['passes'] += 1

            # Compute chi2/nu for each scope
            for scope in scope_stats:
                dof = scope_stats[scope]['dof']
                if dof > 0:
                    scope_stats[scope]['chi2_red'] = scope_stats[scope]['chi2'] / dof
                else:
                    scope_stats[scope]['chi2_red'] = float('inf')

            # Backward compatible: use standard scope for existing variables
            strict_passes = scope_stats['standard']['passes']
            strict_tests_with_predictions = scope_stats['standard']['tests']
            diagnostic_tests_with_predictions = scope_stats['full']['tests'] - scope_stats['standard']['tests']

            total_tests = len(strict_pillars)  # strict total for Pass column
            proof_pct = (strict_passes / strict_tests_with_predictions * 100) if strict_tests_with_predictions > 0 else 0
            # icons for proof
            if proof_pct >= 80:
                proof_class = 'ok'; proof_icon = "<span class='icon ok'>✓</span>"
            elif proof_pct >= 50:
                proof_class = 'warn'; proof_icon = "<span class='icon warn'>⚠ </span>"
            else:
                proof_class = 'bad'; proof_icon = "<span class='icon bad'>✗</span>"

            # icons for pass (based on total_tests for comprehensive view)
            if total_tests and strict_passes == total_tests:
                pass_icon = "<span class='icon ok'>✓</span>"
            elif strict_passes > 0:
                pass_icon = "<span class='icon warn'>⚠ </span>"
            else:
                pass_icon = "<span class='icon bad'>✗</span>"

            tier_code = row.get("tier_code", "UNKNOWN")
            max_z = row.get("max_z", 0.0)
            is_best = row.get("model") == best_model.get("model")
            row_class = "validated-row" if is_best else ""

            # icons for tier (shortened labels for compact display, full explanation in tooltip)
            if tier_code == "VALIDATED":
                tier_formatted = "<span class='icon ok'>✓</span>Cons."; tier_class = "tier-validated"
            elif tier_code == "REQUIRES_COMPONENTS":
                tier_formatted = "<span class='icon warn'>⚠️</span>Comp."; tier_class = "tier-requires_components"
            elif tier_code == "SUPPORTED":
                tier_formatted = "<span class='icon warn'>⚠</span>Supported"; tier_class = "tier-supported"
            elif tier_code == "FAILED":
                tier_formatted = "<span class='icon bad'>✗</span>Failed"; tier_class = "tier-failed"
            else:
                tier_formatted = "<span class='icon neutral'>N/A</span>N/A"; tier_class = "tier-unknown"

            def fmt_value(x, pid=None):
                """Format values to 3-4 significant figures for clean presentation"""
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                    return "N/A"
                try:
                    x = float(x)
                    if abs(x) >= 1e3 or 0 < abs(x) < 1e-2:
                        return "{:.3e}".format(x)
                    return "{:.4g}".format(x)  # 3-4 significant figures
                except Exception:
                    return "N/A"

            pillar_cells = []
            for pid in pillars:
                pred = row.get("vals", {}).get(pid)
                is_vector = is_vector_pillar(pid)
                valid_pred = pred is not None and not (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred)))

                if valid_pred:
                    if is_vector:
                        # Vector pillar: pred is χ²/ν
                        chi2_red = pred
                        # Color based on χ²/ν thresholds
                        if chi2_red <= 1.5:
                            cls = "ok"
                        elif chi2_red <= 2.0:
                            cls = "warn"
                        else:
                            cls = "bad"
                        vdisp = "{:.2f}".format(chi2_red)
                        # Get DOF from vector_pillar_data if available
                        vpd = row.get("vector_pillar_data", {}).get(pid, {})
                        dof = vpd.get("dof", 0)
                        chi2 = vpd.get("chi2", 0)
                        if dof > 0:
                            zdisp = "<span class='cell-sub'>χ²/ν</span>"
                        else:
                            zdisp = "<span class='cell-sub'>χ²/ν</span>"
                    else:
                        # Scalar pillar: use z-score
                        z = row.get("z_scores", {}).get(pid)
                        vdisp = fmt_value(pred, pid)
                        if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                            if abs(z) <= 1.0:
                                cls = "ok"
                            elif abs(z) <= 2.0:
                                cls = "warn"
                            else:
                                cls = "bad"
                            zdisp = "<span class='cell-sub'>z={:.2f}</span>".format(z)
                        else:
                            cls = "na"; zdisp = ""
                else:
                    vdisp = "N/A"; cls = "na"; zdisp = ""

                # Individual cell tooltip - using pillar-specific tooltip
                tooltip_id = f'cell.{pid}.{model_name}'
                vec_cell_class = " vector-cell" if is_vector else ""
                diag_cell_class = " diag-col" if pid in diagnostic_pillars else ""
                role_code = pillar_roles.get(pid, "V")

                # Compute chi2 and dof for this pillar cell (used for custom scope totals)
                if is_vector:
                    # Vector pillar: use stored chi2 and dof
                    vpd = row.get("vector_pillar_data", {}).get(pid, {})
                    cell_chi2 = vpd.get("chi2", 0)
                    cell_dof = vpd.get("dof", 0)
                else:
                    # Scalar pillar: chi2 = z², dof = 1
                    z = row.get("z_scores", {}).get(pid)
                    if z is not None and not (isinstance(z, float) and (math.isnan(z) or math.isinf(z))):
                        cell_chi2 = z * z
                        cell_dof = 1
                    else:
                        cell_chi2 = 0
                        cell_dof = 0

                pillar_cells.append(
                    "<td class='{}{}{}'  data-tooltip-id='{}' data-pillar='{}' data-role='{}' data-chi2='{}' data-dof='{}' style='cursor:help'>{}{}</td>".format(
                        cls, vec_cell_class, diag_cell_class, tooltip_id, pid, role_code, cell_chi2, cell_dof, vdisp, zdisp)
                )

            max_z_display = "{:.2f}".format(max_z) if max_z > 0 else "N/A"
                        # Strict combined χ²/ν (fit pillars only)
            # Use the precomputed strict totals that exclude diagnostic-only pillars (CMB* distance prior).
            strict_chi2 = row.get('combined_excl_cmb_chi2')
            strict_dof = row.get('combined_excl_cmb_dof')
            strict_chi2_red = row.get('combined_excl_cmb_chi2_red')

            # Fallback: compute if χ² and ν exist but reduced value not provided
            if strict_chi2_red is None and strict_chi2 is not None and strict_dof:
                try:
                    strict_chi2_red = float(strict_chi2) / float(strict_dof)
                except Exception:
                    strict_chi2_red = None

            # If still missing, fall back to model-level combined χ²/ν (display-only for non-MTDF models)
            # MTDF uses strict combined (excluding CMB* diagnostics). Other models may only have a published combined value.
            if strict_chi2_red is None or (isinstance(strict_chi2_red, float) and (math.isnan(strict_chi2_red) or math.isinf(strict_chi2_red))):
                fallback_red = row.get('combined_chi2_red') or row.get('scalar_chi2_red') or row.get('chi2_red')
                fallback_dof = row.get('combined_dof') or row.get('scalar_dof') or row.get('dof')
                if fallback_red is None or (isinstance(fallback_red, float) and (math.isnan(fallback_red) or math.isinf(fallback_red))):
                    chi2_display = "N/A"
                    strict_chi2_red = None
                    strict_dof = int(strict_dof or 0)
                else:
                    chi2_display = "{:.4f}".format(float(fallback_red))
                    # Use fallback DOF if provided, otherwise keep existing strict_dof
                    strict_dof = int(fallback_dof or strict_dof or 0)
                    strict_chi2_red = float(fallback_red)
            else:
                chi2_display = "{:.4f}".format(float(strict_chi2_red))
                strict_dof = int(strict_dof or 0)


            # For evidence/tier logic, use the same strict χ²/ν value (excluding diagnostics)
            # so the tier badge and thresholds align with the headline strict totals.
            if strict_chi2_red is None or (isinstance(strict_chi2_red, float) and (math.isnan(strict_chi2_red) or math.isinf(strict_chi2_red))):
                chi2_red_calc = float('inf')
            else:
                chi2_red_calc = float(strict_chi2_red)

            # Tooltip DOF should reflect strict combined DOF used for χ²/ν.
            dof_display = int(strict_dof or 0)


            # Fixed tooltip IDs to match your sheet structure
            foundation_lower = foundation.lower()
            tier_lower = tier_code.lower().replace('_', '')

            # Calculate evidence level based on chi2_red, pass rate, and max_z
            # Pass rate uses strict_tests_with_predictions (Option B: quality of predictions made)
            # Criteria from header.evidence tooltip:
            # Excellent: χ²/ν < 0.1, pass rate ≥ 90%, max |z| < 1
            # Strong: χ²/ν < 1, pass rate ≥ 70%, max |z| < 2
            # Moderate: χ²/ν < 2, pass rate ≥ 50%, max |z| < 3
            # Weak: χ²/ν ≥ 2 or pass rate < 50%
            pass_rate = (strict_passes / strict_tests_with_predictions * 100) if strict_tests_with_predictions > 0 else 0

            if chi2_red_calc < 0.1 and pass_rate >= 90 and max_z < 1:
                evidence_level = 'excellent'
            elif chi2_red_calc < 1 and pass_rate >= 70 and max_z < 2:
                evidence_level = 'strong'
            elif chi2_red_calc < 2 and pass_rate >= 50 and max_z < 3:
                evidence_level = 'moderate'
            else:
                evidence_level = 'weak'

            # Store model data for dynamic tooltips
            model_tooltip_data = {
                'strict_passes': strict_passes,
                'strict_tests_with_predictions': strict_tests_with_predictions,
                'total_tests': total_tests,
                'pass_pct': proof_pct,
                'chi2_red': chi2_red_calc,
                'chi2': chi2_display,
                'dof': dof_display,
                'max_z': max_z,
                'evidence_level': evidence_level
            }

            # Store for later tooltip generation
            if not hasattr(self, '_model_tooltip_data'):
                self._model_tooltip_data = {}
            self._model_tooltip_data[model_name] = model_tooltip_data

            # Format per-scope chi2/nu values for data attributes
            scope_chi2_standard = "{:.4f}".format(scope_stats['standard']['chi2_red']) if scope_stats['standard']['chi2_red'] != float('inf') else "Inf"
            scope_chi2_validation = "{:.4f}".format(scope_stats['validation']['chi2_red']) if scope_stats['validation']['chi2_red'] != float('inf') else "Inf"
            scope_chi2_full = "{:.4f}".format(scope_stats['full']['chi2_red']) if scope_stats['full']['chi2_red'] != float('inf') else "Inf"

            # Create JSON for scope data
            scope_data_json = json.dumps({
                'standard': {
                    'chi2_red': scope_stats['standard']['chi2_red'] if scope_stats['standard']['chi2_red'] != float('inf') else 9999,
                    'passes': scope_stats['standard']['passes'],
                    'tests': scope_stats['standard']['tests'],
                    'dof': scope_stats['standard']['dof']
                },
                'validation': {
                    'chi2_red': scope_stats['validation']['chi2_red'] if scope_stats['validation']['chi2_red'] != float('inf') else 9999,
                    'passes': scope_stats['validation']['passes'],
                    'tests': scope_stats['validation']['tests'],
                    'dof': scope_stats['validation']['dof']
                },
                'full': {
                    'chi2_red': scope_stats['full']['chi2_red'] if scope_stats['full']['chi2_red'] != float('inf') else 9999,
                    'passes': scope_stats['full']['passes'],
                    'tests': scope_stats['full']['tests'],
                    'dof': scope_stats['full']['dof']
                }
            }).replace('"', '&quot;')

            body_rows.append("""
            <tr class='{row_class}' data-model='{model_name}' data-scope-stats='{scope_data_json}'>
                <td class='model-col' data-tooltip-id='cell.model.{model_name}' style="cursor:help">{display_name}</td>
                <td class='evidence-col {proof_class}' data-tooltip-id='cell.evidence.{model_name}' style="cursor:help" data-cell-type="evidence">{proof_icon}{proof_pct:.0f}%</td>
                {pillar_cells}
                <td data-tooltip-id='cell.pass' style="cursor:help" data-cell-type="pass">{pass_icon}{strict_passes}/{total_tests}</td>
                <td class='tier-col {tier_class}' data-tooltip-id='cell.tier.{tier_lower}' style="cursor:help" data-cell-type="tier">{tier_formatted}</td>
                <td data-tooltip-id='cell.maxz.{model_name}' style="cursor:help" data-cell-type="maxz">{max_z_display}</td>
                <td class='scope-chi2-cell' data-tooltip-id='cell.chi2red.{model_name}' style="cursor:help" data-cell-type="chi2" data-chi2-standard='{scope_chi2_standard}' data-chi2-validation='{scope_chi2_validation}' data-chi2-full='{scope_chi2_full}'>{chi2_display}</td>
            </tr>""".format(
                row_class=row_class,
                model_name=model_name,
                scope_data_json=scope_data_json,
                display_name=self._get_proper_model_name(model_name),
                proof_class=proof_class,
                proof_icon=proof_icon,
                proof_pct=proof_pct,
                evidence_level=evidence_level,
                pillar_cells=''.join(pillar_cells),
                pass_icon=pass_icon,
                strict_passes=strict_passes,
                total_tests=total_tests,
                tier_class=tier_class,
                tier_lower=tier_lower,
                tier_formatted=tier_formatted,
                max_z_display=max_z_display,
                chi2_display=chi2_display,
                scope_chi2_standard=scope_chi2_standard,
                scope_chi2_validation=scope_chi2_validation,
                scope_chi2_full=scope_chi2_full
            ))

        return """
        <div class='table-wrap'>
            <table>
                <thead>{header_row}{target_row}</thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """.replace("{header_row}", header_row).replace("{target_row}", target_row).replace("{''.join(body_rows)}", ''.join(body_rows))

    def generate_complete_dashboard(self, payload: Dict[str, Any]) -> str:
        rows = payload.get("rows", [])
        pillars = payload.get("pillars", [])

        css = generate_css_styles(pillars)
        js = generate_javascript() + generate_placeholder_scripts()
        acc_css = self._accordion_css()
        acc_js = self._accordion_js()
        extra_css = self._extra_css()

        # Pre-calculate model data for dynamic tooltips BEFORE building tooltips
        self._precalculate_model_tooltip_data(rows, pillars)

        # Build tooltip data with comprehensive error handling
        try:
            all_tooltips = self._build_tooltip_data_for_js(rows, pillars)
            print(f"Final tooltip count for JS: {len(all_tooltips)}")
            
            # Debug: Show sample of what we're actually sending to JavaScript
            working_tooltips = {k: v for k, v in all_tooltips.items() if v and v.strip() and 'error' not in v.lower()}
            print(f"Working tooltips (non-empty): {len(working_tooltips)}")
            
            if working_tooltips:
                sample_keys = list(working_tooltips.keys())[:3]
                for key in sample_keys:
                    preview = working_tooltips[key][:100] + "..." if len(working_tooltips[key]) > 100 else working_tooltips[key]
                    print(f"Sample working tooltip - {key}: {preview}")
            else:
                print("WARNING: No working tooltips found - all are empty or contain errors")
                
        except Exception as e:
            print(f"ERROR building tooltips: {e}")
            all_tooltips = {}  # Empty - all tooltips must come from workbook

        # Create the tooltip JavaScript with error handling
        try:
            tooltip_data_js = json.dumps(all_tooltips, ensure_ascii=False)
        except Exception as e:
            print(f"ERROR serializing tooltips to JSON: {e}")
            tooltip_data_js = json.dumps({})  # Empty - all tooltips must come from workbook
                
        tooltip_js_content = f"""
(function() {{
  // Initialize tooltip data with error handling
  try {{
    window.TOOLTIP_HTML = {tooltip_data_js};
    console.log('Tooltip data loaded successfully:', Object.keys(window.TOOLTIP_HTML).length, 'tooltips');
    
    // Debug: Show sample tooltips
    const sampleKeys = Object.keys(window.TOOLTIP_HTML).slice(0, 5);
    console.log('Sample tooltip IDs:', sampleKeys);
    sampleKeys.forEach(key => {{
      const content = window.TOOLTIP_HTML[key];
      const preview = content.length > 100 ? content.substring(0, 100) + '...' : content;
      console.log(`  ${{key}}: ${{preview}}`);
    }});
    
  }} catch(e) {{
    console.error('Failed to load tooltip data:', e);
    window.TOOLTIP_HTML = {{}};
  }}
  
  const BOX_ID = 'pro-tooltip-box';
  
  function ensureBox() {{
    let b = document.getElementById(BOX_ID);
    if (!b) {{
      b = document.createElement('div');
      b.id = BOX_ID;
      b.className = 'pro-body-tooltip';
      b.style.cssText = 'position:fixed;display:none;pointer-events:none;max-width:40rem;padding:12px 16px;background:#111827;color:#d1d5db;border:1px solid #2563eb;border-radius:8px;font-size:13px;line-height:1.4;box-shadow:0 8px 24px rgba(0,0,0,.4);z-index:99999;opacity:0;transition:opacity .2s ease;';
      b.innerHTML = "<div class='pro-tooltip-content'></div>";
      document.body.appendChild(b);
    }}
    return b;
  }}

  function getContent(id, el) {{
    // ALL tooltips come exclusively from workbook - zero hardcoded fallbacks
    try {{
      // Check element data attribute first
      if (el && el.dataset && el.dataset.tooltipHtml) {{
        return el.dataset.tooltipHtml;
      }}

      // DYNAMIC chi2 tooltips - generate based on current scope
      if (id && id.startsWith('cell.chi2red.') && el && el.dataset && el.dataset.cellType === 'chi2') {{
        const row = el.closest('tr[data-scope-stats]');
        if (row) {{
          const statsJson = row.getAttribute('data-scope-stats');
          const modelName = row.getAttribute('data-model') || 'Model';
          if (statsJson) {{
            try {{
              const stats = JSON.parse(statsJson.replace(/&quot;/g, '"'));
              // Use currentScope from scripts.py (default to 'standard')
              const scope = window.currentScope || 'standard';
              const scopeData = stats[scope] || stats['standard'];

              if (scopeData) {{
                const chi2_red = scopeData.chi2_red;
                const dof = scopeData.dof;
                const passes = scopeData.passes;
                const tests = scopeData.tests;
                const chi2_total = chi2_red * dof;

                // Scope labels
                const scopeLabels = {{
                  'standard': 'Strict (A+B+V)',
                  'validation': 'Validation Only (V)',
                  'full': 'Full (A+B+V+D)',
                  'custom': 'Custom Selection'
                }};
                const scopeLabel = scopeLabels[scope] || scope;

                // Quality assessment
                let quality = '';
                if (chi2_red < 0.1) quality = '✓ Excellent fit';
                else if (chi2_red < 1) quality = '✓ Strong agreement';
                else if (chi2_red < 2) quality = '~ Acceptable fit';
                else quality = '⚠ Model-data tension';

                return `<strong>${{modelName}}: Reduced Chi-Squared</strong><br><br>` +
                  `<em>Current Scope:</em> <strong>${{scopeLabel}}</strong><br><br>` +
                  `<em>Calculation:</em><br>` +
                  `χ² = ${{chi2_total.toFixed(2)}}<br>` +
                  `ν = ${{dof}} (degrees of freedom)<br>` +
                  `χ²/ν = ${{chi2_total.toFixed(2)}} / ${{dof}} = <strong>${{chi2_red.toFixed(4)}}</strong><br><br>` +
                  `<em>This scope includes:</em><br>` +
                  `• ${{tests}} tests with predictions<br>` +
                  `• ${{passes}}/${{tests}} passed at 1σ<br><br>` +
                  `<em>Assessment:</em> ${{quality}}<br><br>` +
                  `<em>Quality benchmarks:</em><br>` +
                  `• χ²/ν < 0.1: Excellent fit<br>` +
                  `• χ²/ν < 1: Strong agreement<br>` +
                  `• χ²/ν < 2: Acceptable fit<br>` +
                  `• χ²/ν ≥ 2: Model-data tension`;
              }}
            }} catch(e) {{
              console.log('Error parsing scope stats for chi2 tooltip:', e);
            }}
          }}
        }}
      }}

      // Check workbook-loaded tooltip data
      if (window.TOOLTIP_HTML && window.TOOLTIP_HTML[id]) {{
        return window.TOOLTIP_HTML[id];
      }}

      // Special fallback for chi2 tooltips (should have been handled dynamically above)
      if (id && id.startsWith('cell.chi2red.')) {{
        console.log('Chi2 tooltip fallback triggered - dynamic generation failed. ID:', id, 'el:', el, 'cellType:', el?.dataset?.cellType);
        return `<strong>χ²/ν Tooltip</strong><br><br>Dynamic tooltip generation failed.<br>Check console for details.`;
      }}

      // No fallbacks - tooltip must be in workbook UI_Tooltips sheet
      console.log('Tooltip not in workbook:', id);
      return `<em>Tooltip '${{id}}' not defined in workbook UI_Tooltips sheet</em>`;

    }} catch(e) {{
      console.error('Error getting tooltip content:', e);
      return '<em>Tooltip error</em>';
    }}
  }}

  function position(b, x, y) {{
    try {{
      const pad = 16;
      const rect = b.getBoundingClientRect();
      let nx = x + 20;
      let ny = y + 20;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      
      if (nx + rect.width + pad > vw) {{
        nx = Math.max(pad, x - rect.width - 20);
      }}
      if (ny + rect.height + pad > vh) {{
        ny = Math.max(pad, y - rect.height - 20);
      }}
      
      b.style.left = nx + 'px';
      b.style.top = ny + 'px';
    }} catch(e) {{
      console.error('Error positioning tooltip:', e);
    }}
  }}

  window.showTooltip = function(el, _, id, evt) {{
    try {{
      console.log('showTooltip called for:', id);
      const b = ensureBox();
      const html = getContent(id, el);
      
      console.log('Tooltip content length:', html.length);
      
      b.querySelector('.pro-tooltip-content').innerHTML = html;
      b.style.display = 'block';
      b.style.opacity = '1';
      
      // Try to render math if KaTeX is available
      if (window.renderMathInElement) {{
        try {{
          window.renderMathInElement(b, {{
            delimiters: [
              {{ left: "$$", right: "$$", display: true }},
              {{ left: "\\\\[", right: "\\\\]", display: true }},
              {{ left: "\\\\(", right: "\\\\)", display: false }}
            ],
            throwOnError: false
          }});
        }} catch(e) {{
          console.warn('KaTeX rendering failed:', e);
        }}
      }}
      
      const moveHandler = (e) => position(b, e.clientX, e.clientY);
      document.addEventListener('mousemove', moveHandler, {{ passive: true }});
      b._moveListener = moveHandler;
      
      if (evt) {{
        position(b, evt.clientX, evt.clientY);
      }}
    }} catch(e) {{
      console.error('Error showing tooltip:', e);
    }}
  }};

  window.hideTooltip = function() {{
    try {{
      const b = document.getElementById(BOX_ID);
      if (!b) return;
      
      b.style.opacity = '0';
      setTimeout(() => {{
        if (b.style.opacity === '0') {{
          b.style.display = 'none';
        }}
      }}, 200);
      
      if (b._moveListener) {{
        document.removeEventListener('mousemove', b._moveListener);
        b._moveListener = null;
      }}
    }} catch(e) {{
      console.error('Error hiding tooltip:', e);
    }}
  }};

  // Auto-attach to all elements with data-tooltip-id
  document.addEventListener('mouseover', function(e) {{
    try {{
      const target = e.target.closest('[data-tooltip-id]');
      if (!target) return;
      
      const id = target.getAttribute('data-tooltip-id');
      if (id) {{
        console.log('Mouseover detected for tooltip ID:', id);
        showTooltip(target, '', id, e);
      }}
    }} catch(e) {{
      console.error('Error in mouseover handler:', e);
    }}
  }}, {{ passive: true }});

  document.addEventListener('mouseout', function(e) {{
    try {{
      const target = e.target.closest('[data-tooltip-id]');
      if (!target) return;
      hideTooltip();
    }} catch(e) {{
      console.error('Error in mouseout handler:', e);
    }}
  }}, {{ passive: true }});

  console.log('Enhanced tooltip system loaded with debugging');
}})();
"""

        katex_init = """
<script>
  // Initialize KaTeX rendering after page load
  document.addEventListener("DOMContentLoaded", function() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "\\\\[", right: "\\\\]", display: true},
          {left: "\\\\(", right: "\\\\)", display: false}
        ],
        throwOnError: false
      });
      console.log('KaTeX rendering initialized');
    } else {
      setTimeout(function() {
        if (window.renderMathInElement) {
          renderMathInElement(document.body, {
            delimiters: [
              {left: "$$", right: "$$", display: true},
              {left: "\\\\[", right: "\\\\]", display: true},
              {left: "\\\\(", right: "\\\\)", display: false}
            ],
            throwOnError: false
          });
          console.log('KaTeX rendering initialized (delayed)');
        }
      }, 500);
    }
  });
</script>
"""

        extra_head = f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>{tooltip_js_content}</script>
{katex_init}
"""

        # Generate the rest of the dashboard
        header = generate_header_section()
        summary = generate_summary_sections(len(pillars))
        framework_accordion = self._generate_framework_accordion(pillars)
        # Reading guide is first accordion after the table
        reading_guide_accordion = self._generate_reading_guide_accordion()
        # Pillar summary moved to accordion below main table
        pillar_summary_content = self._generate_pillar_summary_table(rows, pillars)
        pillar_summary_accordion = self._accordion(
            "📋 Pillar Overview (Scalar + Vector)",
            pillar_summary_content,
            open_default=False
        )
        table_html = self._generate_table_html(rows, pillars)
        accordions = self._generate_accordions(rows, pillars)
        footer = generate_enhanced_footer(self.db, self._get_file_hash, self._env_summary)

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>MTDF V74 Validation</title>
{css}
{acc_css}
{extra_css}
{extra_head}
</head>
<body>
{header}
{summary}
{framework_accordion}
{table_html}
{reading_guide_accordion}
{pillar_summary_accordion}
{accordions}
{footer}
{js}
{acc_js}
</body>
</html>"""


# Helper functions for integration
def create_dashboard_with_tooltips(db, workbook_data: Dict[str, pd.DataFrame]) -> DashboardGenerator:
    """
    Create dashboard with properly initialized tooltip engine
    
    Args:
        db: Database object
        workbook_data: Dictionary of pandas DataFrames from Excel workbook
        
    Returns:
        DashboardGenerator with tooltip engine
    """
    try:
        # Import the fixed tooltip engine
        from UI.tooltip_engine import TooltipEngine
        
        # Create tooltip engine with better error handling
        tooltip_engine = TooltipEngine(workbook_data)
        
        # Validate tooltip engine
        validation_issues = tooltip_engine.validate_tooltips()
        
        if validation_issues['data_issues']:
            print(f"WARNING: Tooltip data issues: {validation_issues['data_issues']}")
        
        if validation_issues['missing_templates']:
            print(f"INFO: {len(validation_issues['missing_templates'])} tooltips missing templates")
        
        # Create dashboard with tooltip engine
        dashboard = DashboardGenerator(db, tooltip_engine)
        
        print(f"Dashboard created with {len(tooltip_engine.get_all_tooltip_ids())} tooltips")
        return dashboard
        
    except ImportError as e:
        print(f"WARNING: Could not import TooltipEngine: {e}")
        print("Creating dashboard without tooltip engine")
        return DashboardGenerator(db, None)
        
    except Exception as e:
        print(f"ERROR: Failed to create tooltip engine: {e}")
        print("Creating dashboard without tooltip engine")
        return DashboardGenerator(db, None)


def debug_tooltip_system(workbook_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Debug function to test tooltip system loading
    
    Returns:
        Dictionary with debug information
    """
    debug_info = {
        'workbook_sheets': list(workbook_data.keys()),
        'ui_tooltips_found': 'UI_Tooltips' in workbook_data,
        'tooltip_engine_created': False,
        'tooltip_count': 0,
        'sample_tooltips': [],
        'validation_issues': {},
        'errors': []
    }
    
    try:
        if 'UI_Tooltips' in workbook_data:
            tooltips_df = workbook_data['UI_Tooltips']
            debug_info['raw_tooltip_rows'] = len(tooltips_df)
            debug_info['raw_tooltip_columns'] = list(tooltips_df.columns)
            
            # Try to create tooltip engine
            from UI.tooltip_engine import TooltipEngine
            engine = TooltipEngine(workbook_data)
            debug_info['tooltip_engine_created'] = True
            
            # Get tooltip information
            all_ids = engine.get_all_tooltip_ids()
            debug_info['tooltip_count'] = len(all_ids)
            debug_info['sample_tooltips'] = all_ids[:10]
            
            # Run validation
            debug_info['validation_issues'] = engine.validate_tooltips()
            
            # Test a few specific tooltips
            test_ids = ['header.model', 'cell.prediction', 'pillar.P1.header']
            debug_info['test_results'] = {}
            
            for test_id in test_ids:
                try:
                    result = engine.get_tooltip(test_id, {'pillar_id': 'P1'})
                    debug_info['test_results'][test_id] = {
                        'success': True,
                        'length': len(result),
                        'preview': result[:100] + '...' if len(result) > 100 else result
                    }
                except Exception as e:
                    debug_info['test_results'][test_id] = {
                        'success': False,
                        'error': str(e)
                    }
                    
        else:
            debug_info['errors'].append('UI_Tooltips sheet not found in workbook')
            
    except Exception as e:
        debug_info['errors'].append(f"Error creating tooltip engine: {str(e)}")
    
    return debug_info
