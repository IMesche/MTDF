#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
UI/tooltip_engine.py - COMPLETE FIX FOR YOUR EXCEL FORMAT
Dynamic tooltip engine that properly handles your UI_Tooltips sheet structure
"""

import pandas as pd
import re
from typing import Dict, Any, Optional, List
import html
import json

class TooltipEngine:
    def __init__(self, workbook_data: Dict[str, pd.DataFrame]):
        """
        Initialize tooltip engine with workbook data
        
        Args:
            workbook_data: Dictionary containing all Excel sheets including UI_Tooltips
        """
        self.workbook_data = workbook_data
        
        # Handle the UI_Tooltips sheet structure
        raw_df = workbook_data.get('UI_Tooltips', pd.DataFrame())
        
        if not raw_df.empty:
            print(f"Raw UI_Tooltips DataFrame shape: {raw_df.shape}")
            print(f"Raw columns: {list(raw_df.columns)}")
            
            # Check if first row contains title "UI Tooltips" instead of data
            first_cell = raw_df.iloc[0, 0] if len(raw_df) > 0 else None
            is_title_row = str(first_cell).strip().lower() in ["ui tooltips", "ui_tooltips"]
            
            if is_title_row and len(raw_df) > 2:
                print("Detected title row structure - applying fix")
                # Row 0: Title "UI Tooltips"  
                # Row 1: Headers
                # Row 2+: Data
                
                # Get headers from row 1
                header_row = raw_df.iloc[1]
                
                # Get data starting from row 2
                data_rows = raw_df.iloc[2:].copy()
                
                # Set the proper column names
                data_rows.columns = header_row.values
                
                # Reset index
                data_rows.reset_index(drop=True, inplace=True)
                
                self.tooltips_df = data_rows
                print(f"Fixed DataFrame shape: {self.tooltips_df.shape}")
                print(f"Fixed columns: {list(self.tooltips_df.columns)}")
                
            else:
                # DataFrame is already in correct format
                self.tooltips_df = raw_df.copy()
                print("Using DataFrame as-is (no title row detected)")
            
            # Handle 'active' column conversion properly
            if 'active' in self.tooltips_df.columns:
                # Convert various representations of "true" to boolean True
                def convert_active(val):
                    if pd.isna(val):
                        return False
                    # Handle numeric values (1.0 = True, 0.0 = False)
                    if isinstance(val, (int, float)):
                        return val > 0
                    # Handle string values
                    val_str = str(val).strip().lower()
                    return val_str in ['true', '1', '1.0', 'yes', 't', 'y']
                
                self.tooltips_df['active'] = self.tooltips_df['active'].map(convert_active)
                
                active_count = self.tooltips_df['active'].sum()
                print(f"Active tooltips after conversion: {active_count}")
                
                # Show sample of active tooltip IDs
                if active_count > 0:
                    active_tooltips = self.tooltips_df[self.tooltips_df['active'] == True]
                    sample_ids = active_tooltips['tooltip_id'].head(10).tolist()
                    print(f"Sample active tooltip IDs: {sample_ids}")
                    
        else:
            self.tooltips_df = pd.DataFrame()
            print("WARNING: UI_Tooltips sheet is empty or missing")
        
        self.pillar_cache = {}
        self.params_cache = {}
        self._build_pillar_cache()
        self._build_params_cache()
    
    def _fix_sheet_headers(self, df, sheet_name):
        """Fix sheets that have title row instead of headers"""
        if df.empty:
            return df
            
        # Check if first row looks like a title (long text with description)
        first_cell = str(df.iloc[0, 0]) if len(df) > 0 else ""
        
        # If first cell contains "Defines" or is very long, it's likely a title row
        if "Defines" in first_cell or len(first_cell) > 50:
            print(f"  Fixing headers for {sheet_name} sheet")
            
            # The actual headers are in row 0 (after title), data starts at row 1
            if len(df) > 1:
                # Get the actual headers from what pandas thinks is the first data row
                actual_headers = df.iloc[0].values
                
                # Get the data starting from what pandas thinks is row 1
                data_df = df.iloc[1:].copy()
                
                # Set the correct column names
                data_df.columns = actual_headers
                
                # Reset index
                data_df.reset_index(drop=True, inplace=True)
                
                print(f"    Fixed columns: {list(data_df.columns)[:5]}...")
                return data_df
        
        return df
    
    def _build_pillar_cache(self):
        """Build cache of pillar data for fast lookup"""
        # Cache pillar formulas
        if 'Pillar_Formulas' in self.workbook_data:
            formulas_df = self.workbook_data['Pillar_Formulas']
            print(f"Loading {len(formulas_df)} pillar formulas")
            for _, row in formulas_df.iterrows():
                pid = row.get('pillar_id', '')
                if pid:
                    # Store all formula data with various key aliases
                    formula_data = {
                        'pillar_id': pid,
                        'pillar_name': row.get('name', ''),
                        'name': row.get('name', ''),
                        'latex': row.get('latex', ''),
                        'latex_formula': row.get('latex', ''),
                        'python_expr': row.get('python_expr', row.get('expression', '')),
                        'expression': row.get('expression', row.get('python_expr', '')),
                        'description': row.get('description', ''),
                        'reference_doi': row.get('reference_doi', ''),
                        'doi': row.get('reference_doi', '')
                    }
                    self.pillar_cache[f"{pid}_formula"] = formula_data
                    # Also store directly under pillar ID for easier access
                    self.pillar_cache[pid] = formula_data
        
        # Cache pillar targets
        if 'Pillar_Targets' in self.workbook_data:
            targets_df = self.workbook_data['Pillar_Targets']
            print(f"Loading {len(targets_df)} pillar targets")
            for _, row in targets_df.iterrows():
                target_id = row.get('target_id', '')
                pid = target_id.replace('target:', '') if target_id else ''
                if pid:
                    # Get values with proper type conversion
                    target_value = row.get('value', '')
                    uncertainty_value = row.get('uncertainty', '')
                    
                    # Convert to float if possible
                    try:
                        target_value = float(target_value) if target_value != '' else target_value
                    except (TypeError, ValueError):
                        pass
                    
                    try:
                        uncertainty_value = float(uncertainty_value) if uncertainty_value != '' else uncertainty_value
                    except (TypeError, ValueError):
                        pass
                    
                    target_data = {
                        'pillar_id': pid,
                        'target': target_value,
                        'target_value': target_value,
                        'observed': target_value,  # Numeric value alias
                        'sigma': uncertainty_value,
                        'uncertainty': uncertainty_value,
                        'sigma_target': uncertainty_value,  # Alias
                        'target_uncert': uncertainty_value,  # Template placeholder
                        'unit': row.get('unit', ''),
                        'units': row.get('unit', ''),
                        'target_unit': row.get('unit', ''),  # Template placeholder
                        'source_doi': row.get('source_doi', ''),
                        'doi': row.get('source_doi', ''),
                        'reference_doi': row.get('source_doi', ''),  # Template placeholder
                        'dataset': row.get('dataset', ''),
                        'target_dataset': row.get('dataset', ''),  # Template placeholder
                        'observable': row.get('observable', ''),
                        'target_observable': row.get('observable', '')  # Template placeholder - the NAME of what's measured
                    }
                    
                    # Debug print to see what we're storing
                    if pid in ['P1', 'P1B']:
                        print(f"  Storing for {pid}: target={target_value}, uncertainty={uncertainty_value}")
                    
                    self.pillar_cache[f"{pid}_target"] = target_data
                    # Merge with existing pillar data if present
                    if pid in self.pillar_cache:
                        self.pillar_cache[pid].update(target_data)
                    else:
                        self.pillar_cache[pid] = target_data
        
        # Fix and cache pillar tests
        if 'Pillar_Tests' in self.workbook_data:
            # Apply header fix if needed
            tests_df = self.workbook_data['Pillar_Tests']
            if len(tests_df) > 0:
                first_cell = str(tests_df.iloc[0, 0]) if len(tests_df.columns) > 0 else ""
                if "Defines" in first_cell or len(first_cell) > 50:
                    tests_df = self._fix_sheet_headers(tests_df, 'Pillar_Tests')
                    
            print(f"Loading {len(tests_df)} pillar tests")
            for _, row in tests_df.iterrows():
                pid = row.get('pillar_id', '')
                if pid:
                    test_data = {
                        'pillar_id': pid,
                        'pillar_name': row.get('name', ''),  # For coverage templates
                        'what': row.get('what', ''),
                        'why': row.get('why', ''),
                        'how': row.get('how', ''),
                        'when': row.get('when', ''),
                        'caveats': row.get('caveats', ''),  # For coverage templates
                        'proof_condition_text': row.get('proof_condition_text', ''),
                        'test_description': row.get('what', '')  # Alias
                    }
                    self.pillar_cache[f"{pid}_test"] = test_data
                    # Merge with existing pillar data
                    if pid in self.pillar_cache:
                        self.pillar_cache[pid].update(test_data)
                    else:
                        self.pillar_cache[pid] = test_data
        
        print(f"Pillar cache built with {len(self.pillar_cache)} entries")
    
    def _build_params_cache(self):
        """Build cache of parameter data"""
        param_sheets = ['Params_Fundamental', 'Params_Constants', 'Params_Observational', 'Params_Coefficients']
        
        for sheet_name in param_sheets:
            if sheet_name in self.workbook_data:
                df = self.workbook_data[sheet_name]
                for _, row in df.iterrows():
                    param_name = row.get('name', row.get('parameter', ''))
                    if param_name:
                        self.params_cache[param_name] = {
                            'value': row.get('value', ''),
                            'unit': row.get('unit', ''),
                            'description': row.get('description', '')
                        }
        
        print(f"Parameters cache built with {len(self.params_cache)} entries")
    
    def get_tooltip(self, tooltip_id: str, context: Dict[str, Any] = None) -> str:
        """
        Get rendered tooltip for given ID with improved error handling
        """
        if context is None:
            context = {}
        
        try:
            # Check if tooltips are loaded
            if self.tooltips_df.empty:
                return self._get_fallback_tooltip(tooltip_id, context)
            
            # Check required columns
            if 'tooltip_id' not in self.tooltips_df.columns:
                return self._get_fallback_tooltip(tooltip_id, context)
            
            # Find matching tooltip
            if 'active' in self.tooltips_df.columns:
                matching_tooltips = self.tooltips_df[
                    (self.tooltips_df['tooltip_id'] == tooltip_id) &
                    (self.tooltips_df['active'] == True)
                ]
            else:
                matching_tooltips = self.tooltips_df[
                    self.tooltips_df['tooltip_id'] == tooltip_id
                ]

            if 'priority' in self.tooltips_df.columns:
                matching_tooltips = matching_tooltips.sort_values('priority')

            # Process matching tooltips
            for _, tooltip_row in matching_tooltips.iterrows():
                if self._evaluate_show_condition(tooltip_row, context):
                    result = self._render_tooltip(tooltip_row, context)

                    # Check for actual error messages, not scientific terms like "measurement-error"
                    if result and result.strip():
                        has_error = any(err in result.lower() for err in ['error:', 'error -', 'missing data', 'data missing'])
                        if not has_error:
                            return result
            
            # Try fallback in tooltip data
            if not matching_tooltips.empty:
                fallback_id = matching_tooltips.iloc[0].get('fallback_tooltip_id')
                if pd.notna(fallback_id) and fallback_id.strip() and fallback_id != '-':
                    return self.get_tooltip(fallback_id, context)

            # Smart fallback for model variants (e.g., "MTDF (EFE)" -> "MTDF")
            if matching_tooltips.empty and '(' in tooltip_id:
                # Try stripping parenthetical suffix from model name
                # e.g., "cell.model.MTDF (EFE)" -> "cell.model.MTDF"
                # e.g., "cell.P1.MTDF (EFE)" -> "cell.P1.MTDF"
                import re
                base_id = re.sub(r'\s*\([^)]+\)$', '', tooltip_id)
                if base_id != tooltip_id:
                    base_result = self.get_tooltip(base_id, context)
                    # Only use if it's not a generic fallback
                    if base_result and '<em>Cell information</em>' not in base_result:
                        return base_result

            return self._get_fallback_tooltip(tooltip_id, context)
            
        except Exception as e:
            print(f"ERROR in get_tooltip for {tooltip_id}: {e}")
            return self._get_fallback_tooltip(tooltip_id, context)
    
    def _get_fallback_tooltip(self, tooltip_id: str, context: Dict[str, Any] = None) -> str:
        """
        Minimal fallback for tooltips not found in workbook.

        All tooltip content should be defined in UI_Tooltips sheet.
        This fallback only provides a generic shell when workbook lookup fails.
        """
        if context is None:
            context = {}

        # Extract any useful info from tooltip_id for the generic message
        parts = tooltip_id.split('.')

        # For pillar tooltips, try to get name from cache
        if tooltip_id.startswith('pillar.') and len(parts) >= 2:
            pillar_id = parts[1]
            pillar_data = self.pillar_cache.get(pillar_id, {})
            pillar_name = pillar_data.get('pillar_name', pillar_data.get('name', pillar_id))

            tooltip_type = parts[2] if len(parts) > 2 else 'info'
            return f"<strong>{pillar_name}</strong><br><em>{tooltip_type.title()} information</em>"

        # For header/cell tooltips, extract the column name
        if len(parts) >= 2 and parts[0] in ['header', 'cell']:
            col_name = parts[1].replace('_', ' ').title()
            tooltip_type = parts[0].title()
            return f"<strong>{col_name}</strong><br><em>{tooltip_type} information</em>"

        # Ultimate generic fallback
        return f"<em>{tooltip_id}</em>"
    
    def _evaluate_show_condition(self, tooltip_row: pd.Series, context: Dict[str, Any]) -> bool:
        """Evaluate whether tooltip should be shown based on conditions"""
        try:
            field1 = tooltip_row.get('show_when_field')
            if pd.isna(field1) or str(field1).strip() in ['-', '', 'nan']:
                return True
            
            value1 = context.get(str(field1).strip())
            operator1_raw = tooltip_row.get('show_when_operator', '==')
            # Handle NaN operator - default to '=='
            operator1 = '==' if pd.isna(operator1_raw) or str(operator1_raw).strip() in ['', 'nan', '-'] else str(operator1_raw).strip()
            expected1 = tooltip_row.get('show_when_value')
            
            if not self._evaluate_condition(value1, operator1, expected1):
                return False
            
            # Check second condition if present
            field2 = tooltip_row.get('show_when_field2')
            if pd.notna(field2) and str(field2).strip() not in ['-', '', 'nan']:
                value2 = context.get(str(field2).strip())
                operator2 = str(tooltip_row.get('show_when_operator2', 'and')).strip()
                expected2 = tooltip_row.get('show_when_value2')
                
                if operator2.lower() == 'and':
                    return self._evaluate_condition(value2, '==', expected2)
                elif operator2.lower() == 'or':
                    return True  # First condition already passed
            
            return True
                
        except Exception:
            return True  # Show tooltip if condition evaluation fails
    
    def _evaluate_condition(self, value: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single condition"""
        try:
            if value is None or pd.isna(expected):
                return False
            
            # Convert to comparable types
            if isinstance(value, (int, float)):
                try:
                    expected = float(expected)
                except:
                    return False
            else:
                expected = str(expected)
                value = str(value)
            
            # Evaluate
            if operator == '==':
                return value == expected
            elif operator == '!=':
                return value != expected
            elif operator in ['<', '<=', '>', '>=']:
                return eval(f"{value} {operator} {expected}")
            else:
                return False
        except:
            return False
    
    def _render_tooltip(self, tooltip_row: pd.Series, context: Dict[str, Any]) -> str:
        """Render tooltip template with proper variable resolution"""
        try:
            # Get template content
            template = tooltip_row.get('template_content', '')
            if pd.isna(template) or not str(template).strip():
                return ""
            
            template_str = str(template)
            
            # Build comprehensive token dictionary
            tokens = self._build_comprehensive_tokens(tooltip_row, context)
            
            # Fix template syntax issues
            template_str = self._fix_template_syntax(template_str)
            
            # Render the template
            try:
                rendered = self._safe_format(template_str, tokens)

                # Clean up any issues
                rendered = self._clean_rendered_output(rendered)

                return rendered
                
            except Exception as e:
                print(f"Template render error: {e}")
                # Try a very safe fallback rendering
                return self._ultra_safe_render(template_str, tokens)
            
        except Exception as e:
            print(f"ERROR rendering tooltip: {e}")
            return ""
    
    def _fix_template_syntax(self, template: str) -> str:
        """Fix common template syntax issues"""
        # Fix f-string style format specifiers
        template = re.sub(r'\{(\w+):f\}', r'{\1}', template)  # Remove :f
        template = re.sub(r'\{(\w+):\d+f\}', r'{\1}', template)  # Remove :Nf
        template = re.sub(r'\{(\w+):%\}', r'{\1}', template)  # Remove :%
        
        # Fix percentage formatting
        template = re.sub(r'\{evidence:.*?\}%', '{evidence}%', template)
        template = re.sub(r'\{proof_pct:.*?\}%', '{proof_pct}%', template)
        
        return template
    
    def _safe_format(self, template: str, tokens: Dict[str, Any]) -> str:
        """Safe string formatting that handles missing keys and format errors"""
        # First pass: Replace all found tokens
        result = template
        
        # Find all placeholders in the template
        placeholders = re.findall(r'\{([^}:]+)(?::[^}]*)?\}', template)
        
        for placeholder in placeholders:
            # Get the value
            value = tokens.get(placeholder)
            
            if value is None:
                # Try variations of the key
                for key in tokens:
                    if key.lower() == placeholder.lower():
                        value = tokens[key]
                        break
            
            # Format the value appropriately
            if value is not None:
                if isinstance(value, (int, float)):
                    # Format numbers nicely
                    try:
                        if abs(value) >= 1000 or (0 < abs(value) < 0.01):
                            formatted = f"{value:.2e}"
                        elif value == int(value):
                            formatted = str(int(value))
                        else:
                            formatted = f"{value:.3g}"
                    except:
                        formatted = str(value)
                else:
                    formatted = str(value)
                
                # Replace all variations of this placeholder
                patterns = [
                    f"{{{placeholder}}}",
                    f"{{{placeholder}:f}}",
                    f"{{{placeholder}:.1f}}",
                    f"{{{placeholder}:.2f}}",
                    f"{{{placeholder}:g}}",
                    f"{{{placeholder}:%}}"
                ]
                
                for pattern in patterns:
                    result = result.replace(pattern, formatted)
        
        return result
    
    def _ultra_safe_render(self, template: str, tokens: Dict[str, Any]) -> str:
        """Ultra-safe rendering when other methods fail"""
        result = template
        
        # Just replace what we can find
        for key, value in tokens.items():
            if value is not None:
                # Simple string replacement
                result = result.replace(f"{{{key}}}", str(value))
                result = result.replace(f"{{{key}:f}}", str(value))
        
        # Remove any remaining placeholders
        result = re.sub(r'\{[^}]+\}', '', result)
        
        return result if result.strip() else "Tooltip data unavailable"
    
    def _clean_rendered_output(self, rendered: str) -> str:
        """Clean up rendered output"""
        # Protect LaTeX formulas from placeholder removal
        latex_formulas = []
        def save_latex(match):
            latex_formulas.append(match.group(0))
            return f"__LATEX_{len(latex_formulas)-1}__"

        # Temporarily replace LaTeX formulas with placeholders
        # Match $$...$$ first (higher priority), then $...$
        # Use negative lookahead to prevent matching partial delimiters
        rendered = re.sub(r'\$\$(?:(?!\$\$).)*?\$\$', save_latex, rendered, flags=re.DOTALL)
        # Also match single $ delimiters
        rendered = re.sub(r'(?<!\$)\$(?!\$)(?:(?!\$).)*?(?<!\$)\$(?!\$)', save_latex, rendered, flags=re.DOTALL)

        # Remove any remaining unresolved placeholders (but not LaTeX)
        rendered = re.sub(r'\{[^}]+\}', '', rendered)

        # Restore LaTeX formulas
        for i, formula in enumerate(latex_formulas):
            placeholder = f"__LATEX_{i}__"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, formula)

        # Fix common HTML issues
        rendered = rendered.replace('&lt;', '<').replace('&gt;', '>')
        rendered = rendered.replace('\\n', '<br>')

        # Remove error messages (but not scientific terms like "measurement-error")
        # Only remove lines that look like actual error messages
        if ('error:' in rendered.lower() or 'error -' in rendered.lower() or
            'missing data' in rendered.lower() or 'data missing' in rendered.lower()):
            # Try to extract any useful content
            parts = rendered.split('<br>')
            clean_parts = [p for p in parts if not any(err in p.lower() for err in ['error:', 'error -', 'missing data', 'data missing'])]
            if clean_parts:
                rendered = '<br>'.join(clean_parts)

        return rendered.strip()
    
    def _build_comprehensive_tokens(self, tooltip_row: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive token dictionary from all available sources"""
        tokens = {}

        # 1. Start with context
        tokens.update(context)

        # 2. Add pillar-specific data
        pillar_id = context.get('pillar_id', '')
        if pillar_id:
            # Get all pillar data
            if pillar_id in self.pillar_cache:
                tokens.update(self.pillar_cache[pillar_id])

            # Also check individual caches
            for suffix in ['_formula', '_target', '_test']:
                cache_key = f"{pillar_id}{suffix}"
                if cache_key in self.pillar_cache:
                    tokens.update(self.pillar_cache[cache_key])
        
        # 3. Add model-specific data if available
        model_name = context.get('model', '')
        if model_name and hasattr(self, 'db'):
            # Try to get model predictions
            pass  # This would need database access
        
        # 4. Parse input tokens from tooltip definition
        input_tokens_str = tooltip_row.get('input_token', '')
        if pd.notna(input_tokens_str) and str(input_tokens_str).strip() not in ['-', '', 'nan']:
            token_list = [t.strip() for t in str(input_tokens_str).split(';') if t.strip()]
            
            for token_name in token_list:
                if token_name not in tokens:
                    # Try to resolve from various sources
                    resolved_value = self._resolve_token_value(token_name, context)
                    if resolved_value is not None:
                        tokens[token_name] = resolved_value
        
        # 5. Add common aliases and derived values
        self._add_common_aliases(tokens)

        # 6. Ensure proper typing for numeric values
        self._ensure_numeric_types(tokens)

        return tokens
    
    def _resolve_token_value(self, token_name: str, context: Dict[str, Any]) -> Any:
        """Resolve a token value from various sources"""
        # Check context first
        if token_name in context:
            return context[token_name]
        
        # Check parameters cache
        if token_name in self.params_cache:
            return self.params_cache[token_name].get('value')
        
        # Check pillar cache with current pillar
        pillar_id = context.get('pillar_id', '')
        if pillar_id:
            pillar_data = self.pillar_cache.get(pillar_id, {})
            if token_name in pillar_data:
                return pillar_data[token_name]
        
        return None
    
    def _add_common_aliases(self, tokens: Dict[str, Any]):
        """Add common aliases for tokens"""
        # Evidence aliases
        if 'proof_pct' in tokens and 'evidence' not in tokens:
            tokens['evidence'] = tokens['proof_pct']
        if 'evidence' in tokens and 'evidence_pct' not in tokens:
            tokens['evidence_pct'] = tokens['evidence']
        
        # Pass/fail aliases
        if 'passes' in tokens and 'pass_count' not in tokens:
            tokens['pass_count'] = tokens['passes']
        if 'total' in tokens and 'total_count' not in tokens:
            tokens['total_count'] = tokens['total']
        
        if 'passes' in tokens and 'total' in tokens:
            tokens['pass_fraction'] = f"{tokens['passes']}/{tokens['total']}"
            if 'pass_rate' not in tokens:
                try:
                    tokens['pass_rate'] = (tokens['passes'] / tokens['total']) * 100
                except:
                    pass
        
        # Target aliases
        if 'target' in tokens:
            if 'target_observable' not in tokens:
                tokens['target_observable'] = tokens['target']
            if 'observed' not in tokens:
                tokens['observed'] = tokens['target']
        
        # Model name aliases
        if 'model' in tokens:
            model = str(tokens['model'])
            tokens['model_name'] = model
            tokens['model_display'] = self._get_display_name(model)
        
        # Chi-squared aliases
        if 'chi2_red' in tokens and 'chi2_reduced' not in tokens:
            tokens['chi2_reduced'] = tokens['chi2_red']
    
    def _ensure_numeric_types(self, tokens: Dict[str, Any]):
        """Ensure numeric values are properly typed"""
        numeric_keys = [
            'target', 'sigma', 'prediction', 'z_score', 'chi2_red', 'chi2_reduced',
            'max_z', 'evidence', 'evidence_pct', 'proof_pct', 'passes', 'total',
            'pass_rate', 'target_observable', 'observed', 'uncertainty'
        ]
        
        for key in numeric_keys:
            if key in tokens and tokens[key] is not None:
                try:
                    tokens[key] = float(tokens[key])
                except (TypeError, ValueError):
                    pass  # Keep original value if not convertible
    
    def _get_display_name(self, model_key: str) -> str:
        """Get display name for a model"""
        display_names = {
            'MTDF': 'MTDF V74',
            'lcdm_pred_value': 'ΛCDM',
            'mond_pred_value': 'MOND',
            'ede_pred_value': 'Early Dark Energy',
            'fdm_pred_value': 'Fuzzy DM',
            'sidm_pred_value': 'Self-Interacting DM'
        }
        return display_names.get(model_key, model_key)
    
    def get_all_tooltip_ids(self) -> List[str]:
        """Get list of all available tooltip IDs"""
        if self.tooltips_df.empty or 'tooltip_id' not in self.tooltips_df.columns:
            return []
        return self.tooltips_df['tooltip_id'].dropna().unique().tolist()

    def export_tooltips_json(self, output_path: str = None) -> Dict[str, str]:
        """
        Export all rendered tooltips to a JSON file for debugging and Phase 2 migration.

        This dumps the final tooltip HTML map so it can be used as ground truth
        when consolidating all tooltip text back into the workbook.

        Args:
            output_path: Optional path to write JSON file (default: tooltips_effective.json)

        Returns:
            Dictionary mapping tooltip_id to rendered HTML content
        """
        from pathlib import Path

        tooltip_map = {}

        for tooltip_id in self.get_all_tooltip_ids():
            try:
                # Render with empty context to get base template
                html_content = self.get_tooltip(tooltip_id, {})
                if html_content and 'not found' not in html_content.lower():
                    tooltip_map[tooltip_id] = html_content
            except Exception as e:
                tooltip_map[tooltip_id] = f"<!-- Error: {e} -->"

        # Write to JSON file if path provided
        if output_path is None:
            output_path = Path(__file__).parent.parent / "tooltips_effective.json"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tooltip_map, f, indent=2, ensure_ascii=False)
            print(f"Exported {len(tooltip_map)} tooltips to {output_path}")
        except Exception as e:
            print(f"Warning: Could not write tooltip JSON: {e}")

        return tooltip_map

    def validate_tooltips(self) -> Dict[str, List[str]]:
        """Validate tooltip configuration and return issues"""
        issues = {
            'missing_templates': [],
            'invalid_fallbacks': [],
            'format_errors': [],
            'data_issues': []
        }
        
        try:
            if self.tooltips_df.empty:
                issues['data_issues'].append('No tooltip data loaded')
                return issues
            
            if 'tooltip_id' not in self.tooltips_df.columns:
                issues['data_issues'].append('Missing tooltip_id column')
                return issues
            
            for _, row in self.tooltips_df.iterrows():
                tooltip_id = row.get('tooltip_id', '')
                
                # Check for missing template content
                template = row.get('template_content', '')
                if pd.isna(template) or not str(template).strip():
                    issues['missing_templates'].append(tooltip_id)
                else:
                    # Check for format syntax issues
                    template_str = str(template)
                    if ':f}' in template_str or ':%}' in template_str:
                        issues['format_errors'].append(f"{tooltip_id}: Uses f-string syntax")
                
                # Check fallback references
                fallback = row.get('fallback_tooltip_id', '')
                if pd.notna(fallback) and str(fallback).strip() and fallback != '-':
                    if fallback not in self.tooltips_df['tooltip_id'].values:
                        issues['invalid_fallbacks'].append(f"{tooltip_id} -> {fallback}")
                        
        except Exception as e:
            issues['data_issues'].append(f"Validation error: {str(e)}")
        
        return issues


# Integration helper for dashboard.py
def integrate_tooltip_engine(workbook_data: Dict[str, pd.DataFrame]) -> TooltipEngine:
    """Create and return tooltip engine for use in dashboard"""
    try:
        engine = TooltipEngine(workbook_data)
        
        # Validate the engine was created successfully
        validation_issues = engine.validate_tooltips()
        
        if validation_issues['data_issues']:
            print(f"WARNING: Tooltip engine has data issues: {validation_issues['data_issues']}")
        
        if validation_issues['format_errors']:
            print(f"INFO: {len(validation_issues['format_errors'])} tooltips use f-string syntax (will be auto-fixed)")
        
        # Test a few critical tooltips
        test_ids = ['header.model', 'cell.prediction', 'pillar.P1.header']
        working_count = 0
        
        for test_id in test_ids:
            test_context = {
                'pillar_id': 'P1',
                'model': 'MTDF',
                'evidence': 95.5,
                'passes': 14,
                'total': 14
            }
            result = engine.get_tooltip(test_id, test_context)
            if result and 'error' not in result.lower():
                working_count += 1
                print(f"✓ Tooltip {test_id} works")
            else:
                print(f"✗ Tooltip {test_id} failed")
        
        print(f"Tooltip engine validation: {working_count}/{len(test_ids)} test tooltips working")
        
        return engine
        
    except Exception as e:
        print(f"ERROR: Failed to create tooltip engine: {e}")
        return None