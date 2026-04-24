#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

# SPDX-License-Identifier: MIT
# Tooltips/tooltips.py - Complete workbook-first tooltip manager

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import re

# Try to import WorkbookLoader
try:
    from Data.loader import WorkbookLoader
except Exception:
    try:
        from loader import WorkbookLoader
    except Exception:
        WorkbookLoader = None

def load_id_map(json_text: str) -> Dict[str, str]:
    """Load ID mapping from JSON."""
    try:
        data = json.loads(json_text)
        return {item["id"]: item["key"] for item in data.get("map", [])}
    except Exception as e:
        print(f"Warning: Could not load ID map: {e}")
        return {}

def build_context_from_rows(rows_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build context dictionary from workbook rows."""
    ctx = {
        "pillars": {},
        "models": {},
        "constants": {}
    }
    
    # Process pillar data
    for id_key, row in rows_by_id.items():
        if row.get("entity_type") == "pillar":
            # Extract P-number from ID
            pid = id_key
            if "pillar:" in id_key:
                pid = id_key.split("pillar:")[-1]
            
            # Build pillar context
            ctx["pillars"][pid] = {
                "name": row.get("name", f"Pillar {pid}"),
                "description": row.get("what", ""),
                "why": row.get("why", ""),
                "how": row.get("how", ""),
                "when": row.get("when", ""),
                "source_observation": row.get("source_observation_value_si", ""),
                "prediction": row.get("validated_prediction_value_si", ""),
                "unit": row.get("unit", ""),
                "citation": row.get("citation_DOI/Links", ""),
                "proof_condition": row.get("proof_condition_text", ""),
                "formula_tex": row.get("formula_tex", ""),
                "caveats": row.get("caveats", "")
            }
    
    return ctx

class TooltipBuilder:
    """Build tooltips from context and ID mappings."""
    
    def __init__(self, ctx: Dict[str, Any], id_map: Dict[str, str]):
        self.ctx = ctx
        self.id_map = id_map
        self.tooltips = self._build_all_tooltips()
    
    def _build_all_tooltips(self) -> Dict[str, str]:
        """Pre-build all tooltips for performance."""
        tooltips = {}
        
        # Build pillar tooltips
        for pid, pillar_data in self.ctx["pillars"].items():
            tooltips[f"pillar.{pid}.header"] = self._build_pillar_header_tooltip(pid, pillar_data)
            tooltips[f"pillar.{pid}.target"] = self._build_pillar_target_tooltip(pid, pillar_data)
        
        # Build standard tooltips
        tooltips.update(self._build_standard_tooltips())
        
        return tooltips
    
    def _build_pillar_header_tooltip(self, pid: str, data: Dict[str, Any]) -> str:
        """Build pillar header tooltip."""
        name = data.get("name", f"Pillar {pid}")
        description = data.get("description", "")
        why = data.get("why", "")
        
        return f"""
        <div class="tooltip-content">
            <h4>{pid}: {name}</h4>
            <p><strong>What:</strong> {description}</p>
            <p><strong>Why:</strong> {why}</p>
        </div>
        """
    
    def _build_pillar_target_tooltip(self, pid: str, data: Dict[str, Any]) -> str:
        """Build pillar target tooltip."""
        source_obs = data.get("source_observation", "")
        citation = data.get("citation", "")
        how = data.get("how", "")
        
        return f"""
        <div class="tooltip-content">
            <h4>{pid} Target Value</h4>
            <p><strong>Source observation:</strong> {source_obs}</p>
            <p><strong>Method:</strong> {how}</p>
            {f'<p><strong>Reference:</strong> <a href="{citation}" target="_blank">DOI</a></p>' if citation else ''}
        </div>
        """
    
    def _build_standard_tooltips(self) -> Dict[str, str]:
        """Build standard UI tooltips.

        NOTE: The current dashboard policy requires tooltip text to live in the Excel workbook
        (UI_Tooltips) rather than being hardcoded in Python. This legacy module therefore
        returns an empty set for standard UI tooltips, and should not be used by the main
        dashboard pipeline (which uses TooltipEngine).
        """
        return {}
    
    def build(self, element_id: str) -> str:
        """Build tooltip for given element ID."""
        return self.tooltips.get(element_id, f"<em>Tooltip not found: {element_id}</em>")
    
    def generate_json_blob(self) -> str:
        """Generate JSON blob for client-side tooltips."""
        return json.dumps(self.tooltips, indent=2)

class TooltipManager:
    """Main tooltip manager using workbook data."""
    
    def __init__(self, workbook_path, id_map_path):
        self.workbook_path = Path(workbook_path)
        self.id_map_path = Path(id_map_path)
        self.rows_by_id: Dict[str, Dict[str, Any]] = {}
        self.ctx: Dict[str, Any] = {}
        self.id_map: Dict[str, str] = {}
        self.builder: Optional[TooltipBuilder] = None
        self.tooltips: Dict[str, str] = {}  # Add this for compatibility

    def init(self) -> "TooltipManager":
        """Initialize the tooltip manager."""
        if WorkbookLoader is None:
            print("Warning: WorkbookLoader not available, using empty tooltips")
            self.builder = TooltipBuilder({}, {})
        else:
            L = WorkbookLoader(str(self.workbook_path)).load()
            self.rows_by_id = L.by_id
            self.ctx = build_context_from_rows(self.rows_by_id)
            
            # Load ID map
            try:
                self.id_map = load_id_map(self.id_map_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: Could not load ID map: {e}")
                self.id_map = {}
            
            self.builder = TooltipBuilder(self.ctx, self.id_map)
        
        # Make tooltips available for compatibility
        self.tooltips = self.builder.tooltips if self.builder else {}
        return self

    def get(self, element_id: str, active_pillar: Optional[str] = None) -> str:
        """Get tooltip for element ID."""
        if active_pillar:
            self.ctx["active_pillar"] = active_pillar
        return self.builder.build(element_id) if self.builder else "<em>tooltip system not initialized</em>"
    
    def generate_json_blob(self) -> str:
        """Generate JSON blob for client-side tooltips."""
        return self.builder.generate_json_blob() if self.builder else "{}"

# Quick CLI for debugging
if __name__ == "__main__":
    wb = Path("Data/DB_Workbook.xlsx")
    idmap = Path("Tooltips/db_tooltip_id_map.json")
    tm = TooltipManager(wb, idmap).init()
    # Example prints
    print("P1 header:", tm.get("pillar.P1.header"))
    print("P1 target:", tm.get("pillar.P1.target"))
    print("Header tier:", tm.get("header.tier"))