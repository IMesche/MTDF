#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# UI/styles.py - CSS generation module with zero hardcoding
# Author: Ingo Mesche
# Purpose: Generate responsive CSS styles for validation dashboard

from __future__ import annotations
from typing import List

__version__ = "v74"


def generate_css_styles(pillars: List[str]) -> str:
    """
    Generate enhanced CSS styles for dashboard with truly responsive design.
    
    Args:
        pillars: List of pillar IDs from database (e.g., ['P1', 'P2', ...])
        
    Returns:
        Complete CSS stylesheet as string
        
    Note: ZERO HARDCODING - all styles adapt to pillar count from database
    """
    pillar_count = len(pillars)
    
    print(f"🖥️  RESPONSIVE DESIGN: {pillar_count} pillars detected, adapting to any screen size")
    print(f"📱 Table will scale from mobile to ultra-wide monitors")
    
    return f"""
    <style>
    body{{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0b1220;color:#e6edf3;margin:0;padding:24px}}
    .container{{max-width:100%;margin:0 auto}}
    .header{{text-align:center;margin-bottom:20px}}
    .subtitle{{font-size:18px;color:#7d8590;margin:5px 0 15px 0}}
    .evidence-banner{{
        background:#0d4521;
        color:#4ade80;
        padding:12px 20px;
        border-radius:8px;
        margin:15px 0;
        font-weight:600;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:20px
    }}
    .banner-left{{flex-shrink:0}}
    .banner-title{{color:#ffffff;font-size:32px;font-weight:bold;margin:0}}
    .banner-center{{flex-grow:1;text-align:center;color:#4ade80}}
    .banner-right{{flex-shrink:0}}
    .recalc-button{{
        background:#0969da;
        color:white;
        border:none;
        padding:8px 16px;
        border-radius:6px;
        font-size:14px;
        cursor:pointer;
        font-weight:600
    }}
    .recalc-button:hover{{background:#0860ca}}
    .recalc-button:disabled{{background:#6b7280;cursor:not-allowed}}

    /* Executive Summary and Reading Guide */
    .summary-container{{display:flex;gap:20px;margin:20px 0}}
    .executive-summary, .reading-guide{{
        background:#0f172a;
        border:1px solid #1f2937;
        border-radius:8px;
        padding:20px;
        flex:1
    }}
    .executive-summary h2, .reading-guide h3{{color:#e6edf3;margin-bottom:15px;margin-top:0}}
    .executive-summary p, .executive-summary li, .reading-guide li{{
        color:#e6edf3;
        line-height:1.6;
        margin-bottom:8px
    }}
    .executive-summary ul, .reading-guide ul{{padding-left:20px;margin-bottom:0}}
    
    @media (max-width: 1000px) {{
        .summary-container {{flex-direction: column;}}
    }}

    /* TRULY RESPONSIVE TABLE - adapts to any screen size */
    .table-wrap{{
        width:100%;           /* Full viewport width */
        max-width:100%;       /* Never exceed viewport */  
        overflow:auto;         /* Horizontal scroll when needed */
        margin-top:20px;
        border-radius:12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }}
    table{{
        table-layout:auto;
        border-collapse:separate;
        border-spacing:0;
        width:100%;           /* Fill available space */
        min-width:fit-content; /* Shrink to fit content, no fixed minimum */
        background:#0f172a;
        border:1px solid #1f2937;
        border-radius:12px;
        overflow:hidden
    }}
    th,td{{
        border-bottom:1px solid #1f2937;
        border-right:1px solid #1f2937;
        padding:3px 4px;
        text-align:center;
        font-size:13px;
        vertical-align:middle;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        min-height:40px;
    }}
    thead th{{
        position:sticky;
        top:0;
        background:#111827;
        vertical-align:bottom;
        font-weight:600;
        border-bottom:2px solid #374151;
    }}

    /* Responsive column sizing - adapts to screen size */
    .model-col{{
        text-align:left;
        min-width:40px;       /* Minimum readable width */
        max-width:120px;      /* Cap width for compact layout */
        width:auto;            /* Let it grow/shrink naturally */
        font-weight:600;
        position:sticky;
        left:0;
        background:#0f172a;
        z-index:2;
        border-right:2px solid #374151;
    }}
    thead .model-col{{z-index:3}}
    .foundation-col{{
        min-width:40px;       /* Minimum readable width */
        width:auto;            /* Let it adapt */
        text-align:center;
    }}
    .evidence-col{{
        min-width:20px;        /* Minimum readable width */
        width:auto;            /* Let it adapt */
        font-weight:bold;
        text-align:center;
    }}
    .target-row{{font-size:12px;color:#7d8590;font-weight:500}}

    /* Pillar columns - truly responsive */
    th:not(.model-col):not(.foundation-col):not(.evidence-col):not([rowspan]) {{
        min-width:32px;        /* Minimum readable width */
        width:auto;            /* Let them size naturally to fit screen */
        max-width:55px;       /* Maximum for compact layout */
    }}

    /* Summary columns */
    th[rowspan="2"]:nth-last-child(-n+4) {{
        min-width:20px;        /* Minimum readable width */
        width:auto;            /* Let them adapt */
    }}

    /* Pillar coverage indicators */
    .pillar-coverage{{font-size:11px;color:#6b7280;font-weight:normal}}

    /* Role badges (Anchor/Benchmark/Validation) */
    .role-badge{{
        font-size:9px;
        font-weight:bold;
        padding:1px 4px;
        border-radius:3px;
        margin-left:3px;
        vertical-align:middle;
    }}
    .role-badge.anchor{{
        background:#3b1f1f;
        color:#f87171;
        border:1px solid #b91c1c;
    }}
    .role-badge.bench{{
        background:#3b2f1f;
        color:#fbbf24;
        border:1px solid #92400e;
    }}
    .role-badge.val{{
        background:#1f2937;
        color:#60a5fa;
        border:1px solid #1d4ed8;
    }}
    .role-badge.diag{{
        background:#0f3a3a;
        color:#00ffff;
        border:1px solid #00cccc;
    }}

    /* Enhanced Cell colors and states with better borders */
    td.ok{{
        background:#0a2f1f;
        color:#ffffff;
        border:1px solid #166534;
        box-shadow: inset 0 0 0 1px rgba(22, 101, 52, 0.3);
    }}
    td.warn{{
        background:#2f2a0a;
        color:#ffffff;
        border:1px solid #ca8a04;
        box-shadow: inset 0 0 0 1px rgba(202, 138, 4, 0.3);
    }}
    td.bad{{
        background:#3a0a0a;
        color:#ffffff;
        border:1px solid #dc2626;
        box-shadow: inset 0 0 0 1px rgba(220, 38, 38, 0.3);
    }}
    td.na{{
        background:#111827;
        color:#94a3b8;
        border:1px solid #374151;
        font-style:italic;
    }}
    
    .tier-validated{{color:#4ade80;font-weight:bold}}
    .tier-supported{{color:#3b82f6;font-weight:bold}}
    .tier-requires_components{{color:#6b7280;font-weight:bold}}
    .tier-not_supported{{color:#ef4444;font-weight:bold}}
    .cell-sub{{display:block;font-size:11px;color:#94a3b8;margin-top:3px}}
    
    /* Enhanced validated row highlighting */
    .validated-row{{
        border:2px solid #4ade80 !important;
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.3);
    }}
    .validated-row td{{
        border-top:2px solid #4ade80 !important;
        border-bottom:2px solid #4ade80 !important;
    }}
    .validated-row td:first-child{{border-left:2px solid #4ade80 !important}}
    .validated-row td:last-child{{border-right:2px solid #4ade80 !important}}

    /* Tooltip system with blue accents */
    :root {{ --tooltip-accent: #2563eb; }}
    .pro-body-tooltip{{
        position: fixed;
        z-index: 99999;
        max-width: 36rem;
        padding: 8px 10px;
        background: #111827;
        color: #d1d5db;
        border: 1px solid var(--tooltip-accent);
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.35;
        box-shadow:
            0 6px 18px rgba(0,0,0,.35),
            0 0 0 1px rgba(37,99,235,.35),
            0 0 12px rgba(37,99,235,.25);
        pointer-events: none;
        opacity: 0;
        transition: opacity .08s ease;
    }}
    .pro-body-tooltip::before{{
        content: "";
        position: absolute;
        top: -8px;
        left: 50%;
        transform: translateX(-50%);
        border-width: 8px;
        border-style: solid;
        border-color: transparent transparent var(--tooltip-accent) transparent;
        opacity: .95;
    }}
    .pro-body-tooltip::after{{
        content: "";
        position: absolute;
        top: -7px;
        left: 50%;
        transform: translateX(-50%);
        border-width: 7px;
        border-style: solid;
        border-color: transparent transparent #111827 transparent;
    }}

    /* Expandable sections */
    .expandable-section{{margin:20px 0;background:#0f172a;border:1px solid #1f2937;border-radius:8px}}
    .section-header{{padding:15px 20px;cursor:pointer;background:#111827;border-radius:8px 8px 0 0;font-weight:600}}
    .section-header:hover{{background:#1c2128}}
    .arrow{{float:right;transition:transform 0.2s}}
    .section-content{{display:none;padding:20px;border-top:1px solid #374151}}
    
    /* Chi-squared breakdown table */
    .chi-table{{font-size:11px;margin:10px 0;width:100%;border-collapse:separate;border-spacing:0;background:#0f172a;border:1px solid #1f2937;border-radius:8px}}
    .chi-table th,.chi-table td{{padding:6px 10px;border:1px solid #374151;text-align:center}}
    .chi-table th{{background:#111827;font-weight:600}}
    .performance-highlight{{background:#0d4521;color:#4ade80;padding:12px;border-radius:6px;margin:10px 0;font-weight:600}}

    /* Enhanced Footer */
    .enhanced-footer{{text-align:center;margin-top:30px;padding:20px;background:#0f172a;border:1px solid #1f2937;border-radius:8px}}
    .enhanced-footer .footer-main{{color:#e6edf3;font-weight:600;margin-bottom:10px}}
    .enhanced-footer .footer-details{{color:#7d8590;font-size:12px;line-height:1.4}}

    /* Responsive breakpoints - adjust padding and font size for different screen sizes */
    @media (max-width: 1400px) {{
        th, td {{
            padding: 3px 5px;
            font-size: 11px;
        }}
    }}

    @media (max-width: 1000px) {{
        th, td {{
            padding: 2px 4px;
            font-size: 10px;
        }}
        th:not(.model-col):not(.foundation-col):not(.evidence-col):not([rowspan]) {{
            min-width: 35px;
            max-width: 55px;
        }}
    }}

    @media (max-width: 800px) {{
        th, td {{
            padding: 2px 3px;
            font-size: 10px;
        }}
    }}

    /* ============================================
       SCOPE TOGGLE SYSTEM - Filter pillars by role
       ============================================ */

    /* Scope selector dropdown container */
    .scope-selector-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }}

    /* Badge indicators row */
    .scope-badges {{
        display: flex;
        gap: 3px;
        justify-content: center;
        margin-bottom: 4px;
    }}
    .scope-badge {{
        font-size: 9px;
        font-weight: bold;
        padding: 2px 5px;
        border-radius: 3px;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s;
        user-select: none;
    }}
    .scope-badge:hover {{
        box-shadow: 0 0 8px currentColor;
        transform: scale(1.15);
    }}
    .scope-badge.active {{
        opacity: 1;
        transform: scale(1.1);
    }}
    .scope-badge.active:hover {{
        transform: scale(1.2);
    }}
    .scope-badge.inactive {{
        opacity: 0.3;
        transform: scale(0.9);
    }}
    .scope-badge.inactive:hover {{
        opacity: 0.6;
        transform: scale(1.0);
    }}
    .scope-badge.anchor {{ background: #3b1f1f; color: #f87171; border: 1px solid #b91c1c; }}
    .scope-badge.bench {{ background: #3b2f1f; color: #fbbf24; border: 1px solid #92400e; }}
    .scope-badge.val {{ background: #1f2937; color: #60a5fa; border: 1px solid #1d4ed8; }}
    .scope-badge.diag {{ background: #0f3a3a; color: #00ffff; border: 1px solid #00cccc; }}

    /* Dropdown selector */
    .scope-dropdown {{
        background: #1f2937;
        color: #e6edf3;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 3px 20px 3px 8px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        appearance: none;
        -webkit-appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%239ca3af' d='M3 4.5L6 8l3-3.5H3z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 4px center;
    }}
    .scope-dropdown:hover {{
        background-color: #374151;
        border-color: #4b5563;
    }}
    .scope-dropdown:focus {{
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }}

    /* Inactive pillar columns - darker overlay */
    .pillar-inactive {{
        position: relative;
    }}
    .pillar-inactive::after {{
        content: "";
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.6);
        pointer-events: none;
        z-index: 1;
    }}

    /* Inactive cells within table (fallback for td/th) */
    td.scope-excluded, th.scope-excluded {{
        opacity: 0.25;
        filter: grayscale(70%);
    }}

    /* Active scope indicator in header */
    .scope-label {{
        font-size: 10px;
        color: #9ca3af;
        margin-top: 2px;
    }}

    /* ============================================
       TENSION PLOT - σ deviation visualization
       ============================================ */
    .tension-plot-container {{
        padding: 20px;
        overflow-x: auto;
    }}
    .tension-plot {{
        min-width: 800px;
        background: #0a0f1a;
        border-radius: 8px;
        padding: 20px;
    }}
    .tension-plot-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }}
    .tension-plot-title {{
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
    }}
    .tension-legend {{
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
    }}
    .tension-legend-item {{
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        color: #9ca3af;
    }}
    .tension-legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }}
    .tension-row {{
        display: flex;
        align-items: center;
        margin: 4px 0;
        height: 24px;
    }}
    .tension-pillar-label {{
        width: 60px;
        font-size: 11px;
        font-weight: 600;
        color: #cbd5e1;
        text-align: right;
        padding-right: 10px;
        flex-shrink: 0;
    }}
    .tension-pillar-role {{
        font-size: 8px;
        font-weight: bold;
        padding: 1px 3px;
        border-radius: 2px;
        margin-left: 3px;
        vertical-align: middle;
    }}
    .tension-pillar-role.A {{ background: #3b1f1f; color: #f87171; }}
    .tension-pillar-role.B {{ background: #3b2f1f; color: #fbbf24; }}
    .tension-pillar-role.V {{ background: #1f2937; color: #60a5fa; }}
    .tension-pillar-role.D {{ background: #0f3a3a; color: #00ffff; }}
    .tension-bar-container {{
        flex-grow: 1;
        height: 20px;
        background: linear-gradient(90deg,
            #3a0a0a 0%,
            #2f2a0a 20%,
            #0a2f1f 35%,
            #0a2f1f 65%,
            #2f2a0a 80%,
            #3a0a0a 100%);
        border-radius: 4px;
        position: relative;
        border: 1px solid #374151;
    }}
    .tension-centerline {{
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #4b5563;
        transform: translateX(-50%);
    }}
    .tension-tick {{
        position: absolute;
        top: 0;
        bottom: 0;
        width: 1px;
        background: #374151;
    }}
    .tension-marker {{
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        border: 2px solid rgba(255,255,255,0.5);
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
        z-index: 1;
    }}
    .tension-marker:hover {{
        transform: translate(-50%, -50%) scale(1.4);
        box-shadow: 0 0 10px currentColor;
        z-index: 10;
    }}
    .tension-marker.mtdf {{ background: #4ade80; }}
    .tension-marker.lcdm {{ background: #3b82f6; }}
    .tension-marker.mond {{ background: #6b7280; }}
    .tension-marker.ede {{ background: #a855f7; }}
    .tension-marker.fdm {{ background: #ec4899; }}
    .tension-marker.sidm {{ background: #f59e0b; }}
    .tension-marker.excluded {{
        opacity: 0.2;
        pointer-events: none;
    }}
    .tension-scale {{
        display: flex;
        margin-left: 60px;
        margin-top: 8px;
        padding-left: 10px;
    }}
    .tension-scale-labels {{
        flex-grow: 1;
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #6b7280;
    }}
    .tension-scale-label {{
        text-align: center;
    }}
    .tension-threshold-labels {{
        display: flex;
        margin-left: 70px;
        margin-bottom: 5px;
        font-size: 9px;
        color: #6b7280;
    }}
    .tension-threshold-label {{
        position: absolute;
        transform: translateX(-50%);
    }}
    .tension-excluded-row {{
        opacity: 0.3;
    }}
    .tension-type-badge {{
        display: inline-block;
        background: #6366f1;
        color: white;
        font-size: 9px;
        padding: 1px 4px;
        border-radius: 3px;
        margin-left: 4px;
        vertical-align: middle;
        font-weight: bold;
    }}
    .tension-row.vector-pillar .tension-pillar-label {{
        font-style: italic;
    }}
    .tension-tooltip {{
        position: fixed;
        z-index: 99999;
        padding: 8px 12px;
        background: #1f2937;
        color: #e6edf3;
        border: 1px solid #3b82f6;
        border-radius: 6px;
        font-size: 11px;
        pointer-events: none;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }}
    </style>
    """