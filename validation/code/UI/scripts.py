# UI/scripts.py
# Exposes small utility scripts as inline <script> blocks.
#
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Date: December 2025
# Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html

def generate_javascript() -> str:
    """Main JS bundle for the page. Safe for file:// viewing."""
    return r"""
<script>
(function wireRecalc() {
  document.addEventListener("click", function (ev) {
    const el = ev.target.closest("#recalc-btn");
    if (!el) return;
    ev.preventDefault();
    recalculate();
  });
})();

(function injectBanner() {
  const bannerId = "workbook-banner";
  if (document.getElementById(bannerId)) return;

  const bar = document.createElement("div");
  bar.id = bannerId;
  bar.style.background = "#0f172a";
  bar.style.color = "#cbd5e1";
  bar.style.padding = "6px 12px";
  bar.style.fontSize = "13px";
  bar.style.borderBottom = "1px solid #1f2937";
  bar.style.letterSpacing = ".2px";
  bar.innerHTML = `
    <strong>Workbook</strong>:
    <span id="wb-path"></span>
    &nbsp; | &nbsp;
    <strong>Last loaded</strong>:
    <span id="wb-time"></span>
  `;

  const header = document.querySelector(".header, .topbar, header") || null;
  if (header && header.parentNode) header.parentNode.insertBefore(bar, header.nextSibling);
  else document.body.insertBefore(bar, document.body.firstChild);

  const wbSpan = document.getElementById("wb-path");
  const tsSpan = document.getElementById("wb-time");

  function setFallback() {
    if (tsSpan) tsSpan.textContent = document.lastModified || "";
    if (wbSpan) wbSpan.textContent = "";
  }

  // Do not fetch when opened via file:// to avoid CORS noise
  if (location.protocol === "file:") {
    setFallback();
    return;
  }

  fetch("/api/payload", { cache: "no-store" })
    .then(r => (r.ok ? r.json() : Promise.reject()))
    .then(data => {
      if (wbSpan && data && data.workbook) wbSpan.textContent = data.workbook;
      if (tsSpan && data && data.timestamp) tsSpan.textContent = data.timestamp;
    })
    .catch(setFallback);
})();

// Collapsible sections
function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  const arrow = document.querySelector(`[onclick="toggleSection('${sectionId}')"] .arrow`);
  if (section && arrow) {
    if (section.style.display === 'none' || section.style.display === '') {
      section.style.display = 'block'; arrow.textContent = '▼';
    } else { section.style.display = 'none'; arrow.textContent = '▶'; }
  }
}

// Do NOT define showTooltip/hideTooltip here - the real tooltip system is injected in <head>

// ============================================
// SCOPE TOGGLE SYSTEM
// ============================================
// Scopes:
//   standard (Strict): A+B+V (excludes D) - matches paper's strict χ²/ν
//   validation: V only (strictest)
//   full: A+B+V+D (everything)
//   custom: user-defined combination

const SCOPE_PRESETS = {
  standard: ['A', 'B', 'V'],
  validation: ['V'],
  full: ['A', 'B', 'V', 'D']
};

// Type-based scopes (filter by pillar type instead of role)
const TYPE_SCOPES = ['scalar', 'vector'];

// Expose to window for tooltip system access
window.currentScope = 'standard';
let currentScope = window.currentScope;
let activeRoles = ['A', 'B', 'V'];  // Track currently active roles (matches standard preset)
let activeType = null;  // 'scalar' or 'vector' when in type mode, null otherwise

// Toggle a single badge on/off
function toggleBadge(role) {
  // Reset type scope when manually toggling badges (user wants role-based filtering)
  activeType = null;

  const idx = activeRoles.indexOf(role);
  if (idx >= 0) {
    // Don't allow removing the last role
    if (activeRoles.length > 1) {
      activeRoles.splice(idx, 1);
    }
  } else {
    activeRoles.push(role);
  }

  // Check if combination matches a preset
  const matchedPreset = findMatchingPreset(activeRoles);
  const selector = document.getElementById('scope-selector');

  if (matchedPreset) {
    currentScope = matchedPreset;
    window.currentScope = matchedPreset;  // Update window for tooltip system
    if (selector) selector.value = matchedPreset;
  } else {
    currentScope = 'custom';
    window.currentScope = 'custom';  // Update window for tooltip system
    // Add custom option if not exists
    if (selector && !selector.querySelector('option[value="custom"]')) {
      const customOpt = document.createElement('option');
      customOpt.value = 'custom';
      customOpt.textContent = 'Custom';
      selector.appendChild(customOpt);
    }
    if (selector) selector.value = 'custom';
  }

  applyActiveRoles();
}

// Find if current roles match a preset
function findMatchingPreset(roles) {
  const sorted = [...roles].sort().join('');
  for (const [preset, presetRoles] of Object.entries(SCOPE_PRESETS)) {
    if ([...presetRoles].sort().join('') === sorted) {
      return preset;
    }
  }
  return null;
}

function changeScope(scope) {
  currentScope = scope;
  window.currentScope = scope;  // Update window for tooltip system

  // Handle custom scope (don't change activeRoles)
  if (scope === 'custom') {
    activeType = null;
    applyActiveRoles();
    return;
  }

  // Handle type-based scopes (scalar/vector)
  if (TYPE_SCOPES.includes(scope)) {
    activeType = scope;
    activeRoles = ['A', 'B', 'V', 'D'];  // All roles visible in type mode
    applyActiveRoles();
    return;
  }

  // Handle role-based preset scopes
  activeType = null;
  activeRoles = [...(SCOPE_PRESETS[scope] || SCOPE_PRESETS.standard)];
  applyActiveRoles();
}

function applyActiveRoles() {
  const roles = activeRoles;
  const isTypeScope = TYPE_SCOPES.includes(currentScope);

  // Helper to check if a pillar cell matches the type filter
  function matchesTypeFilter(cell) {
    if (!activeType) return true;  // No type filter active
    // Check both vector-cell (data cells) and vector-pillar (headers)
    const isVector = cell.classList.contains('vector-cell') || cell.classList.contains('vector-pillar');
    // In vector mode, exclude CMB (role D) since it's a ΛCDM-compressed diagnostic
    if (activeType === 'vector') {
      const role = cell.getAttribute('data-role');
      if (role === 'D') return false;  // Exclude diagnostic (CMB*) in vector mode
    }
    return (activeType === 'vector') === isVector;
  }

  // 1. Update badge indicators
  document.querySelectorAll('.scope-badge').forEach(badge => {
    const role = badge.getAttribute('data-scope-role');
    if (isTypeScope) {
      // In type mode, dim all role badges
      badge.classList.remove('active');
      badge.classList.add('inactive');
    } else if (roles.includes(role)) {
      badge.classList.add('active');
      badge.classList.remove('inactive');
    } else {
      badge.classList.remove('active');
      badge.classList.add('inactive');
    }
  });

  // 2. Update pillar column visibility (add/remove scope-excluded class)
  document.querySelectorAll('[data-role]').forEach(cell => {
    const role = cell.getAttribute('data-role');
    if (isTypeScope) {
      // In type mode, filter by pillar type (scalar/vector)
      if (matchesTypeFilter(cell)) {
        cell.classList.remove('scope-excluded');
      } else {
        cell.classList.add('scope-excluded');
      }
    } else {
      // In role mode, filter by role
      if (roles.includes(role)) {
        cell.classList.remove('scope-excluded');
      } else {
        cell.classList.add('scope-excluded');
      }
    }
  });

  // 3. Update chi2/nu display for each model row
  document.querySelectorAll('tr[data-scope-stats]').forEach(row => {
    const cell = row.querySelector('.scope-chi2-cell');
    if (!cell) return;

    // For role-based presets, use precomputed values
    if (!isTypeScope && currentScope !== 'custom') {
      const chi2Value = cell.getAttribute('data-chi2-' + currentScope);
      if (chi2Value && chi2Value !== 'Inf') {
        cell.textContent = parseFloat(chi2Value).toFixed(4);
      } else {
        cell.textContent = 'N/A';
      }
      return;
    }

    // For type scopes or custom, compute from individual pillar data
    // These attributes store the actual chi2 contribution and DOF for each pillar:
    // - Scalar pillars: chi2 = z², dof = 1
    // - Vector pillars: chi2 = actual chi2, dof = actual DOF (e.g., SNe has DOF=1700)
    let totalChi2 = 0, totalDof = 0;

    // Get pillar cells in this row
    row.querySelectorAll('td[data-role][data-chi2][data-dof]').forEach(pillarCell => {
      const role = pillarCell.getAttribute('data-role');
      const matchesRole = roles.includes(role);
      const matchesType = matchesTypeFilter(pillarCell);

      // Include pillar if:
      // - Type scope: matches the type filter (scalar/vector)
      // - Custom/role scope: matches the active roles
      const include = isTypeScope ? matchesType : matchesRole;

      if (include) {
        const chi2 = parseFloat(pillarCell.getAttribute('data-chi2')) || 0;
        const dof = parseInt(pillarCell.getAttribute('data-dof')) || 0;
        if (dof > 0) {
          totalChi2 += chi2;
          totalDof += dof;
        }
      }
    });

    if (totalDof > 0) {
      cell.textContent = (totalChi2 / totalDof).toFixed(4);
    } else {
      cell.textContent = 'N/A';
    }
  });

  // 4. Update Pass counts and Evidence for each row
  document.querySelectorAll('tr[data-scope-stats]').forEach(row => {
    // For type scopes or custom, compute dynamically from visible cells
    if (isTypeScope || currentScope === 'custom') {
      let passes = 0, tests = 0;
      row.querySelectorAll('td[data-role]').forEach(pillarCell => {
        const role = pillarCell.getAttribute('data-role');
        const matchesRole = roles.includes(role);
        const matchesType = matchesTypeFilter(pillarCell);

        // Include pillar based on scope type
        const include = isTypeScope ? matchesType : matchesRole;

        if (include && !pillarCell.classList.contains('na')) {
          tests++;
          if (pillarCell.classList.contains('ok') || pillarCell.classList.contains('warn')) {
            // Check if within threshold (ok = within 1σ, warn = within 2σ)
            if (pillarCell.classList.contains('ok')) passes++;
            else if (pillarCell.classList.contains('warn')) {
              // Check z-score for scalar or chi2/nu for vector
              const subEl = pillarCell.querySelector('.cell-sub');
              if (subEl && subEl.textContent.includes('z=')) {
                const zMatch = subEl.textContent.match(/z=([+-]?\d+\.?\d*)/);
                if (zMatch && Math.abs(parseFloat(zMatch[1])) <= 2.0) passes++;
              } else if (subEl && subEl.textContent.includes('χ²/ν')) {
                passes++;  // warn means it's borderline but passing
              }
            }
          }
        }
      });

      const passCell = row.querySelector('[data-cell-type="pass"]');
      if (passCell) {
        if (tests > 0) {
          const passIcon = passes === tests ?
            "<span class='icon ok'>✓</span>" :
            (passes > 0 ? "<span class='icon warn'>⚠ </span>" : "<span class='icon bad'>✗</span>");
          passCell.innerHTML = passIcon + passes + '/' + tests;
        } else {
          passCell.innerHTML = "<span class='icon na'>—</span>N/A";
        }
      }

      const evidenceCell = row.querySelector('[data-cell-type="evidence"]');
      if (evidenceCell) {
        if (tests > 0) {
          const passRate = (passes / tests) * 100;
          let evidenceClass = passRate >= 80 ? 'ok' : (passRate >= 50 ? 'warn' : 'bad');
          const icon = evidenceClass === 'ok' ? "✓" : (evidenceClass === 'warn' ? "⚠ " : "✗");
          evidenceCell.innerHTML = "<span class='icon " + evidenceClass + "'>" + icon + "</span>" + passRate.toFixed(0) + '%';
          evidenceCell.className = 'evidence-col ' + evidenceClass;
        } else {
          evidenceCell.innerHTML = "<span class='icon na'>—</span>N/A";
          evidenceCell.className = 'evidence-col na';
        }
      }
      return;
    }

    // For role-based presets, use stored stats
    const statsJson = row.getAttribute('data-scope-stats');
    if (!statsJson) return;

    try {
      const stats = JSON.parse(statsJson.replace(/&quot;/g, '"'));
      const scopeStats = stats[currentScope];
      if (!scopeStats) return;

      // Update Pass cell
      const passCell = row.querySelector('[data-cell-type="pass"]');
      if (passCell && scopeStats.tests > 0) {
        const passIcon = scopeStats.passes === scopeStats.tests ?
          "<span class='icon ok'>✓</span>" :
          (scopeStats.passes > 0 ? "<span class='icon warn'>⚠ </span>" : "<span class='icon bad'>✗</span>");
        passCell.innerHTML = passIcon + scopeStats.passes + '/' + scopeStats.tests;
      }

      // Update Evidence cell
      const evidenceCell = row.querySelector('[data-cell-type="evidence"]');
      if (evidenceCell && scopeStats.tests > 0) {
        const passRate = (scopeStats.passes / scopeStats.tests) * 100;

        // Determine evidence level based on pass rate (matches original dashboard logic)
        let evidenceClass = 'bad';
        if (passRate >= 80) evidenceClass = 'ok';
        else if (passRate >= 50) evidenceClass = 'warn';

        const icon = evidenceClass === 'ok' ? "✓" : (evidenceClass === 'warn' ? "⚠ " : "✗");
        evidenceCell.innerHTML = "<span class='icon " + evidenceClass + "'>" + icon + "</span>" + passRate.toFixed(0) + '%';
        evidenceCell.className = 'evidence-col ' + evidenceClass;
      }
    } catch (e) {
      console.warn('Failed to parse scope stats:', e);
    }
  });

  // 5. Re-sort table by new chi2/nu values
  sortTableByScope();

  // 6. Store current scope for export
  window.currentValidationScope = currentScope;
  window.currentValidationRoles = [...roles];
  window.currentValidationType = activeType;

  // Log scope change with appropriate info
  if (isTypeScope) {
    console.log('Scope changed to:', currentScope, '- Type:', activeType);
  } else {
    console.log('Scope changed to:', currentScope, '- Roles:', roles.join('+'));
  }
}

function sortTableByScope() {
  const tbody = document.querySelector('table tbody');
  if (!tbody) return;

  const rows = Array.from(tbody.querySelectorAll('tr[data-scope-stats]'));

  rows.sort((a, b) => {
    // First sort by tier (VALIDATED first)
    const aTier = a.classList.contains('validated-row') ? 0 : 1;
    const bTier = b.classList.contains('validated-row') ? 0 : 1;
    if (aTier !== bTier) return aTier - bTier;

    // Then sort by chi2/nu - use displayed value for accuracy with custom scopes
    const aChi2Cell = a.querySelector('.scope-chi2-cell');
    const bChi2Cell = b.querySelector('.scope-chi2-cell');

    const aChi2 = aChi2Cell ? parseFloat(aChi2Cell.textContent) || 9999 : 9999;
    const bChi2 = bChi2Cell ? parseFloat(bChi2Cell.textContent) || 9999 : 9999;

    return aChi2 - bChi2;
  });

  // Re-append sorted rows
  rows.forEach(row => tbody.appendChild(row));
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Add click handlers to badges for custom toggling
  document.querySelectorAll('.scope-badge').forEach(badge => {
    badge.style.cursor = 'pointer';
    badge.addEventListener('click', function(e) {
      e.stopPropagation();
      const role = this.getAttribute('data-scope-role');
      if (role) toggleBadge(role);
    });
  });

  // Set initial scope from dropdown
  const selector = document.getElementById('scope-selector');
  if (selector) {
    changeScope(selector.value);
  }

  // Initialize tension plot
  updateTensionPlot();
});

// ============================================
// TENSION PLOT SYSTEM
// ============================================

function updateTensionPlot() {
  const container = document.getElementById('tension-plot-rows');
  if (!container) return;

  const isTypeScope = TYPE_SCOPES.includes(currentScope);

  // Update row visibility based on scope type
  // Tension plot now shows both scalar and vector pillars (vector converted to z_eff)
  container.querySelectorAll('.tension-row').forEach(row => {
    const role = row.getAttribute('data-pillar-role');
    const isVectorPillar = row.classList.contains('vector-pillar');
    // Detect CMB by checking if any marker has P_CMB_DIST pillar
    const marker = row.querySelector('.tension-marker');
    const pillarId = marker?.getAttribute('data-pillar') || '';
    const isCMB = pillarId.includes('CMB');

    let visible = false;
    if (isTypeScope) {
      // In type mode: filter by pillar type (scalar vs vector)
      if (activeType === 'scalar') {
        visible = !isVectorPillar;  // Show only scalar rows
      } else {
        visible = isVectorPillar && !isCMB;   // Show vector rows but exclude CMB*
      }
    } else {
      // In role mode: filter by role
      visible = activeRoles.includes(role);
    }

    if (visible) {
      row.classList.remove('tension-excluded-row');
    } else {
      row.classList.add('tension-excluded-row');
    }
  });

  // Update marker visibility
  container.querySelectorAll('.tension-marker').forEach(marker => {
    const row = marker.closest('.tension-row');
    const role = row?.getAttribute('data-pillar-role');
    const isVectorPillar = row?.classList.contains('vector-pillar');
    // Detect CMB by checking pillar ID
    const pillarId = marker.getAttribute('data-pillar') || '';
    const isCMB = pillarId.includes('CMB');

    let visible = false;
    if (isTypeScope) {
      // In type mode: filter by pillar type
      if (activeType === 'scalar') {
        visible = !isVectorPillar;
      } else {
        visible = isVectorPillar && !isCMB;  // Exclude CMB* in vector mode
      }
    } else {
      visible = role && activeRoles.includes(role);
    }

    if (visible) {
      marker.classList.remove('excluded');
    } else {
      marker.classList.add('excluded');
    }
  });
}

// Hook into scope changes
const originalApplyActiveRoles = applyActiveRoles;
applyActiveRoles = function() {
  originalApplyActiveRoles();
  updateTensionPlot();
};

// Tension marker tooltip
(function initTensionTooltips() {
  let tooltip = null;

  document.addEventListener('mouseover', function(e) {
    const marker = e.target.closest('.tension-marker');
    if (!marker) return;

    const model = marker.getAttribute('data-model');
    const pillar = marker.getAttribute('data-pillar');
    const z = marker.getAttribute('data-z');
    const pred = marker.getAttribute('data-pred');
    const target = marker.getAttribute('data-target');
    const sigma = marker.getAttribute('data-sigma');

    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.className = 'tension-tooltip';
      document.body.appendChild(tooltip);
    }

    tooltip.innerHTML = `
      <strong>${model}</strong> @ ${pillar}<br>
      z = <strong>${z}</strong>σ<br>
      Pred: ${pred} | Target: ${target} ± ${sigma}
    `;
    tooltip.style.opacity = '1';

    const rect = marker.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width/2 - tooltip.offsetWidth/2) + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 8) + 'px';
  });

  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('.tension-marker') && tooltip) {
      tooltip.style.opacity = '0';
    }
  });
})();
</script>
"""


def generate_placeholder_scripts() -> str:
    """Tiny safety net. Does not override tooltip functions."""
    return r"""
<script>
function recalculate() {
  alert('To recalculate from workbook:\\n\\n' +
        '1. Open a terminal in the code/ directory\\n' +
        '2. Run:\\n\\n' +
        '   python run_validate.py --workbook ../data/DB_Workbook_STRICT_V18.xlsx --out ../output/Validation_Dashboard_V74.html\\n\\n' +
        '3. Refresh this page (F5) to see updated results\\n\\n' +
        'Note: Browser security prevents direct execution of local scripts.');
}
function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  const arrow = document.querySelector(`[onclick="toggleSection('${sectionId}')"] .arrow`);
  if (section && arrow) {
    if (section.style.display === 'none' || section.style.display === '') {
      section.style.display = 'block'; arrow.textContent = '▼';
    } else { section.style.display = 'none'; arrow.textContent = '▶'; }
  }
}
</script>
"""
