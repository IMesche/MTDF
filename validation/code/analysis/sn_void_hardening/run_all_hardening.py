#!/usr/bin/env python3
"""
MTDF SN x Void Hardening Test Suite
====================================

Runs all six hardening tests designed to make the SN x void environment
signal harder to dismiss. Each test targets a specific referee objection:

  Test 1: Scrambled void geometry     -> "Is the signal tied to real voids?"
  Test 2: Fake transition redshift    -> "Is z~0.04 really special?"
  Test 3: Population controls (x1, c) -> "Is it a SN population artefact?"
  Test 4: Wrong-sign metric           -> "Does the sign/direction matter?"
  Test 5: Alternative metrics         -> "Does it depend on metric choice?"
  Test 6: Cross-catalogue overlap     -> "Do the same SNe drive all results?"

Usage:
  python run_all_hardening.py              # Run all tests
  python run_all_hardening.py --test 1 3   # Run only tests 1 and 3
  python run_all_hardening.py --quick      # Reduced realisations for speed

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import sys
import os
import time
import argparse
from datetime import datetime

# Add parent to path for common imports
sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')


def run_test(test_num, quick=False):
    """Import and run a single test."""
    start = time.time()

    if test_num == 1:
        from test_scrambled_voids import run
    elif test_num == 2:
        from test_fake_z_transition import run
    elif test_num == 3:
        from test_population_controls import run
    elif test_num == 4:
        from test_wrong_sign_metric import run
    elif test_num == 5:
        from test_alt_environment_metrics import run
    elif test_num == 6:
        from test_cross_catalogue_overlap import run
    else:
        print(f"Unknown test number: {test_num}")
        return None

    results = run(OUTPUT_DIR)
    elapsed = time.time() - start
    print(f"\n  [Test {test_num} completed in {elapsed:.1f}s]\n")
    return results


def main():
    parser = argparse.ArgumentParser(description='MTDF SN x Void Hardening Suite')
    parser.add_argument('--test', nargs='+', type=int, default=None,
                        help='Run specific tests (e.g. --test 1 3)')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced realisations for speed')
    args = parser.parse_args()

    tests = args.test or [1, 2, 3, 4, 5, 6]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("MTDF SN x Void Hardening Test Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Tests to run: {tests}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    total_start = time.time()
    all_results = {}

    for t in tests:
        try:
            all_results[f'test_{t}'] = run_test(t, quick=args.quick)
        except Exception as e:
            print(f"\n  *** Test {t} FAILED: {e} ***\n")
            import traceback
            traceback.print_exc()
            all_results[f'test_{t}'] = {'error': str(e)}

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 70)
    print("HARDENING SUITE COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Tests run: {len(tests)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    # Quick pass/fail summary
    print("\nTest Results Overview:")
    labels = {
        1: "Scrambled Voids      (signal requires real void geometry?)",
        2: "Fake z-Transition    (z~0.04 uniquely special?)",
        3: "Population Controls  (survives x1/c matching?)",
        4: "Wrong-Sign Metric    (correct direction only?)",
        5: "Alternative Metrics  (robust across constructions?)",
        6: "Cross-Catalogue      (same SNe drive all catalogues?)",
    }
    for t in tests:
        key = f'test_{t}'
        if key in all_results and 'error' not in (all_results[key] or {}):
            print(f"  Test {t}: {labels.get(t, '?'):<55} DONE")
        else:
            print(f"  Test {t}: {labels.get(t, '?'):<55} FAILED")

    return all_results


if __name__ == '__main__':
    main()
