"""
Comprehensive stress test runner for the Hawkes EM estimator.
Tests scale, sparsity, data limits, and extreme criticality.
"""
import subprocess
import sys
import json
from pathlib import Path

# We test 5 distinct edge-case scenarios:
configs = [
    {
        "label": "Baseline Benchmark",
        "nodes": 15, "rho": 0.70, "edge": 0.30, "events": 100000, "project": False,
        "expect_pass": True
    },
    {
        "label": "High Dimensionality",
        "nodes": 40, "rho": 0.70, "edge": 0.15, "events": 200000, "project": False,
        "expect_pass": True
    },
    {
        "label": "High Density Network",
        "nodes": 15, "rho": 0.70, "edge": 0.80, "events": 100000, "project": False,
        "expect_pass": True
    },
    {
        "label": "Data Starvation",
        "nodes": 15, "rho": 0.70, "edge": 0.30, "events": 5000, "project": False,
        "expect_pass": False  # Expecting relative error thresholds to fail due to noise
    },
    {
        "label": "Extreme Super-Criticality",
        "nodes": 15, "rho": 1.50, "edge": 0.30, "events": 50000, "project": True,
        "expect_pass": True   # Expecting to pass 'safety' checks only
    },
]

results = []
all_expected_outcomes_met = True

out_dir = Path("output/synthetic")
out_dir.mkdir(parents=True, exist_ok=True)

for c in configs:
    print(f"\n{'='*70}")
    print(f"  RUNNING: {c['label']}")
    print(f"  (N={c['nodes']}, rho={c['rho']}, edge={c['edge']}, events={c['events']})")
    print(f"{'='*70}\n")

    cmd = [
        "python", "src/synthetic_recovery.py",
        "--output-dir", "output/synthetic/stress_test",
        "--nodes", str(c["nodes"]),
        "--target-rho", str(c["rho"]),
        "--edge-prob", str(c["edge"]),
        "--T", "500000",  # Always high so simulation naturally hits max-events
        "--max-events", str(c["events"]),
        "--iterations", "60",
    ]
    if not c["project"]:
        cmd.append("--no-project")

    # We do NOT use check=True here because we expect the Data Starvation test to fail its strict assertions
    result = subprocess.run(cmd, capture_output=False)
    passed_strict_checks = result.returncode == 0
    
    # Did the script behave as we expected?
    # Data starvation is explicitly expected to fail the relative error checks.
    met_expectations = passed_strict_checks == c["expect_pass"]
    
    results.append({
        "regime": c["label"], 
        "rho": c["rho"], 
        "N": c["nodes"],
        "passed": passed_strict_checks,
        "expected_pass": c["expect_pass"],
        "met_expectations": met_expectations
    })
    
    if not met_expectations:
        all_expected_outcomes_met = False

print("\n\n" + "=" * 70)
print("  STRESS TEST CAMPAIGN SUMMARY")
print("=" * 70)
for r in results:
    status_str = "PASS [OK]" if r["passed"] else "FAIL [FAILED]"
    expected_str = "(Expected)" if r["met_expectations"] else "(UNEXPECTED!)"
    print(f"  [{status_str:13s}]  {r['regime']:25s} | {expected_str}")
print("=" * 70)

# Save summary log
with open(out_dir / "stress_test_summary.json", "w") as f:
    json.dump(results, f, indent=4)

if all_expected_outcomes_met:
    print("ALL SCENARIOS BEHAVED AS THEORETICALLY EXPECTED [OK]")
else:
    print("SOME SCENARIOS BEHAVED UNEXPECTEDLY [FAILED]")
    sys.exit(1)
