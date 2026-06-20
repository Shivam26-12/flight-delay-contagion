"""
Phase 0 Sweep Runner
Runs `synthetic_recovery.py` across 10 random seeds for rho in [0.65, 0.90, 1.05].
Uses multiprocessing to run seeds concurrently.
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import time

def run_single_seed(args):
    rho, seed, events, project = args
    cmd = [
        "python", "src/synthetic_recovery.py",
        "--output-dir", "output/synthetic/phase0",
        "--nodes", "15",
        "--target-rho", str(rho),
        "--edge-prob", "0.30",
        "--T", "500000",
        "--max-events", str(events),
        "--iterations", "60",
        "--seed", str(seed)
    ]
    if not project:
        cmd.append("--no-project")
    
    # We capture output to avoid interleaving console spam from 10 parallel processes
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the metrics CSV that synthetic_recovery.py saves
    out_dir = Path("output/synthetic/phase0") / f"rho_{rho:.2f}_events_{events}"
    # Wait, the output directory name in synthetic_recovery.py is:
    # run_name = f"rho_{args.target_rho:.2f}_events_{args.max_events}"
    # But wait, if multiple seeds write to the SAME directory simultaneously, they will overwrite each other's metrics.csv and npy files!
    # I need to modify the sweep to pass a custom subfolder or just read the stdout instead of relying on the file.
    # Actually, synthetic_recovery.py prints the metrics dict to stdout at the very end.
    
    # Let's extract the metrics from the stdout.
    metrics = {}
    try:
        # Looking for the pandas series output at the end of stdout
        lines = result.stdout.strip().split("\n")
        # Find the line after "SYNTHETIC RECOVERY METRICS"
        start_idx = -1
        for i, line in enumerate(lines):
            if "SYNTHETIC RECOVERY METRICS" in line:
                start_idx = i + 2
                break
        
        if start_idx != -1:
            for line in lines[start_idx:]:
                if "---" in line or "PASS / FAIL CHECKS" in line:
                    break
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    val = parts[-1]
                    try:
                        metrics[key] = float(val) if '.' in val or 'e' in val else int(val)
                    except ValueError:
                        metrics[key] = val == 'True'
        else:
            print(f"Failed to parse metrics for rho={rho}, seed={seed}. Stdout:\n{result.stdout}")
    except Exception as e:
        print(f"Error parsing rho={rho}, seed={seed}: {e}")
        
    return {"rho": rho, "seed": seed, "metrics": metrics, "returncode": result.returncode}

def main():
    events = 250000
    seeds = [100 + i for i in range(10)]
    rhos = [(0.65, False), (0.90, False), (1.05, True)]
    
    tasks = []
    for rho, project in rhos:
        for seed in seeds:
            tasks.append((rho, seed, events, project))
            
    print(f"Starting Phase 0 Sweep: 3 regimes x 10 seeds = 30 total simulations.")
    print(f"Events per simulation: {events}")
    print(f"Running concurrently with {min(len(tasks), 10)} workers...")
    
    start_time = time.time()
    
    results = {0.65: [], 0.90: [], 1.05: []}
    
    with Pool(processes=10) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_seed, tasks)):
            rho = res["rho"]
            seed = res["seed"]
            metrics = res["metrics"]
            if metrics:
                results[rho].append(metrics)
            print(f"[{i+1}/{len(tasks)}] Finished rho={rho:.2f}, seed={seed}")
            
    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f} seconds.")
    
    # Aggregate and print results
    summary = {}
    print("\n" + "="*80)
    print("PHASE 0 AGGREGATED METRICS (Mean ± Std over 10 seeds)")
    print("="*80)
    
    metrics_to_track = [
        "spectral_radius_abs_error",
        "alpha_allowed_spearman",
        "alpha_rel_rmse",
        "mu_rel_error",
        "ll_monotone_fraction",
        "mask_violations",
        "projection_fired"
    ]
    
    for rho in [0.65, 0.90, 1.05]:
        print(f"\nTarget rho = {rho:.2f}")
        print("-" * 40)
        rho_summary = {}
        for metric in metrics_to_track:
            vals = [r.get(metric, np.nan) for r in results[rho]]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) == 0:
                print(f"{metric:30s}: N/A")
                continue
                
            mean = np.mean(vals)
            std = np.std(vals)
            rho_summary[metric] = {"mean": mean, "std": std}
            
            if metric in ["mask_violations", "projection_fired"]:
                print(f"{metric:30s}: {mean:.2f} ± {std:.2f}")
            else:
                print(f"{metric:30s}: {mean:.4f} ± {std:.4f}")
        summary[rho] = rho_summary

    out_dir = Path("output/synthetic/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase0_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nAggregated summary saved to output/synthetic/phase0/phase0_sweep_summary.json")

if __name__ == "__main__":
    main()
