"""
Phase 1 Sweep Runner
Sweeps across multiple projection ceilings on the real flight data.
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

def main():
    ceilings = [0.70, 0.85, 0.95, 0.98]
    train_events = 200000
    test_events = 50000
    max_events = train_events + test_events
    iterations = 20
    
    results = {}
    base_out = Path("output/phase1")
    base_out.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for ceil in ceilings:
        print(f"\n{'='*60}")
        print(f"RUNNING SWEEP: Ceiling = {ceil:.2f}")
        print(f"{'='*60}")
        
        ceil_dir = base_out / f"ceil_{ceil:.2f}"
        ceil_dir.mkdir(parents=True, exist_ok=True)
        model_dir = ceil_dir / "models"
        
        # 1. Train
        train_cmd = [
            "python", "src/train.py",
            "--max-events", str(train_events),
            "--iterations", str(iterations),
            "--stationarity-target", str(ceil),
            "--output-dir", str(ceil_dir),
            "--model-dir", str(model_dir)
        ]
        print("Training model...")
        subprocess.run(train_cmd, check=True)
        
        # 2. Extract Training Metrics
        train_log = pd.read_csv(ceil_dir / "training_log.csv")
        clamp_count = int(train_log["projection_applied"].sum())
        final_rho_pre = float(train_log.iloc[-1]["rho_pre_projection"])
        final_rho_post = float(train_log.iloc[-1]["rho_post_projection"])
        
        # 3. Extract Top Contagion Edges
        alpha = np.load(model_dir / "alpha.npy")
        N = alpha.shape[0]
        # Get indices of top 5 edges
        flat_indices = np.argsort(alpha.flatten())[::-1][:5]
        top_edges = []
        for idx in flat_indices:
            src = idx // N
            tgt = idx % N
            weight = alpha[src, tgt]
            top_edges.append(f"Node {src}->{tgt} ({weight:.3f})")
            
        # 4. Evaluate
        eval_cmd = [
            "python", "src/evaluate.py",
            "--max-events", str(max_events),
            "--train-events", str(train_events),
            "--test-events", str(test_events),
            "--output-dir", str(ceil_dir),
            "--model-dir", str(model_dir)
        ]
        print("Evaluating on test set...")
        subprocess.run(eval_cmd, check=True)
        
        eval_summary = pd.read_csv(ceil_dir / "evaluation_summary.csv", index_col=0).squeeze("columns")
        ll_gain = float(eval_summary["loglik_gain_vs_poisson"])
        mae_imp = float(eval_summary["mae_improvement_percent"])
        
        results[ceil] = {
            "clamp_count": clamp_count,
            "final_rho_pre": final_rho_pre,
            "final_rho_post": final_rho_post,
            "top_edges": top_edges,
            "ll_gain": ll_gain,
            "mae_improvement_percent": mae_imp
        }
        print(f"Results for ceiling {ceil}:")
        print(f"  Clamp triggered {clamp_count} times.")
        print(f"  Final Rho: {final_rho_pre:.4f} -> clamped to -> {final_rho_post:.4f}")
        print(f"  LL Gain: {ll_gain:,.1f}")
        print(f"  MAE Improvement: {mae_imp:.2f}%")
        
    elapsed = time.time() - start_time
    print(f"\nPhase 1 sweep completed in {elapsed/60:.1f} minutes.")
    
    with open(base_out / "phase1_sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
