import subprocess
import pandas as pd
import json
from pathlib import Path
import time
from multiprocessing import Pool

def evaluate_ceiling(ceil):
    train_events = 200000
    test_events = 50000
    max_events = train_events + test_events
    base_out = Path("output/phase1")
    
    print(f"Starting Re-eval for Ceiling {ceil}...")
    ceil_dir = base_out / f"ceil_{ceil:.2f}"
    model_dir = ceil_dir / "models"
    
    eval_cmd = [
        "python", "src/evaluate.py",
        "--max-events", str(max_events),
        "--train-events", str(train_events),
        "--test-events", str(test_events),
        "--output-dir", str(ceil_dir),
        "--model-dir", str(model_dir)
    ]
    subprocess.run(eval_cmd, check=True)
    
    eval_summary = pd.read_csv(ceil_dir / "evaluation_summary.csv", index_col=0).squeeze("columns")
    ll_gain = float(eval_summary["loglik_gain_vs_poisson"])
    mae_imp = float(eval_summary["mae_improvement_percent"])
    
    print(f"Finished Ceiling {ceil}: MAE Imp = {mae_imp:.2f}%")
    return ceil, ll_gain, mae_imp

def main():
    ceilings = [0.70, 0.85, 0.95, 0.98]
    base_out = Path("output/phase1")
    
    with open(base_out / "phase1_sweep_results.json", "r") as f:
        results = json.load(f)
        
    start_time = time.time()
    
    # Run evaluations in parallel
    with Pool(len(ceilings)) as pool:
        eval_results = pool.map(evaluate_ceiling, ceilings)
        
    for ceil, ll_gain, mae_imp in eval_results:
        str_ceil = str(ceil)
        if str_ceil not in results:
            str_ceil = f"{ceil:.1f}" if ceil == 0.7 else f"{ceil:.2f}"
            
        results[str_ceil]["ll_gain"] = ll_gain
        results[str_ceil]["mae_improvement_percent"] = mae_imp
        
    elapsed = time.time() - start_time
    print(f"\nParallel Re-evaluation completed in {elapsed/60:.1f} minutes.")
    
    with open(base_out / "phase1_sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
