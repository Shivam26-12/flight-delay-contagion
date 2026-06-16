import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from data import load_arrays, load_node_index
from hawkes_em import NetworkConstrainedHawkesEM
from diagnostics import residuals_for_node

def main():
    ap = argparse.ArgumentParser(description="Generate Exponential QQ plots for residuals.")
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output/plots")
    ap.add_argument("--max-events", type=int, default=230000)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and data...")
    model = NetworkConstrainedHawkesEM.load(args.model_dir)
    times, nodes, adj, T, df = load_arrays(args.data_dir, max_events=args.max_events)
    node_index = load_node_index(args.data_dir)
    top_nodes = pd.Series(nodes).value_counts().head(args.top_k).index.tolist()
    
    for u in top_nodes:
        print(f"Processing node {u}...")
        res = residuals_for_node(model, times, nodes, int(u))
        if len(res) < 20:
            continue
            
        # Filter 99th percentile to remove overnight artifacts
        res_positive = np.maximum(res, 0.0)
        p99 = np.percentile(res_positive, 99)
        res_filtered = res_positive[res_positive < p99]
            
        label = str(u)
        if not node_index.empty and "NODE" in node_index.columns:
            m = node_index[node_index["NODE"] == u]
            if len(m):
                for col in ["iata_code", "IATA", "node_iata", "iata"]:
                    if col in m.columns:
                        label = str(m.iloc[0][col])
                        break
                        
        plt.figure(figsize=(6, 6))
        stats.probplot(res_filtered, dist="expon", plot=plt)
        plt.title(f"Exponential Q-Q Plot for {label} (Filtered)")
        plt.ylabel("Observed Residuals")
        plt.xlabel("Theoretical Exponential Quantiles")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Add diagonal line for y=x
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
        plt.legend()
        
        plot_path = out_dir / f"qq_{label}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {plot_path}")
        
    print("QQ Plots generated successfully.")

if __name__ == "__main__":
    main()
