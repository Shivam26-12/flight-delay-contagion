import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import load_arrays, load_node_index
from hawkes_em import NetworkConstrainedHawkesEM

def calculate_hourly_predictions(model, times, nodes, train_end_idx, test_end_idx, target_nodes, window_hours=1.0, max_hours=168.0):
    beta = model.beta
    mu = model.mu
    alpha = model.alpha
    N = len(mu)
    
    start = float(times[train_end_idx])
    end = min(float(times[test_end_idx - 1]), start + max_hours)
    
    W = int(np.ceil((end - start) / window_hours))
    true_counts = np.zeros((W, N))
    test_t = times[train_end_idx:test_end_idx]
    test_u = nodes[train_end_idx:test_end_idx]
    
    for t, u in zip(test_t, test_u):
        if t >= end:
            continue
        w = int((t - start) // window_hours)
        if 0 <= w < W:
            true_counts[w, u] += 1
            
    pred = np.zeros((W, N))
    hist_t = times[:test_end_idx]
    hist_u = nodes[:test_end_idx]
    
    for w in range(W):
        a = start + w * window_hours
        b = min(a + window_hours, end)
        pred[w, :] += mu * (b - a)
        
        idx_end = np.searchsorted(hist_t, b, side="left")
        for t, u in zip(hist_t[:idx_end], hist_u[:idx_end]):
            lo = max(a, t)
            if lo < b:
                effect = np.exp(-beta * (lo - t)) - np.exp(-beta * (b - t))
                pred[w, :] += alpha[u, :] * effect
                
    return true_counts, pred, W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output/plots")
    ap.add_argument("--max-events", type=int, default=230000)
    ap.add_argument("--train-events", type=int, default=200000)
    ap.add_argument("--test-events", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=2)
    args = ap.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and data...")
    model = NetworkConstrainedHawkesEM.load(args.model_dir)
    times, nodes, adj, T, df = load_arrays(args.data_dir, max_events=args.max_events)
    node_index = load_node_index(args.data_dir)
    
    train_end = min(args.train_events, len(times) - 1)
    test_end = min(train_end + args.test_events, len(times))
    
    # Pick top nodes from train set
    top_nodes = pd.Series(nodes[:train_end]).value_counts().head(args.top_k).index.tolist()
    
    print(f"Calculating predictions for top {args.top_k} nodes over a 7-day test window...")
    true_counts, pred, W = calculate_hourly_predictions(model, times, nodes, train_end, test_end, top_nodes, window_hours=1.0, max_hours=168.0)
    
    for u in top_nodes:
        label = str(u)
        if not node_index.empty and "NODE" in node_index.columns:
            m = node_index[node_index["NODE"] == u]
            if len(m):
                for col in ["iata_code", "IATA", "node_iata", "iata"]:
                    if col in m.columns:
                        label = str(m.iloc[0][col])
                        break
                        
        plt.figure(figsize=(10, 5))
        hours = np.arange(W)
        plt.plot(hours, true_counts[:, u], label="Actual Delays", color="black", alpha=0.7, linewidth=1.5)
        plt.plot(hours, pred[:, u], label="Hawkes Predicted", color="red", linestyle="--", linewidth=2.0)
        
        plt.title(f"Real vs Predicted Hourly Delays: {label} (7-Day Holdout)")
        plt.xlabel("Hours into Test Window")
        plt.ylabel("Hourly Delay Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        plot_path = out_dir / f"real_vs_pred_{label}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved {plot_path}")

if __name__ == "__main__":
    main()
