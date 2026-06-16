from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kstest
from data import load_arrays, load_node_index
from hawkes_em import NetworkConstrainedHawkesEM


def residuals_for_node(model, times, nodes, target_node, max_events=None):
    beta = model.beta
    mu_i = model.mu[target_node]
    alpha_col = model.alpha[:, target_node]
    N = len(model.mu)
    R = np.zeros(N)
    last_global = float(times[0])
    acc = 0.0
    residuals = []
    count = 0
    for t, u in zip(times, nodes):
        dt = float(t - last_global)
        if dt > 0:
            # Integral over the interval before processing current event.
            acc += mu_i * dt + float(alpha_col @ R) * (1.0 - np.exp(-beta * dt)) / beta
            R *= np.exp(-beta * dt)
        if u == target_node:
            residuals.append(acc)
            acc = 0.0
            count += 1
            if max_events and count >= max_events:
                break
        R[u] += beta
        last_global = float(t)
    return np.asarray(residuals[1:], dtype=float)  # first residual is sensitive to initialization


def main():
    ap = argparse.ArgumentParser(description="Time-rescaling diagnostics for selected nodes.")
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--max-events", type=int, default=100000)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = NetworkConstrainedHawkesEM.load(args.model_dir)
    times, nodes, adj, T, df = load_arrays(args.data_dir, max_events=args.max_events)
    node_index = load_node_index(args.data_dir)
    top_nodes = pd.Series(nodes).value_counts().head(args.top_k).index.tolist()
    rows = []
    for u in top_nodes:
        res = residuals_for_node(model, times, nodes, int(u))
        if len(res) < 20:
            continue
        z = 1.0 - np.exp(-np.maximum(res, 0.0))
        ks = kstest(z, "uniform")
        label = str(u)
        if not node_index.empty and "NODE" in node_index.columns:
            m = node_index[node_index["NODE"] == u]
            if len(m):
                
                for col in ["iata_code", "IATA", "node_iata", "iata"]:
                    if col in m.columns:
                        label = str(m.iloc[0][col])
                        break
        rows.append({
            "node": int(u),
            "label": label,
            "residual_count": int(len(res)),
            "mean_exp_residual": float(np.mean(res)),
            "ks_statistic_uniform_transform": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        })
    out = pd.DataFrame(rows)
    out.to_csv(Path(args.output_dir) / "time_rescaling_diagnostics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
