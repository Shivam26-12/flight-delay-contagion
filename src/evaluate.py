from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_arrays
from hawkes_em import NetworkConstrainedHawkesEM


def conditional_loglik(model, times, nodes, train_end_idx, test_end_idx):
    beta = model.beta
    mu = model.mu
    alpha = model.alpha
    N = len(mu)
    start = float(times[train_end_idx])
    end = float(times[test_end_idx - 1])
    R = np.zeros(N, dtype=float)
    last = float(times[0])
    # Build state up to test start.
    for t, u in zip(times[:train_end_idx], nodes[:train_end_idx]):
        dt = float(t - last)
        if dt > 0:
            R *= np.exp(-beta * dt)
        R[u] += beta
        last = float(t)
    # Decay state to test start.
    if start > last:
        R *= np.exp(-beta * (start - last))
    last = start

    ll_events = 0.0
    eps = 1e-12
    for t, target in zip(times[train_end_idx:test_end_idx], nodes[train_end_idx:test_end_idx]):
        dt = float(t - last)
        if dt > 0:
            R *= np.exp(-beta * dt)
        lam = mu[target] + float(R @ alpha[:, target])
        ll_events += np.log(max(lam, eps))
        R[target] += beta
        last = float(t)

    # Integral over [start, end]
    interval = max(end - start, 1e-9)
    integral = mu.sum() * interval
    denom_by_source = np.zeros(N, dtype=float)
    all_t = times[:test_end_idx]
    all_u = nodes[:test_end_idx]
    for t, u in zip(all_t, all_u):
        if t >= end:
            continue
        a = max(start, float(t))
        if a < end:
            # integral of beta exp(-beta(s-t)) ds from a to end
            denom_by_source[u] += np.exp(-beta * (a - t)) - np.exp(-beta * (end - t))
    integral += float(denom_by_source @ alpha.sum(axis=1))
    return float(ll_events - integral)


def poisson_loglik(train_times, train_nodes, test_times, test_nodes, num_nodes):
    T_train = train_times[-1] - train_times[0]
    start, end = test_times[0], test_times[-1]
    T_test = max(end - start, 1e-9)
    counts = np.bincount(train_nodes, minlength=num_nodes)
    mu = np.maximum(counts / max(T_train, 1e-9), 1e-12)
    ll = float(np.log(mu[test_nodes]).sum() - mu.sum() * T_test)
    return ll


def hourly_mae(model, times, nodes, train_end_idx, test_end_idx, window_hours=1.0):
    beta = model.beta
    mu = model.mu
    alpha = model.alpha
    N = len(mu)
    start = float(times[train_end_idx])
    end = float(times[test_end_idx - 1])
    if end <= start:
        return np.nan, np.nan
    W = int(np.ceil((end - start) / window_hours))
    true_counts = np.zeros((W, N))
    test_t = times[train_end_idx:test_end_idx]
    test_u = nodes[train_end_idx:test_end_idx]
    for t, u in zip(test_t, test_u):
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
        # integrate kernels for events before b
        idx_end = np.searchsorted(hist_t, b, side="left")
        for t, u in zip(hist_t[:idx_end], hist_u[:idx_end]):
            lo = max(a, t)
            if lo < b:
                effect = np.exp(-beta * (lo - t)) - np.exp(-beta * (b - t))
                pred[w, :] += alpha[u, :] * effect
    hawkes_mae = float(np.mean(np.abs(pred - true_counts)))
    baseline_mean = true_counts.mean(axis=0, keepdims=True)
    baseline_mae = float(np.mean(np.abs(baseline_mean - true_counts)))
    return hawkes_mae, baseline_mae


def main():
    ap = argparse.ArgumentParser(description="Evaluate fitted Hawkes model.")
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--max-events", type=int, default=70000, help="Load this many events for train+test evaluation.")
    ap.add_argument("--train-events", type=int, default=50000)
    ap.add_argument("--test-events", type=int, default=10000)
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = NetworkConstrainedHawkesEM.load(args.model_dir)
    times, nodes, adj, T, df = load_arrays(args.data_dir, max_events=args.max_events)
    train_end = min(args.train_events, len(times) - 1)
    test_end = min(train_end + args.test_events, len(times))
    if test_end <= train_end + 5:
        raise ValueError("Not enough test events. Increase --max-events or reduce --train-events.")
    hawkes_ll = conditional_loglik(model, times, nodes, train_end, test_end)
    pois_ll = poisson_loglik(times[:train_end], nodes[:train_end], times[train_end:test_end], nodes[train_end:test_end], adj.shape[0])
    h_mae, b_mae = hourly_mae(model, times, nodes, train_end, test_end)
    res = {
        "train_events": train_end,
        "test_events": test_end - train_end,
        "hawkes_conditional_loglik": hawkes_ll,
        "poisson_loglik": pois_ll,
        "loglik_gain_vs_poisson": hawkes_ll - pois_ll,
        "hawkes_hourly_mae": h_mae,
        "mean_count_baseline_hourly_mae": b_mae,
        "mae_improvement_percent": 100.0 * (b_mae - h_mae) / b_mae if b_mae and not np.isnan(b_mae) else np.nan,
        "spectral_radius": model.spectral_radius(model.alpha),
        "mask_violations": int(((model.alpha > 1e-12) & (model.adj_matrix == 0)).sum()),
    }
    pd.Series(res).to_csv(Path(args.output_dir) / "evaluation_summary.csv")
    print(pd.Series(res).to_string())


if __name__ == "__main__":
    main()
