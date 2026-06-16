from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from hawkes_em import NetworkConstrainedHawkesEM


def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def make_stable_alpha(N, edge_prob=0.25, target_rho=0.65, seed=123):
    rng = np.random.default_rng(seed)
    adj = (rng.random((N, N)) < edge_prob).astype(float)
    np.fill_diagonal(adj, 1.0)
    alpha = rng.gamma(shape=1.5, scale=0.05, size=(N, N)) * adj
    rho = spectral_radius(alpha)
    alpha *= target_rho / max(rho, 1e-12)
    return adj, alpha


def simulate_hawkes(mu, alpha, beta, T, seed=123, max_events=200000):
    rng = np.random.default_rng(seed)
    N = len(mu)
    t = 0.0
    R = np.zeros(N)
    times = []
    nodes = []
    while t < T and len(times) < max_events:
        total = float(mu.sum() + R @ alpha.sum(axis=1))
        if total <= 0:
            break
        wait = rng.exponential(1.0 / total)
        t_new = t + wait
        if t_new > T:
            break
        R *= np.exp(-beta * wait)
        lam = mu + R @ alpha
        total_new = float(lam.sum())
        if rng.random() <= min(1.0, total_new / total):
            probs = lam / total_new
            u = int(rng.choice(N, p=probs))
            times.append(t_new)
            nodes.append(u)
            R[u] += beta
        t = t_new
    return np.asarray(times, dtype=float), np.asarray(nodes, dtype=int)


def main():
    ap = argparse.ArgumentParser(description="Synthetic recovery test for the EM estimator.")
    ap.add_argument("--output-dir", default="output/synthetic")
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    N = args.nodes
    adj, alpha_true = make_stable_alpha(N, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    mu_true = rng.uniform(0.02, 0.06, N)
    times, nodes = simulate_hawkes(mu_true, alpha_true, args.beta, args.T, seed=args.seed + 1)
    if len(times) < 100:
        raise RuntimeError("Synthetic simulation generated too few events; increase T or mu.")
    model = NetworkConstrainedHawkesEM(
        num_nodes=N,
        adj_matrix=adj,
        beta=args.beta,
        seed=args.seed + 2,
        stationarity_project=True,
        stationarity_target=0.98,
    )
    hist = model.fit(times, nodes, T=args.T, max_iter=args.iterations, verbose=True)
    allowed = adj.astype(bool)
    alpha_rmse = float(np.sqrt(np.mean((model.alpha[allowed] - alpha_true[allowed]) ** 2)))
    mu_rmse = float(np.sqrt(np.mean((model.mu - mu_true) ** 2)))
    rho_true = spectral_radius(alpha_true)
    rho_est = spectral_radius(model.alpha)
    corr = spearmanr(alpha_true[allowed], model.alpha[allowed]).correlation
    metrics = {
        "events": int(len(times)),
        "nodes": N,
        "T": args.T,
        "beta_true_fixed": args.beta,
        "spectral_radius_true": rho_true,
        "spectral_radius_est": rho_est,
        "spectral_radius_abs_error": abs(rho_est - rho_true),
        "alpha_allowed_rmse": alpha_rmse,
        "mu_rmse": mu_rmse,
        "alpha_allowed_spearman": float(corr),
        "mask_violations": int(((model.alpha > 1e-12) & (adj == 0)).sum()),
    }
    pd.Series(metrics).to_csv(out / "synthetic_recovery_metrics.csv")
    hist.to_csv(out / "synthetic_training_log.csv", index=False)
    np.save(out / "alpha_true.npy", alpha_true)
    np.save(out / "alpha_est.npy", model.alpha)
    np.save(out / "mu_true.npy", mu_true)
    np.save(out / "mu_est.npy", model.mu)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
