from __future__ import annotations
import argparse
import sys
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
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--edge-prob", type=float, default=0.25)
    ap.add_argument("--target-rho", type=float, default=0.65)
    ap.add_argument("--no-project", action="store_true", help="Disable stationarity projection")
    ap.add_argument("--max-events", type=int, default=200000)
    args = ap.parse_args()

    run_name = f"rho_{args.target_rho:.2f}_events_{args.max_events}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    N = args.nodes
    is_supercritical = args.target_rho >= 1.0

    adj, alpha_true = make_stable_alpha(N, edge_prob=args.edge_prob, target_rho=args.target_rho, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    mu_true = rng.uniform(0.02, 0.06, N)
    times, nodes = simulate_hawkes(mu_true, alpha_true, args.beta, args.T, seed=args.seed + 1, max_events=args.max_events)
    if len(times) < 100:
        raise RuntimeError("Synthetic simulation generated too few events; increase T or mu.")

    # ── FIX: Use the actual observation window ──────────────────────────
    # If the simulator hit max_events before reaching T, the real
    # observation window ends at the last event time (plus a tiny margin).
    # Passing the original T to EM would create a fake "dead zone" of
    # zero events at the tail, biasing mu downward and alpha upward.
    T_actual = float(times[-1]) + 1.0   # +1 so last event is interior
    T_eff = min(args.T, T_actual)
    print(f"[info] Requested T={args.T:.0f}, actual last event at {times[-1]:.1f}, using T_eff={T_eff:.1f}")

    # ── Model setup ─────────────────────────────────────────────────────
    # For supercritical ground truth, the clamp MUST be active so the
    # estimator can safely contain the explosive cascade data.
    use_projection = not args.no_project if not is_supercritical else True
    model = NetworkConstrainedHawkesEM(
        num_nodes=N,
        adj_matrix=adj,
        beta=args.beta,
        seed=args.seed + 2,
        stationarity_project=use_projection,
        stationarity_target=0.98,
    )
    hist = model.fit(times, nodes, T=T_eff, max_iter=args.iterations, verbose=True)

    # ── Metrics ─────────────────────────────────────────────────────────
    allowed = adj.astype(bool)
    alpha_rmse = float(np.sqrt(np.mean((model.alpha[allowed] - alpha_true[allowed]) ** 2)))
    alpha_mean = float(np.mean(alpha_true[allowed]))
    alpha_rel_rmse = alpha_rmse / max(alpha_mean, 1e-12)
    mu_rmse = float(np.sqrt(np.mean((model.mu - mu_true) ** 2)))
    mu_mean = float(np.mean(mu_true))
    mu_rel_err = mu_rmse / max(mu_mean, 1e-12)
    rho_true = spectral_radius(alpha_true)
    rho_est = spectral_radius(model.alpha)
    corr = spearmanr(alpha_true[allowed], model.alpha[allowed]).correlation
    mask_violations = int(((model.alpha > 1e-12) & (adj == 0)).sum())

    ll_vals = hist["log_likelihood"].values
    ll_diffs = np.diff(ll_vals)
    ll_monotone_frac = float(np.sum(ll_diffs >= -1e-8) / max(len(ll_diffs), 1))

    projection_fired = bool(hist["projected"].any())

    metrics = {
        "events": int(len(times)),
        "nodes": N,
        "T_requested": args.T,
        "T_effective": T_eff,
        "beta_true_fixed": args.beta,
        "iterations_used": int(len(hist)),
        "spectral_radius_true": rho_true,
        "spectral_radius_est": rho_est,
        "spectral_radius_abs_error": abs(rho_est - rho_true),
        "alpha_allowed_rmse": alpha_rmse,
        "alpha_rel_rmse": alpha_rel_rmse,
        "mu_rmse": mu_rmse,
        "mu_rel_error": mu_rel_err,
        "alpha_allowed_spearman": float(corr),
        "ll_monotone_fraction": ll_monotone_frac,
        "mask_violations": mask_violations,
        "projection_fired": projection_fired,
    }

    # ── Save outputs ────────────────────────────────────────────────────
    pd.Series(metrics).to_csv(out / "synthetic_recovery_metrics.csv")
    hist.to_csv(out / "synthetic_training_log.csv", index=False)
    np.save(out / "alpha_true.npy", alpha_true)
    np.save(out / "alpha_est.npy", model.alpha)
    np.save(out / "mu_true.npy", mu_true)
    np.save(out / "mu_est.npy", model.mu)

    # ── Pass / Fail Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SYNTHETIC RECOVERY METRICS")
    print("=" * 60)
    print(pd.Series(metrics).to_string())

    if is_supercritical:
        # For rho >= 1.0, we only check safety: clamp fired, LL monotone,
        # no mask violations, and rho_est is safely below 1.0.
        checks = {
            "projection_fired":      (projection_fired == True,   f"Expected True, got {projection_fired}"),
            "rho_est < 1.0":         (rho_est < 1.0,              f"Expected < 1.0, got {rho_est:.4f}"),
            "mask_violations == 0":  (mask_violations == 0,       f"Expected 0, got {mask_violations}"),
            "ll_monotone >= 0.95":   (ll_monotone_frac >= 0.95,   f"Expected >= 0.95, got {ll_monotone_frac:.4f}"),
        }
    else:
        # For stable / near-critical: full accuracy checks
        checks = {
            "rho_abs_error <= 0.10": (abs(rho_est - rho_true) <= 0.10,  f"Expected <= 0.10, got {abs(rho_est - rho_true):.4f}"),
            "alpha_spearman >= 0.70":(corr >= 0.70,                      f"Expected >= 0.70, got {corr:.4f}"),
            "alpha_rel_rmse <= 0.35":(alpha_rel_rmse <= 0.35,            f"Expected <= 0.35, got {alpha_rel_rmse:.4f}"),
            "mu_rel_error <= 0.25":  (mu_rel_err <= 0.25,                f"Expected <= 0.25, got {mu_rel_err:.4f}"),
            "mask_violations == 0":  (mask_violations == 0,              f"Expected 0, got {mask_violations}"),
            "ll_monotone >= 0.95":   (ll_monotone_frac >= 0.95,          f"Expected >= 0.95, got {ll_monotone_frac:.4f}"),
            "projection_not_fired":  (projection_fired == False,         f"Expected False, got {projection_fired}"),
        }

    print("\n" + "-" * 60)
    print("PASS / FAIL CHECKS")
    print("-" * 60)
    all_passed = True
    for name, (passed, detail) in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name:30s} | {detail}")

    print("-" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED [OK]")
    else:
        print("RESULT: SOME CHECKS FAILED [FAILED]")
        sys.exit(1)


if __name__ == "__main__":
    main()
