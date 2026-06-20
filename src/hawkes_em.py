from __future__ import annotations
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class EMHistoryRow:
    iteration: int
    log_likelihood: float
    beta: float
    rho_pre_projection: float
    rho_post_projection: float
    projection_applied: bool
    projection_scale: float
    elapsed_seconds: float
    mu_min: float
    mu_max: float
    alpha_sum: float


class NetworkConstrainedHawkesEM:
    """
    Exact fixed-beta EM for a network-constrained multivariate Hawkes process
    with normalized exponential kernel.

    Orientation:
        alpha[source, target]
        adj[source, target] = 1 allows source events to excite target intensity.

    Intensity:
        lambda_target(t) = mu[target] + sum_source alpha[source, target] * R_source(t)
        R_source(t) = sum_{events m at source, t_m < t} beta * exp(-beta * (t - t_m))

    This avoids hourly bins and avoids neural attention. The graph is a hard support
    constraint only; all nonzero edge strengths are estimated from event timestamps.
    """

    def __init__(
        self,
        num_nodes: int,
        adj_matrix: np.ndarray,
        beta: float = 0.2,
        seed: int = 7,
        stationarity_project: bool = True,
        stationarity_target: float = 0.98,
        init_alpha_scale: float = 0.02,
        init_mu_scale: float = 0.01,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.num_nodes = int(num_nodes)
        self.adj_matrix = (np.asarray(adj_matrix) > 0).astype(float)
        if self.adj_matrix.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("adj_matrix shape must be (num_nodes, num_nodes)")
        self.beta = float(beta)
        self.stationarity_project = bool(stationarity_project)
        self.stationarity_target = float(stationarity_target)
        rng = np.random.default_rng(seed)
        self.mu = rng.uniform(0.2 * init_mu_scale, init_mu_scale, self.num_nodes)
        self.alpha = rng.uniform(0.0, init_alpha_scale, (self.num_nodes, self.num_nodes)) * self.adj_matrix
        self.history: list[EMHistoryRow] = []
        self.was_projected_last_iter = False

    @staticmethod
    def spectral_radius(alpha: np.ndarray) -> float:
        vals = np.linalg.eigvals(alpha)
        return float(np.max(np.abs(vals)))

    def _denominator_by_source(self, times: np.ndarray, nodes: np.ndarray, T: float) -> np.ndarray:
        denom = np.zeros(self.num_nodes, dtype=float)
        terms = 1.0 - np.exp(-self.beta * np.maximum(T - times, 0.0))
        np.add.at(denom, nodes, terms)
        return denom

    def log_likelihood(self, times: np.ndarray, nodes: np.ndarray, T: float) -> float:
        times = np.asarray(times, dtype=float)
        nodes = np.asarray(nodes, dtype=int)
        R = np.zeros(self.num_nodes, dtype=float)
        last_t = float(times[0]) if len(times) else 0.0
        ll = 0.0
        eps = 1e-12
        for t, target in zip(times, nodes):
            dt = float(t - last_t)
            if dt < -1e-12:
                raise ValueError("times must be sorted")
            if dt > 0:
                R *= np.exp(-self.beta * dt)
            lam = self.mu[target] + float(R @ self.alpha[:, target])
            ll += np.log(max(lam, eps))
            R[target] += self.beta
            last_t = float(t)
        denom = self._denominator_by_source(times, nodes, T)
        integral = float(self.mu.sum() * T + denom @ self.alpha.sum(axis=1))
        return float(ll - integral)

    def fit(self, times: np.ndarray, nodes: np.ndarray, T: float, max_iter: int = 20, tol: float = 1e-5, verbose: bool = True):
        times = np.asarray(times, dtype=float)
        nodes = np.asarray(nodes, dtype=int)
        order = np.argsort(times, kind="mergesort")
        times = times[order]
        nodes = nodes[order]
        if len(times) != len(nodes):
            raise ValueError("times and nodes lengths differ")
        if nodes.min() < 0 or nodes.max() >= self.num_nodes:
            raise ValueError("node id outside model dimension")
        if T <= 0:
            raise ValueError("T must be positive")

        denom_by_source = self._denominator_by_source(times, nodes, T)
        denom_safe = np.maximum(denom_by_source, 1e-12)
        prev_ll = -np.inf
        self.history = []

        if verbose:
            print(f"EM fit: events={len(times):,}, nodes={self.num_nodes}, T_hours={T:.2f}, beta={self.beta:.6f}")
            print("Kernel: normalized exponential, fixed beta; graph is hard source->target support mask.")

        for it in range(1, int(max_iter) + 1):
            tic = time.time()
            R = np.zeros(self.num_nodes, dtype=float)
            last_t = float(times[0])
            sum_bg = np.zeros(self.num_nodes, dtype=float)
            sum_trig = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
            ll_events = 0.0
            eps = 1e-12

            for t, target in zip(times, nodes):
                dt = float(t - last_t)
                if dt > 0:
                    R *= np.exp(-self.beta * dt)
                elif dt < -1e-12:
                    raise ValueError("times must be sorted")

                source_contrib = R * self.alpha[:, target]
                lam = self.mu[target] + float(source_contrib.sum())
                lam = max(lam, eps)
                ll_events += np.log(lam)
                sum_bg[target] += self.mu[target] / lam
                sum_trig[:, target] += source_contrib / lam

                # Current event can trigger future events at the same continuous timestamp only after this event.
                R[target] += self.beta
                last_t = float(t)

            integral = float(self.mu.sum() * T + denom_by_source @ self.alpha.sum(axis=1))
            ll = float(ll_events - integral)

            new_mu = sum_bg / T
            new_alpha = sum_trig / denom_safe[:, None]
            new_alpha *= self.adj_matrix
            new_alpha = np.maximum(new_alpha, 0.0)

            rho_pre = self.spectral_radius(new_alpha)
            projected = False
            scale = 1.0
            rho_post = rho_pre
            if self.stationarity_project and rho_pre >= self.stationarity_target:
                scale = self.stationarity_target / max(rho_pre, 1e-12)
                new_alpha *= scale
                rho_post = self.spectral_radius(new_alpha)
                projected = True

            self.mu = np.maximum(new_mu, 1e-12)
            self.alpha = new_alpha
            self.was_projected_last_iter = projected

            elapsed = time.time() - tic
            row = EMHistoryRow(
                iteration=it,
                log_likelihood=ll,
                beta=self.beta,
                rho_pre_projection=rho_pre,
                rho_post_projection=rho_post,
                projection_applied=projected,
                projection_scale=scale,
                elapsed_seconds=elapsed,
                mu_min=float(self.mu.min()),
                mu_max=float(self.mu.max()),
                alpha_sum=float(self.alpha.sum()),
            )
            self.history.append(row)
            if verbose:
                flag = f" [CLAMPED {rho_pre:.4f} -> {rho_post:.4f}]" if projected else ""
                print(f"iter={it:02d} ll={ll:,.2f} rho_pre={rho_pre:.4f}{flag} alpha_sum={self.alpha.sum():.2f} time={elapsed:.1f}s")
            if it > 1 and abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
                if verbose:
                    print("Convergence tolerance reached.")
                break
            prev_ll = ll
        return pd.DataFrame([asdict(r) for r in self.history])

    def save(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "alpha.npy", self.alpha)
        np.save(model_dir / "mu.npy", self.mu)
        np.save(model_dir / "beta.npy", np.array([self.beta], dtype=float))
        np.save(model_dir / "adj_source_target.npy", self.adj_matrix)
        if self.history:
            pd.DataFrame([asdict(r) for r in self.history]).to_csv(model_dir / "training_log.csv", index=False)
        meta = {
            "orientation": "alpha[source,target]",
            "kernel": "normalized exponential beta*exp(-beta*dt)",
            "beta": self.beta,
            "stationarity_project": self.stationarity_project,
            "stationarity_target": self.stationarity_target,
            "spectral_radius": self.spectral_radius(self.alpha),
            "projected_last_iteration": self.was_projected_last_iter,
        }
        pd.Series(meta).to_json(model_dir / "model_metadata.json", indent=2)

    @classmethod
    def load(cls, model_dir: str | Path):
        model_dir = Path(model_dir)
        alpha = np.load(model_dir / "alpha.npy")
        mu = np.load(model_dir / "mu.npy")
        beta = float(np.load(model_dir / "beta.npy")[0])
        adj_path = model_dir / "adj_source_target.npy"
        adj = np.load(adj_path) if adj_path.exists() else (alpha > 0).astype(float)
        obj = cls(num_nodes=alpha.shape[0], adj_matrix=adj, beta=beta)
        obj.alpha = alpha
        obj.mu = mu
        return obj
