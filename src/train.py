from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_arrays, validate_event_log
from hawkes_em import NetworkConstrainedHawkesEM


def main():
    ap = argparse.ArgumentParser(description="Train network-constrained fixed-beta EM Hawkes model.")
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--max-events", type=int, default=50000, help="Default cap for practical local runs. Use --full for all events.")
    ap.add_argument("--full", action="store_true", help="Use all events. This may be slow.")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.2, help="Exponential decay rate per hour. 0.2 => half-life 3.47h.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-stationarity-project", action="store_true", help="Disable projection of alpha to spectral radius < target.")
    ap.add_argument("--stationarity-target", type=float, default=0.98)
    args = ap.parse_args()

    report = validate_event_log(args.data_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.Series(report).to_csv(Path(args.output_dir) / "data_validation_report.csv")
    max_events = None if args.full else args.max_events
    times, nodes, adj, T, df = load_arrays(args.data_dir, max_events=max_events)

    model = NetworkConstrainedHawkesEM(
        num_nodes=adj.shape[0],
        adj_matrix=adj,
        beta=args.beta,
        seed=args.seed,
        stationarity_project=not args.no_stationarity_project,
        stationarity_target=args.stationarity_target,
    )
    hist = model.fit(times, nodes, T=T, max_iter=args.iterations, verbose=True)
    model.save(args.model_dir)
    hist.to_csv(Path(args.output_dir) / "training_log.csv", index=False)
    print(f"Saved model to {args.model_dir}")
    print(f"Final spectral radius: {model.spectral_radius(model.alpha):.4f}")


if __name__ == "__main__":
    main()
