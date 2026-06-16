# Reverse2 Final Deployable: Event-Level Network-Constrained Hawkes

This is the fixed deployable version of the reverse2 Hawkes work.

It uses a **flight-level continuous-time event log**, not hourly training bins. The model is a **statistical multivariate Hawkes process**, not a neural model.

## What is fixed

- Uses `TIME_HOURS` consistently.
- Includes `beta.npy` after training.
- Uses source-to-target orientation consistently: `alpha[source, target]`.
- Uses a hard route-network support mask only. Edge strengths are estimated from data.
- Adds self-loops in the adjacency matrix.
- Removes GAT/neural attention/circular graph learning.
- Uses exact fixed-beta EM recursion for the normalized exponential kernel.
- Avoids finite-lookback EM mismatch.
- Includes synthetic recovery before real-data claims.
- Includes held-out log-likelihood, hourly count MAE, and time-rescaling diagnostics.

## Data included

`processed_data/events.csv.gz` contains the train-ready top-30 airport event log from the v2 workaround.

Important columns:

- `TIME_HOURS`: UTC-normalized event time in hours from the first event.
- `NODE`: integer airport node id.
- `node_iata`: source/origin airport IATA code.
- `destination_iata`: destination airport.
- `mark_delay_minutes`: departure delay mark.
- `valid_rotation`: whether the previous same-tail flight is physically valid.
- `prev_was_event_valid_rotation`: safe rotation-propagation indicator.

Adjacency:

- `processed_data/adj_source_target.npy`
- `processed_data/adj_source_target.csv`

Orientation:

```text
alpha[source, target]
adj[source, target] = 1
```

A delay event at `source` may excite future delay intensity at `target`.

## Install

From inside this folder:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## First test run

Run this first. It is small and should finish quickly on a normal laptop.

```bash
python src/run_all.py --quick
```

This runs:

1. synthetic recovery,
2. real-data training on 10,000 events,
3. held-out evaluation,
4. time-rescaling diagnostics.

Outputs:

- `models/alpha.npy`
- `models/mu.npy`
- `models/beta.npy`
- `models/training_log.csv`
- `models/model_metadata.json`
- `output/synthetic/synthetic_recovery_metrics.csv`
- `output/evaluation_summary.csv`
- `output/time_rescaling_diagnostics.csv`

## Standard run

```bash
python src/run_all.py
```

Default real-data training uses 50,000 events and 10 EM iterations.

## Larger run

```bash
python src/train.py --max-events 200000 --iterations 15 --beta 0.2
python src/evaluate.py --max-events 230000 --train-events 200000 --test-events 20000
python src/diagnostics.py --max-events 230000 --top-k 5
```

## Full event-log run

The full event log has hundreds of thousands of events. Use this only on a stronger machine.

```bash
python src/train.py --full --iterations 20 --beta 0.2
```

## Model equation

For target airport `i`:

```text
lambda_i(t) = mu_i + sum_j alpha[j, i] R_j(t)
R_j(t) = sum_{events m at j, t_m < t} beta * exp(-beta * (t - t_m))
```

The route network only controls which `alpha[j, i]` are allowed to be nonzero. It does not learn the strength.

## Important scientific boundary

This package fixes the code and data pipeline. It does not magically prove the paper claim.

Before claiming recovered contagion, report:

- synthetic recovery metrics,
- spectral radius,
- mask violations,
- held-out log-likelihood vs Poisson,
- time-rescaling diagnostics,
- comparison with univariate Hawkes and older GNN foil if used in the paper.

Stationarity projection is enabled by default. If projection happens, disclose it as a constrained stationary fit, not as unconstrained MLE.

## Common commands

Train only:

```bash
python src/train.py --max-events 50000 --iterations 10
```

Evaluate:

```bash
python src/evaluate.py --max-events 70000 --train-events 50000 --test-events 10000
```

Diagnostics:

```bash
python src/diagnostics.py --max-events 100000 --top-k 5
```

Plot alpha heatmap:

```bash
python src/plot_alpha.py
```
