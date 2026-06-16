# Manifest

## Core code

- `src/data.py` — loader + validation for event-level log.
- `src/hawkes_em.py` — exact fixed-beta EM estimator for network-constrained Hawkes.
- `src/synthetic_recovery.py` — synthetic ground-truth recovery test.
- `src/train.py` — real-data training.
- `src/evaluate.py` — held-out likelihood and hourly count evaluation.
- `src/diagnostics.py` — time-rescaling KS diagnostics.
- `src/plot_alpha.py` — alpha heatmap.
- `src/run_all.py` — complete quick/standard workflow.

## Data

- `processed_data/events.csv.gz` — v2 continuous-time top-30 event log.
- `processed_data/adj_source_target.npy` — source-to-target route mask with self-loops.
- `processed_data/node_index.csv` — node id to airport mapping.

## Model/output folders

The included `models/` and `output/` folders contain a successful quick smoke-test run. You can delete them and rerun.
