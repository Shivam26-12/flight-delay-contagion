# Network-Constrained Hawkes Process: Flight Delay Contagion

This repository contains the complete, deployable pipeline for modeling the cascading contagion of flight delays across the U.S. airport network using a **Network-Constrained Multivariate Hawkes Process**.

Unlike basic machine learning approaches that use arbitrary time-binning or "black-box" neural networks, this model uses **continuous-time exact event logs** and enforces strict physical route masking. It mathematically proves that flight delays are contagious and physically bounds that contagion to actual flight paths.

---

## What Does This Model Do?

1. **Exact Timing:** Uses a normalized exponential kernel to model continuous-time events recursively.
2. **Physical Constraints:** Enforces a hard physical network adjacency mask. Airports cannot mathematically infect each other unless a direct flight physically connects them (Zero Mask Violations).
3. **Rigorous Validation:** Includes automated Synthetic Recovery tests to prove algorithmic identifiability before running on empirical data.
4. **Predictive Tracking:** Forecasts true hourly delays over a blind hold-out window, massively outperforming standard baselines.

---

## Installation & Setup

From inside the root directory, create and activate your virtual environment:

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the strict numerical requirements:
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The codebase includes an automated orchestrator (`src/run_all.py`) that handles synthetic recovery, training, evaluation, and statistical diagnostics in a single command.

### 1. The "Quick Test" Run (Sanity Check)
Run this first. It processes a small subset of events (10,000) to ensure your environment is configured correctly.

```bash
python src/run_all.py --quick
```

### 2. The Standard Research Run
The standard pipeline trains the Expectation-Maximization (EM) algorithm on 50,000 events and evaluates the next 20,000 held-out events.

```bash
python src/run_all.py
```

### 3. The Full Massive-Scale Run
To replicate the full-scale deployment (hundreds of thousands of continuous-time events), run the components manually:

```bash
# 1. Train on 200k historical events
python src/train.py --max-events 200000 --iterations 15 --beta 0.2

# 2. Evaluate blindly on the next 20k future events
python src/evaluate.py --max-events 230000 --train-events 200000 --test-events 20000

# 3. Generate Time-Rescaling Goodness-of-Fit Diagnostics
python src/diagnostics.py --max-events 230000 --top-k 5
```

---

## Visualizations & Reporting

After running the pipeline, a suite of visualization scripts is available to generate research-grade plots.

```bash
# 1. Plot the Contagion Heatmap (Physical Route Validation)
python src/plot_alpha.py

# 2. Plot the EM Convergence Curves (Mathematical Stability)
python src/plot_convergence.py

# 3. Plot the Filtered Daytime Q-Q Plots (Time-Rescaling Diagnostics)
python src/plot_qq.py --max-events 230000 --top-k 5

# 4. Plot Real vs. Predicted Delays (Time-Series Holdout Validation)
python src/plot_predictions.py
```
*Note: All plots are automatically saved directly to the `output/plots/` directory.*

For a full academic writeup of what these plots and metrics mean, see the generated [FINAL_REPORT.md](FINAL_REPORT.md).

---

## The Mathematical Model

For target airport `i`, the instantaneous probability (intensity) of a delay at time `t` is defined as:

```text
lambda_i(t) = mu_i + sum_j alpha[j, i] R_j(t)
R_j(t) = sum_{events m at j, t_m < t} beta * exp(-beta * (t - t_m))
```

* `mu_i`: The independent baseline delay rate for airport `i`.
* `alpha[j, i]`: The learned contagion strength from source `j` to target `i`.
* `R_j(t)`: The fading exponential risk caused by past delays at `j`.

---

## Security & Reliability

This codebase has been verified secure using Bandit (0 High/Medium vulnerabilities). It relies on modern `numpy>=1.23` which safely defaults `allow_pickle=False` against arbitrary `.npy` deserialization attacks.
