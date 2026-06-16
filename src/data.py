from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_EVENT_COLUMNS = {"TIME_HOURS", "NODE"}


def load_events(data_dir: str | Path = "processed_data", max_events: int | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    candidates = [data_dir / "events.csv.gz", data_dir / "events.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No events.csv.gz or events.csv found in {data_dir}")
    df = pd.read_csv(path)

    # Backward compatibility with older workaround files.
    if "TIME_HOURS" not in df.columns and "TIME_MINUTES" in df.columns:
        df["TIME_HOURS"] = df["TIME_MINUTES"].astype(float) / 60.0
    if "NODE" not in df.columns and "node" in df.columns:
        df["NODE"] = df["node"].astype(int)

    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Event log missing required columns: {sorted(missing)}")

    df = df.sort_values("TIME_HOURS").reset_index(drop=True)
    df["TIME_HOURS"] = df["TIME_HOURS"].astype(float)
    df["NODE"] = df["NODE"].astype(int)
    if max_events is not None:
        df = df.head(int(max_events)).copy()
    return df


def load_arrays(data_dir: str | Path = "processed_data", max_events: int | None = None):
    data_dir = Path(data_dir)
    df = load_events(data_dir, max_events=max_events)
    adj_path = data_dir / "adj_source_target.npy"
    if not adj_path.exists():
        adj_path = data_dir / "adj.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"No adjacency file found in {data_dir}")
    adj = np.load(adj_path).astype(float)
    times = df["TIME_HOURS"].to_numpy(dtype=float)
    times = times - times[0]
    nodes = df["NODE"].to_numpy(dtype=int)
    T = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    if T <= 0:
        raise ValueError("Observation horizon T must be positive after sorting.")
    return times, nodes, adj, T, df


def load_node_index(data_dir: str | Path = "processed_data") -> pd.DataFrame:
    p = Path(data_dir) / "node_index.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def validate_event_log(data_dir: str | Path = "processed_data") -> dict:
    df = load_events(data_dir)
    adj = np.load(Path(data_dir) / "adj_source_target.npy")
    times = df["TIME_HOURS"].to_numpy(float)
    nodes = df["NODE"].to_numpy(int)
    report = {
        "rows": int(len(df)),
        "nodes_in_log": int(df["NODE"].nunique()),
        "adj_shape": str(tuple(adj.shape)),
        "allowed_edges_including_self_loops": int(adj.sum()),
        "min_time_hours": float(times.min()),
        "max_time_hours": float(times.max()),
        "duration_days": float((times.max() - times.min()) / 24.0),
        "global_time_monotone_after_sort": bool(np.all(np.diff(times) >= 0)),
        "same_node_nonpositive_gaps": 0,
        "node_id_min": int(nodes.min()),
        "node_id_max": int(nodes.max()),
    }
    bad_gaps = 0
    for u, g in df.groupby("NODE"):
        t = g["TIME_HOURS"].to_numpy(float)
        bad_gaps += int(np.sum(np.diff(t) <= 0))
    report["same_node_nonpositive_gaps"] = bad_gaps
    if report["node_id_min"] < 0 or report["node_id_max"] >= adj.shape[0]:
        raise ValueError("NODE ids are outside adjacency matrix dimensions.")
    if bad_gaps != 0:
        raise ValueError(f"Found {bad_gaps} non-positive within-node gaps; event-time jitter failed.")
    return report
