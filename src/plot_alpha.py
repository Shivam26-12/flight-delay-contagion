from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--data-dir", default="processed_data")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    alpha = np.load(Path(args.model_dir) / "alpha.npy")
    labels = [str(i) for i in range(alpha.shape[0])]
    idx = Path(args.data_dir) / "node_index.csv"
    if idx.exists():
        df = pd.read_csv(idx)
        for col in ["iata_code", "IATA", "node_iata", "iata"]:
            if col in df.columns:
                labels = df.sort_values("NODE")[col].astype(str).tolist()
                break
    plt.figure(figsize=(9, 7))
    plt.imshow(alpha, aspect="auto")
    plt.colorbar(label="alpha[source,target]")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("Target airport")
    plt.ylabel("Source airport")
    plt.title("Network-constrained Hawkes alpha matrix")
    plt.tight_layout()
    plt.savefig(out / "alpha_heatmap.png", dpi=200)
    print(f"saved {out / 'alpha_heatmap.png'}")

if __name__ == "__main__":
    main()
