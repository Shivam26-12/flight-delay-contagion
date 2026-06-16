import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default="models/training_log.csv")
    ap.add_argument("--output-dir", default="output/plots")
    args = ap.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return
        
    df = pd.read_csv(log_path)
    
    # Plot Log-Likelihood
    plt.figure(figsize=(8, 5))
    plt.plot(df['iteration'], df['log_likelihood'], marker='o', linestyle='-', color='b')
    plt.title('EM Convergence: Log-Likelihood')
    plt.xlabel('EM Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / "convergence_loglik.png", dpi=150)
    print(f"Saved {out / 'convergence_loglik.png'}")
    
    # Plot Spectral Radius
    plt.figure(figsize=(8, 5))
    plt.plot(df['iteration'], df['spectral_radius'], marker='s', linestyle='-', color='r')
    plt.axhline(y=0.98, color='k', linestyle='--', label='Stationarity Target (0.98)')
    plt.title('EM Convergence: Spectral Radius')
    plt.xlabel('EM Iteration')
    plt.ylabel('Spectral Radius')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / "convergence_rho.png", dpi=150)
    print(f"Saved {out / 'convergence_rho.png'}")

if __name__ == "__main__":
    main()
