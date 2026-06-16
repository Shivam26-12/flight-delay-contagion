from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run validation, synthetic recovery, training, evaluation, diagnostics.")
    ap.add_argument("--quick", action="store_true", help="Small run suitable for first test.")
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if not args.skip_synthetic:
        syn_args = [py, "src/synthetic_recovery.py", "--T", "300", "--iterations", "6"] if args.quick else [py, "src/synthetic_recovery.py"]
        run(syn_args)
    if args.quick:
        run([py, "src/train.py", "--max-events", "10000", "--iterations", "4"])
        run([py, "src/evaluate.py", "--max-events", "15000", "--train-events", "10000", "--test-events", "3000"])
        run([py, "src/diagnostics.py", "--max-events", "15000", "--top-k", "3"])
    else:
        run([py, "src/train.py", "--max-events", "50000", "--iterations", "10"])
        run([py, "src/evaluate.py", "--max-events", "70000", "--train-events", "50000", "--test-events", "10000"])
        run([py, "src/diagnostics.py", "--max-events", "100000", "--top-k", "5"])
    print("\nDone. See output/ and models/.")


if __name__ == "__main__":
    main()
