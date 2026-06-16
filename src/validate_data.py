from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from data import validate_event_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='processed_data')
    ap.add_argument('--output-dir', default='output')
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = validate_event_log(args.data_dir)
    s = pd.Series(report)
    s.to_csv(Path(args.output_dir) / 'data_validation_report.csv')
    print(s.to_string())

if __name__ == '__main__':
    main()
