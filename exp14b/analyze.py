"""Reuse Exp14 summaries; metrics and final rows retain scale diagnostics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from exp14.analyze import save


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    save(pd.read_csv(args.output / 'metrics.csv').to_dict('records'),
         pd.read_csv(args.output / 'geometry.csv').to_dict('records'), args.output)
