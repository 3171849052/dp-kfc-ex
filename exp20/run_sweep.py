"""Sequential balanced sweep; each run gets a fresh interpreter."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp20.config import POWERS, SEEDS, POWER_ORDER, FORMAL_RUN_COUNT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = Path('exp20/results/smoke' if args.smoke else 'exp20/results')
    output.mkdir(parents=True, exist_ok=True)
    seeds = (42,) if args.smoke else SEEDS
    order = {seed: POWERS if args.smoke else POWER_ORDER[seed] for seed in seeds}
    (output/'config.json').write_text(json.dumps(dict(smoke=args.smoke, seeds=seeds,
        power_order=order, expected_runs=8 if args.smoke else FORMAL_RUN_COUNT), indent=2)+'\n')
    # Preserve previous aggregates until the complete replacement sweep is available,
    # but mark them explicitly as belonging to the previous protocol.
    archive = output/'previous_aggregate'
    for path in list(output.iterdir()):
        if path.is_file() and path.name != 'config.json':
            archive.mkdir(exist_ok=True)
            path.replace(archive/path.name)
    flags = ['--smoke'] if args.smoke else []
    for seed, powers in order.items():
        for power in powers:
            subprocess.run([sys.executable, 'exp20/run_one.py', '--p', str(power),
                            '--seed', str(seed), *flags], check=True)
    subprocess.run([sys.executable, 'exp20/analyze.py', *flags], check=True)
