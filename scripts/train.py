#!/usr/bin/env python
"""Prepare a standalone run, inspect its GPU, or train a prepared directory."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-run', action='store_true')
    modes.add_argument('--run-dir', type=Path)
    modes.add_argument('--print-gpu', action='store_true')
    modes.add_argument('--validate-gpu', action='store_true')
    modes.add_argument('--tmux-session-name', type=Path, metavar='RUN_DIR')
    args = parser.parse_args()
    # dp_kfac's existing __init__ imports torch, so load YAML before importing it.
    import yaml
    if args.tmux_session_name:
        from dp_kfac.standalone.run_logging import format_tmux_session_name
        print(format_tmux_session_name(args.tmux_session_name))
        return
    if args.config is None:
        parser.error('--config is required')
    raw = yaml.safe_load(args.config.read_text())
    if args.validate_gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(raw.get('runtime', {}).get('gpu', 0))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from dp_kfac.standalone.config import load_config
    c = load_config(args.config)
    if args.print_gpu:
        print(c['runtime']['gpu'])
    elif args.validate_gpu:
        import torch
        if c['runtime']['device'] == 'cuda':
            gpu = c['runtime']['gpu']
            if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
                raise ValueError(f'physical GPU {gpu} is unavailable')
            torch.empty(1, device=f'cuda:{gpu}')
        print(c['runtime']['gpu'])
    elif args.prepare_run:
        from dp_kfac.standalone.run_logging import prepare_run
        print(prepare_run(c, args.config))
    else:
        from dp_kfac.standalone.trainer import train
        train(c, args.run_dir)


if __name__ == '__main__':
    main()
