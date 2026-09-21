"""Exactly one fresh process/run per physical GPU."""
import argparse
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33 import config as cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=tuple(cfg.GPU_METHODS), required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from exp33.run import run
    import torch
    torch.set_num_threads(4)
    run(cfg.GPU_METHODS[args.gpu])

if __name__ == '__main__':
    main()
