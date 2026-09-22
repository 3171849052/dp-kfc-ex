"""Save checkpoints at existing boundaries of the unchanged Exp30 loop."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36.runtime import setup, ROOT


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    setup()
    import torch
    import pandas as pd
    from exp35 import vit
    from exp35.adapters import bind
    torch.set_num_threads(4)
    destination = ROOT/'checkpoints'
    destination.mkdir(parents=True, exist_ok=False)
    rows = []

    def save(model, epoch):
        path = destination/f'epoch_{epoch}.pt'
        torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, path)
        rows.append(dict(epoch=epoch, checkpoint=str(path.relative_to(ROOT)), seed=42,
                         method='dp_adamw', physical_gpu=3))
        (ROOT/'results').mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(ROOT/'results/checkpoints.csv', index=False)

    def initialize(seed, device):
        model = vit.reference.initialize(seed, device)
        save(model, 0)
        return model

    def evaluate(model, loader, device):
        result = vit.reference.evaluate(model, loader, device)
        save(model, len(rows))
        return result

    vit.reference.run = bind(vit.reference.run, initialize=initialize, evaluate=evaluate)
    vit.run('dp_adamw')


if __name__ == '__main__':
    main()
