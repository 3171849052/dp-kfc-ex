from dp_kfac.optimizer import generate_pink_noise
import torch


def synthetic_cache(batches, batch_size, device, seed):
    torch.manual_seed(seed)
    return [generate_pink_noise(batch_size, (1, 28, 28), device) for _ in range(batches)]
