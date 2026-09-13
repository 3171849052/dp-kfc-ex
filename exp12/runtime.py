"""Exp12 CUDA execution settings; initialization stays outside measured builds."""
from contextlib import contextmanager
import torch


@contextmanager
def runtime(device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Run VJPs on the caller thread, whose CUDA context is initialized here.
    # This avoids lazy cuBLAS context creation on an autograd worker thread.
    with torch.autograd.set_multithreading_enabled(False):
        if torch.device(device).type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.init()
                a = torch.ones(2, 2, device=device, requires_grad=True)
                torch.autograd.grad((a @ a).sum(), a)
                torch.cuda.synchronize(device)
                del a
        yield
