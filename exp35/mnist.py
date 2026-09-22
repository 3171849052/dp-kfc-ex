"""Exp27 loader protocol and unchanged paper CNN BK training loop."""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.paper import exp_cnn_mnist_a as reference
from exp35 import config as cfg
from exp35.adapters import bind, prepare, MetricsSink
from exp35.geometry import spectral_operator, diagnostics


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train = datasets.MNIST(cfg.DATA_ROOT, train=True, download=False, transform=transform)
    test = datasets.MNIST(cfg.DATA_ROOT, train=False, download=False, transform=transform)
    return (DataLoader(train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True))


def run(method):
    directory, _ = prepare('mnist', method)
    state, gains = {}, {}
    variant = method.removeprefix('dp_kfc_a_')

    def model_factory(**kwargs):
        model = reference.SimpleCNN(**kwargs)
        state['model'] = model
        state['initial'] = {n: p.detach().clone() for n, p in model.named_parameters()}
        return model

    def compute(covariance, power, damping):
        assert (power, damping) == (.4, .001)
        matrix, gain = spectral_operator(covariance, variant)
        gains[str(len(gains))] = gain
        return matrix

    def covariance(activation):
        augmented = torch.cat((activation, torch.ones_like(activation[:, :1])), dim=1)
        return augmented.T @ augmented / len(augmented)

    geometry = bind(reference.build_geometry, compute_a_operator=compute, compute_a_covariance=covariance)

    def build(*args):
        gains.clear()
        ua, ug = geometry(*args)
        # Replace numeric call order by the exact reference layer names.
        named_gains = dict(zip(ua, gains.values()))
        state['diagnostics'] = diagnostics(named_gains)
        state['diagnostics']['operator_state_bytes'] = sum(t.numel()*t.element_size() for t in ua.values())
        return ua, ug

    def extra():
        model = state['model']
        updated = [not torch.equal(state['initial'][n].to(p.device), p.detach()) for n,p in model.named_parameters()]
        return dict(state['diagnostics'], parameters_finite=all(torch.isfinite(p).all().item() for p in model.parameters()),
                    parameters_updated=any(updated), all_parameters_updated=all(updated))

    loop = bind(reference.run_one, LR=.002, SimpleCNN=model_factory, build_geometry=build,
                pd=MetricsSink(directory, 'mnist', method, extra))
    train, test = load_data()
    assert len(train.dataset) == 60000 and len(test.dataset) == 10000
    loop(train, test, {}, 'base' if method == 'dp_adam' else 'a_only', 'pink',
         'bk', 2, 42, 5, torch.device('cuda:0'), directory, profile=True)
