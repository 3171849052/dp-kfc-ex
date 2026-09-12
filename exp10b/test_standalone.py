"""Regression for the user-requested standalone Equil bound removal."""
import unittest
from unittest.mock import patch
import torch
from exp10 import run_exp10 as base
from dp_kfac.standalone.config import load_config
from dp_kfac.standalone.run_logging import format_run_name
from dp_kfac.standalone.trainer import build_equil


class StandaloneTest(unittest.TestCase):
    def test_configs_names_and_unbounded_formula(self):
        torch.set_num_threads(4)
        for path in ("configs/standalone/mnist_dp_equil.yaml",
                     "configs/standalone/mnist_dp_equil_adamw.yaml"):
            c = load_config(path)
            self.assertEqual(set(c["equil"]), {"probes", "tau"})
            self.assertNotIn("cap", format_run_name(c))
        device = torch.device("cuda:0")
        torch.manual_seed(42)
        model = base.GradSampleModule(base.SimpleCNN().to(device), loss_reduction="sum")
        c["synthetic"]["samples"] = 4
        c["equil"]["tau"] = 1.
        batches = [(torch.randn(4, 1, 28, 28, device=device), torch.arange(4, device=device))]
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(c["seed"]+30001)
            probes = base.rademacher(model.parameters(), c["equil"]["probes"])
        _, statistic = base.fisher_statistics(model, batches, device, probes)
        values = torch.cat([v.flatten() for v in statistic.values()])
        raw = {p: (v+values.median()).rsqrt() for p, v in statistic.items()}
        gm = (sum(v.log().sum() for v in raw.values())/values.numel()).exp()
        with patch("dp_kfac.standalone.trainer.pink_batches", return_value=iter(batches)):
            scales = build_equil(model, c, device, 1)
        for p in scales:
            torch.testing.assert_close(scales[p], raw[p]/gm, rtol=0, atol=0)
        self.assertAlmostEqual(torch.cat([s.flatten() for s in scales.values()]).double().log().mean().exp().item(), 1, places=6)
        model.remove_hooks()


if __name__ == "__main__":
    unittest.main(verbosity=2)
