"""CUDA numerical and exact-grad_sample regression tests."""
import unittest
from unittest.mock import patch
import torch
from exp10 import run_exp10 as base
from exp10b import operators as op
from exp10b.config import METHODS, STRUCTURED, LAYERS, PROBES, TAU


class OperatorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cls.device = torch.device("cuda:0")
        torch.manual_seed(42)
        cls.model = base.GradSampleModule(base.SimpleCNN().to(cls.device), loss_reduction="sum")
        cls.params = list(cls.model.parameters())
        probes = base.rademacher(cls.params, PROBES)
        cls.batches = [(torch.randn(4, 1, 28, 28, device=cls.device),
                       torch.arange(4, device=cls.device))]
        cls.statistic = base.layerwise_statistics(cls.model, cls.batches, cls.device, probes)
        cls.probes = probes
        cls.target = op.augmented(cls.model, op.equil_scales(cls.statistic))

    @classmethod
    def tearDownClass(cls):
        cls.model.remove_hooks()

    def test_01_exp10_estimator_and_unbounded_scale(self):
        self.assertIs(op.layerwise_statistics, base.layerwise_statistics)
        actual = op.layerwise_statistics(self.model, self.batches, self.device, self.probes)
        for p in self.params:
            torch.testing.assert_close(actual[p], self.statistic[p], rtol=0, atol=0)
        values = torch.cat([v.flatten() for v in actual.values()])
        raw = {p: (v + TAU*values.median()).rsqrt() for p, v in actual.items()}
        gm = (sum(v.log().sum() for v in raw.values())/values.numel()).exp()
        new = op.equil_scales(actual)
        old, _ = base.equil_scales(actual, TAU)
        for p in self.params:
            torch.testing.assert_close(new[p], raw[p]/gm, rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(new[p].clamp(.1, 10), old[p], rtol=2e-6, atol=1e-7)
        # An extreme valid statistic must retain gains beyond the old bounds.
        extreme = {p: torch.ones_like(p) for p in self.params}
        extreme[self.params[0]].flatten()[0] = 1e12
        self.assertLess(op.equil_scales(extreme)[self.params[0]].flatten()[0].item(), .1)

    def test_02_factorization_least_squares_and_bias(self):
        operator = op.Operator(STRUCTURED[1], self.target)
        dense = operator.dense()
        for name, (r, c) in operator.data.items():
            self.assertEqual(r.ndim, 1)
            self.assertEqual(c.ndim, 1)
            torch.testing.assert_close(dense[name], r[:, None]*c[None, :])
            residual = self.target[name].double().log()-dense[name].double().log()
            torch.testing.assert_close(residual.mean(0), torch.zeros_like(residual.mean(0)), atol=2e-7, rtol=0)
            torch.testing.assert_close(residual.mean(1), torch.zeros_like(residual.mean(1)), atol=2e-7, rtol=0)

    def test_03_scalar_and_coordinate_weighted_gm(self):
        for kind in STRUCTURED:
            operator = op.Operator(kind, self.target)
            values = torch.cat([v.flatten() for v in operator.dense().values()])
            self.assertAlmostEqual(values.double().log().mean().exp().item(), 1, places=6)
            if kind == STRUCTURED[2]:
                self.assertTrue(all(s.numel() == 1 for s in operator.data.values()))
                self.assertTrue(all(v.unique().numel() == 1 for v in operator.dense().values()))

    def test_04_same_targets_rng_and_full_gm(self):
        hashes = []
        for method in METHODS[1:]:
            before = torch.cuda.get_rng_state().clone()
            operator, gains, records = op.build(self.model, method, 42, 1, self.device)
            self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
            self.assertAlmostEqual(gains["gain_geomean"], 1, places=6)
            self.assertEqual(len({r["target_sha256"] for r in records}), 1)
            self.assertEqual(len({r["statistic_sha256"] for r in records}), 1)
            if method in STRUCTURED:
                self.assertEqual({r["method"] for r in records}, set(STRUCTURED))
                hashes.append((records[0]["statistic_sha256"], records[0]["target_sha256"]))
        self.assertEqual(len(set(hashes)), 1)

    def test_05_private_update_matches_dense_reference_without_materialization(self):
        for method in METHODS:
            self.model.zero_grad(set_to_none=True)
            x, y = self.batches[0]
            base.F.cross_entropy(self.model(x), y, reduction="sum").backward()
            operator = None if method == METHODS[0] else op.Operator(method, self.target)
            originals = {p: p.grad_sample.clone() for p in self.params}
            if operator is not None:
                dense = operator.dense()
                for name in LAYERS:
                    m = getattr(self.model._module, name)
                    originals[m.weight].mul_(dense[name][:, :-1].reshape_as(m.weight))
                    originals[m.bias].mul_(dense[name][:, -1])
                with patch.object(op.Operator, "dense", side_effect=AssertionError("Training materialized operator")):
                    operator.apply(self.model)
                for p in self.params:
                    torch.testing.assert_close(p.grad_sample, originals[p], rtol=2e-6, atol=1e-7)
            norm_sq = sum(g.flatten(1).norm(2, dim=1).square() for g in originals.values())
            factors = base._compute_clip_factors(norm_sq, 1.)
            expected = {p: (g.flatten(1)*factors[:, None]).mean(0).view_as(p)
                        for p, g in originals.items()}
            before = {p: p.detach().clone() for p in self.params}
            base.clip_and_noise_gradients(self.model, 0., 1., len(x))
            for p in self.params:
                torch.testing.assert_close(p.grad, expected[p], rtol=3e-6, atol=1e-7)
            torch.optim.SGD(self.params, lr=.5).step()
            for p in self.params:
                torch.testing.assert_close(p, before[p]-.5*expected[p], rtol=3e-6, atol=1e-7)
                self.assertTrue(torch.isfinite(p).all())

    def test_06_dof(self):
        self.assertEqual(op.Operator(STRUCTURED[0], self.target).dof, 206921)
        self.assertEqual(op.Operator(STRUCTURED[1], self.target).dof, 2034)
        self.assertEqual(op.Operator(STRUCTURED[2], self.target).dof, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
