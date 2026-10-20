"""Basic checks for bws_sim.core. Run: python -m pytest -q  (or: python tests/test_core.py)"""
import os
import sys

import numpy as np
from scipy.optimize import approx_fprime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bws_sim.core import (bibd_13_4_1, bibd_7_4_2, gdd_2_7, check_design, simulate,
                          Data, Spec, fit, counts, _core)


def test_designs():
    assert check_design(bibd_13_4_1(), 13) == dict(sets=13, k=4, occ_min=4, occ_max=4, pair_min=1, pair_max=1)
    assert check_design(bibd_7_4_2(), 7) == dict(sets=7, k=4, occ_min=4, occ_max=4, pair_min=2, pair_max=2)
    d = check_design(gdd_2_7(), 14)
    assert (d["sets"], d["occ_min"], d["occ_max"], d["pair_min"], d["pair_max"]) == (14, 4, 4, 0, 1)


def _two_group_data(seed=1):
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, 13)
    d0 = simulate(bibd_13_4_1(), np.arange(13), beta, 120, rng, lam=1.0, group=0)
    d1 = simulate(bibd_13_4_1(), np.arange(13), beta, 120, rng, lam=0.7, group=1)
    return Data.concat([d0, d1])


def test_gradient_matches_finite_differences():
    D = _two_group_data()
    rng = np.random.default_rng(0)
    for sp in (Spec(13, 6, n_groups=2, scale=True, delta_items=[2]), Spec(13, 6, n_groups=2, sep_beta=True)):
        th = rng.normal(0, 0.3, sp.P)
        num = approx_fprime(th, lambda x: _core(x, D, sp)[0], 1e-6)
        _, grad, rows = _core(th, D, sp, need_rows=True)
        assert np.max(np.abs(num - grad)) / np.max(np.abs(grad)) < 1e-4
        assert np.allclose(rows.sum(0), grad)


def test_parameter_recovery_without_heterogeneity():
    rng = np.random.default_rng(3)
    beta = np.linspace(-1, 1, 13)
    d = simulate(bibd_13_4_1(), np.arange(13), beta, 2000, rng, eta_sd=0.0)
    sp = Spec(13, 6)
    o = fit(d, sp)
    est = np.zeros(13)
    est[sp.free] = o["theta"]
    assert np.max(np.abs(est - (beta - beta[6]))) < 4 * o["se"].max()


def test_scale_ratio_recovery_without_heterogeneity():
    rng = np.random.default_rng(4)
    beta = np.linspace(-1, 1, 13)
    d0 = simulate(bibd_13_4_1(), np.arange(13), beta, 1500, rng, eta_sd=0.0, group=0)
    d1 = simulate(bibd_13_4_1(), np.arange(13), beta, 1500, rng, eta_sd=0.0, lam=0.6, group=1)
    sp = Spec(13, 6, n_groups=2, scale=True)
    o = fit(Data.concat([d0, d1]), sp)
    assert abs(np.exp(o["theta"][-1]) - 0.6) < 0.05


def test_same_seed_same_data():
    a = simulate(bibd_13_4_1(), np.arange(13), np.zeros(13), 10, np.random.default_rng(9))
    b = simulate(bibd_13_4_1(), np.arange(13), np.zeros(13), 10, np.random.default_rng(9))
    assert np.array_equal(a.items, b.items) and np.array_equal(a.best, b.best) and np.array_equal(a.worst, b.worst)


def test_counts_range():
    rng = np.random.default_rng(5)
    d = simulate(bibd_13_4_1(), np.arange(13), np.linspace(-1, 1, 13), 50, rng)
    agg, per = counts(d, 13)
    assert np.all(np.abs(agg) <= 1) and np.nanmax(np.abs(per)) <= 1
    assert np.isclose(agg.sum(), 0)          # every set has one best and one worst


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
