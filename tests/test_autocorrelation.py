import numpy as np

from src.measure import (
    autocorrelation,
    autocorrelation_inflation,
    integrated_autocorrelation_time,
    jackknife_columns,
    jackknife_mean,
)


def ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    """Stationary AR(1): rho(t) = phi^t, so tau_int = 1/2 + phi/(1 - phi)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(1.0 - phi**2), size=n)
    v = np.empty(n)
    v[0] = rng.normal(0.0, 1.0)
    for i in range(1, n):
        v[i] = phi * v[i - 1] + noise[i]
    return v


def test_autocorrelation_of_independent_series_is_delta() -> None:
    rng = np.random.default_rng(1)
    rho = autocorrelation(rng.normal(size=4000), max_lag=5)
    assert abs(rho[0] - 1.0) < 1e-12
    assert np.all(np.abs(rho[1:]) < 0.05)


def test_tau_int_of_independent_series_is_one_half() -> None:
    rng = np.random.default_rng(2)
    tau, _ = integrated_autocorrelation_time(rng.normal(size=4000))
    assert abs(tau - 0.5) < 0.15
    assert abs(autocorrelation_inflation(rng.normal(size=4000)) - 1.0) < 0.15


def test_tau_int_matches_ar1_analytic_value() -> None:
    for phi in (0.5, 0.8, 0.9):
        expected = 0.5 + phi / (1.0 - phi)
        tau, window = integrated_autocorrelation_time(ar1(20000, phi, seed=3))
        assert abs(tau - expected) < 0.2 * expected, (phi, tau, expected)
        assert window > 0


def test_jackknife_error_matches_variance_of_repeated_ar1_means() -> None:
    # ground truth: the spread of the mean over independent AR(1) chains. Plain
    # jackknife would report ~sqrt(2 tau_int) = 3x less than this.
    phi, n = 0.8, 2000
    means, errs = [], []
    for seed in range(60):
        v = ar1(n, phi, seed=seed)
        m, e = jackknife_mean(v)
        means.append(m)
        errs.append(e)
    empirical = float(np.std(means, ddof=1))
    reported = float(np.mean(errs))
    assert abs(reported - empirical) < 0.25 * empirical, (reported, empirical)


def test_constant_and_short_series_do_not_break() -> None:
    assert integrated_autocorrelation_time([1.0, 1.0, 1.0, 1.0]) == (0.5, 0)
    assert integrated_autocorrelation_time([2.0]) == (0.5, 0)
    mean, err = jackknife_mean([3.0, 3.0, 3.0, 3.0])
    assert mean == 3.0 and err == 0.0
    assert np.isnan(jackknife_mean([7.0])[1])


def test_jackknife_columns_uses_slowest_column() -> None:
    fast = ar1(3000, 0.0, seed=5)
    slow = ar1(3000, 0.9, seed=6)
    rows = np.stack([fast, slow], axis=1)
    _, err_both = jackknife_columns(rows, lambda c: float(c[0]))
    _, err_fast = jackknife_columns(fast[:, None], lambda c: float(c[0]))
    assert err_both > 2.0 * err_fast
