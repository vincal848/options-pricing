"""Binomial and Crank-Nicolson against the analytic price, against each other,
and against the regressions that the original implementation failed.
"""
from __future__ import annotations

import numpy as np
import pytest

from options_pricing import binomial as bn
from options_pricing import black_scholes as bs
from options_pricing import crank_nicolson as cn

ATM = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.20, q=0.0)

# High-accuracy American put for the ATM case, from a 20000-step CRR tree.
AMERICAN_PUT_ATM = 6.08999


# --- regressions for the two bugs in the original coursework code ---------------

def test_risk_neutral_probability_is_a_probability():
    """The original wrote exp((r-q)dt - d)/(u-d), subtracting d inside the exp.

    That gave p = 6.69 at 50 steps and 13.19 at 200. p must lie in (0,1) and
    approach 0.5 as dt shrinks.
    """
    prev = None
    for n in (10, 50, 200, 1000, 5000):
        _, _, _, p, _ = bn.crr_params(1.0, 0.05, 0.20, n)
        assert 0.0 < p < 1.0, f"p={p} not a probability at {n} steps"
        if prev is not None:
            assert abs(p - 0.5) <= abs(prev - 0.5) + 1e-12, "p should move toward 0.5"
        prev = p
    assert bn.crr_params(1.0, 0.05, 0.20, 5000)[3] == pytest.approx(0.5, abs=2e-3)


def test_price_converges_rather_than_diverges():
    """Error must shrink as steps are added.

    The original diverged: put 2.79 at 50 steps, 1.40 at 200, against a true
    5.57. This is the test that would have caught it.
    """
    exact = float(bs.price(kind="put", **ATM))
    steps = [100, 200, 400, 800, 1600, 3200]
    errs = [abs(bn.price(kind="put", n_steps=n, **ATM) - exact) for n in steps]

    # CRR is first order, so n * err is constant and halving the step size halves
    # the error. Asserting the rate is stricter than asserting any one tolerance:
    # the original code's error *grew* with n, which fails this outright.
    for n, e in zip(steps, errs):
        assert n * e == pytest.approx(2.0, rel=0.05), f"not O(1/n) at n={n}: n*err={n * e}"
    for a, b in zip(errs, errs[1:]):
        assert b < a, f"error must fall as steps are added: {errs}"
    assert errs[-1] < 1e-3


def test_early_exercise_uses_the_current_level():
    """The original compared against price_tree[t, i] rather than the induction level.

    With t=1 that pins the comparison to one row of the lattice. The observable
    consequence: an American put priced below its European counterpart, which
    violates the dominance bound, since the American holder can always just wait.
    """
    euro = float(bs.price(kind="put", **ATM))
    amer = bn.price(kind="put", n_steps=800, exercise="american", **ATM)
    assert amer > euro, "American put must dominate European"
    assert amer == pytest.approx(AMERICAN_PUT_ATM, abs=2e-3)


# --- agreement between the three methods ---------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,tau,r,sigma,q", [
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 110.0, 0.50, 0.03, 0.35, 0.02),
    (90.0, 100.0, 2.00, 0.06, 0.15, 0.00),
    (120.0, 100.0, 0.25, 0.01, 0.45, 0.04),
])
def test_binomial_matches_analytic(kind, S, K, tau, r, sigma, q):
    exact = float(bs.price(S, K, tau, r, sigma, kind, q))
    got = bn.price(S, K, tau, r, sigma, kind, q, n_steps=2000)
    assert got == pytest.approx(exact, abs=5e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,tau,r,sigma,q", [
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 110.0, 0.50, 0.03, 0.35, 0.02),
    (90.0, 100.0, 2.00, 0.06, 0.15, 0.00),
])
def test_crank_nicolson_matches_analytic(kind, S, K, tau, r, sigma, q):
    exact = float(bs.price(S, K, tau, r, sigma, kind, q))
    got = cn.price(S, K, tau, r, sigma, kind, q, n_space=600, n_time=600)
    assert got == pytest.approx(exact, abs=5e-3)


def test_american_put_agrees_across_methods():
    b = bn.price(kind="put", n_steps=2000, exercise="american", **ATM)
    c = cn.price(kind="put", n_space=800, n_time=800, exercise="american", **ATM)
    assert b == pytest.approx(c, abs=5e-3)


def test_american_call_no_dividend_equals_european():
    """With q=0 early exercise of a call is never optimal (Hull ch. 11.5)."""
    euro = bn.price(kind="call", n_steps=500, exercise="european", **ATM)
    amer = bn.price(kind="call", n_steps=500, exercise="american", **ATM)
    assert amer == pytest.approx(euro, abs=1e-10)


def test_american_call_with_dividend_exceeds_european():
    a = dict(S=100.0, K=100.0, tau=1.0, r=0.03, sigma=0.25, q=0.08)
    euro = bn.price(kind="call", n_steps=800, exercise="european", **a)
    amer = bn.price(kind="call", n_steps=800, exercise="american", **a)
    assert amer > euro


# --- Greeks --------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_greeks_match_analytic(kind):
    want = bs.greeks(kind=kind, **ATM)
    got = bn.greeks(kind=kind, n_steps=1500, **ATM)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert got[name] == pytest.approx(float(want[name]), rel=5e-3), name


@pytest.mark.parametrize("kind", ["call", "put"])
def test_crank_nicolson_greeks_match_analytic(kind):
    want = bs.greeks(kind=kind, **ATM)
    got = cn.greeks(kind=kind, n_space=600, n_time=600, **ATM)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert got[name] == pytest.approx(float(want[name]), rel=5e-3), name


def test_gamma_is_not_swamped_by_lattice_noise():
    """Guards the node-based Greeks in binomial.greeks.

    A naive 0.1%-of-spot bump returns gamma = 0.265 against a true 0.0188,
    because rescaling S shifts every lattice node and the 1/h^2 divisor
    amplifies the resulting sawtooth.
    """
    truth = float(bs.greeks(kind="call", **ATM)["gamma"])
    assert bn.greeks(kind="call", n_steps=1000, **ATM)["gamma"] == pytest.approx(truth, rel=5e-3)


# --- structural properties -----------------------------------------------------

def test_crank_nicolson_grid_puts_spot_on_a_node():
    s_nodes, values, j = cn.solve(kind="call", **ATM)
    assert s_nodes[j] == pytest.approx(ATM["S"], rel=1e-12)
    assert len(s_nodes) == len(values)


def test_crank_nicolson_solution_is_monotone_and_convex_in_spot():
    s_nodes, values, _ = cn.solve(kind="call", n_space=400, n_time=400, **ATM)
    interior = values[1:-1]
    assert np.all(np.diff(interior) >= -1e-9), "call value must rise with spot"
    assert np.all(np.diff(interior, 2) >= -1e-6), "call value must be convex in spot"


def test_american_value_dominates_payoff_everywhere():
    s_nodes, values, _ = cn.solve(kind="put", exercise="american", **ATM)
    assert np.all(values >= bs.intrinsic(s_nodes, ATM["K"], "put") - 1e-9)


def test_coarse_steps_are_rejected_not_silently_wrong():
    """A dt too coarse for the drift pushes p out of (0,1); that must raise."""
    with pytest.raises(ValueError, match="outside"):
        bn.crr_params(tau=10.0, r=0.9, sigma=0.02, n_steps=1)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="exercise"):
        bn.price(100, 100, 1, 0.05, 0.2, "call", 0.0, 100, "bermudan")
    with pytest.raises(ValueError, match="n_steps"):
        bn.crr_params(1.0, 0.05, 0.2, 0)
    with pytest.raises(ValueError, match="tau"):
        cn.solve(100, 100, 0.0, 0.05, 0.2, "call")
