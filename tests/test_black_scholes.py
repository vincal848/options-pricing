"""Analytic formulas against published values and against identities that must
hold exactly regardless of the numbers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import pytest

import black_scholes as bs

# Hull's worked example, ch. 15: S=42, K=40, r=10%, sigma=20%, tau=0.5 years.
# Hull reports call 4.76, put 0.81.
HULL = dict(S=42.0, K=40.0, tau=0.5, r=0.10, sigma=0.20)


def test_hull_worked_example():
    assert float(bs.price(kind="call", **HULL)) == pytest.approx(4.76, abs=5e-3)
    assert float(bs.price(kind="put", **HULL)) == pytest.approx(0.81, abs=5e-3)


def test_atm_reference_values():
    """S=K=100, tau=1, r=5%, sigma=20% -- the case used throughout the docs."""
    a = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.20)
    assert float(bs.price(kind="call", **a)) == pytest.approx(10.450583572, abs=1e-8)
    assert float(bs.price(kind="put", **a)) == pytest.approx(5.573526022, abs=1e-8)


@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("S,K,tau,r,sigma", [
    (100, 100, 1.0, 0.05, 0.20),
    (42, 40, 0.5, 0.10, 0.20),
    (7, 10, 2.0, 0.01, 0.60),
    (150, 100, 0.25, 0.04, 0.15),
])
def test_put_call_parity(S, K, tau, r, sigma, q):
    """C - P = S e^{-q tau} - K e^{-r tau}, exactly, for any parameters."""
    c = float(bs.price(S, K, tau, r, sigma, "call", q))
    p = float(bs.price(S, K, tau, r, sigma, "put", q))
    assert c - p == pytest.approx(S * math.exp(-q * tau) - K * math.exp(-r * tau), abs=1e-10)


def test_parity_holds_for_greeks():
    """Differencing parity: delta_c - delta_p = e^{-q tau}, and gamma/vega match."""
    a = dict(S=100.0, K=95.0, tau=0.75, r=0.04, sigma=0.25, q=0.02)
    gc = bs.greeks(kind="call", **a)
    gp = bs.greeks(kind="put", **a)
    assert float(gc["delta"] - gp["delta"]) == pytest.approx(math.exp(-0.02 * 0.75), abs=1e-12)
    assert float(gc["gamma"]) == pytest.approx(float(gp["gamma"]), abs=1e-12)
    assert float(gc["vega"]) == pytest.approx(float(gp["vega"]), abs=1e-12)
    # rho_c - rho_p = K tau e^{-r tau}
    assert float(gc["rho"] - gp["rho"]) == pytest.approx(
        95.0 * 0.75 * math.exp(-0.04 * 0.75), abs=1e-10)


def test_greek_signs():
    """Signs a correct implementation cannot get wrong.

    The coursework version returned a positive put rho; a put loses value when
    rates rise, so it must be negative.
    """
    a = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.2)
    gc, gp = bs.greeks(kind="call", **a), bs.greeks(kind="put", **a)
    assert 0 < float(gc["delta"]) < 1
    assert -1 < float(gp["delta"]) < 0
    assert float(gc["gamma"]) > 0 and float(gp["gamma"]) > 0
    assert float(gc["vega"]) > 0 and float(gp["vega"]) > 0
    assert float(gc["rho"]) > 0, "call rho must be positive"
    assert float(gp["rho"]) < 0, "put rho must be negative"
    assert float(gc["theta"]) < 0, "an ATM call decays"


def test_greeks_match_finite_differences():
    """Every analytic Greek against a central difference of the analytic price."""
    S, K, tau, r, sigma, q = 100.0, 105.0, 1.5, 0.03, 0.28, 0.01
    g = bs.greeks(S, K, tau, r, sigma, "call", q)
    f = lambda **kw: float(bs.price(**{**dict(S=S, K=K, tau=tau, r=r, sigma=sigma,
                                              kind="call", q=q), **kw}))
    h = 1e-5
    assert float(g["delta"]) == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-5)
    assert float(g["gamma"]) == pytest.approx(
        (f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert float(g["vega"]) == pytest.approx(
        (f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h), rel=1e-5)
    assert float(g["rho"]) == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-5)
    # theta = -dV/dtau
    assert float(g["theta"]) == pytest.approx(-(f(tau=tau + h) - f(tau=tau - h)) / (2 * h),
                                              rel=1e-5)


def test_monotonicity():
    """Call value rises in S and sigma; put value falls in S."""
    base = dict(K=100.0, tau=1.0, r=0.05, sigma=0.2)
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    assert np.all(np.diff(bs.price(S=spots, kind="call", **base)) > 0)
    assert np.all(np.diff(bs.price(S=spots, kind="put", **base)) < 0)
    vols = np.array([0.1, 0.2, 0.3, 0.5])
    rising = bs.price(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=vols, kind="call")
    assert np.all(np.diff(rising) > 0)


def test_no_arbitrage_bounds():
    """max(S e^-qt - K e^-rt, 0) <= C <= S e^-qt, for a range of parameters."""
    for S in (60.0, 100.0, 140.0):
        for sigma in (0.1, 0.4):
            c = float(bs.price(S, 100.0, 1.0, 0.05, sigma, "call", 0.02))
            lo = max(S * math.exp(-0.02) - 100.0 * math.exp(-0.05), 0.0)
            assert lo - 1e-12 <= c <= S * math.exp(-0.02) + 1e-12


def test_expiry_is_intrinsic():
    assert float(bs.price(110.0, 100.0, 0.0, 0.05, 0.2, "call")) == pytest.approx(10.0)
    assert float(bs.price(90.0, 100.0, 0.0, 0.05, 0.2, "call")) == pytest.approx(0.0)
    assert float(bs.price(90.0, 100.0, 0.0, 0.05, 0.2, "put")) == pytest.approx(10.0)


def test_vectorised_matches_scalar():
    spots = np.array([80.0, 100.0, 120.0])
    vec = bs.price(spots, 100.0, 1.0, 0.05, 0.2, "call")
    for s, v in zip(spots, vec):
        assert float(bs.price(float(s), 100.0, 1.0, 0.05, 0.2, "call")) == pytest.approx(v)


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.75, 1.5])
def test_implied_vol_round_trip(sigma):
    p = float(bs.price(100.0, 110.0, 1.0, 0.03, sigma, "call"))
    assert bs.implied_vol(p, 100.0, 110.0, 1.0, 0.03, "call") == pytest.approx(sigma, abs=1e-6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="call.*or.*put"):
        bs.price(100, 100, 1, 0.05, 0.2, "straddle")
    with pytest.raises(ValueError, match="sigma"):
        bs.price(100, 100, 1, 0.05, 0.0, "call")
    with pytest.raises(ValueError, match="positive"):
        bs.price(-1, 100, 1, 0.05, 0.2, "call")
    with pytest.raises(ValueError, match="tau"):
        bs.price(100, 100, -1, 0.05, 0.2, "call")
    with pytest.raises(ValueError, match="outside attainable"):
        bs.implied_vol(1e6, 100, 100, 1, 0.05, "call")
