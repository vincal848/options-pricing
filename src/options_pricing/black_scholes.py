"""Analytic Black-Scholes-Merton prices and Greeks for European options on a
dividend-paying asset (Hull, *Options, Futures and Other Derivatives*, ch. 15 and 19;
Stefanica, *A Primer for the Mathematics of Financial Engineering*, ch. 3).

Conventions used throughout this package:

* ``tau`` is time to expiry in **years**. The original coursework code passed a
  period index ``t`` and a period count ``N`` and used ``N - t`` as both a year
  count and a step count; the two are separated here.
* ``sigma`` and ``r`` are continuously compounded annual rates.
* ``q`` is a continuous dividend yield.
* Vega is reported per **1.00** change in volatility and rho per 1.00 change in
  the rate. Divide by 100 for the per-percentage-point figures brokers quote.
* Theta is per year. Divide by 365 for the per-calendar-day figure.

All functions accept scalars or NumPy arrays and broadcast.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

Kind = Literal["call", "put"]

__all__ = ["price", "greeks", "d1_d2", "implied_vol", "intrinsic"]


def _check_kind(kind: str) -> str:
    k = kind.lower()
    if k not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return k


def intrinsic(S, K, kind: Kind):
    """Payoff if exercised now. Used for the tau == 0 boundary."""
    k = _check_kind(kind)
    S, K = np.asarray(S, float), np.asarray(K, float)
    return np.maximum(S - K, 0.0) if k == "call" else np.maximum(K - S, 0.0)


def d1_d2(S, K, tau, r, sigma, q=0.0):
    """The two standardised moneyness terms.

    d1 = [ln(S/K) + (r - q + sigma^2/2) tau] / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)
    """
    S, K, tau = np.asarray(S, float), np.asarray(K, float), np.asarray(tau, float)
    r, sigma, q = np.asarray(r, float), np.asarray(sigma, float), np.asarray(q, float)
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")
    if np.any(tau < 0):
        raise ValueError("tau must be non-negative")
    vol = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / vol
    d2 = d1 - vol
    return d1, d2


def price(S, K, tau, r, sigma, kind: Kind = "call", q=0.0):
    """European option price.

    >>> round(float(price(100, 100, 1.0, 0.05, 0.20, "call")), 6)
    10.450584
    >>> round(float(price(100, 100, 1.0, 0.05, 0.20, "put")), 6)
    5.573526
    """
    k = _check_kind(kind)
    tau = np.asarray(tau, float)
    if np.all(tau == 0):
        return intrinsic(S, K, k)
    d1, d2 = d1_d2(S, K, tau, r, sigma, q)
    S, K = np.asarray(S, float), np.asarray(K, float)
    df_r, df_q = np.exp(-np.asarray(r, float) * tau), np.exp(-np.asarray(q, float) * tau)
    if k == "call":
        out = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        out = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    # At tau == 0 the formula is 0/0; fall back to the payoff.
    return np.where(tau == 0, intrinsic(S, K, k), out)


def greeks(S, K, tau, r, sigma, kind: Kind = "call", q=0.0) -> dict[str, np.ndarray]:
    """The five first- and second-order sensitivities, as a dict.

    Gamma and vega do not depend on ``kind``; delta, theta and rho do. The
    coursework version computed vega with an extra 1/sqrt(2*pi) on the call leg
    but not the put leg, and gave put rho a positive sign; both are corrected.
    """
    k = _check_kind(kind)
    d1, d2 = d1_d2(S, K, tau, r, sigma, q)
    S, K, tau = np.asarray(S, float), np.asarray(K, float), np.asarray(tau, float)
    r, sigma, q = np.asarray(r, float), np.asarray(sigma, float), np.asarray(q, float)
    df_r, df_q = np.exp(-r * tau), np.exp(-q * tau)
    pdf_d1, sqrt_tau = norm.pdf(d1), np.sqrt(tau)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_tau)
    vega = S * df_q * sqrt_tau * pdf_d1
    common_theta = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_tau)

    if k == "call":
        delta = df_q * norm.cdf(d1)
        theta = common_theta + q * S * df_q * norm.cdf(d1) - r * K * df_r * norm.cdf(d2)
        rho = K * tau * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = common_theta - q * S * df_q * norm.cdf(-d1) + r * K * df_r * norm.cdf(-d2)
        rho = -K * tau * df_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target, S, K, tau, r, kind: Kind = "call", q=0.0,
                lo=1e-6, hi=5.0, tol=1e-10, max_iter=200) -> float:
    """Volatility reproducing ``target``, by bisection on a monotone function.

    Bisection rather than Newton: vega collapses for deep out-of-the-money
    options, where a Newton step can leave the bracket entirely.
    """
    k = _check_kind(kind)
    lo_p = float(price(S, K, tau, r, lo, k, q))
    hi_p = float(price(S, K, tau, r, hi, k, q))
    if not (lo_p - tol <= target <= hi_p + tol):
        raise ValueError(
            f"target {target} outside attainable range [{lo_p:.6g}, {hi_p:.6g}] "
            f"for sigma in [{lo}, {hi}]"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(price(S, K, tau, r, mid, k, q)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
