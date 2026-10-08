"""Closed-form Black-Scholes-Merton prices and Greeks for European options.

Conventions used everywhere in this project, because the original conflated them:

  tau     time to expiry in YEARS (the original used `N - t` as both a year count
          and a step count, so changing the discretisation silently changed the
          contract being priced)
  sigma   annual volatility, continuously compounded
  r       annual risk-free rate, continuously compounded
  q       continuous dividend yield

Vega and rho are per 1.00 change, not per percentage point - divide by 100 for the
figures a broker quotes. Theta is per year; divide by 365 for per-day.

Functions take scalars or numpy arrays and broadcast, so a whole surface prices in
one call.

References: Hull, Options Futures and Other Derivatives, ch. 15 and 19;
Stefanica, A Primer for the Mathematics of Financial Engineering, ch. 3.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


def check_kind(kind: str) -> str:
    """Normalize and validate the option type."""
    k = str(kind).lower()
    if k not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put', got %r" % (kind,))
    return k


def intrinsic(S: ArrayLike, K: ArrayLike, kind: str) -> np.ndarray:
    """Payoff if exercised right now. Also the tau == 0 boundary."""
    k = check_kind(kind)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    return np.maximum(S - K, 0.0) if k == "call" else np.maximum(K - S, 0.0)


def d1_d2(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike,
          q: ArrayLike = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The two standardized moneyness terms.

    d1 = [ln(S/K) + (r - q + sigma^2/2) tau] / (sigma sqrt(tau))
    d2 = d1 - sigma sqrt(tau)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")
    if np.any(tau < 0):
        raise ValueError("tau must be non-negative")

    vol = sigma * np.sqrt(tau)
    # tau == 0 divides by zero here; price() handles that case separately and
    # returns the payoff, so the warning is suppressed rather than guarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / vol
    return d1, d1 - vol


def price(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike,
          kind: str = "call", q: ArrayLike = 0.0) -> np.ndarray:
    """European option price.

    >>> round(float(price(100, 100, 1.0, 0.05, 0.20, "call")), 6)
    10.450584
    >>> round(float(price(100, 100, 1.0, 0.05, 0.20, "put")), 6)
    5.573526
    """
    k = check_kind(kind)
    tau = np.asarray(tau, dtype=float)
    if np.all(tau == 0):
        return intrinsic(S, K, k)

    d1, d2 = d1_d2(S, K, tau, r, sigma, q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    df_r = np.exp(-np.asarray(r, dtype=float) * tau)
    df_q = np.exp(-np.asarray(q, dtype=float) * tau)

    if k == "call":
        out = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        out = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)

    return np.where(tau == 0, intrinsic(S, K, k), out)


def greeks(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike,
           kind: str = "call", q: ArrayLike = 0.0) -> dict[str, np.ndarray]:
    """The five sensitivities, as a dict.

    Gamma and vega do not depend on the option type; delta, theta and rho do. The
    original computed call vega with an extra 1/sqrt(2*pi) that put vega did not
    have, so the two branches disagreed with each other, and it returned put rho
    positive when a put must lose value as rates rise.
    """
    k = check_kind(kind)
    d1, d2 = d1_d2(S, K, tau, r, sigma, q)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    pdf_d1 = norm.pdf(d1)
    sqrt_tau = np.sqrt(tau)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_tau)
    vega = S * df_q * sqrt_tau * pdf_d1
    # The decay term is common to both types; the carry terms below are not.
    decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_tau)

    if k == "call":
        delta = df_q * norm.cdf(d1)
        theta = decay + q * S * df_q * norm.cdf(d1) - r * K * df_r * norm.cdf(d2)
        rho = K * tau * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = decay - q * S * df_q * norm.cdf(-d1) + r * K * df_r * norm.cdf(-d2)
        rho = -K * tau * df_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target: ArrayLike, S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike,
                kind: str = "call", q: ArrayLike = 0.0, lo: float = 1e-6, hi: float = 5.0,
                tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Volatility that reproduces an observed price.

    Bisection rather than Newton. Newton converges faster when it works, but vega
    collapses toward zero for deep out-of-the-money options and a Newton step can
    then jump clean out of the bracket. Price is monotone in sigma, so bisection
    cannot fail.
    """
    k = check_kind(kind)
    lo_price = float(price(S, K, tau, r, lo, k, q))
    hi_price = float(price(S, K, tau, r, hi, k, q))

    if not (lo_price - tol <= target <= hi_price + tol):
        raise ValueError(
            "target %g outside attainable range [%.6g, %.6g] for sigma in [%g, %g]"
            % (target, lo_price, hi_price, lo, hi))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(price(S, K, tau, r, mid, k, q)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
