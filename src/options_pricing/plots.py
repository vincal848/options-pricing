"""Plot helpers. Kept out of the pricing modules on purpose.

The original coursework code called ``plt.show()`` from inside the pricing
routine and returned ``print(...)``, i.e. ``None``, so a price could not be used
in any further calculation and pricing anything opened six blocking windows.
Pricing here returns numbers; plotting is a separate, optional call.
"""
from __future__ import annotations

import numpy as np

from options_pricing import GREEK_NAMES, binomial, black_scholes, crank_nicolson
from options_pricing.black_scholes import Kind

__all__ = ["greeks_vs_spot", "value_vs_spot", "convergence", "early_exercise_premium"]

_METHODS = {"analytic": black_scholes, "binomial": binomial, "crank-nicolson": crank_nicolson}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'options-pricing[plots]'"
        ) from exc
    return plt


def value_vs_spot(K, tau, r, sigma, kind: Kind = "call", q=0.0,
                  span=(0.5, 1.5), n=200, exercise="european", ax=None):
    """Option value against spot, with the payoff for reference."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    if exercise == "european":
        vals = black_scholes.price(spots, K, tau, r, sigma, kind, q)
    else:
        vals = [binomial.price(float(s), K, tau, r, sigma, kind, q, 400, exercise)
                for s in spots]
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.plot(spots, vals, label=f"{exercise} {kind}")
    ax.plot(spots, black_scholes.intrinsic(spots, K, kind), "--", lw=1,
            color="0.5", label="payoff at expiry")
    ax.set(xlabel="spot", ylabel="value", title=f"{exercise.title()} {kind}, tau={tau}")
    ax.legend()
    return ax


def greeks_vs_spot(K, tau, r, sigma, kind: Kind = "call", q=0.0,
                   span=(0.5, 1.5), n=200):
    """The five Greeks against spot, one panel each. Analytic, so it is cheap."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    g = black_scholes.greeks(spots, K, tau, r, sigma, kind, q)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, name in zip(axes.ravel(), GREEK_NAMES):
        ax.plot(spots, g[name])
        ax.axvline(K, color="0.7", lw=1, ls=":")
        ax.set(title=name, xlabel="spot")
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"{kind.title()} Greeks, tau={tau}, sigma={sigma}, r={r}")
    return fig


def convergence(S, K, tau, r, sigma, kind: Kind = "call", q=0.0,
                steps=(10, 25, 50, 100, 250, 500, 1000, 2000)):
    """Absolute error of both numerical methods against the analytic price.

    Plotted on log-log axes, where the slope is the order of convergence.
    """
    plt = _plt()
    exact = float(black_scholes.price(S, K, tau, r, sigma, kind, q))
    steps = list(steps)
    bn_err = [abs(binomial.price(S, K, tau, r, sigma, kind, q, n) - exact) for n in steps]
    cn_err = [abs(crank_nicolson.price(S, K, tau, r, sigma, kind, q, n, n) - exact)
              for n in steps]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(steps, bn_err, "o-", label="binomial (steps)")
    ax.loglog(steps, cn_err, "s-", label="Crank-Nicolson (nodes = steps)")
    ax.set(xlabel="discretisation", ylabel="absolute error vs analytic",
           title=f"Convergence, European {kind}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig


def early_exercise_premium(K, tau, r, sigma, q=0.0, span=(0.6, 1.4), n=60):
    """American minus European put value: what the right to exercise early is worth."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    euro = black_scholes.price(spots, K, tau, r, sigma, "put", q)
    amer = [binomial.price(float(s), K, tau, r, sigma, "put", q, 600, "american")
            for s in spots]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(spots, np.asarray(amer) - np.asarray(euro))
    ax.axvline(K, color="0.7", lw=1, ls=":")
    ax.set(xlabel="spot", ylabel="American - European",
           title=f"Early-exercise premium, put, tau={tau}, r={r}")
    return fig
