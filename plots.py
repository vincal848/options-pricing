"""Figures. Deliberately separate from the pricing code.

The original called plt.show() six times from inside the pricing routine and ended
in `return print(...)`, which returns None. Pricing anything opened six blocking
windows and the price could not be fed into anything else - no implied vol, no
surface, no test. Pricing returns numbers here; plotting is a separate call.
"""

import numpy as np

import binomial
import black_scholes as bs
import crank_nicolson as cn
from black_scholes import GREEK_NAMES


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plotting needs matplotlib: pip install matplotlib")
    return plt


def value_vs_spot(K, tau, r, sigma, kind="call", q=0.0, span=(0.5, 1.5), n=200,
                  exercise="european", ax=None):
    """Option value against spot, with the expiry payoff for reference."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    if exercise == "european":
        values = bs.price(spots, K, tau, r, sigma, kind, q)
    else:
        values = [binomial.price(float(s), K, tau, r, sigma, kind, q, 400, exercise)
                  for s in spots]

    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.plot(spots, values, label="%s %s" % (exercise, kind))
    ax.plot(spots, bs.intrinsic(spots, K, kind), "--", lw=1, color="0.5",
            label="payoff at expiry")
    ax.set(xlabel="spot", ylabel="value",
           title="%s %s, tau=%s" % (exercise.title(), kind, tau))
    ax.legend()
    return ax


def greeks_vs_spot(K, tau, r, sigma, kind="call", q=0.0, span=(0.5, 1.5), n=200):
    """The five Greeks against spot, one panel each. Analytic, so it is cheap."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    g = bs.greeks(spots, K, tau, r, sigma, kind, q)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, name in zip(axes.ravel(), GREEK_NAMES):
        ax.plot(spots, g[name])
        ax.axvline(K, color="0.7", lw=1, ls=":")
        ax.set(title=name, xlabel="spot")
    axes.ravel()[-1].axis("off")
    fig.suptitle("%s Greeks, tau=%s, sigma=%s, r=%s" % (kind.title(), tau, sigma, r))
    return fig


def convergence(S, K, tau, r, sigma, kind="call", q=0.0,
                steps=(10, 25, 50, 100, 250, 500, 1000, 2000)):
    """Absolute error of both numerical methods against the analytic price.

    Log-log, so the slope of each line is its order of convergence.
    """
    plt = _plt()
    exact = float(bs.price(S, K, tau, r, sigma, kind, q))
    steps = list(steps)

    bn_err = [abs(binomial.price(S, K, tau, r, sigma, kind, q, n) - exact) for n in steps]
    cn_err = [abs(cn.price(S, K, tau, r, sigma, kind, q, n, n) - exact) for n in steps]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(steps, bn_err, "o-", label="binomial (steps)")
    ax.loglog(steps, cn_err, "s-", label="Crank-Nicolson (nodes = steps)")
    ax.set(xlabel="discretisation", ylabel="absolute error vs analytic",
           title="Convergence, European %s" % kind)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig


def early_exercise_premium(K, tau, r, sigma, q=0.0, span=(0.6, 1.4), n=60):
    """American minus European put value: what the right to exercise early is worth."""
    plt = _plt()
    spots = np.linspace(span[0] * K, span[1] * K, n)
    euro = bs.price(spots, K, tau, r, sigma, "put", q)
    amer = [binomial.price(float(s), K, tau, r, sigma, "put", q, 600, "american")
            for s in spots]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(spots, np.asarray(amer) - np.asarray(euro))
    ax.axvline(K, color="0.7", lw=1, ls=":")
    ax.set(xlabel="spot", ylabel="American - European",
           title="Early-exercise premium, put, tau=%s, r=%s" % (tau, r))
    return fig
