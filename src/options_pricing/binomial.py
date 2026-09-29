"""Cox-Ross-Rubinstein binomial trees for European and American options
(Hull ch. 21; Stefanica ch. 3-4).

The tree is the standard CRR parameterisation over ``n_steps`` of length
``dt = tau / n_steps``::

    u = exp(sigma sqrt(dt)),   d = 1 / u,   p = (exp((r - q) dt) - d) / (u - d)

Two defects in the original coursework implementation are fixed here, both
confirmed against the analytic European price in ``docs/VALIDATION.md``:

1. ``p`` was written ``exp((r - q) * dt - d) / (u - d)``, subtracting ``d``
   *inside* the exponential. The resulting "probability" was 6.69 at 50 steps
   and 13.19 at 200, so the price diverged as steps were added instead of
   converging.
2. The early-exercise comparison indexed the price lattice at ``price_tree[t, i]``
   -- the period argument -- rather than at the backward-induction level ``n``.
   Every node was therefore compared against the intrinsic value of one fixed row.

The backward induction is vectorised over each level, which also removes the
inner Python loop the original used.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from options_pricing.black_scholes import Kind, intrinsic

Exercise = Literal["european", "american"]

__all__ = ["price", "lattice", "crr_params", "greeks"]


def crr_params(tau: float, r: float, sigma: float, n_steps: int, q: float = 0.0):
    """``(dt, u, d, p, discount)`` for a CRR tree.

    Raises if the risk-neutral probability leaves ``(0, 1)``, which happens when
    ``dt`` is too large for the drift -- the condition Hull gives as
    ``sigma > |r - q| sqrt(dt)``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if tau <= 0:
        raise ValueError("tau must be positive; use black_scholes.intrinsic at expiry")
    dt = tau / n_steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (float(np.exp((r - q) * dt)) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.6g} outside (0,1): dt={dt:.6g} is too "
            f"coarse for sigma={sigma:.6g}, r-q={r - q:.6g}. Increase n_steps."
        )
    return dt, u, d, p, float(np.exp(-r * dt))


def lattice(S: float, u: float, d: float, level: int) -> np.ndarray:
    """Asset prices at ``level``, lowest node first: ``S * u^j * d^(level-j)``."""
    j = np.arange(level + 1)
    return S * u**j * d ** (level - j)


def price(S: float, K: float, tau: float, r: float, sigma: float,
          kind: Kind = "call", q: float = 0.0, n_steps: int = 500,
          exercise: Exercise = "european") -> float:
    """Binomial price. ``exercise='american'`` tests early exercise at every node."""
    ex = exercise.lower()
    if ex not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if tau == 0:
        return float(intrinsic(S, K, kind))

    dt, u, d, p, disc = crr_params(tau, r, sigma, n_steps, q)
    values = intrinsic(lattice(S, u, d, n_steps), K, kind).astype(float)

    for level in range(n_steps - 1, -1, -1):
        # values[1:] are the up-children, values[:-1] the down-children of this level.
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if ex == "american":
            values = np.maximum(values, intrinsic(lattice(S, u, d, level), K, kind))

    return float(values[0])


def _induct(S, K, tau, r, sigma, kind, q, n_steps, ex):
    """Backward induction that keeps the last three levels of the tree.

    Returns ``(values_by_level, u, d, dt)`` where ``values_by_level[k]`` holds the
    option values at level ``k`` for ``k`` in 0, 1, 2. Those nine numbers are all
    the Greeks below need.
    """
    dt, u, d, p, disc = crr_params(tau, r, sigma, n_steps, q)
    values = intrinsic(lattice(S, u, d, n_steps), K, kind).astype(float)
    kept: dict[int, np.ndarray] = {}
    for level in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if ex == "american":
            values = np.maximum(values, intrinsic(lattice(S, u, d, level), K, kind))
        if level <= 2:
            kept[level] = values.copy()
    return kept, u, d, dt


def greeks(S: float, K: float, tau: float, r: float, sigma: float,
           kind: Kind = "call", q: float = 0.0, n_steps: int = 500,
           exercise: Exercise = "european") -> dict[str, float]:
    """Greeks for a binomial price (Hull ch. 21.6).

    Delta, gamma and theta are read off the nodes the tree already computed.
    Bumping the spot instead does not work here: rescaling ``S`` moves every
    lattice node, so the price carries a sawtooth in ``S`` that a ``1/h^2``
    divisor amplifies -- a 0.1%% bump returns gamma = 0.265 against a true
    0.0188. The node estimates have no such term.

    Vega and rho have no node equivalent and are still central differences, which
    is fine: they are first-order, so the noise is divided by ``h`` rather than
    ``h^2``.
    """
    ex = exercise.lower()
    if ex not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if n_steps < 3:
        raise ValueError("node-based Greeks need at least 3 steps")

    kept, u, d, dt = _induct(S, K, tau, r, sigma, kind, q, n_steps, ex)
    v0, v1, v2 = kept[0], kept[1], kept[2]

    # Level-1 nodes straddle the spot; level-2 nodes give a delta either side of it.
    s1_up, s1_dn = S * u, S * d
    s2_up, s2_mid, s2_dn = S * u * u, S, S * d * d

    delta = (v1[1] - v1[0]) / (s1_up - s1_dn)
    delta_up = (v2[2] - v2[1]) / (s2_up - s2_mid)
    delta_dn = (v2[1] - v2[0]) / (s2_mid - s2_dn)
    gamma = (delta_up - delta_dn) / (0.5 * (s2_up - s2_dn))
    # v2[1] sits at the same spot as the root, two steps later in calendar time.
    theta = (v2[1] - v0[0]) / (2.0 * dt)

    def f(**kw) -> float:
        args = dict(S=S, K=K, tau=tau, r=r, sigma=sigma, kind=kind, q=q,
                    n_steps=n_steps, exercise=exercise)
        args.update(kw)
        return price(**args)

    hv, hr = 1e-3, 1e-4
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": (f(sigma=sigma + hv) - f(sigma=sigma - hv)) / (2 * hv),
        "rho": (f(r=r + hr) - f(r=r - hr)) / (2 * hr),
        "theta": float(theta),
    }
