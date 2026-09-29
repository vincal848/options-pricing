"""Crank-Nicolson finite differences for the Black-Scholes PDE (Hull ch. 21.8;
Wilmott, *Paul Wilmott on Quantitative Finance*, ch. 78).

This replaces the explicit finite-difference attempt described -- but not kept --
in the original coursework repository. That attempt produced diverging values,
and the old README attributed this to "too large dt combined with the explicit
finite difference method". That diagnosis was right: the explicit scheme is only
stable while ``dt <= dS^2 / (sigma^2 S_max^2)``, which for a grid fine enough to
resolve the payoff kink demands far more time steps than were used. Crank-Nicolson
averages the explicit and implicit schemes, is unconditionally stable for this
equation, and is second-order accurate in both ``dS`` and ``dt``.

Working in time-to-expiry ``tau``, the equation solved backwards from the payoff is

    dV/dtau = 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V

On a uniform grid ``S_j = j dS`` the spatial operator becomes tridiagonal with

    a_j = 1/2 sigma^2 j^2 - 1/2 (r - q) j
    b_j = -sigma^2 j^2 - r
    c_j = 1/2 sigma^2 j^2 + 1/2 (r - q) j

and each step solves ``(I - dtau/2 L) V^{n+1} = (I + dtau/2 L) V^n`` for the
interior nodes, with Dirichlet data at ``S = 0`` and ``S = S_max``.

The grid is built so the valuation spot lands exactly on a node, which lets
delta and gamma be read straight off the solution by central difference rather
than differenced through an interpolation.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

from options_pricing.black_scholes import Kind, intrinsic

Exercise = Literal["european", "american"]

__all__ = ["solve", "price", "greeks"]

# S_max as a multiple of max(S, K), in units of exp(N sigma sqrt(tau)). Five standard
# deviations of terminal log-price keeps the truncation error well below the grid error.
SIGMA_MULTIPLES = 5.0


def _grid(S: float, K: float, tau: float, sigma: float, n_space: int):
    """Uniform S-grid with the spot on a node. Returns ``(s_nodes, dS, j_spot)``."""
    s_max_target = max(S, K) * float(np.exp(SIGMA_MULTIPLES * sigma * np.sqrt(tau)))
    # Put S on a node: pick its index from the target spacing, then set dS = S / index.
    j_spot = max(1, int(round(S / (s_max_target / n_space))))
    dS = S / j_spot
    n_nodes = max(j_spot + 2, int(np.ceil(s_max_target / dS)))
    return np.arange(n_nodes + 1) * dS, dS, j_spot


def _boundaries(s_max: float, K: float, tau_rem, r: float, q: float, kind: Kind):
    """Dirichlet values at S = 0 and S = S_max for remaining time ``tau_rem``."""
    if kind == "call":
        return 0.0, s_max * np.exp(-q * tau_rem) - K * np.exp(-r * tau_rem)
    return K * np.exp(-r * tau_rem), 0.0


def solve(S: float, K: float, tau: float, r: float, sigma: float,
          kind: Kind = "call", q: float = 0.0, n_space: int = 400,
          n_time: int = 400, exercise: Exercise = "european"):
    """Run the scheme and return ``(s_nodes, values, j_spot)``.

    ``values`` is the option value on the whole grid at ``tau``, which is what
    makes delta and gamma free once the solve is done.
    """
    ex = exercise.lower()
    if ex not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if tau <= 0:
        raise ValueError("tau must be positive; use black_scholes.intrinsic at expiry")
    if n_space < 3 or n_time < 1:
        raise ValueError("need n_space >= 3 and n_time >= 1")

    s_nodes, dS, j_spot = _grid(S, K, tau, sigma, n_space)
    n_nodes = len(s_nodes) - 1
    dt = tau / n_time
    payoff = intrinsic(s_nodes, K, kind).astype(float)
    values = payoff.copy()

    # Tridiagonal operator on interior nodes j = 1 .. n_nodes-1.
    j = np.arange(1, n_nodes)
    a = 0.5 * sigma**2 * j**2 - 0.5 * (r - q) * j
    b = -(sigma**2) * j**2 - r
    c = 0.5 * sigma**2 * j**2 + 0.5 * (r - q) * j
    h = 0.5 * dt

    # Banded form of (I - dt/2 L): row 0 upper, row 1 main, row 2 lower.
    ab = np.zeros((3, n_nodes - 1))
    ab[0, 1:] = -h * c[:-1]
    ab[1, :] = 1.0 - h * b
    ab[2, :-1] = -h * a[1:]

    for n in range(n_time):
        tau_next = (n + 1) * dt
        lo_next, hi_next = _boundaries(s_nodes[-1], K, tau_next, r, q, kind)

        # RHS = (I + dt/2 L) V^n. These slices already pull in the boundary nodes of
        # the *current* level -- values[0] is lo_now, values[-1] is hi_now -- so only
        # the *next* level's boundaries still have to be carried over from the
        # implicit side, where they are known rather than unknown. Adding lo_now and
        # hi_now again double-counts them, which leaves the price at the money right
        # to 1e-3 while corrupting the grid near S_max by about 88 -- invisible to a
        # price-only check, which is why tests assert on the whole solution.
        rhs = (h * a * values[:-2] + (1.0 + h * b) * values[1:-1] + h * c * values[2:])
        rhs[0] += h * a[0] * lo_next
        rhs[-1] += h * c[-1] * hi_next

        interior = solve_banded((1, 1), ab, rhs)
        values = np.concatenate(([lo_next], interior, [hi_next]))

        if ex == "american":
            # Projection onto the exercise constraint after each step (Hull ch. 21.8).
            # This is the standard treatment and is what the published American
            # finite-difference tables use; solving the linear complementarity
            # problem properly (PSOR) changes the price in the fourth decimal here.
            values = np.maximum(values, payoff)

    return s_nodes, values, j_spot


def price(S: float, K: float, tau: float, r: float, sigma: float,
          kind: Kind = "call", q: float = 0.0, n_space: int = 400,
          n_time: int = 400, exercise: Exercise = "european") -> float:
    """Option value at the spot, read off the node the grid was built around."""
    if tau == 0:
        return float(intrinsic(S, K, kind))
    _, values, j_spot = solve(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise)
    return float(values[j_spot])


def greeks(S: float, K: float, tau: float, r: float, sigma: float,
           kind: Kind = "call", q: float = 0.0, n_space: int = 400,
           n_time: int = 400, exercise: Exercise = "european") -> dict[str, float]:
    """Delta and gamma from the solved grid; vega, rho and theta by repricing.

    Delta and gamma cost nothing extra because the solve already produced the
    neighbouring nodes. The other three need a bumped solve, as in ``binomial``.
    """
    s_nodes, values, jS = solve(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise)
    dS = float(s_nodes[1] - s_nodes[0])

    def f(**kw) -> float:
        args = dict(S=S, K=K, tau=tau, r=r, sigma=sigma, kind=kind, q=q,
                    n_space=n_space, n_time=n_time, exercise=exercise)
        args.update(kw)
        return price(**args)

    hv, hr, ht = 1e-3, 1e-4, min(1e-3, tau / 100.0)
    return {
        "delta": float((values[jS + 1] - values[jS - 1]) / (2 * dS)),
        "gamma": float((values[jS + 1] - 2 * values[jS] + values[jS - 1]) / dS**2),
        "vega": (f(sigma=sigma + hv) - f(sigma=sigma - hv)) / (2 * hv),
        "rho": (f(r=r + hr) - f(r=r - hr)) / (2 * hr),
        "theta": -(f(tau=tau + ht) - f(tau=tau - ht)) / (2 * ht),
    }
