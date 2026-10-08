"""Crank-Nicolson finite differences for the Black-Scholes PDE.

This is the method I described in my original write-up but could not get to work.
I reported "blown up values" and removed it. The diagnosis I gave at the time was
right: I had used the EXPLICIT scheme with large time steps, and the explicit scheme
is only stable while dt <= dS^2 / (sigma^2 S_max^2), which a grid fine enough to
resolve the payoff kink makes impossible. Crank-Nicolson averages the explicit and
implicit schemes, is unconditionally stable for this equation, and is second order
in both dS and dt. Measured order here is 2.00.

Working in time-to-expiry tau and marching backward from the payoff:

    dV/dtau = 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V

On a uniform grid S_j = j dS the spatial operator is tridiagonal with

    a_j = 1/2 sigma^2 j^2 - 1/2 (r - q) j
    b_j = -sigma^2 j^2 - r
    c_j = 1/2 sigma^2 j^2 + 1/2 (r - q) j

and each step solves (I - dtau/2 L) V^{n+1} = (I + dtau/2 L) V^n on the interior
nodes, with Dirichlet data at S = 0 and S = S_max.

Reference: Hull ch. 21.8; Wilmott, Paul Wilmott on Quantitative Finance, ch. 78.
"""

import numpy as np
from scipy.linalg import solve_banded

from binomial import check_exercise
from black_scholes import intrinsic

# S_max as max(S, K) * exp(SIGMA_MULTIPLES * sigma * sqrt(tau)). Five standard
# deviations of terminal log-price puts the truncation error well below grid error.
SIGMA_MULTIPLES = 5.0


def build_grid(S: float, K: float, tau: float, sigma: float,
               n_space: int) -> tuple[np.ndarray, float, int]:
    """Uniform grid in S with the valuation spot landing exactly on a node.

    Returns (nodes, dS, j_spot). Placing S on a node means delta and gamma can be
    read straight off the solved solution by central difference, instead of being
    differenced through an interpolation that would add its own error.
    """
    s_max_target = max(S, K) * float(np.exp(SIGMA_MULTIPLES * sigma * np.sqrt(tau)))
    # Pick the spot's index from the target spacing, then set dS = S / index so the
    # spot is exactly on node j_spot.
    j_spot = max(1, int(round(S / (s_max_target / n_space))))
    dS = S / j_spot
    n_nodes = max(j_spot + 2, int(np.ceil(s_max_target / dS)))
    return np.arange(n_nodes + 1) * dS, dS, j_spot


def boundaries(s_max: float, K: float, tau_remaining: float, r: float, q: float,
               kind: str) -> tuple[float, float]:
    """Dirichlet values at S = 0 and S = S_max for the remaining time."""
    if kind == "call":
        return 0.0, s_max * np.exp(-q * tau_remaining) - K * np.exp(-r * tau_remaining)
    return K * np.exp(-r * tau_remaining), 0.0


def solve(S: float, K: float, tau: float, r: float, sigma: float, kind: str = "call",
          q: float = 0.0, n_space: int = 400, n_time: int = 400,
          exercise: str = "european") -> tuple[np.ndarray, np.ndarray, int]:
    """Run the scheme; return (nodes, values, j_spot).

    values is the option value across the WHOLE grid at tau, not just at the spot.
    Returning all of it is what makes the grid-wide test in tests/ possible, and
    that test is what caught the boundary bug noted below.
    """
    e = check_exercise(exercise)
    if tau <= 0:
        raise ValueError("tau must be positive; use black_scholes.intrinsic at expiry")
    if n_space < 3 or n_time < 1:
        raise ValueError("need n_space >= 3 and n_time >= 1")

    nodes, dS, j_spot = build_grid(S, K, tau, sigma, n_space)
    n_nodes = len(nodes) - 1
    dt = tau / n_time
    payoff = intrinsic(nodes, K, kind).astype(float)
    values = payoff.copy()

    j = np.arange(1, n_nodes)
    a = 0.5 * sigma ** 2 * j ** 2 - 0.5 * (r - q) * j
    b = -(sigma ** 2) * j ** 2 - r
    c = 0.5 * sigma ** 2 * j ** 2 + 0.5 * (r - q) * j
    half = 0.5 * dt

    # Banded form of (I - dt/2 L): row 0 the upper diagonal, row 1 the main
    # diagonal, row 2 the lower. Built once; only the right-hand side changes.
    ab = np.zeros((3, n_nodes - 1))
    ab[0, 1:] = -half * c[:-1]
    ab[1, :] = 1.0 - half * b
    ab[2, :-1] = -half * a[1:]

    for n in range(n_time):
        tau_next = (n + 1) * dt
        lo_next, hi_next = boundaries(nodes[-1], K, tau_next, r, q, kind)

        # RHS = (I + dt/2 L) V^n. These slices already include the boundary nodes of
        # the CURRENT level - values[0] and values[-1] - so only the NEXT level's
        # boundaries still have to be carried over from the implicit side, where
        # they are known rather than unknown.
        #
        # Adding the current level's boundaries here as well double-counts them.
        # That bug left the at-the-money price accurate to 1e-3 while corrupting the
        # deep-in-the-money end of the grid by about 88, which is why the tests
        # assert on the whole solution and not just on the number of interest.
        rhs = half * a * values[:-2] + (1.0 + half * b) * values[1:-1] + half * c * values[2:]
        rhs[0] += half * a[0] * lo_next
        rhs[-1] += half * c[-1] * hi_next

        interior = solve_banded((1, 1), ab, rhs)
        values = np.concatenate(([lo_next], interior, [hi_next]))

        if e == "american":
            # Projection onto the exercise constraint after each step (Hull ch.
            # 21.8). Solving the linear complementarity problem properly with PSOR
            # changes the price in the fourth decimal here, which is what the
            # cross-method agreement of 6.6e-04 reflects.
            values = np.maximum(values, payoff)

    return nodes, values, j_spot


def price(S: float, K: float, tau: float, r: float, sigma: float, kind: str = "call",
          q: float = 0.0, n_space: int = 400, n_time: int = 400,
          exercise: str = "european") -> float:
    """Option value at the spot, read off the node the grid was built around."""
    if tau == 0:
        return float(intrinsic(S, K, kind))
    _, values, j_spot = solve(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise)
    return float(values[j_spot])


def greeks(S: float, K: float, tau: float, r: float, sigma: float, kind: str = "call",
           q: float = 0.0, n_space: int = 400, n_time: int = 400,
           exercise: str = "european") -> dict[str, float]:
    """Delta and gamma straight off the grid; vega, rho and theta by repricing.

    Delta and gamma are free because the solve already produced the neighbouring
    nodes. The other three need a bumped solve, as in binomial.greeks.
    """
    nodes, values, j_spot = solve(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise)
    dS = float(nodes[1] - nodes[0])

    def reprice(**kw):
        args = dict(S=S, K=K, tau=tau, r=r, sigma=sigma, kind=kind, q=q,
                    n_space=n_space, n_time=n_time, exercise=exercise)
        args.update(kw)
        return price(**args)

    h_sigma, h_r = 1e-3, 1e-4
    h_tau = min(1e-3, tau / 100.0)

    return {
        "delta": float((values[j_spot + 1] - values[j_spot - 1]) / (2 * dS)),
        "gamma": float((values[j_spot + 1] - 2 * values[j_spot] + values[j_spot - 1]) / dS ** 2),
        "vega": (reprice(sigma=sigma + h_sigma) - reprice(sigma=sigma - h_sigma)) / (2 * h_sigma),
        "rho": (reprice(r=r + h_r) - reprice(r=r - h_r)) / (2 * h_r),
        # theta = dV/dt = -dV/dtau, since tau shrinks as calendar time advances.
        "theta": -(reprice(tau=tau + h_tau) - reprice(tau=tau - h_tau)) / (2 * h_tau),
    }
