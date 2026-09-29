"""Cox-Ross-Rubinstein binomial tree, European and American.

Standard CRR over n_steps of length dt = tau / n_steps:

    u = exp(sigma sqrt(dt))    d = 1 / u    p = (exp((r - q) dt) - d) / (u - d)

Two defects in the original are fixed here, both confirmed against the analytic
European price in docs/VALIDATION.md:

  1. p was written exp((r - q) * dt - d) / (u - d), with the down-factor subtracted
     INSIDE the exponential. That gave p = 6.69 at 50 steps and 13.19 at 200, so the
     price moved away from the truth as steps were added.
  2. The early-exercise test indexed the lattice at price_tree[t, i] - the period
     argument the caller passed - instead of at the backward-induction level. Every
     node was compared against one fixed row of the tree.

Reference: Hull ch. 21; Stefanica ch. 3-4.
"""

import numpy as np

from black_scholes import check_kind, intrinsic


def check_exercise(exercise):
    e = str(exercise).lower()
    if e not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american', got %r" % (exercise,))
    return e


def crr_params(tau, r, sigma, n_steps, q=0.0):
    """Return (dt, u, d, p, discount) for a CRR tree.

    Raises if p leaves (0, 1). A risk-neutral probability outside that range is not
    a probability, and pricing with it produces a number that looks like an option
    value but is not one - which is exactly what the original did. Hull gives the
    condition as sigma > |r - q| sqrt(dt).
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
            "risk-neutral probability p=%.6g outside (0,1): dt=%.6g is too coarse "
            "for sigma=%.6g, r-q=%.6g. Increase n_steps." % (p, dt, sigma, r - q))

    return dt, u, d, p, float(np.exp(-r * dt))


def lattice(S, u, d, level):
    """Asset prices at a level of the tree, lowest node first: S * u^j * d^(level-j)."""
    j = np.arange(level + 1)
    return S * u ** j * d ** (level - j)


def price(S, K, tau, r, sigma, kind="call", q=0.0, n_steps=500, exercise="european"):
    """Binomial price. exercise='american' tests early exercise at every node."""
    e = check_exercise(exercise)
    if tau == 0:
        return float(intrinsic(S, K, kind))

    dt, u, d, p, discount = crr_params(tau, r, sigma, n_steps, q)
    values = intrinsic(lattice(S, u, d, n_steps), K, kind).astype(float)

    for level in range(n_steps - 1, -1, -1):
        # values[1:] are the up-children of this level and values[:-1] the down-
        # children, so the whole level discounts in one vectorized step rather than
        # the inner Python loop the original used.
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if e == "american":
            values = np.maximum(values, intrinsic(lattice(S, u, d, level), K, kind))

    return float(values[0])


def induct(S, K, tau, r, sigma, kind, q, n_steps, exercise):
    """Backward induction that keeps the bottom three levels of the tree.

    Returns (levels, u, d, dt) where levels[k] holds the option values at level k
    for k in 0, 1, 2. Those nine numbers are all the Greeks below need.
    """
    e = check_exercise(exercise)
    dt, u, d, p, discount = crr_params(tau, r, sigma, n_steps, q)
    values = intrinsic(lattice(S, u, d, n_steps), K, kind).astype(float)

    kept = {}
    for level in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if e == "american":
            values = np.maximum(values, intrinsic(lattice(S, u, d, level), K, kind))
        if level <= 2:
            kept[level] = values.copy()

    return kept, u, d, dt


def greeks(S, K, tau, r, sigma, kind="call", q=0.0, n_steps=500, exercise="european"):
    """Greeks for a binomial price (Hull ch. 21.6).

    Delta, gamma and theta come off nodes the tree already computed. Bumping the
    spot and repricing does NOT work here: rescaling S moves every lattice node, so
    the price carries a sawtooth in S that a 1/h^2 divisor amplifies. A 0.1% bump
    returns gamma = 0.265 against a true 0.0188. The node estimates have no such
    term.

    Vega and rho have no node equivalent and stay as central differences. That is
    fine - they are first order, so any noise is divided by h rather than h^2.
    """
    e = check_exercise(exercise)
    if n_steps < 3:
        raise ValueError("node-based Greeks need at least 3 steps")

    kept, u, d, dt = induct(S, K, tau, r, sigma, kind, q, n_steps, e)
    v0, v1, v2 = kept[0], kept[1], kept[2]

    # Level-1 nodes straddle the spot; level-2 nodes give a delta either side of it.
    s1_up, s1_down = S * u, S * d
    s2_up, s2_mid, s2_down = S * u * u, S, S * d * d

    delta = (v1[1] - v1[0]) / (s1_up - s1_down)
    delta_up = (v2[2] - v2[1]) / (s2_up - s2_mid)
    delta_down = (v2[1] - v2[0]) / (s2_mid - s2_down)
    gamma = (delta_up - delta_down) / (0.5 * (s2_up - s2_down))
    # v2[1] sits at the same spot as the root, two steps later in calendar time.
    theta = (v2[1] - v0[0]) / (2.0 * dt)

    def reprice(**kw):
        args = dict(S=S, K=K, tau=tau, r=r, sigma=sigma, kind=kind, q=q,
                    n_steps=n_steps, exercise=e)
        args.update(kw)
        return price(**args)

    h_sigma, h_r = 1e-3, 1e-4
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": (reprice(sigma=sigma + h_sigma) - reprice(sigma=sigma - h_sigma)) / (2 * h_sigma),
        "rho": (reprice(r=r + h_r) - reprice(r=r - h_r)) / (2 * h_r),
    }
