"""European and American option pricing by three independent methods.

Each method exposes the same ``price(...)`` and ``greeks(...)`` signature, so they
can be compared directly -- which is the point of the package. The analytic
formulas are exact for European options and serve as the reference the two
numerical methods are checked against in ``tests/``.

    >>> from options_pricing import black_scholes, binomial, crank_nicolson
    >>> round(float(black_scholes.price(100, 100, 1.0, 0.05, 0.2, "call")), 4)
    10.4506
    >>> round(binomial.price(100, 100, 1.0, 0.05, 0.2, "put", n_steps=1000,
    ...                      exercise="american"), 4)
    6.0896

``compare`` runs all three and returns the table the CLI prints.
"""
from __future__ import annotations

from options_pricing import binomial, black_scholes, crank_nicolson
from options_pricing.black_scholes import Kind

__all__ = ["black_scholes", "binomial", "crank_nicolson", "compare", "GREEK_NAMES"]

__version__ = "1.0.0"

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


def compare(S: float, K: float, tau: float, r: float, sigma: float,
            kind: Kind = "call", q: float = 0.0, exercise: str = "european",
            n_steps: int = 1000, n_space: int = 400, n_time: int = 400,
            ) -> dict[str, dict[str, float | None]]:
    """Price and Greeks from every method that applies, keyed by method name.

    The analytic row is omitted for American options, where no closed form exists.
    """
    out: dict[str, dict[str, float | None]] = {}

    if exercise == "european":
        row = {"price": float(black_scholes.price(S, K, tau, r, sigma, kind, q))}
        row |= {g: float(v) for g, v in
                black_scholes.greeks(S, K, tau, r, sigma, kind, q).items()}
        out["analytic"] = row

    out["binomial"] = {
        "price": binomial.price(S, K, tau, r, sigma, kind, q, n_steps, exercise),
        **binomial.greeks(S, K, tau, r, sigma, kind, q, n_steps, exercise),
    }
    out["crank-nicolson"] = {
        "price": crank_nicolson.price(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise),
        **crank_nicolson.greeks(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise),
    }
    return out
