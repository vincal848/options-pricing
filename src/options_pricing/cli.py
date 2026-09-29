"""Command line interface: ``price`` one option, ``compare`` the three methods,
``converge`` to show the error falling with discretisation, or ``iv`` to back out
an implied volatility.
"""
from __future__ import annotations

from typing import Optional

import typer

from options_pricing import GREEK_NAMES, black_scholes, compare as _compare

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)

# Shared options. tau is in years; see black_scholes for the full convention note.
_S = typer.Option(..., "--spot", "-S", help="Spot price of the underlying.")
_K = typer.Option(..., "--strike", "-K", help="Strike price.")
_T = typer.Option(..., "--tau", "-T", help="Time to expiry, in years.")
_R = typer.Option(..., "--rate", "-r", help="Continuously compounded annual risk-free rate.")
_V = typer.Option(..., "--sigma", "-s", help="Annual volatility, e.g. 0.2 for 20%.")
_Q = typer.Option(0.0, "--div", "-q", help="Continuous dividend yield.")
_KIND = typer.Option("call", "--kind", "-k", help="call or put.")
_EX = typer.Option("european", "--exercise", "-e", help="european or american.")
_STEPS = typer.Option(1000, "--steps", help="Binomial time steps.")
_SPACE = typer.Option(400, "--space", help="Crank-Nicolson spatial nodes.")
_TIME = typer.Option(400, "--time-steps", help="Crank-Nicolson time steps.")


def _fmt(x: Optional[float], width: int = 12) -> str:
    return " " * width if x is None else f"{x:>{width}.6f}"


@app.command()
def price(spot: float = _S, strike: float = _K, tau: float = _T, rate: float = _R,
          sigma: float = _V, div: float = _Q, kind: str = _KIND, exercise: str = _EX,
          method: str = typer.Option("auto", "--method", "-m",
                                     help="analytic, binomial, crank-nicolson, or auto."),
          steps: int = _STEPS, space: int = _SPACE, time_steps: int = _TIME) -> None:
    """Price one option and print its Greeks.

    ``--method auto`` uses the analytic formula for European options and the
    binomial tree for American ones.
    """
    from options_pricing import binomial, crank_nicolson

    if method == "auto":
        method = "analytic" if exercise == "european" else "binomial"
    if method == "analytic" and exercise != "european":
        raise typer.BadParameter("no closed form for American options; use binomial or crank-nicolson")

    if method == "analytic":
        p = float(black_scholes.price(spot, strike, tau, rate, sigma, kind, div))
        g = {k: float(v) for k, v in
             black_scholes.greeks(spot, strike, tau, rate, sigma, kind, div).items()}
    elif method == "binomial":
        args = (spot, strike, tau, rate, sigma, kind, div, steps, exercise)
        p, g = binomial.price(*args), binomial.greeks(*args)
    elif method == "crank-nicolson":
        args = (spot, strike, tau, rate, sigma, kind, div, space, time_steps, exercise)
        p, g = crank_nicolson.price(*args), crank_nicolson.greeks(*args)
    else:
        raise typer.BadParameter(f"unknown method {method!r}")

    typer.echo(f"{exercise} {kind} via {method}")
    typer.echo(f"  price {p:.6f}")
    for name in GREEK_NAMES:
        typer.echo(f"  {name:<6}{g[name]:>14.6f}")


@app.command()
def compare(spot: float = _S, strike: float = _K, tau: float = _T, rate: float = _R,
            sigma: float = _V, div: float = _Q, kind: str = _KIND, exercise: str = _EX,
            steps: int = _STEPS, space: int = _SPACE, time_steps: int = _TIME) -> None:
    """Run every applicable method side by side.

    For European options the analytic row is exact, so the other two rows are a
    direct read on the discretisation error.
    """
    table = _compare(spot, strike, tau, rate, sigma, kind, div, exercise,
                     steps, space, time_steps)
    cols = ("price",) + GREEK_NAMES
    typer.echo(f"{exercise} {kind}  S={spot} K={strike} tau={tau} r={rate} sigma={sigma} q={div}")
    typer.echo(f"{'method':<16}" + "".join(f"{c:>12}" for c in cols))
    for name, row in table.items():
        typer.echo(f"{name:<16}" + "".join(_fmt(row.get(c)) for c in cols))

    if "analytic" in table:
        typer.echo("\nabsolute error vs analytic")
        base = table["analytic"]
        for name, row in table.items():
            if name == "analytic":
                continue
            typer.echo(f"{name:<16}" + "".join(
                _fmt(abs(row[c] - base[c]) if row.get(c) is not None else None) for c in cols))


@app.command()
def converge(spot: float = _S, strike: float = _K, tau: float = _T, rate: float = _R,
             sigma: float = _V, div: float = _Q, kind: str = _KIND,
             max_steps: int = typer.Option(2000, help="Largest discretisation to try.")) -> None:
    """Show both numerical methods converging to the analytic European price."""
    from options_pricing import binomial, crank_nicolson

    exact = float(black_scholes.price(spot, strike, tau, rate, sigma, kind, div))
    typer.echo(f"analytic {kind} price {exact:.10f}\n")
    typer.echo(f"{'n':>6}{'binomial':>16}{'error':>14}{'crank-nic':>16}{'error':>14}")
    n = 10
    while n <= max_steps:
        b = binomial.price(spot, strike, tau, rate, sigma, kind, div, n)
        c = crank_nicolson.price(spot, strike, tau, rate, sigma, kind, div, n, n)
        typer.echo(f"{n:>6}{b:>16.8f}{abs(b - exact):>14.2e}{c:>16.8f}{abs(c - exact):>14.2e}")
        n *= 2


@app.command()
def iv(target: float = typer.Option(..., "--price", "-p", help="Observed option price."),
       spot: float = _S, strike: float = _K, tau: float = _T, rate: float = _R,
       div: float = _Q, kind: str = _KIND) -> None:
    """Back out the volatility that reproduces an observed price."""
    v = black_scholes.implied_vol(target, spot, strike, tau, rate, kind, div)
    typer.echo(f"implied volatility {v:.6f}  ({v * 100:.4f}%)")


if __name__ == "__main__":  # pragma: no cover
    app()
