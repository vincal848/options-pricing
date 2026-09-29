"""Price an option, or compare the three methods against each other.

    python run.py compare  -S 100 -K 100 -T 1 -r 0.05 -s 0.2
    python run.py price    -S 100 -K 100 -T 1 -r 0.05 -s 0.2 -k put -e american
    python run.py converge -S 100 -K 100 -T 1 -r 0.05 -s 0.2
    python run.py iv --price 10.4506 -S 100 -K 100 -T 1 -r 0.05
"""

import argparse

import binomial
import black_scholes as bs
import crank_nicolson as cn
from black_scholes import GREEK_NAMES


def compare(S, K, tau, r, sigma, kind="call", q=0.0, exercise="european",
            n_steps=1000, n_space=400, n_time=400):
    """Price and Greeks from every method that applies, keyed by method name.

    The analytic row is left out for American options, where no closed form exists.
    """
    out = {}

    if exercise == "european":
        row = {"price": float(bs.price(S, K, tau, r, sigma, kind, q))}
        for name, value in bs.greeks(S, K, tau, r, sigma, kind, q).items():
            row[name] = float(value)
        out["analytic"] = row

    out["binomial"] = dict(
        price=binomial.price(S, K, tau, r, sigma, kind, q, n_steps, exercise),
        **binomial.greeks(S, K, tau, r, sigma, kind, q, n_steps, exercise))
    out["crank-nicolson"] = dict(
        price=cn.price(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise),
        **cn.greeks(S, K, tau, r, sigma, kind, q, n_space, n_time, exercise))

    return out


def cmd_price(args):
    method = args.method
    if method == "auto":
        method = "analytic" if args.exercise == "european" else "binomial"
    if method == "analytic" and args.exercise != "european":
        raise SystemExit("no closed form for American options; use binomial or crank-nicolson")

    common = (args.spot, args.strike, args.tau, args.rate, args.sigma, args.kind, args.div)

    if method == "analytic":
        value = float(bs.price(*common))
        g = {k: float(v) for k, v in bs.greeks(*common).items()}
    elif method == "binomial":
        value = binomial.price(*common, args.steps, args.exercise)
        g = binomial.greeks(*common, args.steps, args.exercise)
    elif method == "crank-nicolson":
        value = cn.price(*common, args.space, args.time_steps, args.exercise)
        g = cn.greeks(*common, args.space, args.time_steps, args.exercise)
    else:
        raise SystemExit("unknown method %r" % (method,))

    print("%s %s via %s" % (args.exercise, args.kind, method))
    print("  price %.6f" % value)
    for name in GREEK_NAMES:
        print("  %-6s%14.6f" % (name, g[name]))


def cmd_compare(args):
    table = compare(args.spot, args.strike, args.tau, args.rate, args.sigma,
                    args.kind, args.div, args.exercise,
                    args.steps, args.space, args.time_steps)
    cols = ("price",) + GREEK_NAMES

    print("%s %s  S=%s K=%s tau=%s r=%s sigma=%s q=%s"
          % (args.exercise, args.kind, args.spot, args.strike, args.tau,
             args.rate, args.sigma, args.div))
    print("%-16s" % "method" + "".join("%12s" % c for c in cols))
    for name, row in table.items():
        print("%-16s" % name + "".join("%12.6f" % row[c] for c in cols))

    # For European options the analytic row is exact, so the remaining rows read
    # directly as discretisation error.
    if "analytic" in table:
        base = table["analytic"]
        print("\nabsolute error vs analytic")
        for name, row in table.items():
            if name == "analytic":
                continue
            print("%-16s" % name + "".join("%12.6f" % abs(row[c] - base[c]) for c in cols))


def cmd_converge(args):
    exact = float(bs.price(args.spot, args.strike, args.tau, args.rate,
                           args.sigma, args.kind, args.div))
    print("analytic %s price %.10f\n" % (args.kind, exact))
    print("%6s%16s%14s%16s%14s" % ("n", "binomial", "error", "crank-nic", "error"))

    n = 10
    while n <= args.max_steps:
        b = binomial.price(args.spot, args.strike, args.tau, args.rate,
                           args.sigma, args.kind, args.div, n)
        c = cn.price(args.spot, args.strike, args.tau, args.rate,
                     args.sigma, args.kind, args.div, n, n)
        print("%6d%16.8f%14.2e%16.8f%14.2e" % (n, b, abs(b - exact), c, abs(c - exact)))
        n *= 2


def cmd_iv(args):
    v = bs.implied_vol(args.price, args.spot, args.strike, args.tau,
                       args.rate, args.kind, args.div)
    print("implied volatility %.6f  (%.4f%%)" % (v, v * 100))


def add_common(p, need_sigma=True):
    p.add_argument("-S", "--spot", type=float, required=True, help="spot price")
    p.add_argument("-K", "--strike", type=float, required=True, help="strike price")
    p.add_argument("-T", "--tau", type=float, required=True, help="time to expiry in YEARS")
    p.add_argument("-r", "--rate", type=float, required=True, help="annual risk-free rate")
    if need_sigma:
        p.add_argument("-s", "--sigma", type=float, required=True,
                       help="annual volatility, e.g. 0.2 for 20%%")
    p.add_argument("-q", "--div", type=float, default=0.0, help="continuous dividend yield")
    p.add_argument("-k", "--kind", default="call", choices=["call", "put"])


def add_exercise(p):
    p.add_argument("-e", "--exercise", default="european", choices=["european", "american"])
    p.add_argument("--steps", type=int, default=1000, help="binomial time steps")
    p.add_argument("--space", type=int, default=400, help="Crank-Nicolson spatial nodes")
    p.add_argument("--time-steps", type=int, default=400, help="Crank-Nicolson time steps")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="price one option and print its Greeks")
    add_common(p)
    add_exercise(p)
    p.add_argument("-m", "--method", default="auto",
                   choices=["auto", "analytic", "binomial", "crank-nicolson"])
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("compare", help="run every applicable method side by side")
    add_common(p)
    add_exercise(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("converge", help="show both numerical methods converging")
    add_common(p)
    p.add_argument("--max-steps", type=int, default=2000)
    p.set_defaults(func=cmd_converge)

    p = sub.add_parser("iv", help="back out the volatility from an observed price")
    add_common(p, need_sigma=False)
    p.add_argument("-p", "--price", type=float, required=True, help="observed option price")
    p.set_defaults(func=cmd_iv)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
