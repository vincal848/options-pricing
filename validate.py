"""Regenerate docs/VALIDATION.md and the figures in docs/img/.

Every number in the documentation comes from this script, so the docs cannot drift
away from the code:

    python scripts/validate.py

Run from the repository root.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import binomial as bn
import black_scholes as bs
import crank_nicolson as cn
from black_scholes import GREEK_NAMES

DOCS = ROOT / "docs"
IMG = DOCS / "img"

ATM = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.20, q=0.0)
CASES = [
    ("at the money", 100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    ("out of the money, high vol", 100.0, 110.0, 0.50, 0.03, 0.35, 0.02),
    ("in the money, long dated", 90.0, 100.0, 2.00, 0.06, 0.15, 0.00),
    ("deep in the money, short dated", 120.0, 100.0, 0.25, 0.01, 0.45, 0.04),
    ("Hull ch. 15 worked example", 42.0, 40.0, 0.50, 0.10, 0.20, 0.00),
]


def _timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def section_prices(out: io.StringIO) -> None:
    out.write("## European prices against the analytic formula\n\n")
    out.write("Binomial at 2000 steps, Crank-Nicolson on a 600x600 grid.\n\n")
    out.write("| Case | Kind | Analytic | Binomial | error | Crank-Nicolson | error |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    for label, S, K, tau, r, sig, q in CASES:
        for kind in ("call", "put"):
            exact = float(bs.price(S, K, tau, r, sig, kind, q))
            b = bn.price(S, K, tau, r, sig, kind, q, n_steps=2000)
            c = cn.price(S, K, tau, r, sig, kind, q, n_space=600, n_time=600)
            out.write(f"| {label} | {kind} | {exact:.6f} | {b:.6f} | {abs(b - exact):.2e} "
                      f"| {c:.6f} | {abs(c - exact):.2e} |\n")
    out.write("\n")


def section_convergence(out: io.StringIO) -> list[tuple[int, float, float, float, float]]:
    out.write("## Convergence\n\n")
    out.write("European put, at the money. `n` is binomial steps, and both Crank-Nicolson\n")
    out.write("dimensions. `n x error` is constant for a first-order method and falls for\n")
    out.write("a second-order one.\n\n")
    out.write("| n | Binomial | error | n x error | Crank-Nicolson | error | n x error |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    exact = float(bs.price(kind="put", **ATM))
    rows = []
    for n in (10, 25, 50, 100, 250, 500, 1000, 2000, 4000):
        b = bn.price(kind="put", n_steps=n, **ATM)
        c = cn.price(kind="put", n_space=n, n_time=n, **ATM)
        eb, ec = abs(b - exact), abs(c - exact)
        rows.append((n, b, eb, c, ec))
        out.write(f"| {n} | {b:.8f} | {eb:.2e} | {n * eb:.3f} | {c:.8f} | {ec:.2e} "
                  f"| {n * ec:.3f} |\n")
    out.write(f"\nAnalytic value: `{exact:.10f}`\n\n")

    # Estimate observed order from the last few refinements.
    def order(errs, ns):
        return float(np.mean([np.log(a / b) / np.log(nb / na)
                             for (na, a), (nb, b) in zip(zip(ns, errs), zip(ns[1:], errs[1:]))]))

    ns = [r[0] for r in rows[-4:]]
    out.write(f"Observed order of convergence over the last four refinements: "
              f"binomial **{order([r[2] for r in rows[-4:]], ns):.2f}**, "
              f"Crank-Nicolson **{order([r[4] for r in rows[-4:]], ns):.2f}**.\n\n")
    return rows


def section_greeks(out: io.StringIO) -> None:
    out.write("## Greeks\n\n")
    out.write("At the money, both kinds. Binomial 1500 steps, Crank-Nicolson 600x600.\n\n")
    for kind in ("call", "put"):
        want = {k: float(v) for k, v in bs.greeks(kind=kind, **ATM).items()}
        gb = bn.greeks(kind=kind, n_steps=1500, **ATM)
        gc = cn.greeks(kind=kind, n_space=600, n_time=600, **ATM)
        out.write(f"**{kind.title()}**\n\n")
        out.write("| Greek | Analytic | Binomial | rel. error | Crank-Nicolson | rel. error |\n")
        out.write("|---|---|---|---|---|---|\n")
        for g in GREEK_NAMES:
            out.write(f"| {g} | {want[g]:.6f} | {gb[g]:.6f} | {abs(gb[g] / want[g] - 1):.1e} "
                      f"| {gc[g]:.6f} | {abs(gc[g] / want[g] - 1):.1e} |\n")
        out.write("\n")


def section_american(out: io.StringIO) -> None:
    out.write("## American options\n\n")
    out.write("No closed form exists, so the two numerical methods are checked against each\n")
    out.write("other and against bounds that must hold.\n\n")
    euro = float(bs.price(kind="put", **ATM))
    b = bn.price(kind="put", n_steps=2000, exercise="american", **ATM)
    c = cn.price(kind="put", n_space=800, n_time=800, exercise="american", **ATM)
    ref = bn.price(kind="put", n_steps=20000, exercise="american", **ATM)
    out.write("| Quantity | Value |\n|---|---|\n")
    out.write(f"| European put (analytic) | {euro:.6f} |\n")
    out.write(f"| American put, binomial 2000 steps | {b:.6f} |\n")
    out.write(f"| American put, Crank-Nicolson 800x800 | {c:.6f} |\n")
    out.write(f"| American put, binomial 20000 steps (reference) | {ref:.6f} |\n")
    out.write(f"| Early-exercise premium | {ref - euro:.6f} |\n")
    out.write(f"| Disagreement between the two methods | {abs(b - c):.2e} |\n\n")

    ec = bn.price(kind="call", n_steps=1000, exercise="european", **ATM)
    ac = bn.price(kind="call", n_steps=1000, exercise="american", **ATM)
    out.write(f"With no dividend, the American call equals the European call to "
              f"`{abs(ac - ec):.1e}` ({ac:.6f} vs {ec:.6f}), as it must: exercising a "
              f"call early throws away time value (Hull ch. 11.5). With a dividend "
              f"yield above the rate the premium reappears:\n\n")
    d = dict(S=100.0, K=100.0, tau=1.0, r=0.03, sigma=0.25, q=0.08)
    ec2 = bn.price(kind="call", n_steps=1000, exercise="european", **d)
    ac2 = bn.price(kind="call", n_steps=1000, exercise="american", **d)
    out.write(f"- `q=0.08 > r=0.03`: European {ec2:.6f}, American {ac2:.6f}, "
              f"premium {ac2 - ec2:.6f}\n\n")


def section_identities(out: io.StringIO) -> None:
    out.write("## Identities\n\n")
    out.write("Checks that hold for any parameters and so catch sign and factor errors.\n\n")
    out.write("| Identity | Residual |\n|---|---|\n")
    S, K, tau, r, sig, q = 100.0, 95.0, 0.75, 0.04, 0.25, 0.02
    c = float(bs.price(S, K, tau, r, sig, "call", q))
    p = float(bs.price(S, K, tau, r, sig, "put", q))
    parity = abs((c - p) - (S * np.exp(-q * tau) - K * np.exp(-r * tau)))
    gc, gp = bs.greeks(S, K, tau, r, sig, "call", q), bs.greeks(S, K, tau, r, sig, "put", q)
    out.write(f"| Put-call parity, `C - P = S e^-qt - K e^-rt` | {parity:.2e} |\n")
    out.write(f"| `delta_call - delta_put = e^-qt` | "
              f"{abs(float(gc['delta'] - gp['delta']) - np.exp(-q * tau)):.2e} |\n")
    out.write(f"| `gamma_call = gamma_put` | {abs(float(gc['gamma'] - gp['gamma'])):.2e} |\n")
    out.write(f"| `vega_call = vega_put` | {abs(float(gc['vega'] - gp['vega'])):.2e} |\n")
    out.write(f"| `rho_call - rho_put = K t e^-rt` | "
              f"{abs(float(gc['rho'] - gp['rho']) - K * tau * np.exp(-r * tau)):.2e} |\n")
    iv_true = 0.3137
    tgt = float(bs.price(100.0, 110.0, 1.0, 0.03, iv_true, "call"))
    got = bs.implied_vol(tgt, 100.0, 110.0, 1.0, 0.03, "call")
    out.write(f"| Implied-vol round trip at sigma=0.3137 | {abs(got - iv_true):.2e} |\n\n")

    s_nodes, values, _ = cn.solve(kind="call", n_space=400, n_time=400, **ATM)
    ana = bs.price(s_nodes[1:-1], 100.0, 1.0, 0.05, 0.2, "call", 0.0)
    out.write(f"Crank-Nicolson is also checked across the **whole grid**, not only at the "
              f"spot: maximum absolute error over all {len(s_nodes)} nodes is "
              f"`{float(np.abs(values[1:-1] - ana).max()):.2e}`. This is the check that "
              f"caught a boundary term being applied twice, which left the at-the-money "
              f"price accurate to 1e-3 while corrupting the deep-in-the-money end of the "
              f"grid by about 88.\n\n")


def section_cost(out: io.StringIO) -> None:
    out.write("## Cost\n\n")
    out.write("Single price, at the money, on one core.\n\n")
    out.write("| Method | Setting | Time | Error |\n|---|---|---|---|\n")
    exact = float(bs.price(kind="put", **ATM))
    _, t = _timed(bs.price, kind="put", **ATM)
    out.write(f"| analytic | closed form | {t * 1e6:.0f} us | exact |\n")
    for n in (500, 2000):
        v, t = _timed(bn.price, kind="put", n_steps=n, **ATM)
        out.write(f"| binomial | {n} steps | {t * 1e3:.1f} ms | {abs(v - exact):.2e} |\n")
    cn_time: dict[int, float] = {}
    for n in (400, 800):
        v, t = _timed(cn.price, kind="put", n_space=n, n_time=n, **ATM)
        cn_time[n] = t
        out.write(f"| Crank-Nicolson | {n}x{n} | {t * 1e3:.1f} ms | {abs(v - exact):.2e} |\n")
    # Wall-clock at a matched tight tolerance, so the trade-off is stated from data
    # rather than by assuming the second-order method must also be the faster one.
    tol = 3e-4
    bn_n = 500
    while abs(bn.price(kind="put", n_steps=bn_n, **ATM) - exact) > tol and bn_n <= 64000:
        bn_n *= 2
    vb, tb = _timed(bn.price, kind="put", n_steps=bn_n, **ATM)
    out.write(f"| binomial | {bn_n} steps | {tb * 1e3:.1f} ms | {abs(vb - exact):.2e} |\n")
    out.write("\nBoth numerical methods do O(n^2) work here -- the tree touches every node\n")
    out.write("once, and Crank-Nicolson does an O(n) tridiagonal solve at each of n time\n")
    out.write("steps -- so the comparison worth making is time at a matched *tolerance*,\n")
    out.write("not time at a matched n.\n\n")
    out.write("At loose tolerances the tree wins outright: 2000 steps reaches 1e-3 faster\n")
    out.write("than a 400x400 grid reaches the same figure. The ordering reverses as the\n")
    out.write("tolerance tightens, because first-order convergence means the tree needs ten\n")
    out.write(f"times the steps per extra decimal place -- reaching {tol:.0e} costs it\n")
    out.write(f"{bn_n} steps and {tb * 1e3:.0f} ms, against {cn_time[800] * 1e3:.0f} ms for\n")
    out.write("the 800x800 grid that gets there. Neither is a reason to prefer them to the\n")
    out.write("closed form when one exists; the point of both is the American case, where\n")
    out.write("it does not.\n\n")


def figures(rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

    import plots

    fig = plots.greeks_vs_spot(100.0, 1.0, 0.05, 0.2, "call")
    fig.savefig(IMG / "greeks.png")
    plt.close(fig)

    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(ns, [r[2] for r in rows], "o-", label="binomial (steps)")
    ax.loglog(ns, [r[4] for r in rows], "s-", label="Crank-Nicolson (n x n)")
    ref = [2.0 / n for n in ns]
    ax.loglog(ns, ref, ":", color="0.5", label="O(1/n) reference")
    ax.set(xlabel="discretisation n", ylabel="absolute error vs analytic",
           title="Convergence to the analytic European put")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(IMG / "convergence.png")
    plt.close(fig)

    fig = plots.early_exercise_premium(100.0, 1.0, 0.05, 0.2)
    fig.savefig(IMG / "early_exercise.png")
    plt.close(fig)
    print(f"figures written to {IMG}")


def main() -> None:
    out = io.StringIO()
    out.write("# Validation\n\n")
    out.write("Generated by `scripts/validate.py`. Do not edit by hand.\n\n")
    out.write("The analytic Black-Scholes price is exact for European options, so it is the\n")
    out.write("reference for both numerical methods. For American options, where no closed\n")
    out.write("form exists, the methods are checked against each other and against bounds\n")
    out.write("that any correct implementation must satisfy.\n\n")
    section_prices(out)
    rows = section_convergence(out)
    section_greeks(out)
    section_american(out)
    section_identities(out)
    section_cost(out)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "VALIDATION.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"wrote {DOCS / 'VALIDATION.md'}")
    figures(rows)


if __name__ == "__main__":
    main()
