"""The command line entry point runs and prints numbers matching the library."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run  # noqa: E402

ATM = ["-S", "100", "-K", "100", "-T", "1", "-r", "0.05", "-s", "0.2"]


def invoke(*args):
    """Run run.py as a subprocess and return (returncode, stdout + stderr)."""
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), *args],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_price_analytic():
    code, out = invoke("price", *ATM, "-k", "call")
    assert code == 0, out
    assert "10.450584" in out
    assert "delta" in out


def test_price_american_uses_binomial_by_default():
    code, out = invoke("price", *ATM, "-k", "put", "-e", "american")
    assert code == 0, out
    assert "via binomial" in out
    assert "6.08" in out


def test_analytic_american_is_refused():
    """There is no closed form, so asking for one must fail loudly."""
    code, out = invoke("price", *ATM, "-e", "american", "-m", "analytic")
    assert code != 0
    assert "no closed form" in out


def test_compare_prints_every_method_and_the_errors():
    code, out = invoke("compare", *ATM, "--steps", "200", "--space", "100",
                       "--time-steps", "100")
    assert code == 0, out
    for expected in ("analytic", "binomial", "crank-nicolson", "absolute error"):
        assert expected in out


def test_converge_runs():
    code, out = invoke("converge", *ATM, "--max-steps", "80")
    assert code == 0, out
    assert "analytic" in out


def test_iv_recovers_the_input_volatility():
    code, out = invoke("iv", "-p", "10.450584", "-S", "100", "-K", "100",
                       "-T", "1", "-r", "0.05", "-k", "call")
    assert code == 0, out
    assert "0.2000" in out


def test_missing_required_argument_fails():
    code, out = invoke("price", "-S", "100")
    assert code != 0


def test_compare_function_matches_the_library():
    """compare() is the function the CLI wraps; check it directly too."""
    import black_scholes as bs

    table = run.compare(100.0, 100.0, 1.0, 0.05, 0.2, "call",
                        n_steps=500, n_space=200, n_time=200)
    assert set(table) == {"analytic", "binomial", "crank-nicolson"}
    assert table["analytic"]["price"] == pytest.approx(
        float(bs.price(100, 100, 1.0, 0.05, 0.2, "call")), abs=1e-12)
    # All three must agree to within their discretisation error.
    for name in ("binomial", "crank-nicolson"):
        assert table[name]["price"] == pytest.approx(table["analytic"]["price"], abs=1e-2)


def test_compare_omits_analytic_for_american():
    table = run.compare(100.0, 100.0, 1.0, 0.05, 0.2, "put", exercise="american",
                        n_steps=300, n_space=150, n_time=150)
    assert "analytic" not in table
    assert set(table) == {"binomial", "crank-nicolson"}
