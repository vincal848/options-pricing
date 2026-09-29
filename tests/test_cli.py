"""The CLI commands run and print numbers that match the library."""
from __future__ import annotations

from typer.testing import CliRunner

from options_pricing.cli import app

runner = CliRunner()
ATM = ["-S", "100", "-K", "100", "-T", "1", "-r", "0.05", "-s", "0.2"]


def test_price_analytic():
    res = runner.invoke(app, ["price", *ATM, "-k", "call"])
    assert res.exit_code == 0, res.output
    assert "10.450584" in res.output
    assert "delta" in res.output


def test_price_american_binomial():
    res = runner.invoke(app, ["price", *ATM, "-k", "put", "-e", "american"])
    assert res.exit_code == 0, res.output
    assert "6.08" in res.output


def test_analytic_american_is_rejected():
    res = runner.invoke(app, ["price", *ATM, "-e", "american", "-m", "analytic"])
    assert res.exit_code != 0
    assert "no closed form" in res.output


def test_compare_prints_all_methods_and_errors():
    res = runner.invoke(app, ["compare", *ATM, "--steps", "200", "--space", "100",
                              "--time-steps", "100"])
    assert res.exit_code == 0, res.output
    for name in ("analytic", "binomial", "crank-nicolson", "absolute error"):
        assert name in res.output


def test_converge_table_runs():
    res = runner.invoke(app, ["converge", *ATM, "--max-steps", "80"])
    assert res.exit_code == 0, res.output
    assert "analytic" in res.output


def test_iv_recovers_input_vol():
    res = runner.invoke(app, ["iv", "-p", "10.450584", "-S", "100", "-K", "100",
                              "-T", "1", "-r", "0.05", "-k", "call"])
    assert res.exit_code == 0, res.output
    assert "0.2000" in res.output
