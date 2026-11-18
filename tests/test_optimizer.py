"""Checks that the optimizer respects the physics and finds obvious answers.

Run with:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bess import BatterySpec, optimize_dispatch  # noqa: E402
from bess.data import synthetic_prices  # noqa: E402
from bess.price_model import PriceModel  # noqa: E402

B = BatterySpec(power_mw=10, energy_mwh=20, round_trip_eff=0.81, soc_min_frac=0.0,
                soc_max_frac=1.0, soc_start_frac=0.0, degradation_cost=0.0)


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(years=(2022,), seed=1).to_numpy()


def test_flat_prices_no_trading():
    res = optimize_dispatch(np.full(24, 50.0), B)
    assert res.charge.sum() == pytest.approx(0, abs=1e-6)
    assert res.discharge.sum() == pytest.approx(0, abs=1e-6)


def test_simple_arbitrage_by_hand():
    # cheap for 2 hours, expensive for 2 hours. Other hours are $90: buying at $90
    # to sell at $100 loses money with 81% round-trip efficiency (90 / 0.81 > 100).
    p = np.full(24, 90.0)
    p[[2, 3]] = 10.0
    p[[18, 19]] = 100.0
    res = optimize_dispatch(p, B)
    # power limit: 10 MW x 2 h = 20 MWh bought; stored = 20 * 0.9 = 18 MWh;
    # sold = 18 * 0.9 = 16.2 MWh (one-way efficiency = sqrt(0.81) = 0.9)
    assert res.charge[[2, 3]].sum() == pytest.approx(20, abs=1e-6)
    assert res.discharge.sum() == pytest.approx(16.2, abs=1e-6)
    assert res.revenue() == pytest.approx(16.2 * 100 - 20 * 10, abs=1e-4)


def test_physical_limits(prices):
    b = BatterySpec()
    res = optimize_dispatch(prices, b)
    assert res.soc.min() >= b.soc_min - 1e-6
    assert res.soc.max() <= b.soc_max + 1e-6
    assert res.charge.max() <= b.power_mw + 1e-6
    assert res.discharge.max() <= b.power_mw + 1e-6


def test_energy_balance(prices):
    b = BatterySpec()
    res = optimize_dispatch(prices, b)
    prev = np.r_[b.soc_start, res.soc[:-1]]
    prev[::24] = b.soc_start                          # each day starts at soc_start
    expected = prev + b.eta_charge * res.charge - res.discharge / b.eta_discharge
    assert np.allclose(res.soc, expected, atol=1e-5)
    assert np.allclose(res.soc[23::24], b.soc_start, atol=1e-5)   # each day ends at soc_start


def test_profit_never_negative_with_perfect_foresight(prices):
    # doing nothing is always allowed, so optimal profit >= 0 every day
    b = BatterySpec()
    res = optimize_dispatch(prices, b)
    daily = (prices * res.net_power).reshape(-1, 24).sum(1) - \
        b.degradation_cost * res.discharge.reshape(-1, 24).sum(1)
    assert daily.min() >= -1e-4


def test_milp_has_no_simultaneous_charge_discharge():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 40, 24 * 5)                     # many negative prices
    res = optimize_dispatch(p, BatterySpec(), no_simultaneous=True)
    assert not np.any((res.charge > 1e-6) & (res.discharge > 1e-6))


def test_more_energy_is_worth_more(prices):
    b = BatterySpec()
    p1 = optimize_dispatch(prices, b.with_duration(1)).revenue()
    p4 = optimize_dispatch(prices, b.with_duration(4)).revenue()
    assert p4 > p1


def test_price_model_simulates_valid_years():
    p = synthetic_prices(years=(2021, 2022), seed=3)
    m = PriceModel.fit(p)
    sim = next(m.simulate_years(1, seed=0))
    assert len(sim) == 8760
    assert np.isfinite(sim).all()
    assert sim.min() >= m.price_floor and sim.max() <= m.price_cap


def test_plan_and_settle_same_prices_equals_perfect_foresight():
    from bess.backtest import plan_and_settle, run_backtest
    p = synthetic_prices(years=(2022,), seed=2)
    b = BatterySpec()
    a = plan_and_settle(p, p, b)["profit"].sum()
    pf = run_backtest(p, b)["profit"].sum()
    assert a == pytest.approx(pf)


def test_realistic_strategies_never_beat_perfect_foresight():
    from bess.backtest import compare_strategies
    from bess.data import synthetic_dam
    p = synthetic_prices(years=(2022,), seed=4)
    t = compare_strategies(p, BatterySpec(), dam=synthetic_dam(p))
    pf = t.loc["perfect_foresight", "profit_per_year"]
    for name in ["persistence", "weekly_median", "day_ahead_forecast"]:
        assert t.loc[name, "profit_per_year"] <= pf + 1e-6
