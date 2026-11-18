"""Historical backtests of operating strategies.

Every strategy is "plan on one price series, get paid on another":

  strategy                plan (optimize on)          paid at
  ----------------------  --------------------------  ----------------------
  perfect_foresight       actual real-time prices     real-time   (upper bound)
  persistence             yesterday's real-time       real-time
  weekly_median           median of last 7 days       real-time
  day_ahead_market        day-ahead prices            day-ahead
  day_ahead_forecast      day-ahead prices            real-time

perfect_foresight is not achievable: nobody knows tomorrow's real-time prices.
day_ahead_market sells the schedule in the day-ahead market. The prices used
are the cleared day-ahead prices, which assumes the battery's bids clear as
planned (a common simplification; say so in the report).
day_ahead_forecast is fully realistic: day-ahead prices are published the day
before, so they are a free, public forecast of real-time prices.

capture rate = strategy profit / perfect-foresight profit (same days).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .optimizer import optimize_dispatch


# --------------------------------------------------------------------------- forecasts
def forecast_persistence(prices: pd.Series) -> pd.Series:
    """Tomorrow looks like today: forecast for each hour = price 24 hours earlier."""
    return prices.shift(24)


def forecast_weekly_profile(prices: pd.Series, days: int = 7) -> pd.Series:
    """Median price at the same hour over the previous `days` days (robust to spikes)."""
    lagged = pd.concat([prices.shift(24 * k) for k in range(1, days + 1)], axis=1)
    return lagged.median(axis=1, skipna=False)


FORECASTS = {
    "persistence": forecast_persistence,
    "weekly_median": forecast_weekly_profile,
}

# strategies that are paid at real-time prices and so can be compared with perfect foresight
RT_SETTLED = ["perfect_foresight", "persistence", "weekly_median", "day_ahead_forecast"]


# --------------------------------------------------------------------------- core
def _whole_days(plan: pd.Series, settle: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two series on their common hours and keep only complete days."""
    both = pd.concat([plan.rename("plan"), settle.rename("settle")], axis=1).dropna()
    day = both.index.floor("D")
    full = day.map(day.value_counts()) == 24
    both = both[np.asarray(full)]
    if both.empty:
        raise ValueError("no complete days in common between plan and settle prices")
    return both["plan"], both["settle"]


def plan_and_settle(plan: pd.Series, settle: pd.Series, battery: BatterySpec,
                    no_simultaneous: bool = False) -> pd.DataFrame:
    """Optimize on `plan` prices, get paid `settle` prices. Daily results table."""
    plan, settle = _whole_days(plan, settle)
    res = optimize_dispatch(plan.to_numpy(), battery, no_simultaneous=no_simultaneous)
    df = pd.DataFrame({"revenue": settle.to_numpy() * res.net_power * res.dt,
                       "discharged_mwh": res.discharge * res.dt}, index=plan.index)
    daily = df.groupby(df.index.floor("D")).sum()
    daily["degradation_cost"] = battery.degradation_cost * daily["discharged_mwh"]
    daily["profit"] = daily["revenue"] - daily["degradation_cost"]
    daily.index = daily.index.tz_localize(None)
    return daily


def strategy_series(name: str, rt: pd.Series, dam: pd.Series | None = None):
    """(plan, settle) price series for a named strategy."""
    if name == "perfect_foresight":
        return rt, rt
    if name in FORECASTS:
        return FORECASTS[name](rt), rt
    if dam is None:
        raise ValueError(f"strategy {name} needs day-ahead prices")
    if name == "day_ahead_market":
        return dam, dam
    if name == "day_ahead_forecast":
        return dam, rt
    raise ValueError(f"unknown strategy {name}")


def available_strategies(dam: pd.Series | None) -> list[str]:
    names = ["perfect_foresight", *FORECASTS]
    if dam is not None:
        names += ["day_ahead_market", "day_ahead_forecast"]
    return names


def run_backtest(prices: pd.Series, battery: BatterySpec, forecast: str | None = None,
                 dam: pd.Series | None = None, no_simultaneous: bool = False) -> pd.DataFrame:
    """Daily results for one strategy. forecast=None -> perfect foresight."""
    plan, settle = strategy_series(forecast or "perfect_foresight", prices, dam)
    return plan_and_settle(plan, settle, battery, no_simultaneous=no_simultaneous)


# --------------------------------------------------------------------------- comparisons
def all_dailies(prices, battery, dam=None) -> dict[str, pd.DataFrame]:
    """Daily tables for every strategy, restricted to the days they all share."""
    dailies = {n: run_backtest(prices, battery, forecast=None if n == "perfect_foresight" else n,
                               dam=dam) for n in available_strategies(dam)}
    common = None
    for d in dailies.values():
        common = d.index if common is None else common.intersection(d.index)
    return {n: d.loc[common] for n, d in dailies.items()}


def compare_strategies(prices: pd.Series, battery: BatterySpec,
                       dam: pd.Series | None = None) -> pd.DataFrame:
    """Annual profit per strategy ($ and $/kW-year), cycles, and capture rate."""
    dailies = all_dailies(prices, battery, dam)
    out = {}
    for name, daily in dailies.items():
        years = len(daily) / 365.25
        out[name] = {
            "profit_per_year": daily["profit"].sum() / years,
            "profit_per_kw_year": daily["profit"].sum() / years / (battery.power_mw * 1000),
            "cycles_per_year": daily["discharged_mwh"].sum() / years / battery.energy_mwh,
            "paid_at": "day-ahead" if name == "day_ahead_market" else "real-time",
        }
    table = pd.DataFrame(out).T
    for c in ["profit_per_year", "profit_per_kw_year", "cycles_per_year"]:
        table[c] = table[c].astype(float)
    table["capture_rate"] = table["profit_per_year"] / table.loc["perfect_foresight", "profit_per_year"]
    first, last = next(iter(dailies.values())).index[[0, -1]]
    table.attrs["period"] = f"{first.date()} to {last.date()}"
    return table


# strategies that only use information really available at decision time
FULLY_REALISTIC = ["persistence", "weekly_median", "day_ahead_forecast"]


def best_realistic(table: pd.DataFrame) -> tuple[str, float]:
    """Fully realistic strategy with the highest capture rate.

    Excludes perfect_foresight and day_ahead_market (both use prices that are
    not known when the decision is made), so the valuation stays conservative.
    """
    t = table.loc[[n for n in FULLY_REALISTIC if n in table.index]]
    name = t["capture_rate"].astype(float).idxmax()
    return name, float(t.loc[name, "capture_rate"])


def annual_profit_by_year(prices: pd.Series, battery: BatterySpec, forecast=None,
                          dam: pd.Series | None = None) -> pd.Series:
    daily = run_backtest(prices, battery, forecast=forecast, dam=dam)
    return daily["profit"].groupby(daily.index.year).sum()


def sensitivity(prices: pd.Series, battery: BatterySpec, param: str, values) -> pd.DataFrame:
    """Perfect-foresight profit as one battery parameter changes.

    param: "duration_h", "round_trip_eff" or "degradation_cost"
    """
    rows = []
    for v in values:
        b = battery.with_duration(v) if param == "duration_h" else replace(battery, **{param: v})
        daily = run_backtest(prices, b)
        years = len(daily) / 365.25
        rows.append({param: v,
                     "profit_per_year": daily["profit"].sum() / years,
                     "profit_per_kw_year": daily["profit"].sum() / years / (b.power_mw * 1000),
                     "profit_per_kwh_year": daily["profit"].sum() / years / (b.energy_mwh * 1000),
                     "cycles_per_year": daily["discharged_mwh"].sum() / years / b.energy_mwh})
    return pd.DataFrame(rows).set_index(param)


def stress_window(prices: pd.Series, battery: BatterySpec, start: str, end: str,
                  dam: pd.Series | None = None) -> pd.DataFrame:
    """Profit during an extreme event (e.g., Winter Storm Uri) vs. the rest of that year."""
    window = (prices.index >= start) & (prices.index < end)
    if not window.any():
        raise ValueError(f"No data between {start} and {end}")
    year = pd.Timestamp(start).year
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rows = {}
    for name in available_strategies(dam):
        try:
            daily = run_backtest(prices, battery, forecast=None if name == "perfect_foresight" else name,
                                 dam=dam)
        except ValueError:
            continue
        in_w = (daily.index >= s) & (daily.index < e)
        if not in_w.any():
            continue
        rows[name] = {"profit_in_window": daily.loc[in_w, "profit"].sum(),
                      "profit_full_year": daily.loc[daily.index.year == year, "profit"].sum()}
    table = pd.DataFrame(rows).T
    table["share_of_year_pct"] = 100 * table["profit_in_window"] / table["profit_full_year"]
    return table
