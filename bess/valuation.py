"""Monte Carlo valuation, risk metrics and hedging.

Steps:
  1. simulate_annual_profits: simulate many price years, optimize the battery
     on each one, record that year's profit. Perfect foresight on simulated
     prices overstates profit, so multiply by the capture rate measured in
     the historical backtest (forecast profit / perfect profit).
  2. npv_scenarios: build project-life scenarios by drawing a random
     simulated year for each year of the project's life, with capacity fade,
     fixed O&M and discounting. Gives a distribution of NPV and IRR.
  3. risk_metrics: mean, percentiles, VaR / CVaR, probability of loss.
  4. hedge_frontier: rent out part of the battery for a fixed fee (a "toll")
     and see how expected profit and downside risk change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config
from .battery import BatterySpec
from .optimizer import optimize_dispatch
from .price_model import PriceModel


# --------------------------------------------------------------------------- step 1
def simulate_annual_profits(model: PriceModel, battery: BatterySpec,
                            n_years: int = config.N_SIM_YEARS, capture_rate: float = 1.0,
                            seed: int = config.SEED, verbose: bool = True) -> pd.DataFrame:
    """One row per simulated year: perfect-foresight profit and realistic profit."""
    rows = []
    for i, prices in enumerate(model.simulate_years(n_years, seed=seed)):
        res = optimize_dispatch(prices.to_numpy(), battery)
        profit = res.revenue() - res.degradation_cost(battery.degradation_cost)
        rows.append({"perfect_profit": profit,
                     "profit": profit * capture_rate,
                     "cycles": res.energy_discharged() / battery.energy_mwh,
                     "mean_price": prices.mean(),
                     "max_price": prices.max()})
        if verbose and (i + 1) % max(1, n_years // 10) == 0:
            print(f"    simulated {i + 1}/{n_years} years")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 2
def irr(cash_flows: np.ndarray) -> float:
    """Internal rate of return (nan if it does not exist in [-99%, 100%])."""
    npv = lambda r: np.sum(cash_flows / (1 + r) ** np.arange(len(cash_flows)))
    try:
        return brentq(npv, -0.99, 1.0)
    except ValueError:
        return np.nan


def npv_scenarios(annual_profits: np.ndarray, battery: BatterySpec,
                  capex_per_kwh: float = config.CAPEX_PER_KWH,
                  fixed_om_per_kw_yr: float = config.FIXED_OM_PER_KW_YR,
                  life_years: int = config.LIFE_YEARS,
                  discount_rate: float = config.DISCOUNT_RATE,
                  fade_per_yr: float = config.CAPACITY_FADE_PER_YR,
                  n_scenarios: int = config.N_NPV_SCENARIOS,
                  seed: int = config.SEED) -> pd.DataFrame:
    """Project NPV and IRR distribution.

    Each scenario draws `life_years` years at random from annual_profits.
    Capacity fade is applied by scaling profit by (1 - fade)^year. That is an
    approximation (profit is not exactly proportional to capacity); check it
    with the duration sensitivity table from the backtest.
    """
    rng = np.random.default_rng(seed)
    capex = capex_per_kwh * battery.energy_mwh * 1000
    om = fixed_om_per_kw_yr * battery.power_mw * 1000
    years = np.arange(1, life_years + 1)
    fade = (1 - fade_per_yr) ** (years - 1)
    disc = (1 + discount_rate) ** -years

    draws = rng.choice(np.asarray(annual_profits), size=(n_scenarios, life_years))
    cash = draws * fade - om                                    # $ per year
    npv = -capex + cash @ disc
    irrs = np.array([irr(np.r_[-capex, cf]) for cf in cash[: min(n_scenarios, 2000)]])
    out = pd.DataFrame({"npv": npv})
    out["irr"] = np.nan
    out.loc[: len(irrs) - 1, "irr"] = irrs
    return out


def breakeven_capex(annual_profits: np.ndarray, battery: BatterySpec, **kw) -> float:
    """Capex ($/kWh) at which the EXPECTED NPV is zero."""
    base = npv_scenarios(annual_profits, battery, capex_per_kwh=0.0, **kw)["npv"].mean()
    return base / (battery.energy_mwh * 1000)


# --------------------------------------------------------------------------- step 3
def risk_metrics(x, alpha: float = 0.05, name: str = "") -> pd.Series:
    """Summary of a profit or NPV distribution. Higher = better.

    VaR_5%  : the 5th percentile outcome (95% of outcomes are better)
    CVaR_5% : the average of the worst 5% of outcomes (expected shortfall)
    """
    x = np.sort(np.asarray(x, dtype=float))
    q = np.quantile(x, alpha)
    return pd.Series({
        "mean": x.mean(),
        "std": x.std(),
        "p5 (VaR 5%)": q,
        "CVaR 5%": x[x <= q].mean(),
        "median": np.median(x),
        "p95": np.quantile(x, 0.95),
        "prob_negative": (x < 0).mean(),
    }, name=name)


# --------------------------------------------------------------------------- step 4
def hedge_frontier(annual_profits: np.ndarray, battery: BatterySpec,
                   toll_per_kw_month: float | None = None, risk_premium: float = 0.10,
                   fractions=np.linspace(0, 1, 11)) -> pd.DataFrame:
    """Merchant vs. tolled battery.

    A toll: a buyer (e.g., a utility) pays a fixed $/kW-month for control of a
    fraction h of the battery, and keeps the market profit from it.
    Annual profit = (1 - h) * merchant profit + h * toll payments.

    If no toll price is given, it is set to the expected merchant profit
    minus a risk premium (the buyer wants to be paid for taking the risk).
    Merchant profit is assumed to scale linearly with the fraction kept.
    """
    merchant = np.asarray(annual_profits, dtype=float)
    kw = battery.power_mw * 1000
    if toll_per_kw_month is None:
        toll_per_kw_month = merchant.mean() * (1 - risk_premium) / kw / 12
    toll_year = toll_per_kw_month * kw * 12
    rows = []
    for h in fractions:
        m = risk_metrics((1 - h) * merchant + h * toll_year)
        rows.append({"hedged_fraction": h, "expected_profit": m["mean"],
                     "p5": m["p5 (VaR 5%)"], "CVaR_5": m["CVaR 5%"], "std": m["std"]})
    out = pd.DataFrame(rows).set_index("hedged_fraction")
    out.attrs["toll_per_kw_month"] = toll_per_kw_month
    return out
