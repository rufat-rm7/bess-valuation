"""Step 5: Monte Carlo valuation, risk metrics, hedging, and the effect of
extreme (Uri-type) events.

    python scripts/05_valuation.py [--synthetic] [--n-years 200]
    python scripts/05_valuation.py --fit-from 2024    # uses the model from step 4 --fit-from 2024

Needs steps 3 and 4 first (capture rate + fitted price model).
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import add_fit_from, config, get_prices, out_name, parse_args
from bess import BatterySpec
from bess.plots import INK_2, MUTED, SERIES, save
from bess.price_model import PriceModel
from bess.valuation import (breakeven_capex, hedge_frontier, npv_scenarios, risk_metrics,
                            simulate_annual_profits)


def main():
    def extra(ap):
        ap.add_argument("--n-years", type=int, default=config.N_SIM_YEARS)
        add_fit_from(ap)
    args = parse_args(__doc__, extra)
    T, F = config.TABLES, config.FIGURES
    battery = BatterySpec()
    kw = battery.power_mw * 1000

    model_path = config.DATA_PROCESSED / out_name(args, "price_model.pkl", tagged=True)
    capture_path = T / out_name(args, "capture_rate.txt")
    if not model_path.exists():
        fit = f" --fit-from {args.fit_from}" if args.fit_from else ""
        sys.exit(f"ERROR: {model_path.name} not found. Run step 4 first:  "
                 f"python scripts/04_price_model.py{fit}")
    if not capture_path.exists():
        sys.exit(f"ERROR: {capture_path.name} not found. Run step 3 first:  python scripts/03_backtest.py")
    with open(model_path, "rb") as f:
        model: PriceModel = pickle.load(f)
    capture, forecast_name = capture_path.read_text().split()
    capture = float(capture)
    print(f"Capture rate from backtest ({forecast_name}): {capture:.1%}")

    # ---- 1. simulate annual profits
    print(f"\n[1] Simulating {args.n_years} price years and optimizing each...")
    sims = simulate_annual_profits(model, battery, n_years=args.n_years, capture_rate=capture)
    sims.to_csv(T / out_name(args, "simulated_annual_profits.csv", tagged=True), index=False)
    annual = risk_metrics(sims["profit"], name="annual_profit")
    print((annual / np.where(annual.index == "prob_negative", 1, 1e6)).round(3).to_string())
    print("  ($ million, except prob_negative)")

    # ---- 2. NPV / IRR
    print("\n[2] Project NPV and IRR")
    npv = npv_scenarios(sims["profit"].to_numpy(), battery)
    npv_stats = risk_metrics(npv["npv"], name="npv")
    irr_stats = npv["irr"].describe(percentiles=[0.05, 0.5, 0.95])
    be = breakeven_capex(sims["profit"].to_numpy(), battery)
    summary = pd.DataFrame({"value": {
        "capex_$/kWh": config.CAPEX_PER_KWH,
        "expected_npv_$M": npv_stats["mean"] / 1e6,
        "npv_p5_$M": npv_stats["p5 (VaR 5%)"] / 1e6,
        "npv_cvar5_$M": npv_stats["CVaR 5%"] / 1e6,
        "prob_npv_negative": npv_stats["prob_negative"],
        "irr_median": irr_stats["50%"],
        "irr_p5": irr_stats["5%"],
        "irr_p95": irr_stats["95%"],
        "breakeven_capex_$/kWh": be,
    }})
    summary.round(4).to_csv(T / out_name(args, "valuation_summary.csv", tagged=True))
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(npv["npv"] / 1e6, bins=60, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(npv_stats["p5 (VaR 5%)"] / 1e6, color=SERIES[1], lw=1.6, ls="--")
    ax.annotate("5th percentile", (npv_stats["p5 (VaR 5%)"] / 1e6, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_title("Distribution of project NPV")
    ax.set_xlabel("NPV ($ million)")
    ax.set_ylabel("scenarios")
    save(fig, F / out_name(args, tagged=True, name="11_npv_distribution.png"))

    # ---- 3. hedging with a toll
    print("\n[3] Hedging: share of capacity rented out for a fixed toll")
    hedge = hedge_frontier(sims["profit"].to_numpy(), battery)
    hedge.round(0).to_csv(T / out_name(args, "hedge_frontier.csv", tagged=True))
    print(f"  toll price: ${hedge.attrs['toll_per_kw_month']:.2f}/kW-month")
    print((hedge / 1e6).round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(hedge["CVaR_5"] / 1e6, hedge["expected_profit"] / 1e6, marker="o", ms=5, color=SERIES[0])
    for h in (0.0, 0.5, 1.0):
        row = hedge.loc[np.isclose(hedge.index, h)].iloc[0]
        ax.annotate(f"{h:.0%} tolled", (row["CVaR_5"] / 1e6, row["expected_profit"] / 1e6),
                    xytext=(6, -4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_title("Risk vs. return when part of the battery is tolled")
    ax.set_xlabel("CVaR 5%: average of worst 5% of years ($ million)")
    ax.set_ylabel("expected annual profit ($ million)")
    save(fig, F / out_name(args, tagged=True, name="12_hedge_frontier.png"))

    # ---- 4. how much of the value comes from extreme multi-day events?
    print("\n[4] Value with vs. without long (> 48 h) spike episodes such as Uri")
    prices = get_prices(args)
    if args.fit_from:
        prices = prices[prices.index.year >= args.fit_from]
    model_no_ext = PriceModel.fit(prices, max_episode_hours=48)
    sims_no = simulate_annual_profits(model_no_ext, battery, n_years=max(50, args.n_years // 2),
                                      capture_rate=capture, verbose=False)
    comp = pd.DataFrame({
        "with_extreme_events": risk_metrics(sims["profit"]),
        "without_extreme_events": risk_metrics(sims_no["profit"]),
    })
    comp.loc["mean_$/kW-yr"] = comp.loc["mean"] / kw
    comp.round(3).to_csv(T / out_name(args, "extreme_event_effect.csv", tagged=True))
    print(comp.round(0).to_string())


if __name__ == "__main__":
    main()
