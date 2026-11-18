"""Step 3: historical backtest - compare operating strategies (including the
day-ahead market), sensitivities, LP vs MILP check, Winter Storm Uri stress test.

    python scripts/03_backtest.py [--synthetic]

Saves outputs/tables/capture_rate.txt (best realistic strategy), used by step 5.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import config, get_prices, out_name, parse_args
from bess import BatterySpec, optimize_dispatch
from bess.backtest import (annual_profit_by_year, best_realistic, compare_strategies,
                           sensitivity, stress_window)
from bess.data import load_dam_if_available
from bess.plots import INK_2, MUTED, SERIES, save

LABELS = {
    "perfect_foresight": "Perfect foresight\n(upper bound)",
    "persistence": "Yesterday's\nprices",
    "weekly_median": "7-day\nmedian",
    "day_ahead_market": "Day-ahead market\n(bids clear as planned)",
    "day_ahead_forecast": "Day-ahead as\nRT forecast",
}


def main():
    args = parse_args(__doc__)
    prices = get_prices(args)
    dam = load_dam_if_available(hub=args.hub, synthetic=args.synthetic)
    if dam is not None:
        print(f"  day-ahead: {len(dam):,} hours from {dam.index[0].date()} to {dam.index[-1].date()}")
    battery = BatterySpec()
    T = config.TABLES
    T.mkdir(parents=True, exist_ok=True)
    print(f"\nBattery: {battery.power_mw:.0f} MW / {battery.energy_mwh:.0f} MWh, "
          f"RTE {battery.round_trip_eff:.0%}, degradation ${battery.degradation_cost}/MWh")

    # ---- 1. strategies
    print("\n[1] Strategy comparison")
    strat = compare_strategies(prices, battery, dam=dam)
    strat.round(4).to_csv(T / out_name(args, "strategy_comparison.csv"))
    print(f"  period: {strat.attrs['period']}")
    print(strat.round(3).to_string())
    best, capture = best_realistic(strat)
    (T / out_name(args, "capture_rate.txt")).write_text(f"{capture:.4f}\n{best}\n")
    print(f"  best fully realistic strategy: {best}, capture rate {capture:.1%} (saved for step 5)")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(strat.index)
    vals = strat["profit_per_kw_year"].to_numpy()
    colors = [MUTED if n in ("perfect_foresight", "day_ahead_market") else (SERIES[1] if n == best else SERIES[0]) for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, width=0.62)
    for b, n, v in zip(bars, names, vals):
        cr = strat.loc[n, "capture_rate"]
        ax.annotate(f"${v:.0f}" + ("" if n == "perfect_foresight" else f"\n{cr:.0%}"),
                    (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(names)), [LABELS.get(n, n) for n in names], fontsize=9)
    ax.set_ylabel("$/kW-year")
    ax.set_ylim(0, vals.max() * 1.22)
    ax.set_title("Profit by operating strategy (% = capture rate)")
    ax.grid(axis="x", visible=False)
    save(fig, config.FIGURES / out_name(args, "05a_strategy_comparison.png"))

    # ---- 2. profit by calendar year
    print("\n[2] Profit by year")
    by_year = pd.DataFrame({
        "perfect_foresight": annual_profit_by_year(prices, battery),
        best: annual_profit_by_year(prices, battery, forecast=best, dam=dam),
    })
    by_year.round(0).to_csv(T / out_name(args, "profit_by_year.csv"))
    print((by_year / 1e6).round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(by_year))
    w = 0.38
    for i, col in enumerate(by_year.columns):
        ax.bar(x + (i - 0.5) * w, by_year[col] / 1e6, width=w - 0.03, color=[MUTED, SERIES[1]][i],
               label=LABELS.get(col, col).replace("\n", " "))
    ax.set_xticks(x, by_year.index.astype(str))
    ax.set_title("Annual battery profit")
    ax.set_ylabel("$ million")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, config.FIGURES / out_name(args, "05_profit_by_year.png"))

    # ---- 3. sensitivities
    print("\n[3] Sensitivities (perfect foresight)")
    sens = {
        "duration_h": [1, 2, 4, 6],
        "round_trip_eff": [0.80, 0.85, 0.88, 0.92],
        "degradation_cost": [0, 5, 15, 30, 50],
    }
    for param, values in sens.items():
        tab = sensitivity(prices, battery, param, values)
        tab.round(2).to_csv(T / out_name(args, f"sensitivity_{param}.csv"))
        print(f"  {param}:\n{tab.round(1).to_string()}\n")

    # ---- 4. example week of dispatch (a summer week)
    year = prices.index[-1].year
    aug = prices.index >= pd.Timestamp(f"{year}-08-01", tz=prices.index.tz)
    start = prices.index[aug][0] if aug.any() else prices.index[0]
    week = prices[start:start + pd.Timedelta(hours=7 * 24 - 1)]
    res = optimize_dispatch(week.to_numpy(), battery)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    a1.plot(week.index, week.values, color=SERIES[0])
    a1.set_title("Perfect-foresight dispatch, example week")
    a1.set_ylabel("price $/MWh")
    a2.fill_between(week.index, 0, res.soc, color=SERIES[1], alpha=0.25, lw=0)
    a2.plot(week.index, res.soc, color=SERIES[1])
    a2.set_ylabel("stored energy (MWh)")
    a2.set_title("Stored energy: charges when cheap, discharges at peaks",
                 fontsize=10, fontweight="normal", color=INK_2)
    save(fig, config.FIGURES / out_name(args, "06_example_dispatch.png"))

    # ---- 5. LP vs MILP (does the LP charge and discharge at the same time?)
    print("\n[4] LP vs MILP check on the first 90 days")
    sample = prices.iloc[: 90 * 24].to_numpy()
    t0 = time.time()
    lp = optimize_dispatch(sample, battery)
    t_lp = time.time() - t0
    t0 = time.time()
    mi = optimize_dispatch(sample, battery, no_simultaneous=True)
    t_mi = time.time() - t0
    simult = int(np.sum((lp.charge > 1e-6) & (lp.discharge > 1e-6)))
    lp_p = lp.revenue() - lp.degradation_cost(battery.degradation_cost)
    mi_p = mi.revenue() - mi.degradation_cost(battery.degradation_cost)
    pd.Series({"lp_profit": lp_p, "milp_profit": mi_p, "lp_seconds": t_lp, "milp_seconds": t_mi,
               "lp_simultaneous_hours": simult}).to_csv(T / out_name(args, "lp_vs_milp.csv"))
    print(f"  LP profit   ${lp_p:,.0f} ({t_lp:.2f}s), hours with simultaneous charge+discharge: {simult}")
    print(f"  MILP profit ${mi_p:,.0f} ({t_mi:.2f}s)")

    # ---- 6. stress test: Winter Storm Uri
    print("\n[5] Stress test: Winter Storm Uri window")
    if args.synthetic or prices.index[0] <= pd.Timestamp(config.URI_START, tz=prices.index.tz):
        start, end = ("2021-02-15", "2021-02-19") if args.synthetic else (config.URI_START, config.URI_END)
        st = stress_window(prices, battery, start, end, dam=dam)
        st.round(1).to_csv(T / out_name(args, "stress_uri.csv"))
        print(st.round(0).to_string())
    else:
        print("  data does not include Feb 2021 - add 2021 to YEARS in config.py")


if __name__ == "__main__":
    main()
