"""Step 6: collect every result into one readable file, outputs/RESULTS.md.

    python scripts/06_summary.py [--synthetic] [--recent-year 2024]

Use RESULTS.md as the source for the report, README, resume and interviews.
"""

from datetime import date

import pandas as pd

from _common import config, parse_args
from bess import BatterySpec

NICE = {
    "perfect_foresight": "Perfect foresight (upper bound)",
    "persistence": "Yesterday's prices as forecast",
    "weekly_median": "7-day median as forecast",
    "day_ahead_market": "Day-ahead market (bids clear as planned)",
    "day_ahead_forecast": "Day-ahead prices as real-time forecast",
}


def money(x, unit="M"):
    return f"${x / 1e6:,.1f}M" if unit == "M" else f"${x:,.0f}"


def md_table(df: pd.DataFrame, fmt=None) -> str:
    fmt = fmt or {}
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c, v in row.items():
            f = fmt.get(c)
            cells.append(f(v) if f and pd.notna(v) else (f"{v:,.2f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument("--recent-year", type=int,
                                                          default=config.RECENT_FROM))
    pre = "synthetic_" if args.synthetic else ""
    T, F = config.TABLES, config.FIGURES

    def table(name, **kw):
        path = T / f"{pre}{name}"
        return pd.read_csv(path, index_col=0, **kw) if path.exists() else None

    b = BatterySpec()
    kw = b.power_mw * 1000
    out = []
    w = out.append

    w("# Results summary")
    w(f"_Generated {date.today()} by `scripts/06_summary.py`. Re-run it after any change._\n")
    if args.synthetic:
        w("> **SYNTHETIC DATA: these numbers are fake and only test the code. Do not report them.**\n")

    # ---------------------------------------------------------------- setup
    stats = table("price_stats_by_year.csv")
    strat = table("strategy_comparison.csv")
    period = (f"{stats.index[0]}–{stats.index[-2]}" if stats is not None else "?")
    w("## Setup")
    w(f"- Market: ERCOT, hub **{args.hub}**, hourly real-time prices {period}")
    w(f"- Battery: **{b.power_mw:.0f} MW / {b.energy_mwh:.0f} MWh** ({b.duration_h:.0f}-hour), "
      f"round-trip efficiency {b.round_trip_eff:.0%}, degradation cost ${b.degradation_cost:.0f}/MWh discharged")
    w(f"- Finance: capex ${config.CAPEX_PER_KWH:.0f}/kWh, fixed O&M ${config.FIXED_OM_PER_KW_YR:.0f}/kW-yr, "
      f"{config.LIFE_YEARS}-year life, {config.DISCOUNT_RATE:.0%} discount rate, "
      f"{config.CAPACITY_FADE_PER_YR:.0%}/yr capacity fade\n")

    # ---------------------------------------------------------------- key findings
    val_all = table("valuation_summary.csv")
    val_rec = table(f"valuation_summary_from{args.recent_year}.csv")
    by_year = table("profit_by_year.csv")
    stress = table("stress_uri.csv")
    cap_file = T / f"{pre}capture_rate.txt"
    best_name, capture = None, None
    if cap_file.exists():
        c, best_name = cap_file.read_text().split()
        capture = float(c)

    w("## Key findings")
    if strat is not None:
        pf = strat.loc["perfect_foresight", "profit_per_kw_year"]
        w(f"1. **Upper bound:** with perfect foresight the battery earns **${pf:.0f}/kW-year** "
          f"({money(strat.loc['perfect_foresight', 'profit_per_year'])}/year).")
        if best_name:
            bv = strat.loc[best_name, "profit_per_kw_year"]
            w(f"2. **Realistic operation:** the best fully realistic strategy "
              f"({NICE.get(best_name, best_name).lower()}) earns **${bv:.0f}/kW-year**, "
              f"a capture rate of **{capture:.0%}**.")
        if "day_ahead_forecast" in strat.index and "persistence" in strat.index:
            w(f"3. **Forecast quality matters:** using day-ahead prices as the forecast captures "
              f"{strat.loc['day_ahead_forecast', 'capture_rate']:.0%}, vs. "
              f"{strat.loc['persistence', 'capture_rate']:.0%} with yesterday's prices.")
    if by_year is not None and len(by_year) > 1:
        col = by_year.columns[0]
        peak = by_year[col].idxmax()
        w(f"4. **Shrinking spreads:** perfect-foresight profit peaked at "
          f"{money(by_year.loc[peak, col])} in {peak} and was {money(by_year[col].iloc[-1])} "
          f"in {by_year.index[-1]}.")
    if stress is not None and "perfect_foresight" in stress.index:
        w(f"5. **Winter Storm Uri:** one week produced "
          f"**{stress.loc['perfect_foresight', 'share_of_year_pct']:.0f}%** of the year's "
          f"perfect-foresight profit.")
    if val_all is not None:
        v = val_all["value"]
        line = (f"6. **Valuation (all years):** expected NPV {money(v['expected_npv_$M'] * 1e6)}, "
                f"P(NPV < 0) = {v['prob_npv_negative']:.0%}, break-even capex "
                f"**${v['breakeven_capex_$/kWh']:.0f}/kWh** vs. ${v['capex_$/kWh']:.0f}/kWh assumed.")
        if val_rec is not None:
            line += (f" With {args.recent_year}+ prices only: break-even "
                     f"**${val_rec['value']['breakeven_capex_$/kWh']:.0f}/kWh**.")
        w(line)
    w("")
    w("_Why NPV can be negative: the model counts only energy arbitrage. Real ERCOT batteries also "
      "earn ancillary-service revenue (regulation, reserves), which is not modeled here._\n")

    # ---------------------------------------------------------------- tables
    if strat is not None:
        w("## Operating strategies")
        t = strat[["profit_per_year", "profit_per_kw_year", "cycles_per_year", "paid_at", "capture_rate"]].copy()
        t.index = [NICE.get(i, i) for i in t.index]
        t.index.name = "strategy"
        w(md_table(t, {"profit_per_year": money, "profit_per_kw_year": lambda x: f"${x:.1f}",
                       "cycles_per_year": lambda x: f"{x:.0f}", "capture_rate": lambda x: f"{x:.0%}"}))
        w("\n![strategies](figures/" + f"{pre}05a_strategy_comparison.png)\n")

    if by_year is not None:
        w("## Profit by year")
        t = by_year.copy()
        t.columns = [NICE.get(c, c) for c in t.columns]
        t.index.name = "year"
        w(md_table(t, {c: money for c in t.columns}))
        w(f"\n![by year](figures/{pre}05_profit_by_year.png)\n")

    if stress is not None:
        w("## Stress test: Winter Storm Uri")
        t = stress.copy()
        t.index = [NICE.get(i, i) for i in t.index]
        t.index.name = "strategy"
        w(md_table(t, {"profit_in_window": money, "profit_full_year": money,
                       "share_of_year_pct": lambda x: f"{x:.0f}%"}))
        w("")

    sens = {p: table(f"sensitivity_{p}.csv") for p in ["duration_h", "round_trip_eff", "degradation_cost"]}
    if all(v is not None for v in sens.values()):
        w("## Sensitivities (perfect foresight)")
        for p, t in sens.items():
            t = t[["profit_per_kw_year", "profit_per_kwh_year", "cycles_per_year"]]
            w(f"**{p}**\n")
            w(md_table(t, {"profit_per_kw_year": lambda x: f"${x:.1f}",
                           "profit_per_kwh_year": lambda x: f"${x:.1f}",
                           "cycles_per_year": lambda x: f"{x:.0f}"}))
            w("")

    for tag, label in (("", "all years"), (f"_from{args.recent_year}", f"{args.recent_year} onward")):
        v = table(f"price_model_validation{tag}.csv")
        if v is None:
            continue
        w(f"## Price model validation ({label})")
        rows = ["mean", "median", "p5", "p95", "p99", "pct_negative", "pct_above_500",
                "mean_daily_spread", "acf_lag24"]
        w(md_table(v.loc[[r for r in rows if r in v.index], ["real", "simulated"]],
                   {"real": lambda x: f"{x:,.2f}", "simulated": lambda x: f"{x:,.2f}"}))
        w(f"\n![distribution](figures/{pre}08_price_distribution{tag}.png)\n")

    if val_all is not None:
        w("## Valuation")
        cols = {"all years": val_all["value"]}
        if val_rec is not None:
            cols[f"{args.recent_year}+ prices"] = val_rec["value"]
        t = pd.DataFrame(cols)
        t.index.name = "metric"
        w(md_table(t, {c: (lambda x: f"{x:,.3f}") for c in t.columns}))
        w(f"\n![npv](figures/{pre}11_npv_distribution.png)\n")

    hedge = table("hedge_frontier.csv")
    if hedge is not None:
        toll = hedge["expected_profit"].iloc[-1] / kw / 12
        h50 = hedge.iloc[(hedge.index.to_series() - 0.5).abs().argmin()]
        h0 = hedge.iloc[0]
        w("## Hedging with a toll")
        w(f"Toll price: ${toll:.2f}/kW-month (expected merchant profit minus a 10% risk premium). "
          f"Tolling 50% of the battery lowers expected profit from {money(h0['expected_profit'])} to "
          f"{money(h50['expected_profit'])} but raises the worst-5% average (CVaR) from "
          f"{money(h0['CVaR_5'])} to {money(h50['CVaR_5'])}.\n")
        w(f"![hedge](figures/{pre}12_hedge_frontier.png)\n")

    ext = table("extreme_event_effect.csv")
    if ext is not None:
        w("## Extreme events")
        a, bb = ext.loc["mean", "with_extreme_events"], ext.loc["mean", "without_extreme_events"]
        w(f"Removing multi-day spike episodes (> 48 h, like Uri) from the model changes expected "
          f"annual profit from {money(a)} to {money(bb)} ({(a - bb) / a:.0%} of value), and the "
          f"95th-percentile year from {money(ext.loc['p95', 'with_extreme_events'])} to "
          f"{money(ext.loc['p95', 'without_extreme_events'])}.\n")

    # ---------------------------------------------------------------- resume
    if strat is not None and val_all is not None and best_name:
        w("## Resume bullet (check the numbers, then copy)")
        years = len(stats) - 1 if stats is not None else "several"
        w(f"> Built a stochastic optimization and valuation model for a {b.power_mw:.0f} MW / "
          f"{b.energy_mwh:.0f} MWh grid battery on {years} years of ERCOT data (Python, linear "
          f"programming, Monte Carlo); found a realistic capture rate of {capture:.0%} of the "
          f"perfect-foresight value, a break-even capex of ${val_all['value']['breakeven_capex_$/kWh']:.0f}/kWh, "
          f"and quantified downside risk (CVaR) and tolling hedges, including a Winter Storm Uri stress test.\n")

    path = config.ROOT / "outputs" / f"{pre}RESULTS.md"
    path.write_text("\n".join(out))
    print(f"Saved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
