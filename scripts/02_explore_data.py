"""Step 2: describe the price data (tables + figures for the report's data section).

    python scripts/02_explore_data.py [--synthetic]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import config, get_prices, out_name, parse_args
from bess.data import summary_stats
from bess.plots import MUTED, SERIES, save


def main():
    args = parse_args(__doc__)
    prices = get_prices(args)

    # ---- table: statistics per year
    by_year = prices.groupby(prices.index.year).apply(summary_stats).unstack()
    by_year.loc["all"] = summary_stats(prices)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    by_year.round(2).to_csv(config.TABLES / out_name(args, "price_stats_by_year.csv"))
    print(by_year[["mean", "std", "p99", "max", "pct_negative", "mean_daily_spread"]].round(1))

    # ---- figure 1: full series (symlog axis so spikes and normal prices both show)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(prices.index, prices.values, lw=0.5, color=SERIES[0])
    ax.set_yscale("symlog", linthresh=100)
    ax.set_title(f"{args.hub} hourly real-time price")
    ax.set_ylabel("$/MWh (log scale above 100)")
    save(fig, config.FIGURES / out_name(args, "01_price_series.png"))

    # ---- figure 2: average daily shape by quarter (median, robust to spikes)
    df = pd.DataFrame({"p": prices.values, "hour": prices.index.hour, "q": prices.index.quarter})
    prof = df.groupby(["q", "hour"])["p"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}
    for i, q in enumerate(prof.columns):
        ax.plot(prof.index, prof[q], color=SERIES[i], label=names[q])
    ax.set_title("Median price by hour of day")
    ax.set_xlabel("hour of day (UTC-6)")
    ax.set_ylabel("$/MWh")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper left")
    save(fig, config.FIGURES / out_name(args, "02_daily_profile.png"))

    # ---- figure 3: daily max-min spread (what a battery can earn from)
    days = prices.to_numpy()[: len(prices) // 24 * 24].reshape(-1, 24)
    spread = pd.Series(days.max(1) - days.min(1),
                       index=pd.date_range(prices.index[0].date(), periods=len(days), freq="D"))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(spread.index, spread.values, lw=0.4, color=MUTED, label="daily")
    ax.plot(spread.rolling(30).median(), color=SERIES[0], lw=2, label="30-day median")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_title("Daily price spread (max - min)")
    ax.set_ylabel("$/MWh")
    ax.legend(loc="upper right")
    save(fig, config.FIGURES / out_name(args, "03_daily_spread.png"))

    # ---- figure 4: price duration curve (sorted prices)
    fig, ax = plt.subplots(figsize=(7, 4))
    s = np.sort(prices.values)[::-1]
    ax.plot(100 * np.arange(len(s)) / len(s), s, color=SERIES[0])
    ax.set_yscale("symlog", linthresh=100)
    ax.set_title("Price duration curve")
    ax.set_xlabel("% of hours price is at or above this level")
    ax.set_ylabel("$/MWh")
    save(fig, config.FIGURES / out_name(args, "04_price_duration.png"))


if __name__ == "__main__":
    main()
