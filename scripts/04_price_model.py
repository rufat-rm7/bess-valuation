"""Step 4: fit the stochastic price model and check that simulated prices look real.

    python scripts/04_price_model.py [--synthetic] [--n-check 50]
    python scripts/04_price_model.py --fit-from 2024     # "today's market" version

Saves the fitted model to data/processed/price_model.pkl (used by step 5).
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import add_fit_from, config, get_prices, out_name, parse_args
from bess.data import summary_stats
from bess.plots import INK_2, SERIES, save
from bess.price_model import PriceModel


def acf(x, lags):
    s = pd.Series(np.asarray(x))
    return np.array([s.autocorr(k) for k in lags])


def main():
    def extra(ap):
        ap.add_argument("--n-check", type=int, default=50)
        add_fit_from(ap)
    args = parse_args(__doc__, extra)
    prices = get_prices(args)
    if args.fit_from:
        prices = prices[prices.index.year >= args.fit_from]
        print(f"  fitting on {args.fit_from}-{prices.index[-1].year} only")

    model = PriceModel.fit(prices)
    model.fit_from = args.fit_from        # remembered so step 5 uses the same years
    print("\nFitted parameters")
    print(model.describe().round(3).to_string())
    model.describe().to_csv(config.TABLES / out_name(args, "price_model_params.csv", tagged=True))

    path = config.DATA_PROCESSED / out_name(args, "price_model.pkl", tagged=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  saved model to {path.name}")

    # ---- validation: pooled statistics, real vs simulated
    sims = list(model.simulate_years(args.n_check, seed=123))
    pooled = pd.concat(sims)
    table = pd.DataFrame({"real": summary_stats(prices), "simulated": summary_stats(pooled)})
    # spread of the statistic across simulated years (shows sampling uncertainty)
    per_year = pd.DataFrame([summary_stats(s) for s in sims])
    table["sim_p5_across_years"] = per_year.quantile(0.05)
    table["sim_p95_across_years"] = per_year.quantile(0.95)
    table.round(3).to_csv(config.TABLES / out_name(args, "price_model_validation.csv", tagged=True))
    print("\nValidation (real vs simulated)")
    print(table.round(2).to_string())

    # ---- figure: one real year vs one simulated year
    real_year = prices[prices.index.year == prices.index[-1].year]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharey=True)
    axes[0].plot(real_year.index, real_year.values, lw=0.5, color=SERIES[0])
    axes[0].set_title("Real prices (last year of data)")
    axes[1].plot(sims[0].index, sims[0].values, lw=0.5, color=SERIES[1])
    axes[1].set_title("One simulated year")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=100)
        ax.set_ylabel("$/MWh")
    fig.tight_layout()
    save(fig, config.FIGURES / out_name(args, tagged=True, name="07_real_vs_simulated_path.png"))

    # ---- figure: distribution (log-density so the tail is visible)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.r_[np.linspace(-250, 300, 111), np.geomspace(305, config.PRICE_CAP, 40)]
    ax.hist(prices.values, bins=bins, density=True, histtype="step", lw=1.8, color=SERIES[0], label="real")
    ax.hist(pooled.values, bins=bins, density=True, histtype="step", lw=1.8, color=SERIES[1], label="simulated")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("log")
    ax.set_title("Price distribution")
    ax.set_xlabel("$/MWh")
    ax.set_ylabel("density (log)")
    ax.legend()
    save(fig, config.FIGURES / out_name(args, tagged=True, name="08_price_distribution.png"))

    # ---- figure: autocorrelation
    lags = np.arange(1, 24 * 7 + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, acf(prices.values, lags), color=SERIES[0], label="real")
    ax.plot(lags, np.mean([acf(s.values, lags) for s in sims[:10]], axis=0), color=SERIES[1],
            label="simulated (avg of 10 years)")
    ax.set_xticks(range(0, 24 * 7 + 1, 24))
    ax.set_title("Autocorrelation of hourly prices")
    ax.set_xlabel("lag (hours)")
    ax.legend()
    save(fig, config.FIGURES / out_name(args, tagged=True, name="09_autocorrelation.png"))

    # ---- figure: daily shape
    def profile(s):
        return s.groupby(s.index.hour).median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(profile(prices), color=SERIES[0], label="real")
    ax.plot(profile(pooled), color=SERIES[1], label="simulated")
    ax.set_title("Median price by hour of day")
    ax.set_xlabel("hour")
    ax.set_ylabel("$/MWh")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    save(fig, config.FIGURES / out_name(args, tagged=True, name="10_profile_real_vs_sim.png"))


if __name__ == "__main__":
    main()
