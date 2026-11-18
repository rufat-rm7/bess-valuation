"""Load ERCOT prices and turn them into clean hourly series.

Two markets:
  rtm  real-time market, 15-minute prices (report NP6-785-ER)
  dam  day-ahead market, hourly prices set the day before delivery (report NP4-180-ER)

Ways to get prices:
  1. download_ercot()          - automatic download with the gridstatus library
  2. load_ercot_excel_files()  - if gridstatus breaks: download ERCOT's yearly
                                 files by hand and point to them
  3. synthetic_prices() / synthetic_dam()
                               - fake but realistic prices, ONLY for testing the
                                 code. Never report results from synthetic data.

Time convention: everything is converted to fixed Central Standard Time
(UTC-6, no daylight saving). That way every day has exactly 24 hours,
which keeps the daily optimization simple. Mention this in the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FIXED_TZ = "Etc/GMT+6"  # = UTC-6 all year (the sign is inverted in this naming scheme)


# --------------------------------------------------------------------------- download
MARKETS = ("rtm", "dam")


def download_ercot(years=config.YEARS, hub=config.HUB, market: str = "rtm",
                   raw_dir: Path = config.DATA_RAW, verbose=True) -> pd.DataFrame:
    """Download settlement point prices for one hub.

    market="rtm" -> 15-minute real-time prices, market="dam" -> hourly day-ahead.
    Each year is cached in data/raw so it is only downloaded once.
    Returns columns: interval_start (UTC), price.
    """
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}")
    from gridstatus import Ercot  # imported here so the rest works without it

    raw_dir.mkdir(parents=True, exist_ok=True)
    ercot = Ercot()
    frames = []
    for year in years:
        cache = raw_dir / f"ercot_{market}_{hub}_{year}.csv"
        if cache.exists():
            if verbose:
                print(f"  {market} {year}: using cached {cache.name}")
            df = pd.read_csv(cache)
        else:
            if verbose:
                print(f"  {market} {year}: downloading from ERCOT (can take a few minutes)...")
            fetch = ercot.get_rtm_spp if market == "rtm" else ercot.get_dam_spp
            full = fetch(year=year)
            df = full.loc[full["Location"] == hub, ["Interval Start", "SPP"]].copy()
            if df.empty:
                raise ValueError(f"No rows for hub {hub}. Available: "
                                 f"{sorted(full['Location'].unique())[:20]}")
            df.columns = ["interval_start", "price"]
            df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
            df.to_csv(cache, index=False)
        df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ercot_excel_files(paths: list[Path], hub=config.HUB) -> pd.DataFrame:
    """Backup loader for ERCOT's yearly Excel files downloaded by hand.

    Real-time files (NP6-785-ER): Delivery Date, Delivery Hour (1-24, hour
    ENDING), Delivery Interval (1-4), Repeated Hour Flag, Settlement Point Name,
    Settlement Point Price.
    Day-ahead files (NP4-180-ER): Delivery Date, Hour Ending ("01:00"...),
    Repeated Hour Flag, Settlement Point, Settlement Point Price.
    The format is detected from the column names.
    """
    frames = []
    for path in paths:
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        name_col = "Settlement Point Name" if "Settlement Point Name" in df else "Settlement Point"
        df = df[df[name_col] == hub]
        if "Delivery Hour" in df:                                   # real-time
            df = df.dropna(subset=["Delivery Hour"])
            hour_end = df["Delivery Hour"].astype(int)
            minutes = (df["Delivery Interval"].astype(int) - 1) * 15
        else:                                                       # day-ahead
            hour_end = df["Hour Ending"].astype(str).str.split(":").str[0].astype(int)
            minutes = 0
        local = (pd.to_datetime(df["Delivery Date"])
                 + pd.to_timedelta(hour_end - 1, unit="h")
                 + pd.to_timedelta(minutes, unit="min"))
        # Repeated Hour Flag = "Y" marks the second (standard-time) copy of the
        # repeated hour when daylight saving ends. ambiguous=True means "DST".
        ambiguous = (df["Repeated Hour Flag"].astype(str).str.upper() != "Y").to_numpy()
        ts = pd.DatetimeIndex(local).tz_localize(config.TIMEZONE, ambiguous=ambiguous,
                                                 nonexistent="shift_forward")
        frames.append(pd.DataFrame({"interval_start": ts.tz_convert("UTC"),
                                    "price": df["Settlement Point Price"].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- cleaning
def to_hourly(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.Series:
    """15-minute prices -> hourly average price ($/MWh) in fixed UTC-6 time.

    Small gaps (up to max_gap_hours) are filled by linear interpolation;
    the function prints how much was filled so you can report it.
    """
    s = (df.drop_duplicates("interval_start")
           .set_index("interval_start")["price"].sort_index())
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    hourly = s.resample("1h").mean()
    hourly.index = hourly.index.tz_convert(FIXED_TZ)

    # Keep whole days only (in fixed UTC-6 time).
    start = hourly.index[0].ceil("D")
    end = hourly.index[-1].floor("D")
    hourly = hourly[(hourly.index >= start) & (hourly.index < end)]

    n_missing = int(hourly.isna().sum())
    hourly = hourly.interpolate(limit=max_gap_hours)
    n_left = int(hourly.isna().sum())
    print(f"  hourly series: {len(hourly):,} hours, {n_missing} missing, "
          f"{n_missing - n_left} filled by interpolation, {n_left} still missing")
    if n_left:
        hourly = hourly.ffill().bfill()
    hourly.name = "price"
    return hourly


def hourly_path(hub=config.HUB, market: str = "rtm") -> Path:
    # real-time keeps the original file name so older downloads still work
    name = f"prices_hourly_{hub}.csv" if market == "rtm" else f"prices_hourly_{market}_{hub}.csv"
    return config.DATA_PROCESSED / name


def save_hourly(prices: pd.Series, hub=config.HUB, market: str = "rtm") -> Path:
    path = hourly_path(hub, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.rename("price").to_frame().to_csv(path, index_label="time")
    return path


def load_prices(hub=config.HUB, synthetic=False, market: str = "rtm") -> pd.Series:
    """Load a processed hourly series (run scripts/01_download_data.py first).

    synthetic=True returns synthetic prices instead (for testing only).
    """
    if synthetic:
        rt = synthetic_prices()
        return rt if market == "rtm" else synthetic_dam(rt)
    path = hourly_path(hub, market)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/01_download_data.py` first, "
            "or add --synthetic to test the code without real data.")
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert(FIXED_TZ)
    return pd.Series(df["price"].to_numpy(), index=pd.DatetimeIndex(idx), name="price")


def load_dam_if_available(hub=config.HUB, synthetic=False) -> pd.Series | None:
    """Day-ahead prices, or None (with a message) if they were not downloaded."""
    try:
        return load_prices(hub=hub, synthetic=synthetic, market="dam")
    except FileNotFoundError:
        print("  day-ahead prices not found: day-ahead strategies are skipped "
              "(run scripts/01_download_data.py to download them)")
        return None


# --------------------------------------------------------------------------- synthetic
def synthetic_prices(years=(2021, 2022, 2023), seed: int = 7,
                     include_storm=True) -> pd.Series:
    """Fake ERCOT-like hourly prices for testing the code. NOT real data.

    Built from: daily shape (evening peak, solar dip at midday), seasons
    (hot summers), mean-reverting noise, random evening price spikes,
    occasional negative prices, and optionally a fake winter storm in
    mid-February of the first year (to test the stress-test code).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1] + 1}-01-01",
                        freq="1h", tz=FIXED_TZ, inclusive="left")
    n = len(idx)
    hour = idx.hour.to_numpy()
    month = idx.month.to_numpy()
    summer = np.isin(month, [6, 7, 8, 9]).astype(float)
    spring = np.isin(month, [3, 4, 5]).astype(float)

    evening_peak = np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)
    solar_dip = np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)
    night_low = np.exp(-0.5 * ((hour - 4) / 2.5) ** 2)
    base = (32 + 10 * summer
            + (25 + 55 * summer) * evening_peak
            - (8 + 12 * spring) * solar_dip
            - 8 * night_low)

    # multiplicative noise: a slow daily level (weather) and fast hourly noise,
    # both scaled by the base price, like real prices
    n_days = n // 24
    level = np.zeros(n_days)
    e_day = rng.normal(0, 0.12, n_days)
    for d in range(1, n_days):
        level[d] = 0.8 * level[d - 1] + e_day[d]
    hourly = np.zeros(n)
    e_hour = rng.normal(0, 0.08, n)
    for t in range(1, n):
        hourly[t] = 0.7 * hourly[t - 1] + e_hour[t]
    noise = base * (np.exp(np.repeat(level, 24)[:n]) * (1 + hourly) - 1)

    spikes = np.zeros(n)
    p_spike = 0.0015 + 0.02 * summer * evening_peak
    onsets = rng.random(n) < p_spike
    sizes = rng.lognormal(mean=5.3, sigma=1.0, size=n)
    for t in range(1, n):
        spikes[t] = 0.55 * spikes[t - 1] + (sizes[t] if onsets[t] else 0.0)

    neg = (rng.random(n) < 0.012 * spring * solar_dip) * rng.uniform(-40, -15, n)

    p = base + noise + spikes + neg
    if include_storm:
        storm = (idx >= f"{years[0]}-02-15") & (idx < f"{years[0]}-02-19")
        p[storm] = rng.uniform(3500, config.PRICE_CAP, storm.sum())
    p = np.clip(p, config.PRICE_FLOOR, config.PRICE_CAP)
    return pd.Series(p, index=idx, name="price")


def synthetic_dam(rt: pd.Series, seed: int = 11) -> pd.Series:
    """Fake day-ahead prices built from synthetic real-time prices. NOT real data.

    Real day-ahead prices follow the expected daily shape but miss most
    real-time spikes. Here: real-time prices with spikes capped at the
    weekly 97th percentile, plus 10% noise.
    """
    rng = np.random.default_rng(seed)
    cap = rt.rolling(24 * 7, min_periods=24).quantile(0.97).bfill()
    dam = np.minimum(rt, cap) * (1 + rng.normal(0, 0.10, len(rt)))
    return dam.clip(config.PRICE_FLOOR, config.PRICE_CAP).rename("price")


def summary_stats(prices: pd.Series) -> pd.Series:
    """Statistics used to describe the data (and to validate the price model)."""
    x = prices.to_numpy()
    days = x[: len(x) // 24 * 24].reshape(-1, 24)   # consecutive 24-hour blocks
    return pd.Series({
        "mean": x.mean(),
        "std": x.std(),
        "p1": np.percentile(x, 1),
        "p5": np.percentile(x, 5),
        "median": np.median(x),
        "p95": np.percentile(x, 95),
        "p99": np.percentile(x, 99),
        "max": x.max(),
        "pct_negative": 100 * (x < 0).mean(),
        "pct_above_500": 100 * (x > 500).mean(),
        "mean_daily_spread": (days.max(axis=1) - days.min(axis=1)).mean(),
        "acf_lag1": pd.Series(x).autocorr(1),
        "acf_lag24": pd.Series(x).autocorr(24),
    })
