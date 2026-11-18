"""Step 1: download ERCOT prices (real-time and day-ahead) and save clean hourly series.

    python scripts/01_download_data.py                       # years and hub from config.py
    python scripts/01_download_data.py --markets rtm         # real-time only
    python scripts/01_download_data.py --years 2023 2024 --hub HB_WEST
    python scripts/01_download_data.py --excel data/raw/*.xlsx --markets dam   # backup: files downloaded by hand

Outputs:
    data/processed/prices_hourly_<HUB>.csv        real-time (required)
    data/processed/prices_hourly_dam_<HUB>.csv    day-ahead (optional, for the day-ahead strategies)

Years already downloaded are cached in data/raw and are not downloaded again.
"""

import argparse
import sys
from pathlib import Path

from _common import config
from bess.data import download_ercot, load_ercot_excel_files, save_hourly, summary_stats, to_hourly

NAMES = {"rtm": "real-time", "dam": "day-ahead"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=config.YEARS)
    ap.add_argument("--hub", default=config.HUB)
    ap.add_argument("--markets", nargs="+", choices=["rtm", "dam"], default=["rtm", "dam"])
    ap.add_argument("--excel", nargs="+", type=Path,
                    help="ERCOT yearly Excel files downloaded by hand (one market per run)")
    args = ap.parse_args()

    failed = []
    for market in args.markets:
        print(f"\n=== {NAMES[market]} prices, {args.hub} ===")
        try:
            if args.excel:
                raw = load_ercot_excel_files(args.excel, hub=args.hub)
            else:
                raw = download_ercot(years=args.years, hub=args.hub, market=market)
        except Exception as e:                       # keep going: day-ahead is optional
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(market)
            continue
        print(f"  {len(raw):,} rows")
        hourly = to_hourly(raw)
        path = save_hourly(hourly, hub=args.hub, market=market)
        print(f"  saved {path.relative_to(config.ROOT)}")
        print(summary_stats(hourly)[["mean", "median", "p99", "max", "pct_negative"]].round(2).to_string())

    if "rtm" in failed:
        print("\nReal-time download failed. Try again later, or download the files by hand "
              "(see README) and use --excel.")
        sys.exit(1)
    if "dam" in failed:
        print("\nDay-ahead download failed: the project still runs, without the day-ahead "
              "strategies. Re-run with --markets dam to try again.")


if __name__ == "__main__":
    main()
