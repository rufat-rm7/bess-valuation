"""Helpers shared by the numbered scripts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `bess` importable

from bess import config            # noqa: E402
from bess.data import load_prices  # noqa: E402


def parse_args(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic prices to test the code (results are NOT real)")
    ap.add_argument("--hub", default=config.HUB)
    if extra:
        extra(ap)
    return ap.parse_args()


def get_prices(args):
    prices = load_prices(hub=args.hub, synthetic=args.synthetic)
    if args.synthetic:
        print("*** SYNTHETIC DATA: for testing only, do not report these numbers ***")
    print(f"  {len(prices):,} hours from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def out_name(args, name: str, tagged: bool = False) -> str:
    """Prefix outputs from synthetic runs so they are never mixed up with real ones.

    tagged=True also adds the --fit-from year (e.g. price_model_from2024.pkl),
    so the "all years" and "recent market" runs keep separate files.
    """
    if tagged and getattr(args, "fit_from", None):
        stem, dot, ext = name.rpartition(".")
        name = f"{stem}_from{args.fit_from}{dot}{ext}"
    return f"synthetic_{name}" if args.synthetic else name


def add_fit_from(ap):
    ap.add_argument("--fit-from", type=int, default=None,
                    help="only use prices from this year on (e.g. 2024 = today's market)")
