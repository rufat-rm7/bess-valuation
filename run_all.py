"""Run the whole project with checkpoints.

    python run_all.py                  # everything, in order
    python run_all.py --resume         # continue from the first stage that has not passed
    python run_all.py --from 3         # start at stage 3
    python run_all.py --only 5         # run one stage
    python run_all.py --quick          # fewer Monte Carlo years (fast check, ~2 min)
    python run_all.py --synthetic      # test the code on fake data
    python run_all.py --download       # re-run the data download even if data exists

Stages:
    0  setup        Python packages + unit tests
    1  data         download ERCOT real-time + day-ahead prices
    2  explore      price statistics and charts
    3  backtest     operating strategies, sensitivities, Uri stress test
    4  price model  fit + validate (all years, and the recent-market version)
    5  valuation    Monte Carlo NPV/IRR, risk, hedging (both versions)
    6  summary      writes outputs/RESULTS.md

After each stage a CHECKPOINT verifies its results:
    PASS  result looks right
    WARN  it ran, but check this (does not stop the run)
    FAIL  something is wrong: the run stops and shows the end of the log

Full logs: outputs/logs/stage<N>_<name>.log
Checkpoint status: outputs/logs/checkpoints.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bess import config  # noqa: E402

LOGS = ROOT / "outputs" / "logs"
STATE = LOGS / "checkpoints.json"
PY = sys.executable


class Checkpoint:
    def __init__(self):
        self.results = []

    def ok(self, msg):
        self.results.append(("PASS", msg))

    def warn(self, msg):
        self.results.append(("WARN", msg))

    def fail(self, msg):
        self.results.append(("FAIL", msg))

    def check(self, cond, msg, else_msg=None, level="FAIL"):
        if cond:
            self.ok(msg)
        else:
            (self.fail if level == "FAIL" else self.warn)(else_msg or msg)
        return cond

    @property
    def status(self):
        levels = [r[0] for r in self.results]
        return "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")


# --------------------------------------------------------------------------- helpers
def run(cmd: list[str], log: Path) -> int:
    """Run a script, show its output live, and save it to the log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            f.write(line)
            print("      " + line, end="")
        return proc.wait()


def fresh(path: Path, since: float) -> bool:
    return path.exists() and path.stat().st_mtime >= since - 1


def table(name, args):
    import pandas as pd
    path = config.TABLES / f"{prefix(args)}{name}"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def prefix(args):
    return "synthetic_" if args.synthetic else ""


def script(name, args, *extra):
    cmd = [PY, str(ROOT / "scripts" / name)]
    if args.synthetic:
        cmd.append("--synthetic")
    return cmd + [str(e) for e in extra]


def recent_year(args):
    return 2022 if args.synthetic else args.recent_year


# --------------------------------------------------------------------------- stages
def stage_setup(args, cp: Checkpoint, log: Path, t0: float):
    cp.check(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}",
             f"Python {sys.version.split()[0]} is too old: need 3.10 or newer")
    for mod in ["numpy", "pandas", "scipy", "matplotlib"]:
        try:
            importlib.import_module(mod)
            cp.ok(f"{mod} installed")
        except ImportError:
            cp.fail(f"{mod} missing: run  pip install -r requirements.txt")
    try:
        importlib.import_module("gridstatus")
        cp.ok("gridstatus installed")
    except ImportError:
        (cp.warn if args.synthetic else cp.fail)("gridstatus missing: run  pip install -r requirements.txt")
    code = run([PY, "-m", "pytest", "-q", str(ROOT / "tests")], log)
    cp.check(code == 0, "unit tests pass", "unit tests FAILED (see log)")


def stage_data(args, cp: Checkpoint, log: Path, t0: float):
    from bess.data import hourly_path, load_prices
    if args.synthetic:
        cp.ok("synthetic mode: no download needed")
        return
    rt_path = hourly_path(args.hub, "rtm")
    dam_path = hourly_path(args.hub, "dam")
    if args.download or not rt_path.exists() or not dam_path.exists():
        code = run(script("01_download_data.py", argparse.Namespace(synthetic=False)) +
                   ["--hub", args.hub], log)
        if code != 0:
            cp.fail("download script failed")
    else:
        cp.ok("data already downloaded (use --download to refresh)")

    if not cp.check(rt_path.exists(), f"real-time prices: {rt_path.name}",
                    f"{rt_path.name} missing"):
        return
    rt = load_prices(hub=args.hub)
    years = sorted(set(rt.index.year))
    cp.check(len(rt) >= 2 * 8760, f"{len(rt):,} hours, years {years[0]}-{years[-1]}",
             f"only {len(rt):,} hours: need at least 2 years")
    cp.check(not rt.isna().any(), "no missing values", "missing values in price data")
    cp.check(-500 <= rt.min() and rt.max() <= 10000,
             f"prices in a sensible range (${rt.min():,.0f} to ${rt.max():,.0f})",
             f"strange prices: min {rt.min():,.0f}, max {rt.max():,.0f}")
    cp.check(2021 in years, "includes 2021 (Winter Storm Uri)",
             "2021 not included: the Uri stress test will be skipped", level="WARN")

    if cp.check(dam_path.exists(), f"day-ahead prices: {dam_path.name}",
                "day-ahead prices missing: day-ahead strategies will be skipped "
                "(run  python scripts/01_download_data.py --markets dam)", level="WARN"):
        dam = load_prices(hub=args.hub, market="dam")
        overlap = len(rt.index.intersection(dam.index)) / len(rt)
        cp.check(overlap >= 0.8, f"day-ahead covers {overlap:.0%} of real-time hours",
                 f"day-ahead covers only {overlap:.0%} of real-time hours", level="WARN")
        corr = rt.reindex(dam.index).corr(dam)
        cp.check(corr > 0.2, f"day-ahead and real-time are correlated (r = {corr:.2f})",
                 f"day-ahead vs real-time correlation is only {corr:.2f}: check the files", level="WARN")


def stage_explore(args, cp: Checkpoint, log: Path, t0: float):
    code = run(script("02_explore_data.py", args, "--hub", args.hub), log)
    if not cp.check(code == 0, "script finished", "script crashed (see log)"):
        return
    p = prefix(args)
    for f in ["01_price_series", "02_daily_profile", "03_daily_spread", "04_price_duration"]:
        cp.check(fresh(config.FIGURES / f"{p}{f}.png", t0), f"figure {f}.png", f"figure {f}.png not created")
    cp.check(fresh(config.TABLES / f"{p}price_stats_by_year.csv", t0), "statistics table",
             "statistics table not created")


def stage_backtest(args, cp: Checkpoint, log: Path, t0: float):
    code = run(script("03_backtest.py", args, "--hub", args.hub), log)
    if not cp.check(code == 0, "script finished", "script crashed (see log)"):
        return
    s = table("strategy_comparison.csv", args)
    if not cp.check(s is not None, "strategy table saved", "strategy table missing"):
        return
    pf = s.loc["perfect_foresight", "profit_per_year"]
    cp.check(pf > 0, f"perfect-foresight profit ${pf / 1e6:.2f}M/yr", "perfect-foresight profit <= 0")
    rt_settled = [n for n in ["persistence", "weekly_median", "day_ahead_forecast"] if n in s.index]
    cp.check(all(s.loc[n, "profit_per_year"] <= pf * 1.0001 for n in rt_settled),
             "perfect foresight is the upper bound for real-time strategies",
             "a real-time strategy beats perfect foresight: optimizer or data problem")
    cap_file = config.TABLES / f"{prefix(args)}capture_rate.txt"
    if cp.check(fresh(cap_file, t0), "capture rate saved for valuation", "capture_rate.txt missing"):
        c, name = cap_file.read_text().split()
        c = float(c)
        cp.check(0 < c <= 1, f"capture rate {c:.0%} ({name})", f"capture rate {c} outside (0, 1]")
        cp.check(c >= 0.3, "capture rate is reasonable", f"capture rate {c:.0%} is low", level="WARN")
    cp.check("day_ahead_forecast" in s.index, "day-ahead strategies included",
             "day-ahead strategies not included (no day-ahead data)", level="WARN")
    m = table("lp_vs_milp.csv", args)
    if m is not None:
        lp, mi = m.loc["lp_profit"].iloc[0], m.loc["milp_profit"].iloc[0]
        cp.check(abs(lp - mi) <= 0.01 * abs(mi) + 1, "LP and MILP give the same profit",
                 f"LP and MILP differ ({lp:,.0f} vs {mi:,.0f}): consider no_simultaneous=True",
                 level="WARN")
    st = table("stress_uri.csv", args)
    cp.check(st is not None, "Uri stress test saved", "Uri stress test not run", level="WARN")


def _validate_model(v, cp, label):
    real, sim = v["real"], v["simulated"]
    neg_ok = abs(sim["pct_negative"] - real["pct_negative"]) <= max(1.0, 0.5 * real["pct_negative"])
    cp.check(neg_ok, f"{label}: negative-price share {sim['pct_negative']:.1f}% vs {real['pct_negative']:.1f}% real",
             f"{label}: negative-price share {sim['pct_negative']:.1f}% vs {real['pct_negative']:.1f}% real",
             level="WARN")
    for stat, tol in [("median", 0.2), ("p95", 0.25), ("mean_daily_spread", 0.25)]:
        rel = abs(sim[stat] / real[stat] - 1) if real[stat] else 0
        cp.check(rel <= tol, f"{label}: {stat} within {tol:.0%} ({sim[stat]:.1f} vs {real[stat]:.1f})",
                 f"{label}: {stat} off by {rel:.0%} ({sim[stat]:.1f} vs {real[stat]:.1f})", level="WARN")


def stage_model(args, cp: Checkpoint, log: Path, t0: float):
    ry = recent_year(args)
    for extra, label in (([], "all years"), (["--fit-from", ry], f"{ry}+")):
        code = run(script("04_price_model.py", args, "--hub", args.hub, *extra), log)
        if not cp.check(code == 0, f"{label}: script finished", f"{label}: script crashed (see log)"):
            return
        tag = f"_from{ry}" if extra else ""
        v = table(f"price_model_validation{tag}.csv", args)
        if cp.check(v is not None, f"{label}: model saved and validated", f"{label}: validation missing"):
            _validate_model(v, cp, label)


def stage_valuation(args, cp: Checkpoint, log: Path, t0: float):
    ry = recent_year(args)
    n = 50 if args.quick else config.N_SIM_YEARS
    for extra, label in (([], "all years"), (["--fit-from", ry], f"{ry}+")):
        code = run(script("05_valuation.py", args, "--hub", args.hub, "--n-years", n, *extra), log)
        if not cp.check(code == 0, f"{label}: script finished", f"{label}: script crashed (see log)"):
            return
        tag = f"_from{ry}" if extra else ""
        v = table(f"valuation_summary{tag}.csv", args)
        if not cp.check(v is not None, f"{label}: valuation saved", f"{label}: valuation missing"):
            continue
        v = v["value"]
        cp.check(0 <= v["prob_npv_negative"] <= 1, f"{label}: expected NPV ${v['expected_npv_$M']:.1f}M, "
                 f"P(NPV<0) {v['prob_npv_negative']:.0%}", f"{label}: invalid probability")
        cp.check(v["breakeven_capex_$/kWh"] > 0, f"{label}: break-even capex ${v['breakeven_capex_$/kWh']:.0f}/kWh",
                 f"{label}: break-even capex <= 0 (profit does not cover fixed O&M)", level="WARN")
        sims = table(f"simulated_annual_profits{tag}.csv", args)
        if sims is not None:
            cp.check(len(sims) == n, f"{label}: {n} simulated years", f"{label}: {len(sims)} of {n} years")
    if args.quick:
        cp.warn("--quick used: re-run stage 5 without --quick for final numbers")


def stage_summary(args, cp: Checkpoint, log: Path, t0: float):
    code = run(script("06_summary.py", args, "--hub", args.hub, "--recent-year", recent_year(args)), log)
    if not cp.check(code == 0, "script finished", "script crashed (see log)"):
        return
    path = ROOT / "outputs" / f"{prefix(args)}RESULTS.md"
    if cp.check(fresh(path, t0), f"outputs/{path.name} written", "RESULTS.md not written"):
        cp.check("Key findings" in path.read_text(), "key findings included",
                 "RESULTS.md has no key findings (missing tables?)", level="WARN")


STAGES = [
    ("setup", stage_setup),
    ("data", stage_data),
    ("explore", stage_explore),
    ("backtest", stage_backtest),
    ("price_model", stage_model),
    ("valuation", stage_valuation),
    ("summary", stage_summary),
]


# --------------------------------------------------------------------------- main
def load_state():
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=0, help="first stage to run")
    ap.add_argument("--only", type=int, help="run only this stage")
    ap.add_argument("--resume", action="store_true", help="start at the first stage not yet passed")
    ap.add_argument("--quick", action="store_true", help="50 Monte Carlo years instead of the config value")
    ap.add_argument("--synthetic", action="store_true", help="fake data, to test the code")
    ap.add_argument("--download", action="store_true", help="force the data download")
    ap.add_argument("--hub", default=config.HUB)
    ap.add_argument("--recent-year", type=int, default=config.RECENT_FROM)
    args = ap.parse_args()

    mode = "synthetic" if args.synthetic else "real"
    state = load_state()
    mstate = state.setdefault(mode, {})
    if args.only is not None:
        todo = [args.only]
    elif args.resume:
        first = next((i for i, (n, _) in enumerate(STAGES)
                      if mstate.get(n, {}).get("status") not in ("PASS", "WARN")), None)
        if first is None:
            print("All stages already passed. Use --from N to re-run.")
            return
        todo = list(range(first, len(STAGES)))
    else:
        todo = list(range(args.start, len(STAGES)))

    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"Running stages {todo} ({mode} data)" + (" - QUICK MODE" if args.quick else ""))
    summary = []
    for i in todo:
        name, fn = STAGES[i]
        log = LOGS / f"{'synthetic_' if args.synthetic else ''}stage{i}_{name}.log"
        log.write_text("")
        print(f"\n{'=' * 70}\n[Stage {i}/{len(STAGES) - 1}] {name}\n{'=' * 70}")
        t0 = time.time()
        cp = Checkpoint()
        try:
            fn(args, cp, log, t0)
        except Exception as e:  # a checkpoint itself crashed
            cp.fail(f"checkpoint error: {type(e).__name__}: {e}")
        dt = time.time() - t0

        print(f"\n  CHECKPOINT {i} ({name}):")
        for level, msg in cp.results:
            print(f"    [{level}] {msg}")
        print(f"  => {cp.status}  ({dt:.0f}s, log: {log.relative_to(ROOT)})")
        mstate[name] = {"status": cp.status, "seconds": round(dt, 1),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "messages": [f"[{lv}] {m}" for lv, m in cp.results]}
        STATE.write_text(json.dumps(state, indent=2))
        summary.append((i, name, cp.status, dt))

        if cp.status == "FAIL":
            print(f"\nStopped at stage {i} ({name}). Last lines of the log:\n")
            print("".join(log.read_text().splitlines(keepends=True)[-25:]))
            print(f"Fix the problem, then continue with:  python run_all.py --resume"
                  + (" --synthetic" if args.synthetic else ""))
            break

    print(f"\n{'=' * 70}\nSUMMARY")
    for i, name, status, dt in summary:
        print(f"  stage {i}  {name:<12} {status:<5} {dt:6.0f}s")
    if summary and summary[-1][2] != "FAIL" and summary[-1][0] == len(STAGES) - 1:
        print(f"\nDone. Read outputs/{'synthetic_' if args.synthetic else ''}RESULTS.md")
    sys.exit(1 if any(s == "FAIL" for _, _, s, _ in summary) else 0)


if __name__ == "__main__":
    main()
