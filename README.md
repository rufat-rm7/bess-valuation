# Grid-Scale Battery Storage: Dispatch Optimization, Stochastic Valuation and Risk

How much is a 100 MW / 200 MWh battery in the Texas power market (ERCOT) worth,
how should it operate, and how risky is its income?

This project combines **electrical engineering** (battery physics: power limits,
state of charge, efficiency losses, degradation) with **financial engineering**
(stochastic price modeling, Monte Carlo valuation, VaR/CVaR, hedging).

**Key results** (ERCOT Houston hub, 2021–2025, 100 MW / 200 MWh battery):

- With perfect foresight the battery earns **$81/kW-year**; planning on public day-ahead prices captures **60%** of that ($48/kW-year), vs. 36% with yesterday's prices.
- Perfect-foresight profit fell from **$13.1M in 2023 to $2.8M in 2025** as daily price spreads collapsed.
- One week of **Winter Storm Uri** produced 42% of 2021's perfect-foresight profit, while naive forecasts lost money that week.
- On energy arbitrage alone the project breaks even at **$125/kWh** capex (2021–2025 prices) or **$32/kWh** (2024–2025 prices), vs. $300/kWh assumed.

Full numbers: [`outputs/RESULTS.md`](outputs/RESULTS.md) · Report: [`report/BESS_ERCOT_report.pdf`](report/BESS_ERCOT_report.pdf)

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

That one command runs the whole project (about 3 minutes plus the first-time
download) and writes **`outputs/RESULTS.md`**: every key number, table and chart
in one file, plus a filled-in resume bullet.

### Checkpoints

`run_all.py` runs 7 stages. After each one, a **checkpoint** checks the results:

| Stage | What it does | What the checkpoint verifies |
|---|---|---|
| 0 setup | packages + unit tests | Python version, packages installed, 10 tests pass |
| 1 data | download ERCOT real-time + day-ahead prices | ≥ 2 years, no gaps, sensible price range, 2021 included, day-ahead lines up with real-time |
| 2 explore | statistics and charts | all figures and the statistics table created |
| 3 backtest | strategies, sensitivities, Uri | perfect foresight is the upper bound, capture rate in (0, 100%], LP = MILP, stress test saved |
| 4 price model | fit + validate (all years and 2024+) | simulated vs real: negative-price share, median, p95, daily spread |
| 5 valuation | Monte Carlo NPV/IRR, risk, hedging | results saved, probabilities valid, break-even capex, number of simulated years |
| 6 summary | writes `outputs/RESULTS.md` | file written with key findings |

Each check prints **PASS**, **WARN** (it ran, but look at this) or **FAIL**
(the run stops and shows the end of the log). Full logs are in `outputs/logs/`.

```bash
python run_all.py --resume       # after fixing a FAIL: continue where it stopped
python run_all.py --from 3       # re-run from stage 3 (e.g. after changing config.py)
python run_all.py --only 5       # run a single stage
python run_all.py --quick        # 50 Monte Carlo years instead of 200 (fast check)
python run_all.py --synthetic    # test the code on fake data (never report these numbers)
python run_all.py --download     # download the data again
```

### Running steps one by one
Each stage is also a normal script, if you want to run or study one step:

```bash
python scripts/01_download_data.py             # real-time + day-ahead prices (cached after the first time)
python scripts/02_explore_data.py              # data statistics + figures
python scripts/03_backtest.py                  # strategies, sensitivities, LP vs MILP, Uri
python scripts/04_price_model.py               # fit + validate the price model (all years)
python scripts/04_price_model.py --fit-from 2024   # "today's market" version
python scripts/05_valuation.py                 # Monte Carlo NPV, IRR, VaR/CVaR, hedging
python scripts/05_valuation.py --fit-from 2024
python scripts/06_summary.py                   # writes outputs/RESULTS.md
```
`--fit-from` runs save their files with a `_from2024` suffix, so both scenarios
can be compared side by side.

### If the download fails
ERCOT sometimes changes its website. Download the yearly files by hand from the
ERCOT website: report **NP6-785-ER** (real-time, "Historical RTM Load Zone and Hub
Prices") and **NP4-180-ER** (day-ahead, "Historical DAM Load Zone and Hub Prices").
Put them in `data/raw/` and run, for each market:
```bash
python scripts/01_download_data.py --markets rtm --excel data/raw/rtm_*.xlsx
python scripts/01_download_data.py --markets dam --excel data/raw/dam_*.xlsx
```
The project still runs without day-ahead prices; only the day-ahead strategies are skipped.

All assumptions (battery size, efficiency, capex, discount rate...) are in
`bess/config.py`. After changing one, run `python run_all.py --from 3`.

---

## Project structure

```
bess/
  config.py        all assumptions in one place
  data.py          ERCOT download (real-time + day-ahead), cleaning, synthetic test data
  battery.py       battery physical model (BatterySpec)
  optimizer.py     dispatch linear program (LP / MILP, HiGHS solver)
  backtest.py      strategies (plan on one price, paid at another), sensitivities, stress test
  price_model.py   stochastic hourly price model (fit + simulate)
  valuation.py     Monte Carlo profits, NPV/IRR, VaR/CVaR, toll hedging
  plots.py         chart style
scripts/           01-06, the stages (run in order, or all at once with run_all.py)
run_all.py         runs every stage with checkpoints
tests/             unit tests for the optimizer and price model
outputs/           RESULTS.md, figures/, tables/, logs/ (created when you run)
report/            project report (PDF)
```

---

## Method

### 1. Battery model (electrical engineering)
| Symbol | Meaning | Default |
|---|---|---|
| P | max charge/discharge power | 100 MW |
| E | energy capacity | 200 MWh (2-hour battery) |
| η_rt | round-trip efficiency, split as η_c = η_d = √η_rt | 88% |
| s_min, s_max | state-of-charge limits | 5% - 95% of E |
| k | degradation cost per MWh discharged | $15/MWh |

### 2. Dispatch optimization (linear program)
For each day (24 hours, Δt = 1 h):

```
maximize    Σ_t p_t (d_t − c_t) Δt  −  k Σ_t d_t Δt
subject to  s_t = s_{t−1} + η_c c_t Δt − d_t Δt / η_d      energy balance
            s_min ≤ s_t ≤ s_max                              state of charge
            0 ≤ c_t ≤ P,  0 ≤ d_t ≤ P                        power limits
            s_start = s_end = 50% of E                       daily cycle
```
All days are stacked into one sparse LP and solved with HiGHS (a full year in
~0.2 s). Optional binary variables forbid charging and discharging in the
same hour (MILP), which matters when prices are negative.

### 3. Backtests: operating strategies
Every strategy is "plan on one price series, get paid on another":

| Strategy | Optimize on | Paid at | Realistic? |
|---|---|---|---|
| Perfect foresight | actual real-time prices | real-time | no: upper bound |
| Yesterday's prices | real-time prices 24 h earlier | real-time | yes |
| 7-day median | median of the last 7 days | real-time | yes |
| Day-ahead market | day-ahead prices | day-ahead | optimistic: assumes bids clear as planned |
| Day-ahead as forecast | day-ahead prices | real-time | yes: day-ahead prices are public the day before |

**Capture rate** = strategy profit / perfect-foresight profit (same days). The
valuation uses the best *fully realistic* strategy, to stay conservative.

### 4. Stochastic price model
```
p_t = f(t) · exp(D_day) · (1 + H_t) + S_t     (clipped to the ERCOT price floor and cap)
```
- f(t): median price by quarter × weekday/weekend × hour
- D: log daily price level, AR(1) across days around its mean (multi-day weather
  persistence). Logs keep the daily level positive and right-skewed, like real prices.
- H: hourly relative noise, AR(1) across hours (discrete Ornstein–Uhlenbeck)
- S: spike episodes ($/MWh); start probability depends on quarter and hour, and the
  whole shape (height and duration) is resampled from spikes seen in the data
- Shocks are bootstrapped from fitted residuals (keeps fat tails)

Validated by comparing simulated vs. real statistics (percentiles, % negative
prices, daily spread, autocorrelation) and charts. An earlier additive version produced 4x too many negative prices; the
multiplicative/log form fixed it.

### 5. Valuation and risk
- Simulate N years → optimize each (perfect foresight on simulated prices) →
  annual profit × capture rate of the best fully realistic strategy
- Build 5,000 project-life scenarios (15 years, 2%/year capacity fade, fixed O&M,
  8% discount rate) → NPV and IRR distributions, break-even capex
- Risk: VaR 5% and CVaR 5% (expected shortfall) of annual profit and NPV
- Hedging: rent a fraction h of the battery for a fixed toll ($/kW-month);
  show expected profit vs. CVaR as h goes from 0% to 100%
- Extreme events: value with vs. without Uri-type (> 48 h) spike episodes

---

## Limitations

1. **Energy arbitrage only.** Real ERCOT batteries earn a large share of their
   revenue from **ancillary services** (regulation, reserves). Leaving them out
   means NPV is probably *understated*; say so, and treat it as future work.
2. Price-taker assumption: the battery does not move prices.
3. Years are independent; no trend in gas prices, demand, or battery buildout
   (more batteries on the grid shrink spreads over time).
4. Degradation is a linear cost per MWh, not an electrochemical model.
5. Capacity fade scales profit proportionally (approximation).
6. Hourly averages hide 15-minute price spikes, so real-time value is understated.
7. The day-ahead market strategy assumes the battery's bids clear exactly as
   planned; that is why the valuation uses "day-ahead as forecast" instead.

---

## Future work
- **Better forecast**: gradient boosting or ridge regression on lagged prices,
  day-ahead prices and calendar features. Report how the capture rate improves.
- **Compare hubs**: HB_HOUSTON vs HB_WEST (more renewables, more negative prices):
  set `HUB` in `config.py` and run `python run_all.py --download`.
- **Ancillary services**: add a reserve product to the LP (capacity held back
  for a fixed $/MW-h payment). This is the biggest missing revenue stream.
- **Solar + storage**: add a solar profile and show how the battery shifts solar
  energy into the evening peak.
