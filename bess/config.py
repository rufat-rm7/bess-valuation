"""Project-wide settings. Change values here instead of inside the code.

Every number below is an assumption. In the report, list them in a table
and explain where each one comes from (and run sensitivities on the
important ones).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

# ---------------------------------------------------------------- market data
HUB = "HB_HOUSTON"            # ERCOT settlement point (try HB_WEST: more volatile)
YEARS = [2021, 2022, 2023, 2024, 2025]   # 2021 includes Winter Storm Uri (Feb 2021)
TIMEZONE = "US/Central"

# ERCOT price limits. Check the current values on ercot.com before the report:
# the system-wide offer cap was lowered from $9,000 to $5,000/MWh after Uri.
PRICE_CAP = 5000.0            # $/MWh, used to clip simulated prices
PRICE_FLOOR = -250.0          # $/MWh

URI_START = "2021-02-13"
URI_END = "2021-02-20"

# ---------------------------------------------------------------- battery
POWER_MW = 100.0
DURATION_H = 2.0              # energy = power * duration -> 200 MWh
ROUND_TRIP_EFF = 0.88
SOC_MIN_FRAC = 0.05
SOC_MAX_FRAC = 0.95
SOC_START_FRAC = 0.50
DEGRADATION_COST = 15.0       # $/MWh discharged (wear cost of cycling)

# ---------------------------------------------------------------- finance
CAPEX_PER_KWH = 300.0         # $/kWh installed (all-in). Look up a current estimate.
FIXED_OM_PER_KW_YR = 10.0     # $/kW-year
LIFE_YEARS = 15
DISCOUNT_RATE = 0.08
CAPACITY_FADE_PER_YR = 0.02   # usable capacity lost each year

# ---------------------------------------------------------------- Monte Carlo
N_SIM_YEARS = 200             # simulated price years (raise to 500+ for final run)
N_NPV_SCENARIOS = 5000        # project-life scenarios built from the simulated years
SEED = 42

# "today's market" scenario: price model fitted on this year onward (--fit-from)
RECENT_FROM = 2024
