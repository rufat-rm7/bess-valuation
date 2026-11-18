r"""Battery dispatch optimization (linear program, solved with HiGHS via scipy).

Problem for one block of T hours (usually one day, T = 24):

    maximize    sum_t  p_t * (d_t - c_t) * dt  -  k * sum_t d_t * dt
    subject to  s_t = s_{t-1} + eta_c * c_t * dt - d_t * dt / eta_d     (energy balance)
                s_min <= s_t <= s_max                                    (state of charge)
                0 <= c_t <= P,   0 <= d_t <= P                           (power limits)
                s_{-1} = s_0,    s_{T-1} = s_0                           (start/end SoC)

    p_t   price ($/MWh)            c_t, d_t  charge / discharge power (MW)
    s_t   stored energy (MWh)      k         degradation cost ($/MWh discharged)
    eta   one-way efficiencies     dt        hours per step (1.0 for hourly data)

Making the battery start and end every block at the same state of charge
means each day can be solved on its own, and days can be compared fairly.
All days are stacked into ONE sparse LP, which is much faster than solving
them one by one (important for the Monte Carlo).

With negative prices, an LP can "cheat" by charging and discharging at the
same time to burn energy as losses. Set no_simultaneous=True to forbid that
with binary variables (MILP). It is slower; compare both in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .battery import BatterySpec


@dataclass
class DispatchResult:
    charge: np.ndarray      # MW
    discharge: np.ndarray   # MW
    soc: np.ndarray         # MWh at the END of each hour
    prices: np.ndarray      # prices the schedule was optimized on
    dt: float

    @property
    def net_power(self) -> np.ndarray:
        """Positive = selling to the grid, negative = buying."""
        return self.discharge - self.charge

    def revenue(self, settle_prices: np.ndarray | None = None) -> float:
        """Market revenue ($), settled at settle_prices (default: the optimized prices)."""
        p = self.prices if settle_prices is None else np.asarray(settle_prices)
        return float(np.sum(p * self.net_power) * self.dt)

    def degradation_cost(self, k: float) -> float:
        return float(k * self.discharge.sum() * self.dt)

    def energy_discharged(self) -> float:
        return float(self.discharge.sum() * self.dt)

    def to_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame({"price": self.prices, "charge_mw": self.charge,
                             "discharge_mw": self.discharge, "soc_mwh": self.soc},
                            index=index)


def optimize_dispatch(prices, battery: BatterySpec, dt: float = 1.0,
                      block_hours: int = 24, no_simultaneous: bool = False) -> DispatchResult:
    """Optimal schedule for a price series, solved in independent blocks.

    prices       1-D array or Series; length must be a multiple of block_hours/dt
    battery      BatterySpec
    block_hours  length of each independent block (24 = one day)
    """
    p = np.asarray(prices, dtype=float)
    steps = int(round(block_hours / dt))
    n = p.size
    if n == 0 or n % steps:
        raise ValueError(f"length {n} is not a multiple of block size {steps}")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices contain NaN or inf")
    n_blocks = n // steps

    # Variable vector x = [c_0..c_{n-1}, d_0..d_{n-1}, s_0..s_{n-1}] (+ binaries u)
    ec, ed, b = battery.eta_charge, battery.eta_discharge, battery
    ic, id_, is_ = 0, n, 2 * n
    n_var = 3 * n + (n if no_simultaneous else 0)

    # Objective (minimize the negative of profit)
    cost = np.zeros(n_var)
    cost[ic:ic + n] = p * dt
    cost[id_:id_ + n] = -p * dt + b.degradation_cost * dt

    # Energy balance: s_t - s_{t-1} - ec*dt*c_t + dt/ed*d_t = 0  (= s0 at block start)
    t = np.arange(n)
    rows = [t, t, t]
    cols = [is_ + t, ic + t, id_ + t]
    vals = [np.ones(n), -ec * dt * np.ones(n), (dt / ed) * np.ones(n)]
    not_first = t % steps != 0
    rows.append(t[not_first])
    cols.append(is_ + t[not_first] - 1)
    vals.append(-np.ones(not_first.sum()))
    rhs = np.where(not_first, 0.0, b.soc_start)

    # End of each block: s_last = s0
    last = np.arange(steps - 1, n, steps)
    rows.append(n + np.arange(n_blocks))
    cols.append(is_ + last)
    vals.append(np.ones(n_blocks))
    rhs = np.concatenate([rhs, np.full(n_blocks, b.soc_start)])

    A_eq = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n + n_blocks, n_var))

    lower = np.concatenate([np.zeros(2 * n), np.full(n, b.soc_min)])
    upper = np.concatenate([np.full(2 * n, b.power_mw), np.full(n, b.soc_max)])

    if not no_simultaneous:
        res = linprog(cost, A_eq=A_eq, b_eq=rhs, bounds=np.column_stack([lower, upper]),
                      method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        x = res.x
    else:
        # u_t = 1 -> may charge, u_t = 0 -> may discharge
        iu = 3 * n
        lower = np.concatenate([lower, np.zeros(n)])
        upper = np.concatenate([upper, np.ones(n)])
        P = b.power_mw
        A_c = sparse.csr_matrix((np.r_[np.ones(n), -P * np.ones(n)],
                                 (np.r_[t, t], np.r_[ic + t, iu + t])), shape=(n, n_var))
        A_d = sparse.csr_matrix((np.r_[np.ones(n), P * np.ones(n)],
                                 (np.r_[t, t], np.r_[id_ + t, iu + t])), shape=(n, n_var))
        integrality = np.r_[np.zeros(3 * n), np.ones(n)]
        res = milp(cost, integrality=integrality, bounds=Bounds(lower, upper),
                   constraints=[LinearConstraint(A_eq, rhs, rhs),
                                LinearConstraint(A_c, -np.inf, 0),
                                LinearConstraint(A_d, -np.inf, P)])
        if res.status != 0:
            raise RuntimeError(f"MILP failed: {res.message}")
        x = res.x

    charge = np.clip(x[ic:ic + n], 0, None)
    discharge = np.clip(x[id_:id_ + n], 0, None)
    return DispatchResult(charge=charge, discharge=discharge, soc=x[is_:is_ + n],
                          prices=p, dt=dt)
