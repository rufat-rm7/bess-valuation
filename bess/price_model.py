r"""Stochastic model of hourly electricity prices.

    p_t = R_day * (f(t) + g(t) * H_t)  +  S_t,   clipped to [price floor, price cap]
    R_day = exp(D_day)

    f(t)   seasonal shape: median price for each (quarter, weekday/weekend, hour)
    g(t)   scale of the swings = max(f, 25% of the typical level). Swings are
           proportional to the price level (calm periods have small swings);
           the floor stops hours with a near-zero normal price from blowing up.
    D_d    log daily price level, AR(1) across days around its mean mu:
               D_d - mu = phi_d (D_{d-1} - mu) + e_d
           (multi-day weather: a hot week stays hot. Working in logs keeps the
           daily level positive and right-skewed, like real prices.)
    H_t    hourly noise relative to g, AR(1) across hours: H_t = phi_h H_{t-1} + u_t
           (an AR(1) is the discrete-time version of an Ornstein-Uhlenbeck process)
    S_t    spike episodes ($/MWh): with probability lambda(quarter, hour) an
           episode starts, and its whole shape (height AND duration) is copied
           from a randomly chosen episode seen in the data.

The shocks e_d and u_t are bootstrapped (resampled) from the fitted residuals
instead of assumed Gaussian, so fat tails are kept.

How the fit works:
  1. f = robust medians (medians are not pulled up by spikes)
  2. relative deviation z = (p - f) / g; anything above k * MAD is "spike";
     the spike part (in $) is cut into episodes = runs of consecutive spike hours
  3. spike-free price q = f + g * (clipped z)
  4. daily level R_d = sum(q) / sum(f) over the day;  D_d = log(R_d); fit AR(1)
  5. hourly part H = (q - R_d * f) / (R_d * g); fit AR(1)

Known limitations (put these in the report):
  - years are independent: no trends in gas prices, demand or battery buildout
    (more batteries on the grid -> smaller spreads in the future)
  - spike episodes are resampled from history, so the model never produces
    a spike shape that was not seen before (only new combinations)
  - spike timing depends only on quarter and hour, not on weather or load
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def _calendar_keys(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"quarter": index.quarter, "weekend": (index.dayofweek >= 5).astype(int),
                         "hour": index.hour}, index=index)


def _ar1(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares AR(1) coefficient (no intercept) and residuals."""
    x0, x1 = x[:-1], x[1:]
    phi = float(np.dot(x0, x1) / np.dot(x0, x0))
    return phi, x1 - phi * x0


def _episodes(s: np.ndarray, sign: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start positions and shapes of runs where sign*s > 0."""
    on = (sign * s) > 0
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False]) + 1
    return starts, [s[a:b].copy() for a, b in zip(starts, ends)]


@dataclass
class PriceModel:
    seasonal: pd.Series = field(repr=False)                 # by (quarter, weekend, hour)
    scale_floor: float = 1.0                                # minimum of g, $/MWh
    mu_d: float = 0.0
    phi_d: float = 0.0
    phi_h: float = 0.0
    resid_d: np.ndarray = field(default=None, repr=False)
    resid_h: np.ndarray = field(default=None, repr=False)
    h_bounds: tuple = (-1.0, 1.0)                           # range of H seen in the data
    spike_threshold: float = 0.0                            # relative to f
    onset_prob_up: pd.Series = field(default=None, repr=False)   # by (quarter, hour)
    onset_prob_down: pd.Series = field(default=None, repr=False)
    episodes_up: list = field(default_factory=list, repr=False)
    episodes_down: list = field(default_factory=list, repr=False)
    price_floor: float = config.PRICE_FLOOR
    price_cap: float = config.PRICE_CAP

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, prices: pd.Series, k_mad: float = 5.0,
            max_episode_hours: int | None = None,
            lower_quantile: float = 0.005) -> "PriceModel":
        """Fit the model to an hourly price series (whole days).

        k_mad              spike threshold in robust standard deviations
        max_episode_hours  drop spike episodes longer than this from the library
                           (e.g. 48 to test the value WITHOUT a Uri-type event)
        lower_quantile     low quantile used as the floor of the normal part
        """
        if len(prices) % 24:
            raise ValueError("prices must contain whole days")
        keys = _calendar_keys(prices.index)
        seasonal = prices.groupby([keys.quarter, keys.weekend, keys.hour]).median()
        seasonal.index.names = ["quarter", "weekend", "hour"]
        f = seasonal.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        scale_floor = max(0.25 * float(np.median(np.abs(seasonal))), 1.0)
        g = np.maximum(f, scale_floor)

        # 1-2. spikes
        z = (prices.to_numpy() - f) / g
        mad = np.median(np.abs(z - np.median(z))) * 1.4826     # robust std estimate
        thr = k_mad * mad
        low = max(-thr, float(np.quantile(z, lower_quantile)))
        zc = np.clip(z, low, thr)
        s = (z - zc) * g                                        # spike part, $/MWh

        # 3-4. daily level (log)
        q = f + g * zc                                          # spike-free price
        fd = f.reshape(-1, 24).sum(1)
        if np.any(fd <= 0):
            raise ValueError("a day has a non-positive seasonal level; this model needs positive daily levels")
        ratio = np.maximum(q.reshape(-1, 24).sum(1) / fd, 0.05)
        D = np.log(ratio)
        mu_d = float(D.mean())
        phi_d, resid_d = _ar1(D - mu_d)

        # 5. hourly relative noise
        R = np.repeat(ratio, 24)
        H = (q - R * f) / (R * g)
        phi_h, resid_h = _ar1(H)
        h_bounds = (float(np.quantile(H, lower_quantile)), float(np.quantile(H, 1 - lower_quantile)))

        starts_up, ep_up = _episodes(s, +1)
        starts_dn, ep_dn = _episodes(s, -1)
        if max_episode_hours is not None:
            keep = [len(e) <= max_episode_hours for e in ep_up]
            starts_up = starts_up[keep]
            ep_up = [e for e, k in zip(ep_up, keep) if k]

        def onset_table(starts):
            mask = np.zeros(len(s), bool)
            mask[starts] = True
            df = pd.DataFrame({"quarter": keys.quarter.to_numpy(),
                               "hour": keys.hour.to_numpy(), "m": mask})
            g = df.groupby(["quarter", "hour"])["m"]
            prior = mask.mean()
            return (g.sum() + 0.5 * prior) / (g.count() + 0.5)   # light smoothing

        return cls(seasonal=seasonal, scale_floor=scale_floor, mu_d=mu_d, phi_d=phi_d, phi_h=phi_h,
                   resid_d=resid_d, resid_h=resid_h, h_bounds=h_bounds,
                   spike_threshold=thr,
                   onset_prob_up=onset_table(starts_up), onset_prob_down=onset_table(starts_dn),
                   episodes_up=ep_up, episodes_down=ep_dn)

    # ------------------------------------------------------------------ simulate
    def simulate(self, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
        n = len(index)
        if n % 24:
            raise ValueError("index must contain whole days")
        keys = _calendar_keys(index)
        f = self.seasonal.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        f = np.where(np.isnan(f), np.nanmedian(f), f)
        g = np.maximum(f, self.scale_floor)

        # daily log level and hourly noise, both AR(1) with bootstrapped shocks
        e_d = rng.choice(self.resid_d, size=n // 24)
        u_h = rng.choice(self.resid_h, size=n)
        D = np.empty(n // 24)
        H = np.empty(n)
        prev = 0.0
        for i, e in enumerate(e_d):
            prev = self.phi_d * prev + e
            D[i] = prev
        prev = 0.0
        for t, u in enumerate(u_h):
            prev = self.phi_h * prev + u
            H[t] = prev
        H = np.clip(H, *self.h_bounds)
        normal = np.repeat(np.exp(self.mu_d + D), 24) * (f + g * H)

        # spike episodes
        s = np.zeros(n)
        qh = pd.MultiIndex.from_arrays([keys.quarter, keys.hour])
        for probs, library in ((self.onset_prob_up, self.episodes_up),
                               (self.onset_prob_down, self.episodes_down)):
            if not library:
                continue
            lam = probs.reindex(qh).fillna(0).to_numpy()
            for t in np.flatnonzero(rng.random(n) < lam):
                ep = library[rng.integers(len(library))]
                m = min(len(ep), n - t)
                s[t:t + m] += ep[:m]

        p = np.clip(normal + s, self.price_floor, self.price_cap)
        return pd.Series(p, index=index, name="price")

    def simulate_years(self, n_years: int, seed: int = config.SEED, year: int = 2023):
        """Yield n_years independent simulated years (365-day calendar of `year`)."""
        rng = np.random.default_rng(seed)
        index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="1h",
                              tz="Etc/GMT+6", inclusive="left")
        for _ in range(n_years):
            yield self.simulate(index, rng)

    def describe(self) -> pd.Series:
        hl = lambda phi: np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
        lengths = [len(e) for e in self.episodes_up]
        return pd.Series({
            "mu_d (mean log daily level)": self.mu_d,
            "phi_d (daily persistence)": self.phi_d,
            "half_life_daily_level_days": hl(self.phi_d),
            "phi_h (hourly mean reversion)": self.phi_h,
            "half_life_hourly_noise_hours": hl(self.phi_h),
            "spike_threshold (relative to seasonal)": self.spike_threshold,
            "up_episodes_per_year": self.onset_prob_up.mean() * 8760,
            "down_episodes_per_year": self.onset_prob_down.mean() * 8760,
            "median_up_episode_hours": np.median(lengths) if lengths else np.nan,
            "longest_up_episode_hours": max(lengths) if lengths else 0,
            "largest_up_excess_$/MWh": max((e.max() for e in self.episodes_up), default=np.nan),
        })
