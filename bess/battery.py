"""Physical model of a grid-scale battery."""

from dataclasses import dataclass, replace
import math

from . import config


@dataclass(frozen=True)
class BatterySpec:
    """Battery parameters.

    power_mw         max charge / discharge power at the grid connection (MW)
    energy_mwh       nameplate energy capacity (MWh)
    round_trip_eff   energy out / energy in over a full cycle (0-1)
    soc_min_frac     lowest allowed state of charge, as a fraction of energy_mwh
    soc_max_frac     highest allowed state of charge
    soc_start_frac   state of charge at the start (and end) of each optimization block
    degradation_cost wear cost per MWh discharged ($/MWh). A simple way to make
                     the optimizer skip trades that earn less than they wear
                     out the battery.
    """

    power_mw: float = config.POWER_MW
    energy_mwh: float = config.POWER_MW * config.DURATION_H
    round_trip_eff: float = config.ROUND_TRIP_EFF
    soc_min_frac: float = config.SOC_MIN_FRAC
    soc_max_frac: float = config.SOC_MAX_FRAC
    soc_start_frac: float = config.SOC_START_FRAC
    degradation_cost: float = config.DEGRADATION_COST

    def __post_init__(self):
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ValueError("power_mw and energy_mwh must be positive")
        if not 0 < self.round_trip_eff <= 1:
            raise ValueError("round_trip_eff must be in (0, 1]")
        if not 0 <= self.soc_min_frac <= self.soc_start_frac <= self.soc_max_frac <= 1:
            raise ValueError("need 0 <= soc_min <= soc_start <= soc_max <= 1")

    # Split round-trip losses evenly between charging and discharging.
    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip_eff)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_eff)

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_start(self) -> float:
        return self.soc_start_frac * self.energy_mwh

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw

    def with_duration(self, hours: float) -> "BatterySpec":
        """Same power, different duration (used for sensitivity runs)."""
        return replace(self, energy_mwh=self.power_mw * hours)

    def scaled(self, capacity_factor: float) -> "BatterySpec":
        """Battery after capacity fade (energy scaled, power unchanged)."""
        return replace(self, energy_mwh=self.energy_mwh * capacity_factor)
