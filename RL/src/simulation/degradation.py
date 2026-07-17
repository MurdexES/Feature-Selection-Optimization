"""Physical degradation and failure-risk formulas used by ``PumpModel``.

Kept separate from ``pump_model.py`` so the *shape* of these relationships
(how load, age, and mechanical condition drive health loss and failure
risk) can be read, configured, and unit-tested independently of the
stateful simulator that consumes them.

Every constant below is a **simulation assumption** chosen so that an
unmaintained pump running at high load fails after roughly 2000-2500
simulated hours -- a convenient timescale for training an agent, not a
measurement from a real ESP. Treat the defaults as a starting point to
tune, not as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationConfig:
    """Tunable constants behind the per-operating-hour health-loss formula.

    The health lost in one hour is::

        degradation =
            base_rate
            + load_coefficient * load ** 2
            + age_coefficient * min(hours_since_maintenance / age_normalization_hours, age_cap)
            + condition_coefficient * condition_severity
            + max(0, Normal(0, stochastic_std))

    ``load`` enters squared so that running hard (load close to 1.0) costs
    disproportionately more health than running gently -- this is what
    makes "operate at reduced load" a genuine trade-off rather than a
    strictly dominated action.
    """

    base_rate: float = 0.00008
    load_coefficient: float = 0.00018
    age_coefficient: float = 0.00005
    age_normalization_hours: float = 3000.0
    age_cap: float = 1.5
    condition_coefficient: float = 0.00006
    stochastic_std: float = 0.000025


def calculate_degradation(
    config: DegradationConfig,
    load: float,
    hours_since_maintenance: float,
    condition_severity: float,
    rng: np.random.Generator,
) -> float:
    """Health lost during one operating hour at the given load/condition."""

    load_degradation = config.load_coefficient * load**2

    age_factor = min(
        hours_since_maintenance / config.age_normalization_hours,
        config.age_cap,
    )
    age_degradation = config.age_coefficient * age_factor

    condition_degradation = config.condition_coefficient * condition_severity

    # Clipped at zero: degradation should never *heal* the pump on its own.
    stochastic_degradation = max(0.0, rng.normal(0.0, config.stochastic_std))

    return (
        config.base_rate
        + load_degradation
        + age_degradation
        + condition_degradation
        + stochastic_degradation
    )


@dataclass(frozen=True)
class FailureRiskConfig:
    """Tunable constants behind the per-operating-hour failure probability.

    Uses a logistic ("sigmoid") curve in health: failure probability stays
    close to zero while ``health`` is comfortably above ``risk_midpoint``,
    then rises sharply as health drops below it.
    """

    base_probability: float = 0.035
    risk_midpoint: float = 0.22
    risk_steepness: float = 18.0
    condition_weight: float = 0.8
    max_probability: float = 0.65


def calculate_failure_probability(
    config: FailureRiskConfig,
    health: float,
    load: float,
    condition_severity: float,
) -> float:
    """Probability the pump fails during the current operating hour."""

    health_risk = 1.0 / (
        1.0 + np.exp(config.risk_steepness * (health - config.risk_midpoint))
    )

    load_multiplier = 0.5 + load**2
    condition_multiplier = 1.0 + config.condition_weight * condition_severity

    probability = (
        config.base_probability
        * health_risk
        * load_multiplier
        * condition_multiplier
    )

    return float(np.clip(probability, 0.0, config.max_probability))
