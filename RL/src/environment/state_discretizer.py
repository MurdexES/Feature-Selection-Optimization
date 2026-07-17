"""Discretize the continuous observable state for tabular Q-learning.

A Q-table needs a finite, hashable state. This module maps an
``ObservableState`` (continuous sensor floats, counters) onto a small
tuple of bin indices.

Discretization is a lossy compression, and the choice of bins is the
whole game: too few and states that need different actions collapse into
one cell (the agent literally cannot represent the right policy); too
many and each cell is visited too rarely to learn its values. We keep
only the features that carry maintenance-relevant signal, and place bin
edges where the *decision* should change, not at uniform intervals:

  * vibration edges (7.0, 9.5, 12.0): the first two are exactly the
    thresholds the sensor-threshold baseline uses, so the tabular agent
    can represent at least that policy; 12+ means "deep degradation".
  * maintenance-age edges every 500 h up to 2000 h: degradation and the
    age term in the failure model grow on this timescale.
  * inspected health edges (0.55, 0.78): match the simulator's RUBBING
    and UNBALANCE condition boundaries.

Total: 4 * 5 * 4 * 2 * 2 = 320 states -- small enough that every state
gets visited thousands of times in training, large enough to express a
sensible policy. DQN (next phase) removes this bottleneck entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.environment.operations import ObservableState

DiscreteState = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class DiscretizerConfig:
    vibration_edges: tuple[float, ...] = (7.0, 9.5, 12.0)
    maintenance_age_edges: tuple[float, ...] = (500, 1000, 1500, 2000)
    inspected_health_edges: tuple[float, ...] = (0.55, 0.78)
    fresh_inspection_hours: int = 168


@dataclass(frozen=True)
class StateDiscretizer:
    config: DiscretizerConfig = field(default_factory=DiscretizerConfig)

    def discretize(self, state: ObservableState) -> DiscreteState:
        vibration_bin = int(
            np.digitize(
                state.sensors.vibration, self.config.vibration_edges
            )
        )

        maintenance_age_bin = int(
            np.digitize(
                state.hours_since_maintenance,
                self.config.maintenance_age_edges,
            )
        )

        # 0 = no (valid) inspection on file; 1..3 = critical/degraded/healthy.
        if state.last_inspected_health is None:
            inspected_health_bin = 0
        else:
            inspected_health_bin = 1 + int(
                np.digitize(
                    state.last_inspected_health,
                    self.config.inspected_health_edges,
                )
            )

        inspection_is_fresh = (
            state.hours_since_inspection is not None
            and state.hours_since_inspection
            < self.config.fresh_inspection_hours
        )

        in_downtime = state.remaining_downtime > 0

        return (
            vibration_bin,
            maintenance_age_bin,
            inspected_health_bin,
            int(inspection_is_fresh),
            int(in_downtime),
        )

    @property
    def state_space_size(self) -> int:
        return (
            (len(self.config.vibration_edges) + 1)
            * (len(self.config.maintenance_age_edges) + 1)
            * (len(self.config.inspected_health_edges) + 2)
            * 2
            * 2
        )
