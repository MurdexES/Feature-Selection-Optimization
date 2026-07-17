"""The four decisions an agent (or baseline policy) can make each hour.

Defined in its own module because *everything* above the simulator speaks
this vocabulary: baseline policies return these, the Gymnasium environment
exposes them as its action space, and the trained agents choose among them.
"""

from enum import IntEnum


class MaintenanceAction(IntEnum):
    OPERATE_FULL = 0
    OPERATE_REDUCED = 1
    INSPECT = 2
    PREVENTIVE_MAINTENANCE = 3
