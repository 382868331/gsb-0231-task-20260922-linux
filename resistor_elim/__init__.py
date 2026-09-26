"""定向结点电阻网络等效消元库（精确有理，Python 标准库）。

公开接口见 :class:`resistor_elim.Network` 与
:func:`resistor_elim.schur_eliminate`。
"""

from .errors import (
    NetworkError,
    NodeError,
    ConductanceError,
    ResistanceError,
    ZeroResistanceError,
    PortError,
    PotentialError,
    LimitError,
    SingularSystemError,
)
from .errors import StayWeightError, TargetError
from .hitting import INFINITY, HittingTimeResult, hitting_times
from .network import Network
from .reduction import (
    schur_eliminate,
    Reduction,
    ComponentReduction,
    SchurRecord,
    Solution,
)

__all__ = [
    "Network",
    "schur_eliminate",
    "Reduction",
    "ComponentReduction",
    "SchurRecord",
    "Solution",
    "NetworkError",
    "NodeError",
    "ConductanceError",
    "ResistanceError",
    "ZeroResistanceError",
    "PortError",
    "PotentialError",
    "LimitError",
    "SingularSystemError",
    "StayWeightError",
    "TargetError",
    "hitting_times",
    "HittingTimeResult",
    "INFINITY",
]

__version__ = "1.0.0"
