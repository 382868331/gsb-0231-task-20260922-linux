"""定向结点电阻网络等效消元库（精确有理，Python 标准库）。

公开接口见 :class:`resistor_elim.Network` 与
:func:`resistor_elim.schur_eliminate`。
"""

from .errors import (
    NetworkError,
    NodeError,
    ConductanceError,
    StayWeightError,
    ResistanceError,
    ZeroResistanceError,
    PortError,
    PotentialError,
    LimitError,
    SingularSystemError,
)
from .network import Network
from .reduction import (
    schur_eliminate,
    Reduction,
    ComponentReduction,
    SchurRecord,
    Solution,
)
from .random_walk import HittingTimeResult, expected_hitting_times

__all__ = [
    "Network",
    "schur_eliminate",
    "Reduction",
    "ComponentReduction",
    "SchurRecord",
    "Solution",
    "HittingTimeResult",
    "expected_hitting_times",
    "NetworkError",
    "NodeError",
    "ConductanceError",
    "StayWeightError",
    "ResistanceError",
    "ZeroResistanceError",
    "PortError",
    "PotentialError",
    "LimitError",
    "SingularSystemError",
]

__version__ = "1.0.0"
