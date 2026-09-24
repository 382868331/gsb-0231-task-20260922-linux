"""定向结点网络等效消元库（精确有理，Python 标准库）。

电阻（直流/静态）接口见 :class:`resistor_elim.Network` 与
:func:`resistor_elim.schur_eliminate`；R/L/C 正弦稳态相量接口见
:func:`resistor_elim.ac_schur_eliminate`。
"""

from .errors import (
    NetworkError,
    NodeError,
    ConductanceError,
    ResistanceError,
    ZeroResistanceError,
    CapacitanceError,
    InductanceError,
    FrequencyError,
    PortError,
    PotentialError,
    LimitError,
    SingularSystemError,
)
from .linalg import CFrac
from .network import Network
from .reduction import (
    schur_eliminate,
    Reduction,
    ComponentReduction,
    SchurRecord,
    Solution,
)
from .ac import (
    ac_schur_eliminate,
    ACReduction,
    ACComponentReduction,
    ACSchurRecord,
    ACSolution,
)

__all__ = [
    "Network",
    "schur_eliminate",
    "Reduction",
    "ComponentReduction",
    "SchurRecord",
    "Solution",
    "ac_schur_eliminate",
    "ACReduction",
    "ACComponentReduction",
    "ACSchurRecord",
    "ACSolution",
    "CFrac",
    "NetworkError",
    "NodeError",
    "ConductanceError",
    "ResistanceError",
    "ZeroResistanceError",
    "CapacitanceError",
    "InductanceError",
    "FrequencyError",
    "PortError",
    "PotentialError",
    "LimitError",
    "SingularSystemError",
]

__version__ = "1.1.0"
