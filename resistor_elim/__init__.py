"""定向结点网络：电阻等效消元（直流，精确有理）与 RLC 相量分析（交流）。

直流接口见 :class:`resistor_elim.Network` 与
:func:`resistor_elim.schur_eliminate`；交流接口见
:func:`resistor_elim.ac_admittance`（给定非零有限角频率求端口复导纳）。
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
from .network import Network
from .reduction import (
    schur_eliminate,
    Reduction,
    ComponentReduction,
    SchurRecord,
    Solution,
)
from .cx import QComplex
from .ac import (
    ac_admittance,
    AdmittanceResult,
    ACComponentReduction,
    ACSchurRecord,
    ACSolution,
)

__all__ = [
    "Network",
    "schur_eliminate",
    "ac_admittance",
    "Reduction",
    "ComponentReduction",
    "SchurRecord",
    "Solution",
    "AdmittanceResult",
    "ACComponentReduction",
    "ACSchurRecord",
    "ACSolution",
    "QComplex",
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
