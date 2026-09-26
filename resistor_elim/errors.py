"""错误类型层级。

所有库抛出的异常都继承 :class:`NetworkError`；``node`` / ``edge``
属性携带可定位信息（节点标签或 ``(u, v)`` 边端点），便于定位非法输入。
"""

from __future__ import annotations

from typing import Any


class NetworkError(ValueError):
    """本库所有输入/求解错误的基类。"""

    def __init__(self, message: str, *, node: Any = None, edge: Any = None):
        super().__init__(message)
        self.node = node
        self.edge = edge


class NodeError(NetworkError):
    """节点标签非法（未注册、缺失、重复等）。"""


class ConductanceError(NetworkError):
    """电导值非法（非正、NaN/Infinity、类型不符）。"""


class ResistanceError(NetworkError):
    """电阻值非法（非正、NaN/Infinity、类型不符）。"""


class ZeroResistanceError(ResistanceError):
    """明确拒绝零电阻：它不在本库实现的无源正电导子集内。"""


class PortError(NetworkError):
    """端口集合非法（空集、未知节点、重复等）。"""


class TargetError(PortError):
    """随机游走目标集合非法（空集、未知节点、重复等）。"""


class StayWeightError(NetworkError):
    """节点停留权重非法（负数、NaN/Infinity、类型不符）。"""


class PotentialError(NetworkError):
    """给定的端口电势非法（长度/键不符、NaN/Infinity 等）。"""


class LimitError(NetworkError):
    """超过接口上限（节点数 ``MAX_NODES``）。"""


class SingularSystemError(NetworkError):
    """线性系统奇异（例如孤立内部节点没有任何电导相连）。

    本库不会静默丢弃或“任选地掩盖”导致奇异性的组件/节点。
    """
