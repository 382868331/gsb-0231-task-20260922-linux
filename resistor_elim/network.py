"""无源电阻网络的数据模型：校验、拉普拉斯矩阵、连通组件。

数值约定
--------
* 电导/电阻一律使用精确有理数（``int`` 或 ``fractions.Fraction``）。
  不接受 ``float``（含 ``NaN``/``Infinity``），避免二进制浮点被隐式当成
  精确值；端口电势允许有限 ``float``，按 ``Fraction(x)`` 精确转换，
  ``NaN``/正负 ``Infinity`` 被拒绝。
* ``bool`` 是 ``int`` 的子类，但在任何“整数参数/数值参数”位置都被拒绝。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Hashable, Iterable

from .errors import (
    ConductanceError,
    LimitError,
    NodeError,
    ResistanceError,
    ZeroResistanceError,
)

#: 接口硬上限：网络节点数 ≤ 30。
MAX_NODES = 30


def reject_bool(value: Any, what: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{what} 不得为 bool（请显式使用 int）")


def exact_scalar(value: Any, what: str, error_cls: type[Exception]) -> Fraction:
    """把电导/电阻等核心数值转为严格正/待判定的 ``Fraction``。

    接受 ``int``（拒绝 ``bool``）与 ``Fraction``；``float`` 一律拒绝，
    其中 ``NaN``/``Infinity`` 给出专门提示。
    """
    reject_bool(value, what)
    if isinstance(value, float):
        if math.isnan(value):
            raise error_cls(f"{what} 不得为 NaN")
        if math.isinf(value):
            raise error_cls(f"{what} 不得为 Infinity")
        raise error_cls(
            f"{what} 必须是 int 或 fractions.Fraction（精确有理子集），"
            f"不接受 float：收到 {value!r}；请改用 Fraction(...) 或整数"
        )
    if isinstance(value, Fraction):
        f = value
    elif isinstance(value, int):
        f = Fraction(value)
    else:
        raise error_cls(f"{what} 必须是 int 或 Fraction，收到 {type(value).__name__}")
    return f


def potential_scalar(value: Any, what: str) -> Fraction:
    """端口电势：接受 int/Fraction 与**有限** float（精确转换）。"""
    reject_bool(value, what)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"{what} 不得为 NaN")
        if math.isinf(value):
            raise ValueError(f"{what} 不得为 Infinity")
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise ValueError(
        f"{what} 必须是 int、float（有限）或 Fraction，收到 {type(value).__name__}"
    )


class Network:
    """无源自电导电阻网络。

    参数
    ----
    nodes:
        节点标签序列（任意可哈希、非 ``bool``、互不相同），长度 1..30。
        边只能引用已注册节点——拼错的标签会被直接定位为错误，而不是
        隐式新建节点。

    平行边允许：同一对节点之间多次添加电导会相加（基尔霍夫并联）。
    自环、非正电导、零电阻被明确拒绝。
    """

    def __init__(self, nodes: Iterable[Hashable]):
        labels: list[Hashable] = []
        seen: set[Hashable] = set()
        for label in nodes:
            if isinstance(label, bool) or not isinstance(label, Hashable):
                raise NodeError(f"节点标签必须是可哈希且非 bool：{label!r}", node=label)
            if label in seen:
                raise NodeError(f"节点标签重复：{label!r}", node=label)
            seen.add(label)
            labels.append(label)
        if not labels:
            raise NodeError("网络至少需要 1 个节点")
        if len(labels) > MAX_NODES:
            raise LimitError(
                f"节点数 {len(labels)} 超过上限 {MAX_NODES}"
            )
        self._nodes: tuple[Hashable, ...] = tuple(labels)
        self._index: dict[Hashable, int] = {v: i for i, v in enumerate(labels)}
        # 邻接电导（平行边求和）
        self._adj: dict[Hashable, dict[Hashable, Fraction]] = {v: {} for v in labels}
        # 每条添加记录：(u, v, conductance)，便于逐支路核对
        self._branches: list[tuple[Hashable, Hashable, Fraction]] = []

    # ------------------------------------------------------------------ #
    @property
    def nodes(self) -> tuple[Hashable, ...]:
        return self._nodes

    @property
    def branches(self) -> tuple[tuple[Hashable, Hashable, Fraction], ...]:
        return tuple(self._branches)

    def __len__(self) -> int:
        return len(self._nodes)

    def index(self, label: Hashable) -> int:
        try:
            return self._index[label]
        except (KeyError, TypeError) as exc:
            raise NodeError(f"未注册的节点标签：{label!r}", node=label) from exc

    # ------------------------------------------------------------------ #
    def add_conductor(self, u: Hashable, v: Hashable, conductance: Any) -> None:
        """添加一条电导为 ``conductance`` 的电阻支路（要求严格正）。"""
        iu = self.index(u)
        iv = self.index(v)
        if iu == iv:
            raise ConductanceError("不允许自环支路（u == v）", edge=(u, v))
        g = exact_scalar(conductance, f"电导 ({u!r}, {v!r})", ConductanceError)
        if g <= 0:
            raise ConductanceError(
                f"电导必须严格为正，收到 {g}（零电导对应无限大电阻，"
                f"请直接省略该支路）",
                edge=(u, v),
            )
        self._adj[u][v] = self._adj[u].get(v, Fraction(0)) + g
        self._adj[v][u] = self._adj[v].get(u, Fraction(0)) + g
        self._branches.append((u, v, g))

    def add_resistor(self, u: Hashable, v: Hashable, resistance: Any) -> None:
        """添加一条电阻为 ``resistance`` 的支路（要求严格正；零电阻被拒绝）。"""
        # 先做节点定位校验，再做数值语义，错误都可定位。
        self.index(u)
        self.index(v)
        r = exact_scalar(resistance, f"电阻 ({u!r}, {v!r})", ResistanceError)
        if r == 0:
            raise ZeroResistanceError(
                "零电阻（理想短路线）不在本库实现的正电导无源子集内："
                "它使结点电势定义退化；请先在网络定义中合并等势节点",
                edge=(u, v),
            )
        if r < 0:
            raise ResistanceError(f"电阻必须严格为正，收到 {r}", edge=(u, v))
        self.add_conductor(u, v, Fraction(1) / r)

    # ------------------------------------------------------------------ #
    def laplacian(self) -> list[list[Fraction]]:
        """返回结点拉普拉斯矩阵 ``K``（``Fraction``，按节点顺序索引）。

        ``K[i,i]`` 为结点 ``i`` 上所有电导之和；``K[i,j] = -g_ij``。
        """
        n = len(self._nodes)
        k = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for u, neighbors in self._adj.items():
            iu = self._index[u]
            for v, g in neighbors.items():
                iv = self._index[v]
                k[iu][iv] -= g
                k[iu][iu] += g
        return k

    def conductance_between(self, u: Hashable, v: Hashable) -> Fraction:
        self.index(u)
        self.index(v)
        return self._adj[u].get(v, Fraction(0))

    def neighbors(self, u: Hashable) -> dict[Hashable, Fraction]:
        self.index(u)
        return dict(self._adj[u])

    def components(self) -> list[tuple[Hashable, ...]]:
        """返回无向连通组件（按各自最小节点次序排序的标签元组列表）。

        孤立节点自身构成一个组件并显式出现，不会被丢弃。
        """
        seen: set[Hashable] = set()
        result: list[tuple[Hashable, ...]] = []
        for start in self._nodes:
            if start in seen:
                continue
            stack = [start]
            comp: list[Hashable] = []
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in self._adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            order = {label: i for i, label in enumerate(self._nodes)}
            comp.sort(key=lambda label: order[label])
            result.append(tuple(comp))
        return result
