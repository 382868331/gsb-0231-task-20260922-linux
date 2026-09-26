r"""加权随机游走的期望首次到达步数（Poisson 方程，精确有理）。

把网络的正电导归一成每个节点的转移概率：节点 ``u`` 上

    p(u -> v) = g_uv / (stay_u + \sum_w g_uw)   （v 为 u 的邻居）
    p(u -> u) = stay_u / (stay_u + \sum_w g_uw) （停留/自环概率）

其中 ``stay_u`` 是 :meth:`Network.set_stay_weight` 设置的非负停留权重
（默认 0）。平行边电导先相加，因此平行边自然合并为同一邻居的概率质量。

目标集合 ``targets`` 吸收（首达目标即停止，h_target = 0）。对其余节点，
期望首达步数 ``h`` 满足 Poisson 方程（**不是**把有效电阻直接当时间）：

    h_u = 1 + \sum_v p(u -> v) h_v   即   (I - P) h = 1

对每个含目标的连通组件，在暂态节点上精确求解该线性系统；与目标集合
不连通的组件无法到达目标，期望为正无穷（``math.inf``），**不**作为
线性求解失败处理，也不抛奇异错误。

结果附带逐节点方程残差与连通性诊断，全部为精确 ``Fraction``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Hashable, Iterable

from .errors import TargetError
from .linalg import Mat
from .network import Network

#: 不可达目标的期望首达步数哨兵值（正无穷）。
INFINITY = math.inf


@dataclass
class HittingTimeResult:
    """期望首次到达步数的求解结果（精确有理；不可达为 ``math.inf``）。"""

    network: Network
    #: 吸收目标集合（按用户给定顺序）
    targets: tuple[Hashable, ...]
    #: 每个节点的期望首达步数；目标为 0，不可达节点为 math.inf
    expectations: dict[Hashable, Any]
    #: Poisson 方程残差 r_u = h_u - 1 - sum_v p(u->v) h_v；
    #: 目标节点与不可达节点无方程，残差为 None；其余精确为 0
    residuals: dict[Hashable, Fraction | None]
    #: 与目标连通的非目标节点（求解了方程的暂态节点）
    reachable_transient: tuple[Hashable, ...]
    #: 与目标不连通的节点（期望为 inf）
    unreachable: tuple[Hashable, ...]
    #: 每个含目标组件的 (组件标签元组, 其中目标元组)，用于连通性诊断
    target_components: tuple[tuple[tuple[Hashable, ...], tuple[Hashable, ...]], ...] = field(
        default_factory=tuple
    )

    def expectation(self, node: Hashable) -> Any:
        """单个起点的期望首达步数（不可达返回 ``math.inf``）。"""
        self.network.index(node)
        return self.expectations[node]

    def is_reachable(self, node: Hashable) -> bool:
        self.network.index(node)
        return self.expectations[node] != INFINITY


def _validate_targets(network: Network, targets: Iterable[Hashable]) -> list[Hashable]:
    if isinstance(targets, (str, bytes)):
        raise TargetError("targets 必须是标签序列（单目标也请写成 [label]）")
    try:
        target_list = list(targets)
    except TypeError as exc:
        raise TargetError("targets 必须是节点标签的可迭代对象") from exc
    if not target_list:
        raise TargetError("目标集合不能为空")
    seen: set[Hashable] = set()
    for t in target_list:
        if isinstance(t, bool):
            raise TargetError("目标标签不得为 bool", node=t)
        if t in seen:
            raise TargetError(f"目标重复：{t!r}", node=t)
        seen.add(t)
        network.index(t)  # 未注册 → NodeError（可定位）
    return target_list


def _transition_denominators(network: Network) -> dict[Hashable, Fraction]:
    stay = network.stay_weights()
    denom: dict[Hashable, Fraction] = {}
    for u in network.nodes:
        denom[u] = stay[u] + sum(
            (g for g in network.neighbors(u).values()), Fraction(0)
        )
    return denom


def hitting_times(
    network: Network,
    targets: Iterable[Hashable],
) -> HittingTimeResult:
    """计算每个起点到目标集合的期望首次到达步数。

    参数
    ----
    network:
        :class:`Network`（正电导子集；可用
        :meth:`Network.set_stay_weight` 标注非负停留权重）。
    targets:
        吸收目标节点标签序列（非空、无重复、均已注册）。

    返回
    ----
    HittingTimeResult
        ``expectations`` 给出每个起点的期望步数（精确 ``Fraction``；
        与目标不连通的节点为 ``math.inf``），``residuals`` 给出 Poisson
        方程逐节点残差（精确为 0；目标/不可达节点为 ``None``），
        并附连通性诊断。本函数不修改 ``network``。
    """
    if not isinstance(network, Network):
        raise TypeError("network 必须是 resistor_elim.Network 实例")
    target_list = _validate_targets(network, targets)
    target_set = set(target_list)
    denom = _transition_denominators(network)
    stay = network.stay_weights()

    expectations: dict[Hashable, Any] = {}
    residuals: dict[Hashable, Fraction | None] = {}
    reachable: list[Hashable] = []
    unreachable: list[Hashable] = []
    target_components: list[tuple[tuple[Hashable, ...], tuple[Hashable, ...]]] = []

    for comp in network.components():
        comp_targets = tuple(t for t in target_list if t in comp)
        if not comp_targets:
            # 与目标不连通的组件：期望正无穷，不是求解失败。
            for x in comp:
                expectations[x] = INFINITY
                residuals[x] = None
                unreachable.append(x)
            continue

        target_components.append((comp, comp_targets))
        transient = [x for x in comp if x not in target_set]
        for t in comp_targets:
            expectations[t] = Fraction(0)
            residuals[t] = None

        if transient:
            n = len(transient)
            pos = {x: i for i, x in enumerate(transient)}
            a = Mat.zeros(n, n)
            b = [Fraction(1) for _ in range(n)]
            for x in transient:
                i = pos[x]
                d = denom[x]
                # d > 0 恒成立：组件含目标且连通，暂态节点必有边或停留权重
                a[i][i] = Fraction(1) - stay[x] / d
                for y, g in network.neighbors(x).items():
                    if y in pos:
                        a[i][pos[y]] -= g / d
            h = Mat.solve(a, b)
            for x, hx in zip(transient, h):
                expectations[x] = hx
                reachable.append(x)

        # 逐节点 Poisson 方程残差（独立回代验证，精确为 0）
        for x in transient:
            d = denom[x]
            expect_next = stay[x] / d * expectations[x]
            for y, g in network.neighbors(x).items():
                expect_next += g / d * expectations[y]
            residuals[x] = expectations[x] - 1 - expect_next

    return HittingTimeResult(
        network=network,
        targets=tuple(target_list),
        expectations=expectations,
        residuals=residuals,
        reachable_transient=tuple(reachable),
        unreachable=tuple(unreachable),
        target_components=tuple(target_components),
    )
